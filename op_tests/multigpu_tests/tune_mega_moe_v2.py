#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Targeted multi-rank configuration sweeps for MegaMoEV2.

The script keeps one operator instance alive while testing launch configurations,
so weights and MORI workspaces are reused exactly as they are in SGLang.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import torch
import torch.distributed as dist

from aiter.ops.flydsl.kernels.mega_moe import MegaMoEV2
from test_mega_moe_v2 import (
    NETWORKS,
    _barrier,
    _cleanup,
    _make_inputs,
    _quantize_weights,
    _reduce_float,
    _setup_dist,
    _time_graph,
)


def _parse_scalar(value: str):
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    return int(value)


def _parse_overrides(values: list[str]) -> dict[str, object]:
    result = {}
    for item in values:
        name, separator, value = item.partition("=")
        if not separator or not name:
            raise ValueError(f"override must be FIELD=VALUE, got {item!r}")
        result[name] = _parse_scalar(value)
    return result


def _apply_overrides(config, overrides):
    stage1, stage2 = {}, {}
    p2p_quant = config.p2p_quant
    for name, value in overrides.items():
        if name.startswith("stage1."):
            stage1[name.removeprefix("stage1.")] = value
        elif name.startswith("stage2."):
            stage2[name.removeprefix("stage2.")] = value
        elif name == "p2p_quant":
            p2p_quant = str(value)
        else:
            raise ValueError(f"unsupported override {name!r}")
    return replace(
        config,
        stage1=replace(config.stage1, **stage1),
        stage2=replace(config.stage2, **stage2),
        p2p_quant=p2p_quant,
    )


def _dedupe(candidates):
    result = []
    seen = set()
    for name, config, combine_geometry in candidates:
        key = (
            json.dumps(asdict(config), sort_keys=True),
            combine_geometry,
        )
        if key not in seen:
            result.append((name, config, combine_geometry))
            seen.add(key)
    return result


def _candidates(base, sweep, num_cu):
    candidates = [("base", base, None)]
    if sweep == "dispatch-cu":
        for value in (32, 64, 96, 128, 160, 192, 224):
            if value < num_cu and value % 8 == 0:
                candidates.append(
                    (
                        f"dispatch_cu={value}",
                        replace(
                            base,
                            stage1=replace(base.stage1, num_dispatch_cu=value),
                        ),
                        None,
                    )
                )
    elif sweep == "stage1-recipe":
        recipes = (
            ("n256w4-basic", 256, 4, 1, False, False, False),
            ("n256w4-resource", 256, 4, 1, False, False, True),
            ("n256w8", 256, 8, 1, True, True, False),
            ("n256w8-resource", 256, 8, 1, True, True, True),
            ("n512w8", 512, 8, 1, True, True, False),
            ("n512w8-resource", 512, 8, 1, True, True, True),
        )
        for name, tile_n, waves, grid_mult, mfma, async_copy, resource in recipes:
            for b_nt in (0, 3):
                candidates.append(
                    (
                        f"{name}-bnt{b_nt}",
                        replace(
                            base,
                            stage1=replace(
                                base.stage1,
                                tile_n=tile_n,
                                num_waves=waves,
                                grid_mult=grid_mult,
                                mfma_amajor=mfma,
                                async_a_copy=async_copy,
                                use_tile_resource=resource,
                                b_nt=b_nt,
                            ),
                        ),
                        None,
                    )
                )
    elif sweep == "grid":
        for value in (1, 2, 3, 4, 6, 8):
            candidates.append(
                (
                    f"grid_mult={value}",
                    replace(base, stage1=replace(base.stage1, grid_mult=value)),
                    None,
                )
            )
    elif sweep == "payload":
        recipes = (
            ("plain-ws1", 1, False, False, 0, False),
            ("plain-ws4", 4, False, False, 0, False),
            ("plain-ws8", 8, False, False, 0, False),
            ("chunk-ws1", 1, True, False, 384, False),
            ("chunk-ws4", 4, True, False, 384, False),
            ("chunk-ws8", 8, True, False, 384, False),
            ("ready-ws1", 1, True, False, 384, True),
            ("ready-ws4", 4, True, False, 384, True),
            ("ready-ws8", 8, True, False, 384, True),
        )
        for name, shards, grouping, counting, rows, tile_ready in recipes:
            candidates.append(
                (
                    name,
                    replace(
                        base,
                        stage1=replace(
                            base.stage1,
                            work_shards=shards,
                            external_grouping=grouping,
                            external_counting=counting,
                            payload_chunk_rows=rows,
                            payload_tile_ready=tile_ready,
                        ),
                    ),
                    None,
                )
            )
    elif sweep == "stage2":
        for block_n in (128, 256):
            candidates.append(
                (
                    f"stage2-bn{block_n}-nonpersist",
                    replace(
                        base,
                        stage2=replace(
                            base.stage2,
                            block_n=block_n,
                            persist=False,
                            persist_cu=0,
                        ),
                    ),
                    None,
                )
            )
            for persist_cu in (128, 192, 224, 240):
                if persist_cu <= num_cu:
                    candidates.append(
                        (
                            f"stage2-bn{block_n}-pcu{persist_cu}",
                            replace(
                                base,
                                stage2=replace(
                                    base.stage2,
                                    block_n=block_n,
                                    persist=True,
                                    persist_cu=persist_cu,
                                ),
                            ),
                            None,
                        )
                    )
    elif sweep == "combine":
        for block_num in (64, 80, 96, 128, 160, 192, 224, 240):
            if block_num <= num_cu:
                for warps in (4, 8, 16):
                    candidates.append(
                        (
                            f"combine-b{block_num}-w{warps}",
                            base,
                            (block_num, warps),
                        )
                    )
    else:
        raise ValueError(f"unknown sweep {sweep!r}")
    return _dedupe(candidates)


def _relative_l2(actual, expected, device):
    value = float(
        torch.linalg.vector_norm(actual.float() - expected.float())
        / torch.linalg.vector_norm(expected.float())
    )
    return _reduce_float(value, device, dist.ReduceOp.MAX)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", choices=NETWORKS, default="glm52")
    parser.add_argument("--tokens", default="1,4,6,12,24,48,96")
    parser.add_argument(
        "--sweep",
        choices=("dispatch-cu", "stage1-recipe", "grid", "payload", "stage2", "combine"),
        required=True,
    )
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--max-tok-per-rank", type=int, default=256)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Base config override, e.g. stage1.num_dispatch_cu=96",
    )
    parser.add_argument("--output-json")
    args = parser.parse_args()

    tokens_list = [int(value) for value in args.tokens.split(",")]
    if not tokens_list or min(tokens_list) <= 0:
        raise ValueError("--tokens must contain positive integers")
    if max(tokens_list) > args.max_tok_per_rank:
        raise ValueError("--max-tok-per-rank must cover --tokens")
    overrides = _parse_overrides(args.override)

    rank, world, device = _setup_dist()
    try:
        network = NETWORKS[args.network]
        if network["experts"] % world:
            raise ValueError("expert count must be divisible by world size")
        local_experts = network["experts"] // world
        packed = _quantize_weights(
            network["model_dim"],
            network["inter_dim"],
            local_experts,
            rank,
            args.seed,
            device,
        )
        w1, w1_scale, w2, w2_scale = packed[:4]
        x, route_weights, ids = _make_inputs(
            max(tokens_list),
            network["model_dim"],
            network["experts"],
            network["topk"],
            rank,
            args.seed,
            device,
        )
        moe = MegaMoEV2(
            rank=rank,
            world_size=world,
            quant="a8w4",
            w1=w1,
            w1_scale=w1_scale,
            w2=w2,
            w2_scale=w2_scale,
            max_tok_per_rank=args.max_tok_per_rank,
            **network,
        )
        moe.set_weights(w1, w1_scale, w2, w2_scale)
        num_cu = torch.cuda.get_device_properties(device).multi_processor_count
        original_select = moe._select_config
        results = []

        for tokens in tokens_list:
            local_x = x[:tokens].contiguous()
            local_weights = route_weights[:tokens].contiguous()
            local_ids = ids[:tokens].contiguous()
            x_q, x_scale = moe.quantize(local_x)
            default_config = original_select(tokens)
            base = _apply_overrides(default_config, overrides)

            def select_base(_tokens):
                moe._active_config = base
                return base

            moe._select_config = select_base
            expected = moe(local_x, local_weights, local_ids)[:tokens].clone()
            _barrier()

            token_results = []
            for name, config, combine_geometry in _candidates(base, args.sweep, num_cu):
                if combine_geometry is None:
                    moe.comb_cfg.combine_block_num = None
                    moe.comb_cfg.combine_warp_num_per_block = None
                else:
                    (
                        moe.comb_cfg.combine_block_num,
                        moe.comb_cfg.combine_warp_num_per_block,
                    ) = combine_geometry

                def select_candidate(_tokens, selected=config):
                    moe._active_config = selected
                    return selected

                moe._select_config = select_candidate

                if args.sweep in {"dispatch-cu", "stage1-recipe", "grid", "payload"}:

                    def phase():
                        moe._run_fused_stage1(
                            x_q,
                            local_weights,
                            x_scale,
                            local_ids,
                            config=config.stage1,
                        )

                else:
                    moe._run_fused_stage1(
                        x_q,
                        local_weights,
                        x_scale,
                        local_ids,
                        config=config.stage1,
                    )
                    _barrier()

                    def phase():
                        moe._run_stage2(tokens, None, True, config)

                phase_ms = _time_graph(phase, device, args.iters)

                def end_to_end():
                    return moe(local_x, local_weights, local_ids)

                e2e_ms = _time_graph(end_to_end, device, args.iters)
                actual = moe(local_x, local_weights, local_ids)[:tokens]
                _barrier()
                rel_l2 = _relative_l2(actual, expected, device)
                row = {
                    "tokens": tokens,
                    "name": name,
                    "phase_mean_us": phase_ms[0] * 1000,
                    "phase_max_us": phase_ms[1] * 1000,
                    "e2e_mean_us": e2e_ms[0] * 1000,
                    "e2e_max_us": e2e_ms[1] * 1000,
                    "relative_l2": rel_l2,
                    "config": asdict(config),
                    "combine_geometry": combine_geometry,
                }
                token_results.append(row)
                if rank == 0:
                    print("[TUNE] " + json.dumps(row, sort_keys=True), flush=True)

            best = min(token_results, key=lambda row: row["e2e_max_us"])
            base_row = next(row for row in token_results if row["name"] == "base")
            for row in token_results:
                row["e2e_max_speedup_vs_base"] = (
                    base_row["e2e_max_us"] / row["e2e_max_us"]
                )
            results.extend(token_results)
            if rank == 0:
                print(
                    "[BEST] "
                    + json.dumps(
                        {
                            "tokens": tokens,
                            "name": best["name"],
                            "e2e_max_us": best["e2e_max_us"],
                            "base_e2e_max_us": base_row["e2e_max_us"],
                            "speedup": base_row["e2e_max_us"] / best["e2e_max_us"],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

        if rank == 0 and args.output_json:
            output = Path(args.output_json)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(
                json.dumps(
                    {
                        "network": args.network,
                        "world_size": world,
                        "max_tok_per_rank": args.max_tok_per_rank,
                        "sweep": args.sweep,
                        "overrides": overrides,
                        "iters": args.iters,
                        "results": results,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )
    finally:
        _cleanup()


if __name__ == "__main__":
    main()
