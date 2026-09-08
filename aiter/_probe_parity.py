# SPDX-License-Identifier: MIT
"""Ad-hoc 8-rank probe: consecutive MegaMoEV2 launches must be epoch-invariant.

Alternates two different token batches through one workspace so a parity-half
mix-up (stale payload rows from the previous launch) shows up as a relL2 jump or
as a mismatch against the same batch's first result.
"""
from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("MORI_SHMEM_HEAP_SIZE", "40G")

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "op_tests", "multigpu_tests")
)

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

import test_mega_moe_v2 as T  # noqa: E402
from aiter.ops.flydsl.kernels.mega_moe import MegaMoEV2  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bs", type=int, default=84)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--mtpr", type=int, default=256)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    rank, world, device = T._setup_dist()
    try:
        network = T.NETWORKS["glm52"]
        local_experts = network["experts"] // world
        packed = T._quantize_weights(
            network["model_dim"], network["inter_dim"], local_experts, rank, args.seed, device
        )
        w1, w1_scale, w2, w2_scale, w1_q, w1_ref_scale, w2_q, w2_ref_scale = packed
        ref_weights = (w1_q, w1_ref_scale, w2_q, w2_ref_scale)
        batches = []
        for variant in range(2):
            x, weights, ids = T._make_inputs(
                args.bs,
                network["model_dim"],
                network["experts"],
                network["topk"],
                rank,
                args.seed + 977 * variant,
                device,
            )
            batches.append((x.contiguous(), weights.contiguous(), ids.contiguous()))

        moe = MegaMoEV2(
            rank=rank,
            world_size=world,
            quant="a8w4",
            w1=w1,
            w1_scale=w1_scale,
            w2=w2,
            w2_scale=w2_scale,
            max_tok_per_rank=args.mtpr,
            **network,
        )
        references = [
            T._reference(
                x,
                weights,
                ids,
                ref_weights,
                rank,
                world,
                moe.model_dim,
                moe.inter_dim,
                moe.experts,
                moe.swiglu_limit,
            )
            for (x, weights, ids) in batches
        ]

        first = [None, None]
        worst_rel = 0.0
        mismatch = 0
        for epoch in range(args.epochs):
            variant = epoch % 2
            x, weights, ids = batches[variant]
            out = moe(x, weights, ids)[: args.bs].clone()
            rel = float(
                torch.linalg.vector_norm(out.float() - references[variant])
                / torch.linalg.vector_norm(references[variant])
            )
            rel = T._reduce_float(rel, device, dist.ReduceOp.MAX)
            worst_rel = max(worst_rel, rel)
            if first[variant] is None:
                first[variant] = out
            elif not torch.equal(first[variant], out):
                mismatch += 1
            if rank == 0:
                print(f"[PARITY] epoch={epoch} variant={variant} relL2={rel:.6f}", flush=True)
        mismatch = int(T._reduce_float(float(mismatch), device, dist.ReduceOp.SUM))
        if rank == 0:
            print(
                f"[PARITY-SUMMARY] bs={args.bs} epochs={args.epochs} "
                f"worst_relL2={worst_rel:.6f} bitwise_mismatches={mismatch}",
                flush=True,
            )
    finally:
        T._cleanup()


if __name__ == "__main__":
    main()
