# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Focused correctness and graph-replay coverage for FlyDSL sparse MLA decode."""

from __future__ import annotations

import math
from dataclasses import dataclass

import pytest
import torch

H = 16
DV = 512
DT = 64
DIM = DV + DT
BLOCK_I = 64
LOG2E = math.log2(math.e)


@dataclass(frozen=True)
class Case:
    name: str
    seq: int
    width: int
    valid: int
    context: int
    shift: int
    seed: int

    @property
    def ng(self) -> int:
        return self.width // BLOCK_I


CASES = (
    Case("seq1", seq=1, width=2048, valid=2048, context=62000, shift=0, seed=101),
    Case("primary", seq=6, width=2048, valid=2048, context=62000, shift=342, seed=102),
    Case("ctx64", seq=6, width=2048, valid=64, context=64, shift=0, seed=103),
    Case("ctx2000", seq=6, width=2048, valid=2000, context=2000, shift=88, seed=104),
    Case("ctx1", seq=6, width=2048, valid=1, context=1, shift=0, seed=105),
    Case("degen_k1", seq=6, width=64, valid=1, context=64, shift=0, seed=106),
    Case("k2049", seq=6, width=2112, valid=2049, context=62208, shift=341, seed=107),
)

GRAPH_CASES = ("seq1", "k2049")

if not torch.cuda.is_available():
    pytest.skip(
        "FlyDSL sparse MLA decode requires ROCm device access", allow_module_level=True
    )

from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl import (
    flydsl_sparse_mla_decode,
    flydsl_sparse_mla_decode_partial,
    is_flydsl_available,
)

pytestmark = pytest.mark.skipif(
    get_gfx() != "gfx950" or not is_flydsl_available(),
    reason="FlyDSL sparse MLA decode requires gfx950 with FlyDSL installed",
)


def _snr_db(actual: torch.Tensor, ref: torch.Tensor) -> float:
    actual64 = actual.to(torch.float64)
    ref64 = ref.to(torch.float64)
    noise = torch.linalg.vector_norm((actual64 - ref64).reshape(-1))
    signal = torch.linalg.vector_norm(ref64.reshape(-1))
    if noise.item() == 0.0:
        return float("inf")
    if signal.item() == 0.0:
        return float("-inf")
    return 20.0 * math.log10((signal / noise).item())


def _build_case(case: Case):
    device = torch.device("cuda")
    g = torch.Generator(device=device).manual_seed(case.seed)
    q = (torch.randn((case.seq, H, DIM), device=device, generator=g) * 0.25).clamp_(
        -2.0, 2.0
    )
    kv = (torch.randn((case.context, DIM), device=device, generator=g) * 0.25).clamp_(
        -2.0, 2.0
    )
    q_fp8 = q.to(torch.float8_e4m3fn).contiguous()
    kv_fp8 = kv.to(torch.float8_e4m3fn).contiguous()
    indices = torch.full((case.seq, case.width), -1, dtype=torch.int32, device=device)
    base = torch.arange(case.valid, dtype=torch.int32, device=device)
    for row in range(case.seq):
        if case.context > 1 and case.valid > 1:
            indices[row, : case.valid] = (base + row * case.shift) % case.context
        else:
            indices[row, 0] = 0
    return q_fp8, kv_fp8, indices.contiguous()


def _reference_partials(
    q_fp8: torch.Tensor,
    kv_fp8: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    seq, width = map(int, indices.shape)
    ng = width // BLOCK_I
    q = q_fp8.to(torch.float32)
    kv = kv_fp8.to(torch.float32)
    safe_indices = indices.clamp_min(0).to(torch.int64)
    gathered = kv.index_select(0, safe_indices.reshape(-1)).reshape(seq, width, DIM)
    valid = indices >= 0
    scores = torch.einsum("shd,skd->shk", q, gathered) * float(sm_scale)
    scores = scores.masked_fill(~valid.unsqueeze(1), float("-inf"))

    scores = scores.view(seq, H, ng, BLOCK_I).transpose(1, 2).contiguous()
    values = gathered[:, :, :DV].view(seq, ng, BLOCK_I, DV)
    valid_split = valid.view(seq, ng, BLOCK_I)
    split_max = scores.max(dim=-1).values
    live = valid_split.any(dim=-1).unsqueeze(-1)
    max_safe = torch.where(live, split_max, torch.zeros_like(split_max))
    weights = torch.exp(scores - max_safe.unsqueeze(-1)) * valid_split.unsqueeze(2)
    denom = weights.sum(dim=-1)
    inv = torch.where(denom > 0, denom.reciprocal(), torch.zeros_like(denom))
    partial = torch.einsum("sghk,sgkd->sghd", weights * inv.unsqueeze(-1), values)
    lse_nat = torch.where(
        denom > 0,
        torch.log(denom) + max_safe,
        torch.zeros_like(denom),
    )
    partial_lse = torch.where(
        denom > 0,
        lse_nat * LOG2E,
        torch.full_like(denom, -(2**30)),
    )

    scores_by_head = scores.transpose(1, 2).reshape(seq, H, width)
    full_max = scores_by_head.max(dim=-1).values
    full_live = valid.any(dim=-1).unsqueeze(-1)
    full_max_safe = torch.where(full_live, full_max, torch.zeros_like(full_max))
    full_weights = torch.exp(
        scores_by_head - full_max_safe.unsqueeze(-1)
    ) * valid.unsqueeze(1)
    full_denom = full_weights.sum(dim=-1)
    final = torch.einsum("shk,skd->shd", full_weights, gathered[:, :, :DV])
    final = torch.where(
        full_denom.unsqueeze(-1) > 0,
        final / full_denom.unsqueeze(-1),
        torch.zeros_like(final),
    )
    return partial.to(torch.bfloat16), partial_lse, final.to(torch.bfloat16)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_sparse_mla_decode_exact_contract(case: Case):
    q, kv, indices = _build_case(case)
    partial_output = torch.empty(
        (case.seq, case.ng, H, DV), device=q.device, dtype=torch.bfloat16
    )
    partial_lse = torch.empty(
        (case.seq, case.ng, H), device=q.device, dtype=torch.float32
    )
    out = torch.empty((case.seq, H, DV), device=q.device, dtype=torch.bfloat16)
    sm_scale = 1.0 / math.sqrt(DIM)

    ref_partial, ref_lse, ref_out = _reference_partials(q, kv, indices, sm_scale)

    flydsl_sparse_mla_decode_partial(
        q, kv, indices, partial_output, partial_lse, sm_scale
    )
    flydsl_sparse_mla_decode(
        q,
        kv,
        indices,
        out,
        sm_scale,
        partial_output=partial_output,
        partial_lse=partial_lse,
    )
    out_repeat = torch.empty_like(out)
    flydsl_sparse_mla_decode(
        q,
        kv,
        indices,
        out_repeat,
        sm_scale,
        partial_output=partial_output,
        partial_lse=partial_lse,
    )
    torch.cuda.synchronize()

    sentinel_mask = ref_lse == -(2**30)
    assert torch.equal(partial_lse == -(2**30), sentinel_mask)
    live_mask = ~sentinel_mask
    if live_mask.any():
        assert _snr_db(partial_lse[live_mask], ref_lse[live_mask]) >= 80.0
    assert _snr_db(partial_output, ref_partial) >= 30.0
    assert _snr_db(out, ref_out) >= 30.0
    assert torch.isfinite(out.to(torch.float32)).all()
    assert torch.equal(out, out_repeat)


@pytest.mark.parametrize(
    "case_name",
    GRAPH_CASES,
)
def test_sparse_mla_decode_graph_replay(case_name: str):
    case = next(item for item in CASES if item.name == case_name)
    q_a, kv_a, indices_a = _build_case(case)
    q_b, kv_b, indices_b = _build_case(
        Case(
            case.name,
            seq=case.seq,
            width=case.width,
            valid=case.valid,
            context=case.context,
            shift=case.shift,
            seed=case.seed + 1000,
        )
    )
    sm_scale = 1.0 / math.sqrt(DIM)

    out_a = torch.empty((case.seq, H, DV), device="cuda", dtype=torch.bfloat16)
    po_a = torch.empty((case.seq, case.ng, H, DV), device="cuda", dtype=torch.bfloat16)
    pl_a = torch.empty((case.seq, case.ng, H), device="cuda", dtype=torch.float32)
    flydsl_sparse_mla_decode(
        q_a,
        kv_a,
        indices_a,
        out_a,
        sm_scale,
        partial_output=po_a,
        partial_lse=pl_a,
    )
    expected_a = out_a.clone()

    out_b = torch.empty((case.seq, H, DV), device="cuda", dtype=torch.bfloat16)
    po_b = torch.empty((case.seq, case.ng, H, DV), device="cuda", dtype=torch.bfloat16)
    pl_b = torch.empty((case.seq, case.ng, H), device="cuda", dtype=torch.float32)
    flydsl_sparse_mla_decode(
        q_b,
        kv_b,
        indices_b,
        out_b,
        sm_scale,
        partial_output=po_b,
        partial_lse=pl_b,
    )
    expected_b = out_b.clone()

    q_live = q_a.clone()
    kv_live = kv_a.clone()
    indices_live = indices_a.clone()
    out_live = torch.empty((case.seq, H, DV), device="cuda", dtype=torch.bfloat16)
    po_live = torch.empty(
        (case.seq, case.ng, H, DV), device="cuda", dtype=torch.bfloat16
    )
    pl_live = torch.empty((case.seq, case.ng, H), device="cuda", dtype=torch.float32)
    stream = torch.cuda.Stream()

    torch.cuda.synchronize()
    with torch.cuda.stream(stream):
        flydsl_sparse_mla_decode(
            q_live,
            kv_live,
            indices_live,
            out_live,
            sm_scale,
            partial_output=po_live,
            partial_lse=pl_live,
        )
    stream.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        flydsl_sparse_mla_decode(
            q_live,
            kv_live,
            indices_live,
            out_live,
            sm_scale,
            partial_output=po_live,
            partial_lse=pl_live,
        )

    expected_by_tag = {"a": expected_a, "b": expected_b}
    source_by_tag = {
        "a": (q_a, kv_a, indices_a),
        "b": (q_b, kv_b, indices_b),
    }
    for tag in ("a", "b", "a", "b"):
        q_src, kv_src, indices_src = source_by_tag[tag]
        q_live.copy_(q_src)
        kv_live.copy_(kv_src)
        indices_live.copy_(indices_src)
        out_live.zero_()
        po_live.zero_()
        pl_live.zero_()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out_live, expected_by_tag[tag])
        assert torch.isfinite(out_live.to(torch.float32)).all()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
