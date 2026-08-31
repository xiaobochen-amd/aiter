# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness and CUDA-graph coverage for FP8 FlyDSL sparse-MLA prefill."""

import pytest
import torch

from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl import is_flydsl_available

_NUM_HEADS = 16
_V_HEAD_DIM = 512
_ROPE_HEAD_DIM = 64
_HEAD_DIM = _V_HEAD_DIM + _ROPE_HEAD_DIM
_TOPK = 2048
_SOFTMAX_SCALE = 0.0625
_MIN_SNR_DB = 35.3


def _require_gfx950_flydsl():
    if not torch.cuda.is_available() or get_gfx() != "gfx950":
        pytest.skip("requires gfx950")
    if not is_flydsl_available():
        pytest.skip("requires FlyDSL")


def _make_case(num_tokens, num_pages=4096, seed=1234):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randn(
        num_tokens,
        _NUM_HEADS,
        _HEAD_DIM,
        generator=generator,
        dtype=torch.float32,
    ).to(torch.float8_e4m3fn)
    kv = torch.randn(num_pages, _HEAD_DIM, generator=generator, dtype=torch.float32).to(
        torch.float8_e4m3fn
    )
    indices = torch.randint(
        0,
        num_pages,
        (num_tokens, _TOPK),
        generator=generator,
        dtype=torch.int32,
    )
    indices[:, -64:] = -1
    q = q.cuda()
    return (
        q[:, :, :_V_HEAD_DIM].contiguous(),
        q[:, :, _V_HEAD_DIM:].contiguous(),
        kv.cuda().contiguous(),
        indices.cuda(),
    )


def _reference(q_nope, q_rope, kv, indices, token_ids):
    query = torch.cat((q_nope[token_ids], q_rope[token_ids]), dim=-1).double()
    rows = indices[token_ids].long()
    valid = rows >= 0
    gathered = kv.double()[rows.clamp(min=0)]
    scores = torch.einsum("nhd,nkd->nhk", query, gathered) * _SOFTMAX_SCALE
    scores.masked_fill_(~valid[:, None, :], -float("inf"))
    probabilities = torch.softmax(scores, dim=-1)
    return torch.einsum("nhk,nkv->nhv", probabilities, gathered[:, :, :_V_HEAD_DIM])


def _snr_db(actual, expected):
    actual = actual.double()
    expected = expected.double()
    signal = expected.square().sum()
    noise = (actual - expected).square().sum()
    return float(10 * torch.log10(signal / noise))


def test_flydsl_sparse_mla_prefill_correctness():
    _require_gfx950_flydsl()
    from aiter.ops.flydsl import flydsl_sparse_mla_prefill

    q_nope, q_rope, kv, indices = _make_case(512)
    actual = flydsl_sparse_mla_prefill(q_nope, q_rope, kv, indices, _SOFTMAX_SCALE)
    torch.cuda.synchronize()
    token_ids = torch.linspace(0, 511, 16, device="cuda", dtype=torch.long)
    expected = _reference(q_nope, q_rope, kv, indices, token_ids)
    snr = _snr_db(actual[token_ids], expected)
    assert snr >= _MIN_SNR_DB, (
        f"SNR {snr:.6f} dB is below the {_MIN_SNR_DB} dB no-regression gate"
    )


def test_flydsl_sparse_mla_prefill_cuda_graph_replay():
    _require_gfx950_flydsl()
    from aiter.ops.flydsl import flydsl_sparse_mla_prefill

    q_nope, q_rope, kv, indices = _make_case(64)
    output = torch.empty(
        (64, _NUM_HEADS, _V_HEAD_DIM), device="cuda", dtype=torch.bfloat16
    )
    flydsl_sparse_mla_prefill(q_nope, q_rope, kv, indices, _SOFTMAX_SCALE, out=output)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        flydsl_sparse_mla_prefill(
            q_nope, q_rope, kv, indices, _SOFTMAX_SCALE, out=output
        )

    next_q_nope, next_q_rope, _, _ = _make_case(64, seed=4321)
    q_nope.copy_(next_q_nope)
    q_rope.copy_(next_q_rope)
    expected = torch.empty_like(output)
    flydsl_sparse_mla_prefill(q_nope, q_rope, kv, indices, _SOFTMAX_SCALE, out=expected)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
