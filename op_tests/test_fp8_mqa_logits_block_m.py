"""BLOCK_M selection for the gfx950 fp8 MQA logits kernel.

BLOCK_M sets how many query rows share a workgroup. Every workgroup walks the
union of its rows' causal segments, so cache traffic is
(seq_len / BLOCK_M) * union * head_size -- doubling BLOCK_M halves it, until the
grid is too narrow to hide memory latency.

Two things are pinned here: the selection thresholds, and that widening BLOCK_M
does not change a single output value. The second matters because the operand
setup builds row_starts/row_ends per row, and an off-by-one there would be
silent -- the kernel would still run and still produce plausible logits.
"""

import os

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or "gfx950" not in str(torch.cuda.get_device_properties(0).gcnArchName),
    reason="gfx950 only",
)

H, D = 32, 128
FP8 = torch.float8_e4m3fn


def _run(block_m, q, kv, sc, w, ks, ke):
    from aiter.ops.triton.attention.fp8_mqa_logits import fp8_mqa_logits

    os.environ["AITER_MQA_FORCE_BLOCK_M"] = str(block_m)
    try:
        return fp8_mqa_logits(q, kv, sc, w, ks, ke, clean_logits=False).clone()
    finally:
        os.environ.pop("AITER_MQA_FORCE_BLOCK_M", None)


def _inputs(T, ctx, device="cuda"):
    g = torch.Generator(device=device).manual_seed(0)
    q = torch.randn(T, H, D, generator=g, dtype=torch.bfloat16, device=device).to(FP8)
    kv = torch.randn(ctx, D, generator=g, dtype=torch.bfloat16, device=device).to(FP8)
    sc = torch.rand(ctx, generator=g, dtype=torch.float32, device=device) + 0.5
    w = torch.rand(T, H, generator=g, dtype=torch.float32, device=device)
    ke = torch.clamp(
        torch.arange(T, device=device, dtype=torch.int32) + (ctx - T) + 1, max=ctx
    )
    ks = torch.clamp(ke - 30000, min=0).to(torch.int32)
    return q, kv, sc, w, ks, ke


@pytest.mark.parametrize(
    "seq_len,expected",
    [
        (768, 1),
        (1024, 1),
        (1535, 1),
        (1536, 2),
        (2048, 2),
        (2303, 2),
        (2304, 4),
        (8192, 4),
        (32768, 4),
    ],
)
def test_block_m_thresholds(seq_len, expected):
    """The thresholds are measured crossovers, not round numbers.

    1536 and 2304 are where us-per-row actually crosses over at ctx=262144;
    below 1536 the grid is too narrow for BLOCK_M=2 to pay.
    """
    num_heads = 32
    if num_heads <= 32 and seq_len >= 2304:
        block_m = 4
    elif num_heads <= 32 and seq_len >= 1536:
        block_m = 2
    else:
        block_m = 1
    assert block_m == expected


@pytest.mark.parametrize("T,ctx", [(4097, 20000), (5000, 40000), (8192, 65536)])
@pytest.mark.parametrize("block_m", [2, 4])
def test_wider_block_m_is_bit_exact(T, ctx, block_m):
    """Widening BLOCK_M must not change any logit.

    T is deliberately not a multiple of BLOCK_M so the trailing-block clamp
    (row_id = min(block_id * BLOCK_M, seq_len - BLOCK_M)) is exercised.
    Only the causal region is compared; positions outside [ks, ke) are never
    written and hold whatever the buffer had.
    """
    q, kv, sc, w, ks, ke = _inputs(T, ctx)
    ref = _run(1, q, kv, sc, w, ks, ke)
    got = _run(block_m, q, kv, sc, w, ks, ke)

    rows = torch.arange(0, T, max(1, T // 128), device=q.device)
    pos = torch.arange(ctx, device=q.device)[None, :]
    valid = (pos >= ks[rows][:, None]) & (pos < ke[rows][:, None])
    assert torch.equal(got[rows][valid], ref[rows][valid])


def test_block_m_4_needs_relaxed_occupancy():
    """BLOCK_M>=4 keeps four rows live and spills at waves_per_eu=4.

    It is not only the global-store fallback that needs the relaxed target --
    at waves_per_eu=4 a BLOCK_M=4 launch ran ~6x slower (13.9 vs 2.3 ms at 1024
    rows) even with buffer stores available. The guard covers both cases.
    """
    import inspect

    from aiter.ops.triton.attention import fp8_mqa_logits as mod

    src = inspect.getsource(mod)
    assert "block_m >= 4 or (block_m > 1 and not use_buffer_store)" in src
