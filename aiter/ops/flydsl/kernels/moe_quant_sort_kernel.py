# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Token-major fused MXFP4 quant + swizzled E8M0 scale scatter (FlyDSL).

The HIP ``fused_mx_quant_moe_sort_kernel`` is parallelised over *sorted rows*.
In stage 1 the fp4 payload is addressed by ``token_id`` while only the E8M0
scale is addressed by ``sorted_row``, so a token that routes to ``topk``
experts is re-quantised once per expert and rewrites the very same payload
bytes that many times (~27x at the GLM-5.2 EP4 decode shape, where 64 tokens
expand to 1760 sorted rows).

This kernel is parallelised over ``(token, column slice)`` instead: a block
quantises its slice of one token exactly once, keeps the E8M0 bytes of that
slice in LDS, then locates the token's sorted rows and scatters the bytes into
the swizzled layout. Splitting quant and scatter into two kernels instead costs
a second launch boundary plus a global round trip through the per-token scale,
which at this size is more than the redundant work it removes.

The packed sorted id is ``(topk_slot << 24) | token_id`` and a token's slots
are distinct, so the rows found by the scan can be bucketed by slot without an
atomic. Padding rows carry ``token_id == num_tokens`` and therefore match no
block, which reproduces the HIP kernel leaving their scale bytes untouched.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import ReductionOp, T
from flydsl.expr.typing import Vector as Vec

from aiter.ops.flydsl.kernels import buffer_ops

from .kernels_common import get_warp_size
from .tensor_shim import _run_compiled

GROUP = 32  # MX block size
ELEMS_PER_THREAD = 8  # one dwordx4 load of bf16
WARP_SIZE = get_warp_size()

# fp32 bits of 1/max_pos for RoundUp ceil_pow2(amax / max_pos); fp4 max_pos = 6.
_FP4_INV_MAX_POS_BITS = 0x3E2AAAAB

MAX_BLOCK = 1024


def _pick_geometry(cols: int):
    """Split a row across `nsplit` blocks of `block` threads, one chunk each.

    Every block rescans `sorted_ids`, so the narrowest split that still fits a
    workgroup is the cheapest; measured monotonically best down to nsplit=1.
    """
    chunks = cols // ELEMS_PER_THREAD
    for nsplit in range(1, chunks + 1):
        if chunks % nsplit or (cols // GROUP) % nsplit:
            continue
        block = chunks // nsplit
        if block <= MAX_BLOCK and block % WARP_SIZE == 0:
            return nsplit, block
    return None


def _lds_load(raw_ptr, idx):
    return fx.ptr_load(raw_ptr + fx.Int64(idx))


def _lds_store(raw_ptr, val, idx):
    fx.ptr_store(val, raw_ptr + fx.Int64(idx))


@functools.lru_cache(maxsize=64)
def _compile_token_major_quant_sort(
    *,
    cols: int,
    scale_n_pad: int,
    sorted_len: int,
    sort_topk: int,
    has_weights: bool,
    nsplit: int,
    block: int,
):
    n_iters = cols // (nsplit * block * ELEMS_PER_THREAD)
    scale_n = cols // GROUP  # valid scale columns of the whole row
    scale_n_local = scale_n // nsplit  # ... of one column slice
    # Phase 3 lays the block out as (slot group, column) so every lane stores on
    # every pass; one column slice is exactly `block // slots_per_pass` wide.
    slots_per_pass = block // scale_n_local
    n_passes = (sort_topk + slots_per_pass - 1) // slots_per_pass
    tile_bytes = scale_n_pad * GROUP  # bytes of one 32-row swizzle tile
    # The scan is a static dwordx4 sweep of the whole `sorted_ids` allocation.
    # A dynamic `num_valid`-bounded loop serialises one global load per trip;
    # unrolling lets every load issue up front and overlap the activation
    # loads. Clamping the tail index re-reads a few rows, which is harmless
    # because writing a row into its slot is idempotent.
    scan_iters = (sorted_len + block * 4 - 1) // (block * 4)
    scan_clamp = sorted_len - 4

    @fx.struct
    class SharedStorage:
        e8m0: fx.Array[fx.Int32, scale_n_local, 16]
        rows: fx.Array[fx.Int32, sort_topk, 16]

    @flyc.kernel(known_block_size=[block, 1, 1])
    def token_major_quant_sort_kernel(
        out: fx.Tensor,
        scale: fx.Tensor,
        inp: fx.Tensor,
        sorted_ids: fx.Tensor,
        num_valid_ids: fx.Tensor,
        sorted_weights: fx.Tensor,
    ):
        tid = gpu.thread_idx.x
        bid = gpu.block_idx.x
        token = bid // fx.Int32(nsplit) if nsplit > 1 else bid
        part = bid % fx.Int32(nsplit) if nsplit > 1 else fx.Int32(0)

        out_rsrc = buffer_ops.create_buffer_resource(out, max_size=True)
        scale_rsrc = buffer_ops.create_buffer_resource(scale, max_size=True)
        in_rsrc = buffer_ops.create_buffer_resource(inp, max_size=True)
        sid_rsrc = buffer_ops.create_buffer_resource(sorted_ids, max_size=True)
        nv_rsrc = buffer_ops.create_buffer_resource(num_valid_ids, max_size=True)
        sw_rsrc = buffer_ops.create_buffer_resource(sorted_weights, max_size=True)

        c_zero = fx.Int32(0)
        c_one = fx.Int32(1)
        c_wave = fx.Int32(WARP_SIZE)

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        e8m0_mr = lds.e8m0.ptr
        rows_mr = lds.rows.ptr

        if tid < fx.Int32(sort_topk):
            _lds_store(rows_mr, fx.Int32(-1), tid)
        num_valid = buffer_ops.buffer_load(nv_rsrc, c_zero, vec_width=1, dtype=T.i32)
        gpu.barrier()

        # ---- Phase 1: bucket this token's sorted rows by topk slot ----
        c_clamp = fx.Int32(scan_clamp)
        for sit in range_constexpr(scan_iters):
            base = (fx.Int32(sit * block) + tid) << fx.Int32(2)
            base = (base > c_clamp).select(c_clamp, base)
            quad = Vec(
                buffer_ops.buffer_load(sid_rsrc, base, vec_width=4, dtype=T.i32)
            )
            for e in range_constexpr(4):
                row = base + fx.Int32(e)
                fused = quad[e]
                if (row < num_valid) & ((fused & fx.Int32(0xFFFFFF)) == token):
                    _lds_store(rows_mr, row, fused >> fx.Int32(24))

        # ---- Phase 2: quantise this token's column slice, once ----
        slice_chunks = fx.Int32(n_iters * block)
        in_dw_base = token * fx.Int32(cols // 2) + (part * slice_chunks << fx.Int32(2))
        out_dw_base = token * fx.Int32(cols // 8) + part * slice_chunks
        for it in range_constexpr(n_iters):
            vec_idx = fx.Int32(it * block) + tid
            raw = buffer_ops.buffer_load(
                in_rsrc,
                in_dw_base + (vec_idx << fx.Int32(2)),
                vec_width=4,
                dtype=T.i32,
            )
            values = Vec(raw).bitcast(fx.BFloat16).to(fx.Float32)
            local_max = fmath.absf(values).reduce(ReductionOp.MAX)
            local_max = local_max.maximumf(fx.Float32(1e-10))

            # A 32-element MX group spans 4 consecutive lanes. |x| >= 0, so the
            # fp32 bit pattern orders the same way as the value and the reduce
            # can stay on the (always-available) i32 shuffle path.
            lm_i = local_max.bitcast(fx.Int32)
            for sh in range_constexpr(2):
                peer = lm_i.shuffle_xor(fx.Int32(1 << sh), c_wave)
                lm_i = (peer > lm_i).select(peer, lm_i)

            working = (
                lm_i.bitcast(fx.Float32)
                * fx.Int32(_FP4_INV_MAX_POS_BITS).bitcast(fx.Float32)
            ).bitcast(fx.Int32)
            biased_exp = (working >> fx.Int32(23)) & fx.Int32(0xFF)
            e8m0 = ((working & fx.Int32(0x7FFFFF)) != c_zero).select(
                biased_exp + c_one, biased_exp
            )
            e8m0 = (e8m0 > fx.Int32(0xFF)).select(fx.Int32(0xFF), e8m0)

            dequant_scale = (e8m0 << fx.Int32(23)).bitcast(fx.Float32)
            packed = fx.Int32(0)
            for pair in range_constexpr(ELEMS_PER_THREAD // 2):
                packed = rocdl.cvt_scalef32_pk_fp4_f32(
                    T.i32,
                    packed,
                    values[2 * pair],
                    values[2 * pair + 1],
                    dequant_scale,
                    pair,
                )
            buffer_ops.buffer_store(fx.Int32(packed), out_rsrc, out_dw_base + vec_idx)

            if (tid & fx.Int32(3)) == c_zero:
                _lds_store(e8m0_mr, e8m0, vec_idx >> fx.Int32(2))
        gpu.barrier()

        # ---- Phase 3: scatter the E8M0 bytes into the swizzled sorted rows ----
        # mx_scale_shuffle_idx(scale_n_pad, x, y) splits into a row-only and a
        # column-only half; each lane owns one column for `n_passes` slots, so
        # the column half is computed once and every load is issued up front.
        sub = tid // fx.Int32(scale_n_local)
        col = part * fx.Int32(scale_n_local) + tid % fx.Int32(scale_n_local)
        col_addr = (
            ((col >> fx.Int32(3)) << fx.Int32(8))
            + ((col & fx.Int32(3)) << fx.Int32(6))
            + (((col & fx.Int32(7)) >> fx.Int32(2)) << fx.Int32(1))
        )
        c_topk = fx.Int32(sort_topk)
        slot_rows = []
        for pss in range_constexpr(n_passes):
            slot = fx.Int32(pss * slots_per_pass) + sub
            has_slot = slot < c_topk
            row = _lds_load(rows_mr, has_slot.select(slot, c_zero))
            slot_rows.append(has_slot.select(row, fx.Int32(-1)))
        keeps = []
        for row in slot_rows:
            if has_weights:
                w_bits = buffer_ops.buffer_load(
                    sw_rsrc,
                    (row >= c_zero).select(row, c_zero),
                    vec_width=1,
                    dtype=T.i32,
                )
                keeps.append(
                    ((w_bits & fx.Int32(0x7FFFFFFF)) != c_zero).select(c_one, c_zero)
                )
            else:
                keeps.append(c_one)
        val = _lds_load(e8m0_mr, tid % fx.Int32(scale_n_local))
        for pss in range_constexpr(n_passes):
            row = slot_rows[pss]
            if row >= c_zero:
                addr = (
                    (row >> fx.Int32(5)) * fx.Int32(tile_bytes)
                    + ((row & fx.Int32(15)) << fx.Int32(2))
                    + ((row & fx.Int32(31)) >> fx.Int32(4))
                    + col_addr
                )
                buffer_ops.buffer_store(
                    (val * keeps[pss]).to(fx.Uint8),
                    scale_rsrc,
                    addr,
                    offset_is_bytes=True,
                )

    @flyc.jit
    def launch(
        out: fx.Tensor,
        scale: fx.Tensor,
        inp: fx.Tensor,
        sorted_ids: fx.Tensor,
        num_valid_ids: fx.Tensor,
        sorted_weights: fx.Tensor,
        i32_grid: fx.Int32,
        stream: fx.Stream = None,
    ):
        stream = stream if stream is not None else fx.Stream(None)
        launcher = token_major_quant_sort_kernel(
            out, scale, inp, sorted_ids, num_valid_ids, sorted_weights
        )
        launcher.launch(grid=(i32_grid, 1, 1), block=(block, 1, 1), stream=stream)

    return launch


_dummy_weights_cache = {}


def _dummy_weights(device):
    t = _dummy_weights_cache.get(device)
    if t is None:
        t = torch.zeros(1, dtype=torch.float32, device=device)
        _dummy_weights_cache[device] = t
    return t


def can_run_token_major_quant_sort(cols: int, group_size: int) -> bool:
    """True when the token-major kernel supports this shape."""
    return group_size == GROUP and cols % GROUP == 0 and _pick_geometry(cols) is not None


def token_major_mxfp4_quant_moe_sort(
    out: torch.Tensor,
    scale: torch.Tensor,
    inp: torch.Tensor,
    sorted_ids: torch.Tensor,
    num_valid_ids: torch.Tensor,
    sorted_weights,
    sort_topk: int,
    stream=None,
) -> None:
    """Quantise ``inp`` to MXFP4 and scatter its E8M0 scales into ``scale``.

    ``out`` is token-indexed ``[num_tokens, cols // 2]``; ``scale`` is
    sorted-row-indexed ``[pad32(num_sorted_rows), scale_n_pad]`` in the
    swizzled GEMM layout. Only stage 1 (one input row per token) is supported.
    """
    num_tokens, cols = inp.shape
    nsplit, block = _pick_geometry(cols)

    launcher = _compile_token_major_quant_sort(
        cols=cols,
        scale_n_pad=scale.shape[1],
        sorted_len=sorted_ids.shape[0],
        sort_topk=sort_topk,
        has_weights=sorted_weights is not None,
        nsplit=nsplit,
        block=block,
    )
    weights = (
        sorted_weights if sorted_weights is not None else _dummy_weights(inp.device)
    )
    stream = stream if stream is not None else torch.cuda.current_stream(inp.device)
    _run_compiled(
        launcher,
        out.view(torch.uint8),
        scale.view(torch.uint8),
        inp,
        sorted_ids,
        num_valid_ids,
        weights,
        int(num_tokens * nsplit),
        fx.Stream(stream),
    )
