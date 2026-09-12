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
quantises its slice of one token exactly once and scatters the resulting E8M0
bytes into the swizzled layout of that token's sorted rows. Splitting quant and
scatter into two kernels instead costs a second launch boundary plus a global
round trip through the per-token scale, which at this size is more than the
redundant work it removes.

Locating a token's sorted rows is what the sorting kernel's inverse table is
for: ``topk`` direct reads, all issued alongside the activation load. Without it
(oneshot sorting, or a non-FlyDSL sort) the fallback is a static dwordx4 sweep
of the whole ``sorted_ids`` allocation, bucketed by the ``(topk_slot << 24) |
token_id`` packing -- a token's slots are distinct so no atomic is needed, and
padding rows carry ``token_id == num_tokens`` so they match no block, which
reproduces the HIP kernel leaving their scale bytes untouched.

Stage 2 quantises one input row per (token, expert slot) assignment, so it has
no redundant work for this kernel to remove, but ``per_assignment`` mode still
wins on the same table: walking sorted rows means the HIP kernel cannot issue an
activation load until both ``num_valid_ids`` and ``sorted_ids`` have come back,
whereas here the block owns its input row by construction and the single
inverse-table read rides alongside the load it would otherwise gate.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import ReductionOp, T
from flydsl.expr.typing import Vector as Vec

from aiter.ops.flydsl.kernels import buffer_ops

from .kernels_common import ROW_INV_BASE, ROW_INV_ZERO_WEIGHT, get_warp_size
from .tensor_shim import _run_compiled

GROUP = 32  # MX block size
ELEMS_PER_THREAD = 8  # one dwordx4 load of bf16
LANES_PER_GROUP = GROUP // ELEMS_PER_THREAD
ROW_MASK = ~ROW_INV_ZERO_WEIGHT & 0x7FFFFFFF
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


def quantize_mxfp4_chunk(raw):
    """One thread's ``ELEMS_PER_THREAD`` bf16 inputs -> (packed fp4, E8M0).

    A 32-element MX group spans ``LANES_PER_GROUP`` consecutive lanes, so the
    amax reduction is an xor shuffle across them. |x| >= 0, so the fp32 bit
    pattern orders the same way as the value and the reduce can stay on the
    (always-available) i32 shuffle path.
    """
    values = Vec(raw).bitcast(fx.BFloat16).to(fx.Float32)
    local_max = fmath.absf(values).reduce(ReductionOp.MAX)
    local_max = local_max.maximumf(fx.Float32(1e-10))

    lm_i = local_max.bitcast(fx.Int32)
    for sh in range_constexpr(LANES_PER_GROUP.bit_length() - 1):
        peer = lm_i.shuffle_xor(fx.Int32(1 << sh), fx.Int32(WARP_SIZE))
        lm_i = (peer > lm_i).select(peer, lm_i)

    working = (
        lm_i.bitcast(fx.Float32)
        * fx.Int32(_FP4_INV_MAX_POS_BITS).bitcast(fx.Float32)
    ).bitcast(fx.Int32)
    biased_exp = (working >> fx.Int32(23)) & fx.Int32(0xFF)
    e8m0 = ((working & fx.Int32(0x7FFFFF)) != fx.Int32(0)).select(
        biased_exp + fx.Int32(1), biased_exp
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
    return fx.Int32(packed), e8m0


def mx_scale_col_offset(col):
    """Column-only half of ``mx_scale_shuffle_idx``.

    The swizzle separates into a row-only and a column-only term, so a lane
    that owns one column computes this once and reuses it for every row.
    """
    return (
        ((col >> fx.Int32(3)) << fx.Int32(8))
        + ((col & fx.Int32(3)) << fx.Int32(6))
        + (((col & fx.Int32(7)) >> fx.Int32(2)) << fx.Int32(1))
    )


def mx_scale_row_offset(row, tile_bytes):
    """Row-only half of ``mx_scale_shuffle_idx``; see ``mx_scale_col_offset``."""
    return (
        (row >> fx.Int32(5)) * fx.Int32(tile_bytes)
        + ((row & fx.Int32(15)) << fx.Int32(2))
        + ((row & fx.Int32(31)) >> fx.Int32(4))
    )


@functools.lru_cache(maxsize=64)
def _compile_token_major_quant_sort(
    *,
    cols: int,
    scale_n_pad: int,
    sorted_len: int,
    sort_topk: int,
    has_weights: bool,
    has_row_inv: bool,
    per_assignment: bool,
    nsplit: int,
    block: int,
):
    assert has_row_inv or not per_assignment
    scale_n_local = cols // GROUP // nsplit  # scale columns of one column slice
    # `_pick_geometry` splits a row so that one thread owns exactly one dwordx4,
    # which makes the block exactly `LANES_PER_GROUP` lanes wide per MX group.
    assert block == scale_n_local * LANES_PER_GROUP
    # Phase 3 lays the block out as (slot, column) with the slot on the low bits
    # of the lane id: all `LANES_PER_GROUP` lanes of a group hold that group's
    # E8M0 after the reduce, so every pass stores from the register that
    # produced it -- no LDS staging and no barrier.
    n_passes = 1 if per_assignment else (
        (sort_topk + LANES_PER_GROUP - 1) // LANES_PER_GROUP
    )
    tile_bytes = scale_n_pad * GROUP  # bytes of one 32-row swizzle tile
    # Rows the caller allocates: pad32(sorted_len), the shape the swizzled
    # layout is defined on. Used as the scale descriptor's extent below.
    scale_valid_bytes = ((sorted_len + 31) // 32 * 32) * scale_n_pad
    # Fallback scan: a static dwordx4 sweep of the whole `sorted_ids`
    # allocation. A dynamic `num_valid`-bounded loop serialises one global load
    # per trip; unrolling lets every load issue up front. Clamping the tail
    # index re-reads a few rows, which is harmless because writing a row into
    # its slot is idempotent.
    scan_iters = (sorted_len + block * 4 - 1) // (block * 4)
    scan_clamp = sorted_len - 4

    @fx.struct
    class SharedStorage:
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
        # One input row per block: a token in stage 1, a (token, slot)
        # assignment in `per_assignment` mode.
        in_row = bid // fx.Int32(nsplit) if nsplit > 1 else bid
        part = bid % fx.Int32(nsplit) if nsplit > 1 else fx.Int32(0)

        out_rsrc = buffer_ops.create_buffer_resource(out, max_size=True)
        # Real extent, not max_size. Phase 3 below scatters into this buffer at
        # a row read out of the sorting kernel's inverse table, and that table
        # can hand back a row outside the allocation. Under max_size the
        # descriptor spans the whole address space, so such a store lands
        # wherever the address arithmetic points: silent corruption of whatever
        # else is mapped there, or a GPU page fault when nothing is. That was
        # the deployment crash -- every MoE kernel retired cleanly under
        # per-kernel host syncs while the server died downstream, and disabling
        # just this table made it stop. Seeding the table to -1 did not help, so
        # the bad rows are produced, not left over, and the consumer has to
        # bound them. Doing it in the descriptor is free; the same check written
        # as a predicate in the store loop cost 2.8% of the operator score.
        scale_rsrc = buffer_ops.create_buffer_resource(
            scale, num_records_bytes=scale_valid_bytes)
        in_rsrc = buffer_ops.create_buffer_resource(inp, max_size=True)

        c_zero = fx.Int32(0)
        c_one = fx.Int32(1)
        c_topk = fx.Int32(sort_topk)
        c_wflag = fx.Int32(ROW_INV_ZERO_WEIGHT)

        # The activation load feeds the longest chain, so issue it first and let
        # the row lookup below overlap with it.
        in_dw_base = in_row * fx.Int32(cols // 2) + (
            (part * fx.Int32(block)) << fx.Int32(2)
        )
        raw = buffer_ops.buffer_load(
            in_rsrc,
            in_dw_base + (tid << fx.Int32(2)),
            vec_width=4,
            dtype=T.i32,
        )

        # ---- Phase 1: this token's sorted row per topk slot, `-1` if none ----
        # The payload carries ROW_INV_ZERO_WEIGHT for a row whose routed weight
        # is zero, matching the HIP kernel's zero scale byte for those rows.
        sub = tid & fx.Int32(LANES_PER_GROUP - 1)
        slot_rows = []
        if const_expr(per_assignment):
            # The block already owns one assignment, so its sorted row is a
            # single uniform read that the activation load above covers.
            nv_rsrc = buffer_ops.create_buffer_resource(num_valid_ids, max_size=True)
            slot_rows.append(
                buffer_ops.buffer_load(
                    nv_rsrc,
                    fx.Int32(ROW_INV_BASE) + in_row,
                    vec_width=1,
                    dtype=T.i32,
                )
            )
        elif const_expr(has_row_inv):
            nv_rsrc = buffer_ops.create_buffer_resource(num_valid_ids, max_size=True)
            inv_base = fx.Int32(ROW_INV_BASE) + in_row * c_topk
            for pss in range_constexpr(n_passes):
                slot = fx.Int32(pss * LANES_PER_GROUP) + sub
                has_slot = slot < c_topk
                packed_row = buffer_ops.buffer_load(
                    nv_rsrc,
                    inv_base + has_slot.select(slot, c_zero),
                    vec_width=1,
                    dtype=T.i32,
                )
                slot_rows.append(has_slot.select(packed_row, fx.Int32(-1)))
        else:
            sid_rsrc = buffer_ops.create_buffer_resource(sorted_ids, max_size=True)
            nv_rsrc = buffer_ops.create_buffer_resource(num_valid_ids, max_size=True)
            sw_rsrc = buffer_ops.create_buffer_resource(sorted_weights, max_size=True)
            rows_mr = fx.SharedAllocator().allocate(SharedStorage).peek().rows.ptr
            if tid < c_topk:
                _lds_store(rows_mr, fx.Int32(-1), tid)
            num_valid = buffer_ops.buffer_load(
                nv_rsrc, c_zero, vec_width=1, dtype=T.i32
            )
            gpu.barrier()
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
                    # Only stage 1 reaches the scan, where `in_row` is a token.
                    if (row < num_valid) & ((fused & fx.Int32(0xFFFFFF)) == in_row):
                        _lds_store(rows_mr, row, fused >> fx.Int32(24))
            gpu.barrier()
            for pss in range_constexpr(n_passes):
                slot = fx.Int32(pss * LANES_PER_GROUP) + sub
                has_slot = slot < c_topk
                row = _lds_load(rows_mr, has_slot.select(slot, c_zero))
                row = has_slot.select(row, fx.Int32(-1))
                if has_weights:
                    w_bits = buffer_ops.buffer_load(
                        sw_rsrc,
                        (row >= c_zero).select(row, c_zero),
                        vec_width=1,
                        dtype=T.i32,
                    )
                    row = ((w_bits & fx.Int32(0x7FFFFFFF)) == c_zero).select(
                        row | c_wflag, row
                    )
                slot_rows.append(row)

        # ---- Phase 2: quantise this token's column slice, once ----
        packed, e8m0 = quantize_mxfp4_chunk(raw)
        out_dw_base = in_row * fx.Int32(cols // 8) + part * fx.Int32(block)
        buffer_ops.buffer_store(packed, out_rsrc, out_dw_base + tid)

        # ---- Phase 3: scatter the E8M0 bytes into the swizzled sorted rows ----
        col = part * fx.Int32(scale_n_local) + (tid >> fx.Int32(2))
        col_addr = mx_scale_col_offset(col)
        for pss in range_constexpr(n_passes):
            packed_row = slot_rows[pss]
            keep = ((packed_row & c_wflag) == c_zero).select(c_one, c_zero)
            row = packed_row & fx.Int32(ROW_MASK)
            if packed_row >= c_zero:
                buffer_ops.buffer_store(
                    (e8m0 * keep).to(fx.Uint8),
                    scale_rsrc,
                    mx_scale_row_offset(row, tile_bytes) + col_addr,
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


def has_sorted_row_inverse(num_valid_ids: torch.Tensor, num_assignments: int) -> bool:
    """True when ``num_valid_ids`` carries the sorted-row inverse table."""
    return num_valid_ids.numel() >= ROW_INV_BASE + num_assignments


def token_major_mxfp4_quant_moe_sort(
    out: torch.Tensor,
    scale: torch.Tensor,
    inp: torch.Tensor,
    sorted_ids: torch.Tensor,
    num_valid_ids: torch.Tensor,
    sorted_weights,
    sort_topk: int,
    per_assignment: bool = False,
    stream=None,
) -> None:
    """Quantise ``inp`` to MXFP4 and scatter its E8M0 scales into ``scale``.

    ``out`` is indexed like ``inp``; ``scale`` is sorted-row-indexed
    ``[pad32(num_sorted_rows), scale_n_pad]`` in the swizzled GEMM layout.

    A ``num_valid_ids`` sized past its two scalars carries the sorting kernel's
    sorted-row inverse table, which turns the row lookup into direct reads;
    otherwise the kernel scans ``sorted_ids``. ``per_assignment`` switches
    ``inp`` from one row per token (stage 1) to one row per (token, expert slot)
    assignment (stage 2) and requires the table.
    """
    num_rows, cols = inp.shape
    nsplit, block = _pick_geometry(cols)
    num_assignments = num_rows if per_assignment else num_rows * sort_topk

    launcher = _compile_token_major_quant_sort(
        cols=cols,
        scale_n_pad=scale.shape[1],
        sorted_len=sorted_ids.shape[0],
        sort_topk=sort_topk,
        has_weights=sorted_weights is not None,
        has_row_inv=has_sorted_row_inverse(num_valid_ids, num_assignments),
        per_assignment=per_assignment,
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
        int(num_rows * nsplit),
        fx.Stream(stream),
    )
