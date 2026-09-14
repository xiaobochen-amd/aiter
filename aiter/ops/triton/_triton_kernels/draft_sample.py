# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import triton
import triton.language as tl

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

_draft_sample_reduce_repr = make_kernel_repr(
    "_draft_sample_reduce",
    [
        "BLOCK_SIZE",
        "NSPLIT",
        "WITH_SAMPLE",
    ],
)

_draft_sample_finalize_repr = make_kernel_repr(
    "_draft_sample_finalize",
    [
        "BLOCK_SIZE",
        "NSPLIT",
        "WITH_SAMPLE",
    ],
)

# Mirrors fast_sample()'s q.clamp_min_(torch.finfo(torch.float32).tiny): a zero
# exponential draw must not turn into an inf score that argmax would select.
_TINY_F32 = tl.constexpr(1.1754943508222875e-38)


@triton.jit(repr=_draft_sample_reduce_repr)
def _draft_sample_reduce(
    logits_ptr,
    temperature_ptr,
    seed_ptr,
    part_max_ptr,
    part_sum_ptr,
    part_key_ptr,
    part_logit_ptr,
    part_index_ptr,
    n_cols,
    logits_row_stride,
    CHUNK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NSPLIT: tl.constexpr,
    RAND_ROUNDS: tl.constexpr,
    TEMP_REPEAT: tl.constexpr,
    WITH_SAMPLE: tl.constexpr,
):
    """Per-(row, column-chunk) partial softmax statistics and Gumbel-max key.

    One program per chunk instead of torch's one program per row: the deployed
    speculative steps have 8-84 rows against 256 CUs, so the row-per-program
    shape that torch.softmax uses leaves the machine almost entirely idle.

    The Gumbel-max identity lets the draw be decided here, before the row is
    normalized: argmax_i(p_i / q_i) == argmax_i(x_i - log q_i) because p_i =
    exp(x_i - m) / Z and both m and Z are row constants. So no program needs the
    normalizer to pick the sample, and the winning logit is carried out
    alongside the key so its probability can be formed in the second pass.

    ``TEMP_REPEAT`` folds the caller's repeat_interleave of the per-request
    temperature over the draft-token axis into the row indexing; ``WITH_SAMPLE``
    is off on the verify side, which wants q only.
    """
    row = tl.program_id(0)
    part = tl.program_id(1)

    inv_t = 1.0 / tl.load(temperature_ptr + row // TEMP_REPEAT).to(tl.float32)
    row_ptr = logits_ptr + row * logits_row_stride

    col_begin = part * CHUNK
    col_end = tl.maximum(tl.minimum(col_begin + CHUNK, n_cols), col_begin)

    # Lane-local accumulators: the cross-lane reductions happen once at the end
    # instead of four times per block iteration.
    m = tl.full([BLOCK_SIZE], -float("inf"), tl.float32)
    s = tl.zeros([BLOCK_SIZE], tl.float32)
    best_key = tl.full([BLOCK_SIZE], -float("inf"), tl.float32)
    best_logit = tl.full([BLOCK_SIZE], -float("inf"), tl.float32)
    best_index = tl.zeros([BLOCK_SIZE], tl.int64)

    if WITH_SAMPLE:
        seed = tl.load(seed_ptr)
        rand_base = row * n_cols

    for b in tl.range(col_begin, col_end, BLOCK_SIZE):
        cols = b + tl.arange(0, BLOCK_SIZE)
        mask = cols < col_end
        x = (
            tl.load(row_ptr + cols, mask=mask, other=-float("inf"), cache_modifier=".cg")
            .to(tl.float32)
            * inv_t
        )

        m_new = tl.maximum(m, x)
        s = s * tl.exp(m - m_new) + tl.exp(x - m_new)
        m = m_new

        if WITH_SAMPLE:
            u = tl.rand(seed, rand_base + cols, n_rounds=RAND_ROUNDS)
            q = tl.maximum(-tl.log(u), _TINY_F32)
            key = tl.where(mask, x - tl.log(q), -float("inf"))

            take = key > best_key
            best_key = tl.where(take, key, best_key)
            best_logit = tl.where(take, x, best_logit)
            best_index = tl.where(take, cols.to(tl.int64), best_index)

    row_max = tl.max(m, axis=0)
    row_sum = tl.sum(s * tl.exp(m - row_max), axis=0)

    out = row * NSPLIT + part
    tl.store(part_max_ptr + out, row_max)
    tl.store(part_sum_ptr + out, row_sum)
    if WITH_SAMPLE:
        winner = tl.arange(0, BLOCK_SIZE) == tl.argmax(best_key, axis=0)
        tl.store(part_key_ptr + out, tl.max(best_key, axis=0))
        tl.store(part_logit_ptr + out, tl.sum(tl.where(winner, best_logit, 0.0), axis=0))
        tl.store(part_index_ptr + out, tl.sum(tl.where(winner, best_index, 0), axis=0))


@triton.jit(repr=_draft_sample_finalize_repr)
def _draft_sample_finalize(
    logits_ptr,
    probs_ptr,
    temperature_ptr,
    seed_ptr,
    part_max_ptr,
    part_sum_ptr,
    part_key_ptr,
    part_logit_ptr,
    part_index_ptr,
    sample_p_ptr,
    sample_index_ptr,
    n_cols,
    logits_row_stride,
    probs_row_stride,
    CHUNK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NSPLIT: tl.constexpr,
    TEMP_REPEAT: tl.constexpr,
    WITH_SAMPLE: tl.constexpr,
):
    """Reduce the per-chunk partials and write this program's slice of probs.

    Every program re-reduces the row's NSPLIT partials (NSPLIT <= 64 scalars, so
    cheaper than a third dispatch to broadcast them) and then streams its own
    chunk a second time to emit exp(x - m) / Z.
    """
    row = tl.program_id(0)
    part = tl.program_id(1)

    parts = tl.arange(0, NSPLIT)
    base = row * NSPLIT + parts
    p_max = tl.load(part_max_ptr + base)
    p_sum = tl.load(part_sum_ptr + base)

    m = tl.max(p_max, axis=0)
    inv_z = 1.0 / tl.sum(p_sum * tl.exp(p_max - m), axis=0)

    if WITH_SAMPLE:
        if part == 0:
            p_key = tl.load(part_key_ptr + base)
            winner = tl.argmax(p_key, axis=0)
            sel = tl.sum(
                tl.where(parts == winner, tl.load(part_index_ptr + base), 0), axis=0
            )
            sel_logit = tl.sum(
                tl.where(parts == winner, tl.load(part_logit_ptr + base), 0.0), axis=0
            )
            tl.store(sample_index_ptr + row, sel)
            tl.store(
                sample_p_ptr + row,
                (tl.exp(sel_logit - m) * inv_z).to(sample_p_ptr.dtype.element_ty),
            )
            if row == 0:
                # Advance the graph-resident counter so a captured replay draws
                # fresh coins instead of replaying the seed frozen in at capture.
                tl.store(seed_ptr, tl.load(seed_ptr) + 1)

    inv_t = 1.0 / tl.load(temperature_ptr + row // TEMP_REPEAT).to(tl.float32)
    in_ptr = logits_ptr + row * logits_row_stride
    out_ptr = probs_ptr + row * probs_row_stride

    col_begin = part * CHUNK
    col_end = tl.maximum(tl.minimum(col_begin + CHUNK, n_cols), col_begin)
    for b in tl.range(col_begin, col_end, BLOCK_SIZE):
        cols = b + tl.arange(0, BLOCK_SIZE)
        mask = cols < col_end
        x = (
            tl.load(in_ptr + cols, mask=mask, other=-float("inf"), cache_modifier=".cg")
            .to(tl.float32)
            * inv_t
        )
        p = tl.exp(x - m) * inv_z
        tl.store(out_ptr + cols, p.to(out_ptr.dtype.element_ty), mask=mask)


_shard_argmax_partial_repr = make_kernel_repr(
    "_shard_argmax_partial",
    [
        "BLOCK_SIZE",
        "NSPLIT",
    ],
)

_shard_argmax_reduce_repr = make_kernel_repr(
    "_shard_argmax_reduce",
    [
        "NSPLIT",
        "TP_SIZE",
    ],
)

# Sentinel for "this lane holds no candidate index"; any real vocab id is below
# it, so a min-reduction over it is a no-op.
_BIG_INDEX = tl.constexpr(2147483647.0)


@triton.jit(repr=_shard_argmax_partial_repr)
def _shard_argmax_partial(
    logits_ptr,
    part_ptr,
    n_cols,
    vocab_offset,
    vocab_limit,
    n_rows,
    logits_row_stride,
    CHUNK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NSPLIT: tl.constexpr,
):
    """Per-(row, column-chunk) max logit and the lowest column that attains it.

    Runs on this rank's vocab shard only, so the full-vocab all-gather that
    feeds the draft's greedy pick is replaced by an all-gather of these
    partials -- 12 bytes per (row, chunk) instead of two bytes per vocab entry.
    Columns are scanned left to right and only a strictly greater logit
    displaces the incumbent, so the lane-local winner is already the lowest
    index; the cross-lane step keeps that property by minimising over the lanes
    that tie on the maximum.
    """
    row = tl.program_id(0)
    part = tl.program_id(1)

    row_ptr = logits_ptr + row * logits_row_stride
    col_begin = part * CHUNK
    col_end = tl.maximum(tl.minimum(col_begin + CHUNK, n_cols), col_begin)

    m = tl.full([BLOCK_SIZE], -float("inf"), tl.float32)
    best = tl.full([BLOCK_SIZE], _BIG_INDEX, tl.float32)

    for b in tl.range(col_begin, col_end, BLOCK_SIZE):
        cols = b + tl.arange(0, BLOCK_SIZE)
        ids = cols + vocab_offset
        # The lm_head shard is padded up to a multiple of the TP size; entries
        # past the real vocab carry zero weights, so they must not win.
        mask = (cols < col_end) & (ids < vocab_limit)
        x = tl.load(
            row_ptr + cols, mask=mask, other=-float("inf"), cache_modifier=".cg"
        ).to(tl.float32)

        take = x > m
        m = tl.where(take, x, m)
        best = tl.where(take, ids.to(tl.float32), best)

    row_max = tl.max(m, axis=0)
    row_index = tl.min(tl.where(m == row_max, best, _BIG_INDEX), axis=0)

    out = row * NSPLIT + part
    tl.store(part_ptr + out, row_max)
    tl.store(part_ptr + n_rows * NSPLIT + out, row_index)


@triton.jit(repr=_shard_argmax_reduce_repr)
def _shard_argmax_reduce(
    part_ptr,
    index_ptr,
    n_rows,
    NSPLIT: tl.constexpr,
    TP_SIZE: tl.constexpr,
    LANES: tl.constexpr,
):
    """Reduce the gathered ``[TP_SIZE, 2, n_rows, NSPLIT]`` partials to one id.

    The gather is rank-major, so lane ``k`` reads rank ``k // NSPLIT``'s chunk
    ``k % NSPLIT``. Ties on the maximum resolve to the lowest vocab id, which is
    what ``torch.argmax`` on the gathered row would have returned.
    """
    row = tl.program_id(0)

    lanes = tl.arange(0, LANES)
    valid = lanes < TP_SIZE * NSPLIT
    rank = lanes // NSPLIT
    chunk = lanes % NSPLIT
    off = rank * (2 * n_rows * NSPLIT) + row * NSPLIT + chunk

    m = tl.load(part_ptr + off, mask=valid, other=-float("inf"))
    idx = tl.load(part_ptr + off + n_rows * NSPLIT, mask=valid, other=_BIG_INDEX)

    row_max = tl.max(m, axis=0)
    sel = tl.min(tl.where(m == row_max, idx, _BIG_INDEX), axis=0)
    tl.store(index_ptr + row, sel.to(tl.int64))
