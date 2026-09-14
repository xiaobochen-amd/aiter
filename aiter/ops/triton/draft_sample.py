# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from typing import Optional, Tuple

import torch
import triton

from aiter.ops.triton._triton_kernels.draft_sample import (
    _draft_sample_finalize,
    _draft_sample_reduce,
    _shard_argmax_partial,
    _shard_argmax_reduce,
)
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()

# Target program count, rounded DOWN to a power of two. The split factor -- not
# the row count -- is what fills the machine, and the two call sites sit in
# different regimes, so they carry their own targets. Draft (8-14 rows): 256
# programs beats 320 at bs=10 (69.4us vs 87.3us for five calls) because a longer
# per-program chunk amortises the tail iteration. Verify (48-84 rows) is already
# bandwidth-bound and wants the deeper split instead -- swept under graph replay
# at V=154880, 768 puts all three scored row counts on their measured best
# (30.3us at 48, 36.1 at 60, 51.9 at 84, vs 32.0/35.7/62.2 at a target of 256).
_DRAFT_TARGET_PROGRAMS = 256
_VERIFY_TARGET_PROGRAMS = 768
_MAX_NSPLIT = 64
_BLOCK_SIZE = 2048
# Philox rounds, matching torch's own generator. Dropping to 4 buys ~18us of a
# ~550us saving (<0.07% of a decode step), which is not worth weakening the
# draw: rejection-sampling correctness assumes X really is distributed as q.
_RAND_ROUNDS = 10

# One int32 counter per device, advanced inside the finalize kernel. It has to be
# a device tensor: a host-side seed would be baked into the draft CUDA graph at
# capture time and every replay would then reuse the same coins.
_SEEDS: dict = {}


def _seed_tensor(device: torch.device) -> torch.Tensor:
    seed = _SEEDS.get(device)
    if seed is None:
        seed = torch.randint(0, 2**30, (1,), dtype=torch.int32, device=device)
        _SEEDS[device] = seed
    return seed


def _run(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    probs_out: Optional[torch.Tensor],
    target_programs: int,
    temp_repeat: int,
    with_sample: bool,
):
    """Two column-split passes over ``logits``: partial stats, then normalize.

    Both speculative call sites reduce ``[rows, vocab]`` with rows well below the
    CU count, so the split factor is chosen from the target program count rather
    than from the rows. ``temp_repeat`` folds the caller's repeat_interleave of
    the per-request temperature into the row indexing.
    """
    n_rows, n_cols = logits.shape

    want = max(1, target_programs // max(1, n_rows))
    nsplit = min(_MAX_NSPLIT, 1 << (want.bit_length() - 1))
    nsplit = min(nsplit, max(1, triton.cdiv(n_cols, _BLOCK_SIZE)))
    chunk = triton.cdiv(n_cols, nsplit)

    device = logits.device
    # torch.softmax(logits / temperatures) promotes, so a bf16 logit with an
    # fp32 temperature yields fp32 probs; the caller's downstream buffers are
    # typed for that, so the fused op has to promote identically.
    probs_dtype = torch.promote_types(logits.dtype, temperatures.dtype)
    probs = (
        torch.empty(logits.shape, dtype=probs_dtype, device=device)
        if probs_out is None
        else probs_out
    )
    parts = torch.empty((4, n_rows * nsplit), dtype=torch.float32, device=device)
    part_index = torch.empty((n_rows * nsplit,), dtype=torch.int64, device=device)
    sample_p = torch.empty((n_rows, 1), dtype=probs.dtype, device=device)
    sample_index = torch.empty((n_rows, 1), dtype=torch.int64, device=device)
    seed = _seed_tensor(device)

    grid = (n_rows, nsplit)
    _draft_sample_reduce[grid](
        logits,
        temperatures,
        seed,
        parts[0],
        parts[1],
        parts[2],
        parts[3],
        part_index,
        n_cols,
        logits.stride(0),
        CHUNK=chunk,
        BLOCK_SIZE=_BLOCK_SIZE,
        NSPLIT=nsplit,
        RAND_ROUNDS=_RAND_ROUNDS,
        TEMP_REPEAT=temp_repeat,
        WITH_SAMPLE=with_sample,
        num_warps=8,
        waves_per_eu=2,
    )
    _draft_sample_finalize[grid](
        logits,
        probs,
        temperatures,
        seed,
        parts[0],
        parts[1],
        parts[2],
        parts[3],
        part_index,
        sample_p,
        sample_index,
        n_cols,
        logits.stride(0),
        probs.stride(0),
        CHUNK=chunk,
        BLOCK_SIZE=_BLOCK_SIZE,
        NSPLIT=nsplit,
        TEMP_REPEAT=temp_repeat,
        WITH_SAMPLE=with_sample,
        num_warps=8,
        waves_per_eu=2,
    )
    return probs, sample_p, sample_index


def draft_sample(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    probs_out: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused softmax + Gumbel-max draw for the EAGLE draft proposal.

    Replaces the seven full-vocab passes of
    ``softmax -> empty_like -> exponential_ -> clamp_min_ -> div -> argmax ->
    gather`` with two, and gives both of them a grid that fills the device.

    Args:
        logits: ``[n_rows, n_cols]`` draft logits, contiguous along the row.
        temperatures: ``[n_rows]`` or ``[n_rows, 1]`` sampling temperatures.
        probs_out: optional preallocated ``[n_rows, n_cols]`` output for q.

    Returns:
        ``(probs, sample_p, sample_index)`` -- q, q(X) and X, matching
        ``sglang.srt.speculative.spec_utils.sample_draft_proposal``.
    """
    assert logits.dim() == 2 and logits.stride(1) == 1, "logits must be row-contiguous"
    _LOGGER.info(f"DRAFT_SAMPLE: logits={tuple(logits.shape)}")
    dtype = torch.promote_types(logits.dtype, temperatures.dtype)
    if logits.shape[0] == 0:
        probs = (
            torch.empty(logits.shape, dtype=dtype, device=logits.device)
            if probs_out is None
            else probs_out
        )
        return (
            probs,
            torch.empty((0, 1), dtype=dtype, device=logits.device),
            torch.empty((0, 1), dtype=torch.int64, device=logits.device),
        )
    return _run(logits, temperatures, probs_out, _DRAFT_TARGET_PROGRAMS, 1, True)


def verify_target_probs(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    draft_token_num: int,
    probs_out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Fused ``softmax(logits / repeat_interleave(T, draft_token_num))``.

    The EAGLE verify step feeds the rejection sampler a ``[bs * draft_token_num,
    vocab]`` distribution built by three torch kernels -- a repeat_interleave of
    the temperatures, a full-vocab divide, and a one-program-per-row softmax.
    All three collapse into the two passes here.

    Args:
        logits: ``[bs * draft_token_num, vocab]`` target logits, row-contiguous.
        temperatures: ``[bs]`` or ``[bs, 1]`` per-request temperatures.
        draft_token_num: rows per request, i.e. the repeat_interleave factor.
        probs_out: optional preallocated fp32 output.

    Returns:
        ``[bs * draft_token_num, vocab]`` fp32 probabilities.
    """
    assert logits.dim() == 2 and logits.stride(1) == 1, "logits must be row-contiguous"
    _LOGGER.info(f"VERIFY_TARGET_PROBS: logits={tuple(logits.shape)}")
    if logits.shape[0] == 0:
        return (
            torch.empty(logits.shape, dtype=torch.float32, device=logits.device)
            if probs_out is None
            else probs_out
        )
    probs, _, _ = _run(
        logits,
        temperatures,
        probs_out,
        _VERIFY_TARGET_PROGRAMS,
        draft_token_num,
        False,
    )
    return probs


# The greedy draft pick only needs one id per row, so the vocab shard can stay
# where it was produced: each rank reduces its own slice and the ranks exchange
# partials instead of logits. Target program count is picked the same way as the
# sampling path -- the rows (8-14) cannot fill the device on their own.
_ARGMAX_TARGET_PROGRAMS = 256


def _argmax_nsplit(n_rows: int, n_cols: int) -> int:
    want = max(1, _ARGMAX_TARGET_PROGRAMS // max(1, n_rows))
    nsplit = min(_MAX_NSPLIT, 1 << (want.bit_length() - 1))
    return min(nsplit, max(1, triton.cdiv(n_cols, _BLOCK_SIZE)))


def shard_argmax_partials(
    logits: torch.Tensor,
    vocab_offset: int = 0,
    vocab_limit: Optional[int] = None,
) -> Tuple[torch.Tensor, int]:
    """Per-chunk (max logit, lowest winning vocab id) for this rank's shard.

    Args:
        logits: ``[n_rows, shard_cols]`` logits of this rank's vocab shard,
            contiguous along the row.
        vocab_offset: global id of this shard's first column.
        vocab_limit: first global id that is padding; defaults to no padding.

    Returns:
        ``(partials, nsplit)`` where ``partials`` is a flat fp32 tensor viewing
        ``[2, n_rows, nsplit]`` -- maxima then ids. All-gather it along dim 0
        and hand the result to :func:`shard_argmax_reduce`.
    """
    assert logits.dim() == 2 and logits.stride(1) == 1, "logits must be row-contiguous"
    n_rows, n_cols = logits.shape
    nsplit = _argmax_nsplit(n_rows, n_cols)
    parts = torch.empty((2 * n_rows * nsplit,), dtype=torch.float32, device=logits.device)
    _shard_argmax_partial[(n_rows, nsplit)](
        logits,
        parts,
        n_cols,
        vocab_offset,
        vocab_offset + n_cols if vocab_limit is None else vocab_limit,
        n_rows,
        logits.stride(0),
        CHUNK=triton.cdiv(n_cols, nsplit),
        BLOCK_SIZE=_BLOCK_SIZE,
        NSPLIT=nsplit,
        num_warps=8,
        waves_per_eu=2,
    )
    return parts, nsplit


def shard_argmax_reduce(
    partials: torch.Tensor,
    n_rows: int,
    nsplit: int,
    tp_size: int,
) -> torch.Tensor:
    """Reduce gathered shard partials to ``[n_rows, 1]`` global argmax ids."""
    index = torch.empty((n_rows, 1), dtype=torch.int64, device=partials.device)
    lanes = 1 << max(0, (tp_size * nsplit - 1)).bit_length()
    _shard_argmax_reduce[(n_rows,)](
        partials,
        index,
        n_rows,
        NSPLIT=nsplit,
        TP_SIZE=tp_size,
        LANES=max(2, lanes),
        num_warps=1,
    )
    return index


def greedy_argmax(
    logits: torch.Tensor,
    vocab_offset: int = 0,
    vocab_limit: Optional[int] = None,
) -> torch.Tensor:
    """Single-rank ``argmax(logits, dim=-1, keepdim=True)`` over a full row.

    Same lowest-id tie-break as the sharded pair above, which is what the
    speculative draft needs and what ROCm's ``torch.argmax`` does not guarantee
    (sglang #26358).
    """
    _LOGGER.info(f"GREEDY_ARGMAX: logits={tuple(logits.shape)}")
    parts, nsplit = shard_argmax_partials(logits, vocab_offset, vocab_limit)
    return shard_argmax_reduce(parts, logits.shape[0], nsplit, 1)
