# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Public launcher for gfx950 FlyDSL sparse MLA decode."""

from __future__ import annotations

import functools
import math

import flydsl.expr as fx
import torch

from .kernels.sparse_mla_decode import (
    BARRIER_SLOTS,
    BARRIER_STRIDE,
    BLOCK_I,
    DIM,
    DV,
    H,
    compile_sparse_mla_partial,
)
from .kernels.tensor_shim import _run_compiled, ptr_arg
from .mla_reduce_kernels import _flydsl_sparse_mla_decode_combine

# Largest KV pool the producer can address with 32-bit byte offsets.
_BUFFER_MAX_BYTES = 1 << 31
# L2 domains the dispatcher rotates workgroups through on gfx950.
_XCDS = 8


@functools.lru_cache(maxsize=8)
def _num_cu(device_index: int) -> int:
    return int(torch.cuda.get_device_properties(device_index).multi_processor_count)


def _pick_inner_iter(seq: int, ng_total: int) -> int:
    """Return the power-of-two producer grouping factor for this shape.

    `_decode_partial_groups` places the grid directly and only falls back here
    for the shapes where no placement fits the CU array once -- see its
    docstring. Each producer CTA is one wavefront handling `inner_iter` 64-key
    tiles, so
    the grid is `seq * (ng_total // inner_iter)` CTAs. Two effects compete: a
    larger grouping shortens the grid (and the combine, which sees one partial
    row per group) while a smaller one gives the CU array more independent
    tiles to overlap.

    The producer now prefetches tile k+1's gather while tile k's softmax and PV
    run, so a CTA covers its own memory latency as long as it owns more than one
    tile, and the balance moved decisively toward the larger grouping. Measured
    on the ruler at topk=2048 (ng_total 32), speedup per decode shape:

        seq   ii=2    ii=4    ii=8
         48  1.028   1.096   1.086
         60  1.082   1.070   1.099
         72  1.082   1.085   0.837
         84      -   1.153   0.938
         96      -   1.139   0.993

    inner_iter 4 wins or ties everywhere -- seq 48 by 6.6% over the tail-utilisation
    tie-break this function used to apply -- while 8 collapses as soon as the grid
    drops near one CTA per CU (seq 72 and up). So take the largest grouping the
    CTA-count floors allow and stop at 4.
    """
    inner_iter = 1
    while inner_iter < 4:
        candidate = inner_iter * 2
        if ng_total % candidate != 0:
            break
        min_producer_ctas = 192 if candidate == 2 else 384
        if seq * (ng_total // candidate) < min_producer_ctas:
            break
        inner_iter = candidate
    return inner_iter


# Most 64-key tiles one producer CTA is allowed to own. Fatter CTAs win while
# the grid still fits the CU array once, but the fattest split ng=32 admits
# loses at seq 84, so the search stops one step short.
_DECODE_MAX_TILES = 11


def _decode_partial_groups(seq: int, ng_total: int, num_cu: int) -> int:
    """Return how many partial records the producer splits one token into.

    A CTA owns `ceil(ng_total / n_groups)` of the token's 64-key tiles and the
    grid is `seq * n_groups` CTAs, so this one number fixes both the grid and
    the combine's input length. It used to be `ng_total // inner_iter` with
    `inner_iter` a power of two, which left the grid quantised in factor-of-two
    steps. The producer now takes a ragged split, so every count in
    `1..ng_total` is reachable and the grid can be placed exactly.

    Where to place it is sharp. Measured on the full producer+combine call
    (same process, ABBA, min of 5-6 palindromic passes, against the shipped
    grouping, with a duplicate shipped arm at the tail pricing position bias at
    0.03-1.6%):

        seq   n_groups   ctas   ctas/CU   tiles/CTA    delta
         24       8       192      0.75       4        -3.4%
         24      10       240      0.94       4        -1.1%
         32       8       256      1.00       4        -9.2%
         32       9       288      1.13       4        +9.7%
         40       6       240      0.94       6       -14.9%
         40       7       280      1.09       5        +2.7%
         48       5       240      0.94       7        -3.4%
         48       6       288      1.13       6       +18.7%
         60       5       300      1.17       7       +21.6%
         72       3       216      0.84      11       -11.1%
         72       4       288      1.13       8        +4.6%
         84       3       252      0.98      11        -8.1%
         84       4       336      1.31       8        +2.5%

    One CTA per CU is a cliff, not a slope: every count that stays under it
    wins by 3-15% and the first one over it loses by 2-22%, because a CU handed
    a second producer CTA serialises a whole fat CTA behind the first. So take
    the shortest CTA -- the largest `n_groups` -- that still fits the array
    once. Among equally short CTAs take the smallest count, because a ragged
    split pays for its padding tile: at seq 24 counts 8 and 10 both own four
    tiles and 10 spends a quarter of them on padding, which is the whole 2.3%
    between those two rows. The padding costs skeleton, not traffic -- an out
    of range row number is killed by the buffer bounds check, so a padding tile
    moves no bytes.

    What a tile slot is worth, and therefore what a perfectly packed grid could
    buy, is priced at seq 48 by sweeping `ng_total` at a fixed `n_groups` of 5,
    which moves how many of the five CTAs carry a seventh *real* tile while
    every arm from 31 up runs seven slots (control arm 0.06-0.6%):

        ng_total   slots   CTAs at 7 real   us
            30       6            0        14.58
            31       7            1        15.07
            32       7            2        15.28   (ship)
            33       7            3        15.43

    The first real tile on the critical CTA is 0.49 us; every further CTA that
    converts padding to real work is only 0.18 us, which is exactly the 6.5%
    the added rows put on the gather's 2.6 us of exposed DRAM. So the wall is
    the busiest CTA's real tiles plus footprint, and the slack in a 240/256
    grid absorbs the rest.

    Widening that sweep to five slots separates the two costs. At `ng_total`
    25 / 30 / 32 the same grid runs 5 / 6 / 7 slots over 1200 / 1440 / 1536
    real tiles and measures 12.49 / 13.88 / 14.47 us, which resolves to 5.56 us
    a call, 63 ns a slot and 5.5 ns a real tile: the wall tracks the grid's
    total real tiles rather than the busiest CTA's slot count, and a padding
    slot is worth 63 ns, not a tile. A cross-token schedule that packed all
    1536 tiles into 256 CTAs of six slots therefore only drops the three
    padding slots -- 14.41 us against 14.47, or 0.4% -- and not the 3.5% a
    critical-path reading of the rows above suggests, which does not repay the
    second Q block and second partial record such a CTA has to carry.

    When no count fits the array once inside `_DECODE_MAX_TILES` -- which needs
    `seq` past eight times that -- keep the power-of-two grouping.
    """
    best = None
    for n_groups in range(1, ng_total + 1):
        if seq * n_groups > num_cu:
            break
        tiles = -(-ng_total // n_groups)
        if tiles <= _DECODE_MAX_TILES and (best is None or tiles < best[0]):
            best = (tiles, n_groups)
    if best is None:
        return ng_total // _pick_inner_iter(seq, ng_total)
    return best[1]


def _pick_xpf_prime(inner_iter: int, producer_ctas: int, num_cu: int) -> int:
    """Return how many tiles the producer's prologue issues before the Q publish.

    The steady-state pipeline keeps one tile's gather in flight, issued from the
    body of the tile before it. The first two tiles have no earlier body to be
    issued from, so their two index->KV chases run back to back while the CU has
    nothing else to do. Issuing both before the Q publish folds them into one
    burst, which is free in registers (VGPR 162 -> 166, still three CTAs per CU)
    and bit-exact.

    It only pays while the CU has spare gather capacity. Measured on the full
    producer+combine call (same process, ABBA, min of 5 palindromic passes):

        ctas/CU   inner_iter=4   inner_iter=8
            0.63              -          -9.5%
           0.75-1.0           -       -2.2% .. -1.9%
            1.25          -0.0%              -
            1.5           -2.2%              -
           1.75-2.0    -1.7% .. -2.7%        -
            2.25          +2.0%              -
            2.6           +0.8%              -
            3.0           +0.6%              -

    Past two CTAs per CU the third co-resident CTA already has the outstanding
    misses the deeper burst was meant to add, and the extra queueing shows up as
    wall time. Groupings below 4 own too few tiles to prefetch at all: issuing
    two of an inner_iter=2 CTA's tiles up front is the whole CTA at once, which
    is the `_XPF_DEPTH=2` schedule measured at +1.6% on seq 14.

    A grouping that lifts its index column has no chase left to fold, so the
    second tile is pure cost there and the table above inverts: against a
    duplicate shipped arm it reads -6.2% at seven tiles and -5.0% at eight to
    drop back to one, bit identical, while the groupings that still read their
    rows inside the gather keep wanting two (+1.9 .. +2.4% at eleven tiles).
    The chase was throttling the burst: it is now the prologue's only
    outstanding traffic, and doubling it doubles what the first body's LDS
    writes wait on.
    """
    if inner_iter < 4 or _pick_index_prefetch(inner_iter):
        return 1
    return 2 if producer_ctas <= 2 * num_cu else 1


# Groupings whose index rows are all fetched before the Q publish, keyed by
# tiles per CTA. Everything else keeps a tile's rows inside its own gather. The
# depth is all or nothing -- a bounded rolling depth reads flat to worse.
_INDEX_PREFETCH = {2: 2, 7: 7, 8: 8, 11: 11}

# Which of those entries lift only the wide pair rather than all three rows.
# Eleven tiles is the one shape where the full column loses, and the cost there
# is pair-forming, not spill: two registers a tile instead of three.
_INDEX_PREFETCH_ROWS_ONLY = {11}


def _pick_index_rows_only(inner_iter: int) -> bool:
    """Whether this grouping lifts only the wide index pair."""
    return inner_iter in _INDEX_PREFETCH_ROWS_ONLY


def _pick_index_prefetch(inner_iter: int) -> int:
    """Return how many tiles of index rows the producer keeps in flight.

    Zero -- the default -- fetches a tile's rows inside its own gather, so the
    steady-state body pays an index round trip before it can issue the nine
    `buffer_load_dwordx4` that row numbers address. Lifting the whole CTA's
    rows into the prologue removes that, at three live registers per tile, and
    the two effects do not trade off monotonically. Measured against the
    shipped path on the full call (same process, ABBA, min of 6-8, against a
    duplicate shipped arm pricing in-block position bias, all bit identical):

        seq  tiles  n_groups  CTAs/CU   delta
         14      2        16     0.88   -0.75%
         24      4         8     0.75   +1.24%
         36      5         7     0.98   +2.40%
         42      6         6     0.98   +1.64%
         48      7         5     0.94    0.00%
         60      8         4     0.94   -0.81%
         72     11         3     0.84    0.00%
         84     11         3     0.98   +6.31%
         96      4         8     3.00   -0.30%

    So take the depth from a table of the groupings that measured a win and
    leave the rest alone. A depth between the two ends is not a third option:
    rolling four tiles ahead measures +0.8% at seven tiles per CTA and +0.46%
    at eleven, in both cases worse than either lifting all of them or none.

    Nor are the losses register pressure. At the depth this lifts to, eleven
    tiles is 200 VGPRs against 172 and eight tiles is 166 against 170 -- no
    spill anywhere, and all of it inside the 256 that would keep two waves on
    a SIMD -- while the eleven-tile streams differ by ten `v_pk_mul_f32` the
    allocator could no longer pair.

    Parking the column in LDS instead buys the register back and costs
    `inner_iter * 256` bytes of a budget that prices occupancy at 0.1-0.2%
    here, but it puts a `ds_read_b32` back on the gather's address path -- the
    round trip this lever exists to remove -- and reads +6.7% at seven tiles,
    +2.3% at eight and +1.9% at eleven, bit identical each time. The register
    column is the only form that pays.
    """
    return _INDEX_PREFETCH.get(inner_iter, 0)


def _fuse_combine(seq: int, n_groups: int, num_cu: int) -> bool:
    """Report whether the reduction can ride in the producer's epilogue.

    What it buys is one graph node. A node on this device costs 1.5-1.8 us of
    wall whatever it contains -- measured by launching the reducer twice in the
    same graph and differencing -- and the reducer's own body is only 0.2-0.6 us
    of that, so the launch, not the reduction, is what the decode step pays for
    84 times.

    Two conditions:

    * Residency. The fused form rendezvouses across CTAs, which only terminates
      if a token's splits are all resident at once. `num_cu` is the conservative
      bound: the dispatcher walks workgroups round robin over the CU array, so a
      grid no larger than the array puts at most one CTA on a CU, and the
      producer's occupancy -- two CTAs per CU at its LDS and register footprint
      -- covers that twice over.
    * Column length. Past 16 splits the reduction stops being worth carrying:
      the epilogue rebuilds the softmax weights with a cross-lane butterfly
      instead of a scalar column, the per-lane slice narrows to a single dword,
      and 32 CTAs contend for one token's counter. Measured against the two
      kernel path, same block ABBA, all bit identical: 16 splits is -6.1% at
      seq 10 and -7.6% at seq 14, and 32 splits is +3.2% at seq 8.
    """
    return 2 <= n_groups <= 16 and seq * n_groups <= num_cu


@functools.lru_cache(maxsize=8)
def _barrier_scratch(device_index: int) -> torch.Tensor:
    """Per-token rendezvous counters for the fused reduction, one line each.

    Cached per device so a HIP graph captures a stable address, and zeroed only
    here: the barrier is sense reversing, so every launch leaves each counter on
    the value the next one expects whatever the split count. That is the whole
    reason for the sense trick -- counters that had to be cleared between
    launches would need either a second kernel or a host memset inside the
    captured graph, which is the cost the fusion is trying to remove.
    """
    return torch.zeros(
        BARRIER_SLOTS * BARRIER_STRIDE,
        dtype=torch.int32,
        device=torch.device("cuda", device_index),
    )


def _use_split_major(seq: int, n_groups: int, num_cu: int) -> bool:
    """Use split-major ownership once the producer grid is saturated."""
    return seq * n_groups >= 2 * num_cu


def _split_major_folds_q(seq: int, n_groups: int) -> bool:
    """Report whether split-major ownership shrinks the Q fetch fan-out.

    Workgroups go round robin over the device's `_XCDS` L2 domains, so with
    token-major ownership (`owner = tok * n_groups + split`) a token's CTAs land
    on `min(n_groups, _XCDS)` different domains and every one of them pulls the
    token's whole 9 KB Q block off chip. Split-major (`owner = split * seq +
    tok`) steps `owner` by `seq`, so the fan-out drops to
    `_XCDS // gcd(seq, _XCDS)` -- one domain whenever seq is a multiple of
    `_XCDS`, two for the odd multiples of four.

    Measured on the producer alone (same process, ABBA, min of 12): switching
    seq 48 and 60 over is -3.4% and -5.2%, and a constant-Q ablation prices the
    whole off-chip Q fetch at 4.2% and 3.5% there, so nearly all of it is this
    fan-out. The saturation rule above left both shapes token-major.
    """
    return _XCDS // math.gcd(seq, _XCDS) < min(n_groups, _XCDS)


def sparse_mla_decode_workspace_shape(
    seq: int, width: int
) -> tuple[tuple[int, int, int, int], tuple[int, int, int]]:
    """Return the partial-output and partial-LSE shapes for sparse MLA decode."""
    if not 1 <= seq <= 96:
        raise ValueError(f"supported sparse decode seq values are 1..96; got {seq}")
    if width % BLOCK_I != 0:
        raise ValueError(f"index width must be padded to {BLOCK_I}, got {width}")
    ng = width // BLOCK_I
    if not 1 <= ng <= 33:
        raise ValueError(f"supported split count is 1..33, got {ng}")
    ng_partial = _decode_partial_groups(seq, ng, _num_cu(torch.cuda.current_device()))
    return (seq, ng_partial, H, DV), (seq, ng_partial, H)


def _require_cuda_tensor(
    name: str, tensor: torch.Tensor, *, dtype: torch.dtype
) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor)!r}")
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be a ROCm tensor, got {tensor.device}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    if tensor.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}, got {tensor.dtype}")


def _validate_sparse_decode_inputs(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    out: torch.Tensor | None,
) -> tuple[int, int]:
    _require_cuda_tensor("q", q, dtype=torch.float8_e4m3fn)
    _require_cuda_tensor("kv", kv, dtype=torch.float8_e4m3fn)
    _require_cuda_tensor("indices", indices, dtype=torch.int32)
    if out is not None:
        _require_cuda_tensor("out", out, dtype=torch.bfloat16)

    if q.ndim != 3 or tuple(q.shape[1:]) != (H, DIM):
        raise ValueError(f"q must have shape [seq,{H},{DIM}], got {tuple(q.shape)}")
    seq = int(q.shape[0])
    # seq is runtime; kernels are compiled per split count `ng`, not per seq.
    if not 1 <= seq <= 96:
        raise ValueError(f"supported sparse decode seq values are 1..96; got {seq}")
    if not (
        (kv.ndim == 2 and int(kv.shape[1]) == DIM)
        or (kv.ndim == 3 and tuple(kv.shape[1:]) == (1, DIM))
    ):
        raise ValueError(
            f"kv must have shape [P,{DIM}] or [P,1,{DIM}], got {tuple(kv.shape)}"
        )
    if out is not None and (out.ndim != 3 or tuple(out.shape) != (seq, H, DV)):
        raise ValueError(
            f"out must have shape [{seq},{H},{DV}], got {tuple(out.shape)}"
        )
    if (
        q.device != kv.device
        or q.device != indices.device
        or (out is not None and q.device != out.device)
    ):
        raise ValueError("all sparse decode tensors must be on the same device")

    width = int(indices.numel() // seq)
    if tuple(indices.shape) != (seq, width):
        raise ValueError(
            f"indices must have shape [{seq},width], got {tuple(indices.shape)}"
        )
    if width % BLOCK_I != 0:
        raise ValueError(f"index width must be padded to {BLOCK_I}, got {width}")
    ng = width // BLOCK_I
    if not 1 <= ng <= 33:
        raise ValueError(f"supported split count is 1..33, got {ng}")

    arch = str(torch.cuda.get_device_properties(q.device).gcnArchName).split(":")[0]
    if arch != "gfx950":
        raise ValueError(f"FlyDSL sparse MLA decode is gated to gfx950, got {arch}")
    return seq, ng


def _validate_workspace(
    partial_output: torch.Tensor,
    partial_lse: torch.Tensor,
    *,
    seq: int,
    ng_partial: int,
    device: torch.device,
) -> None:
    _require_cuda_tensor("partial_output", partial_output, dtype=torch.bfloat16)
    _require_cuda_tensor("partial_lse", partial_lse, dtype=torch.float32)
    if partial_output.device != device or partial_lse.device != device:
        raise ValueError("sparse decode workspace must share the decode device")
    if tuple(partial_output.shape) != (seq, ng_partial, H, DV):
        raise ValueError(
            f"partial_output must have shape [{seq},{ng_partial},{H},{DV}], got "
            f"{tuple(partial_output.shape)}"
        )
    if tuple(partial_lse.shape) != (seq, ng_partial, H):
        raise ValueError(
            f"partial_lse must have shape [{seq},{ng_partial},{H}], got "
            f"{tuple(partial_lse.shape)}"
        )


def _launch_partial(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    partial_output: torch.Tensor,
    partial_lse: torch.Tensor,
    out: torch.Tensor,
    sm_scale: float,
    *,
    ng: int,
    n_groups: int,
    fuse_combine: bool,
) -> None:
    inner_iter = -(-ng // n_groups)
    seq = int(q.shape[0])
    num_cu = _num_cu(q.device.index)
    split_major = _split_major_folds_q(seq, n_groups) or _use_split_major(
        seq, n_groups, num_cu
    )
    # The gather addresses the pool with 32-bit buffer offsets, which is one
    # VGPR per address instead of a 64-bit pair. Pools that cannot be reached
    # that way fall back to 64-bit pointer arithmetic.
    use_buffer = kv.numel() * kv.element_size() < _BUFFER_MAX_BYTES
    launch = compile_sparse_mla_partial(
        ng,
        inner_iter=inner_iter,
        split_major=split_major,
        use_buffer=use_buffer,
        xpf_prime=_pick_xpf_prime(inner_iter, seq * n_groups, num_cu),
        ixpf_depth=_pick_index_prefetch(inner_iter),
        ixpf_rows_only=_pick_index_rows_only(inner_iter),
        n_groups=n_groups,
        fuse_combine=fuse_combine,
    )
    _run_compiled(
        launch,
        ptr_arg(q, fx.Uint8),
        ptr_arg(kv.reshape(-1, DIM), fx.Uint8),
        ptr_arg(indices, fx.Int32),
        ptr_arg(partial_output, fx.BFloat16),
        ptr_arg(partial_lse, fx.Float32),
        ptr_arg(out, fx.BFloat16),
        ptr_arg(_barrier_scratch(q.device.index), fx.Int32),
        float(sm_scale) * math.log2(math.e),
        int(q.shape[0]),
        fx.Stream(torch.cuda.current_stream(q.device)),
    )


def flydsl_sparse_mla_decode(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    out: torch.Tensor,
    sm_scale: float,
    *,
    partial_output: torch.Tensor | None = None,
    partial_lse: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run sparse MLA decode via FlyDSL partials and the shared reducer.

    Provide persistent ``partial_output`` and ``partial_lse`` buffers when
    capturing the call in a HIP graph. If they are omitted, temporary scratch is
    allocated eagerly for convenience.
    """
    seq, ng = _validate_sparse_decode_inputs(q, kv, indices, out)
    ng_partial = _decode_partial_groups(seq, ng, _num_cu(q.device.index))
    if (partial_output is None) != (partial_lse is None):
        raise ValueError(
            "partial_output and partial_lse must be provided together for "
            "graph-safe scratch reuse"
        )
    if partial_output is None:
        partial_output = torch.empty(
            (seq, ng_partial, H, DV), device=q.device, dtype=torch.bfloat16
        )
        partial_lse = torch.empty(
            (seq, ng_partial, H), device=q.device, dtype=torch.float32
        )
    else:
        _validate_workspace(
            partial_output,
            partial_lse,
            seq=seq,
            ng_partial=ng_partial,
            device=q.device,
        )

    fuse_combine = _fuse_combine(seq, ng_partial, _num_cu(q.device.index))
    _launch_partial(
        q,
        kv,
        indices,
        partial_output,
        partial_lse,
        out,
        sm_scale,
        ng=ng,
        n_groups=ng_partial,
        fuse_combine=fuse_combine,
    )
    if not fuse_combine:
        _flydsl_sparse_mla_decode_combine(
            partial_output.unsqueeze(0),
            partial_lse.unsqueeze(0),
            out.unsqueeze(0),
        )
    return out


__all__ = ["flydsl_sparse_mla_decode", "sparse_mla_decode_workspace_shape"]
