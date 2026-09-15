# SPDX-License-Identifier: MIT

"""gfx950 sparse-MLA decode producer with 64-key split granularity."""

# FlyDSL kernel annotations must be evaluated eagerly.
import functools
from contextlib import contextmanager

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm, scf
from flydsl.expr import arith
from flydsl.expr import math as fly_math
from flydsl.expr.arith import CmpIPredicate
from flydsl.expr.arith import _to_raw as _raw
from flydsl.expr.typing import T

from . import buffer_ops
from .mla_reduce import (
    _pointer_buffer_tensor,
    _store_final_out,
    _tree_reduce,
    _uniform_i32,
)

H = 16
DV = 512
DT = 64
DIM = DV + DT
BLOCK_I = 64
FP8_MAX = 448.0
PARTIAL_THREADS = 256
PARTIAL_WAVES = 4
PITCH = DV + 16
# Tiles whose gather is in flight at once inside a producer CTA. Two is slower
# everywhere it has been measured, and not because of occupancy: at grouping 8
# the grid is 0.75 CTAs per CU, so the second tile's 36 live gather registers
# cost nothing, and it is still +0.4% at seq 48 and +9.2% at seq 84. The
# prologue is the one place a deeper burst pays -- see `_pick_xpf_prime`.
_XPF_DEPTH = 1

# Sense bit of a barrier counter, the poll ceiling that keeps a grid the
# hardware did not co-schedule from wedging the device, and the backoff between
# polls -- see `_arrive_and_wait`.
_BARRIER_SENSE = -(1 << 31)
_BARRIER_MAX_POLLS = 1 << 20
_BARRIER_SLEEP = 1
# i32 slots per counter. One 128 B line each: the counters are hammered by
# uncached polls, and two on a line would serialise two tokens' rendezvous.
BARRIER_STRIDE = 32
BARRIER_SLOTS = 96

# CPol bits: SC0 in bit 0, SC1 in bit 4. Both set makes a store write through
# past every cache and a load read past every cache, which is what a reducer on
# another L2 domain needs. A token's splits only share a domain when the
# dispatcher's `block % 8` fan-out happens to keep them together, which under
# split-major ownership means `seq % 8 == 0`: drop these bits and seq 48 stays
# bit identical while seq 60 and seq 84 lose 0.4% of the output to stale reads,
# with the partials themselves still correct on disk.
#
# Carrying the policy per instruction rather than fencing is what makes the
# fusion pay, and it is not even a trade: an agent-scope fence pair was priced
# at 5.1 us for the release writeback and 1.3 us for the acquire invalidate,
# against 1.3-2.3 us for this whole epilogue, because a fence is a whole-cache
# operation that every CTA issues. Against plain cached access it is still
# 0.3-0.4 us cheaper here -- a partial is read once, by another CU, so leaving
# it in the writer's L2 only buys the reader a miss.
_CPOL_DEVICE = 0x11

def _fused_combine_plan(n_groups: int) -> list[tuple[int, int, int]]:
    """Rounds of `(dv_slices, slots, head_base)` reducing one token's H heads.

    The standalone reducer splits a head across lanes to size its *grid*; here
    the grid is already the producer's, so what the split sizes instead is how
    a token's work divides among the `4 * n_groups` waves that produced it. A
    slot is one 64-lane job holding `DV / (64 * dv)` of one head's values, so a
    round hands each wave at most one slot and `dv` trades store width against
    how many waves a head keeps busy.

    Narrow slots win, so `dv` minimises round count and nothing else: three
    rounds of `dv=2` beat two of `dv=1` by 3.7% at twelve waves, and a
    mixed-width schedule that saves a round costs 0.96%.

    Staging the rounds' reads across the round loop changes what a round costs,
    so this width was rescanned with that in place rather than inherited: at
    seq 84, neither halving nor doubling it separated from a duplicate shipped
    arm, so the choice above stands on its own measurements.

    Which head a wave takes within a round is not a lever. Slots are packed
    head-major, so the `dv` waves sharing a head each read that head's LSE
    column and a wave's head changes every round; cutting instead by DV slice,
    so a wave keeps its head and a chunk reads one column for all `dv` of its
    rounds, removed 37% of the epilogue's uncached LSE loads at seq 84 and
    measured +0.03% against a lazy control arm. The column is issued alongside
    the partials it shares a round with, so it is never the round's critical
    path.
    """
    units = PARTIAL_WAVES * n_groups
    best = None
    for dv in (1, 2, 4):
        cost = -(-(H * dv) // units) / dv
        if best is None or cost < best[0] - 1e-9:
            best = (cost, dv)
    dv = best[1]
    slots = H * dv
    return [
        (dv, min(units, slots - r * units), r * units // dv)
        for r in range(-(-slots // units))
    ]


def _arrive_and_wait(bar_ptr, arrivals: int, is_master):
    """Arrive at, and wait on, one sense-reversing barrier of `arrivals` CTAs.

    Costs one i32 of scratch and needs no reset, which a HIP graph replay could
    not do anyway: the master adds `SENSE - (arrivals - 1)` and everyone else
    adds one, so every launch adds exactly `SENSE`, and the counter's top bit
    alone tells a CTA whether the generation it joined has completed. The bit
    cannot flip early because the partial sums stay below it until the last
    arrival. Correct from any starting value with the low 31 bits clear.

    The caller must have drained its stores to the scope the readers use before
    calling; the atomic here is relaxed on purpose, because an acquire or
    release at agent scope lowers to an L2 writeback-invalidate, which in this
    kernel costs more than the whole reduction (measured 8.6 us against 0.9).

    Poll traffic is the cost that matters, not the atomic: the load has to skip
    every cache to see another CTA's write, so each poll is a round trip to the
    coherence point and the pollers all queue on one line. That is why this is
    called once per token over `n_groups` CTAs rather than once over the grid --
    240 pollers on a single counter cost 9.3 us at seq 48.

    The wait is bounded. Every CTA is resident by construction -- the launcher
    only fuses when the grid fits the CU array once -- but a bound turns a
    violated assumption into a wrong answer instead of a hung device.
    """
    i32 = T.i32
    ptr = fx.to_llvm_ptr(bar_ptr)
    delta = is_master.select(
        fx.Int32(_BARRIER_SENSE - (arrivals - 1)), fx.Int32(1)
    )
    old = llvm.atomicrmw(
        llvm.AtomicBinOp.add,
        ptr,
        _raw(delta),
        llvm.AtomicOrdering.monotonic,
        syncscope="agent",
    )
    zero = arith.constant(0, type=i32)
    sense = arith.constant(_BARRIER_SENSE, type=i32)
    limit = arith.constant(_BARRIER_MAX_POLLS, type=i32)
    loop = scf.WhileOp([i32], [zero])
    test = loop.before.blocks.append(i32)
    with ir.InsertionPoint(test):
        polls = test.arguments[0]
        cur = llvm.load(i32, ptr, volatile_=True)
        pending = arith.cmpi(
            CmpIPredicate.eq, arith.andi(arith.xori(cur, old), sense), zero
        )
        scf.ConditionOp(
            arith.andi(pending, arith.cmpi(CmpIPredicate.slt, polls, limit)), [polls]
        )
    body = loop.after.blocks.append(i32)
    with ir.InsertionPoint(body):
        if fx.const_expr(_BARRIER_SLEEP):
            fx.rocdl.s_sleep(_BARRIER_SLEEP)
        scf.YieldOp([arith.addi(body.arguments[0], arith.constant(1, type=i32))])


def _lse_reads(load_lse, ni, scalar):
    """Issue one (token, head) column's LSE loads for `_split_weights`.

    Split out from the weights so a caller with several columns in flight can
    put every column's loads on the wire before it consumes the first one.
    """
    if scalar:
        return [load_lse(fx.Int32(s)) for s in fx.range_constexpr(ni)]
    lane = fx.Int32(fx.thread_idx.x) % fx.Int32(64)
    return [load_lse((lane < fx.Int32(ni)).select(lane, fx.Int32(0)))]


def _split_weights(lse_vals, ni, scalar):
    """Softmax weights of one (token, head)'s `ni` partial records.

    `lse_vals` is what `_lse_reads` returned for the same `(ni, scalar)`. Both
    branches are the standalone reducer's, kept because the choice between them
    is a property of the column length, not of who is running it: the scalar
    form spends one exp2 per record but reads the column with wave-uniform
    addresses, and the butterfly's two 6-step shuffles cost the same whatever
    `ni` is.
    """
    if scalar:
        max_lse = _tree_reduce(lse_vals, lambda a, b: fx.Float32(a).maximumf(b))
        weights = [
            fx.Float32(fx.rocdl.exp2(T.f32, (v - max_lse).ir_value()))
            for v in lse_vals
        ]
        inv = fx.Float32(
            fx.rocdl.rcp(T.f32, _tree_reduce(weights, lambda a, b: a + b).ir_value())
        )
        return [w * inv for w in weights]

    lane = fx.Int32(fx.thread_idx.x) % fx.Int32(64)
    in_split = lane < fx.Int32(ni)
    lse = in_split.select(lse_vals[0], fx.Float32(float("-inf")))
    max_lse = lse
    for off in [32, 16, 8, 4, 2, 1]:
        max_lse = fx.Float32(max_lse).maximumf(
            fx.Float32(max_lse).shuffle_xor(fx.Int32(off), fx.Int32(64))
        )
    scale = fx.Float32(fx.rocdl.exp2(T.f32, (lse - max_lse).ir_value()))
    scale = in_split.select(scale, fx.Float32(0.0))
    denom = scale
    for off in [32, 16, 8, 4, 2, 1]:
        denom = denom + fx.Float32(denom).shuffle_xor(fx.Int32(off), fx.Int32(64))
    scale = scale * fx.Float32(fx.rocdl.rcp(T.f32, denom.ir_value()))
    return [
        fx.Float32(
            fx.rocdl.readlane(T.f32, scale.ir_value(), fx.Int32(s).ir_value())
        )
        for s in fx.range_constexpr(ni)
    ]


@contextmanager
def _if_then(if_op):
    with ir.InsertionPoint(if_op.then_block):
        try:
            yield if_op.then_block
        finally:
            block = if_op.then_block
            if (not block.operations) or not isinstance(
                block.operations[-1], scf.YieldOp
            ):
                scf.YieldOp([])


def _exp2(value):
    return fx.Float32(fx.rocdl.exp2(T.f32, fx.Float32(value).ir_value()))



def _pack_i32x2(lo, hi):
    return fx.Vector.from_elements([lo, hi], fx.Int32).bitcast(fx.Int64)[0]


def _mfma128(a, b, c):
    """gfx950 full-rate FP8 MFMA with unity scales (ABID=0)."""
    return fx.rocdl.mfma_scale_f32_16x16x128_f8f6f4(
        T.f32x4,
        [a, b, c, 0, 0, 0, fx.Int32(0), 0, fx.Int32(0)],
    )


def _mfma32(a, b, c):
    return fx.rocdl.mfma_f32_16x16x32_fp8_fp8(
        T.f32x4, [_pack_i32x2(a[0], a[1]), _pack_i32x2(b[0], b[1]), c, 0, 0, 0]
    )


# ng ranges over 1..33, so 32 entries evict at the top of the domain and a
# re-entry costs about 31 ms with the disk cache warm. Size to the domain.
@functools.lru_cache(maxsize=64)
def compile_sparse_mla_partial(
    ng: int,
    inner_iter: int = 1,
    waves_per_eu: int = 1,
    split_major: bool = False,
    use_buffer: bool = True,
    xpf_prime: int = 1,
    ixpf_depth: int = 0,
    ixpf_rows_only: bool = False,
    n_groups: int | None = None,
    fuse_combine: bool = False,
):
    """Compile the 64-key BF16-partial, log2-LSE producer.

    `n_groups` is the number of partial records a token is split into, so the
    grid is `seq * n_groups` CTAs and each owns `inner_iter` tiles. Passing it
    explicitly lifts the requirement that it divide `ng`: group `g` owns tiles
    `g, g + n_groups, ...`, and when `ng` is not a multiple the last round has
    `ng % n_groups` real tiles. The short groups run their final tile with the
    index row forced negative, which the existing padding mask already turns
    into an all `-inf` score -- an exactly neutral tile, no separate epilogue.

    `ixpf_depth` is how many tiles of index rows are in flight at once. Zero
    fetches a tile's rows inside its own gather, which is what the KV stage's
    depth used to imply; see `issue_index` and `_pick_index_prefetch`.

    `fuse_combine` folds the reducer in behind a grid-wide barrier instead of
    launching it as a second kernel. Only legal when every CTA is resident --
    the launcher gates on the grid fitting the CU array once -- and only worth
    it because a graph node on this device costs 1.5-1.8 us of wall against a
    reducer whose body is 0.2-0.6 us. Each token is reduced by the very CTAs
    that produced it, so the partials are read back from the L2 that wrote them.
    """
    if not 1 <= ng <= 33:
        raise ValueError(f"sparse MLA decode needs 1..33 splits, got {ng}")
    if n_groups is None:
        if inner_iter < 1 or inner_iter & (inner_iter - 1) or ng % inner_iter != 0:
            raise ValueError(
                f"inner_iter={inner_iter} must be a power-of-two divisor of ng={ng}"
            )
        n_groups = ng // inner_iter
    elif not 1 <= n_groups <= ng or inner_iter != -(-ng // n_groups):
        raise ValueError(
            f"n_groups={n_groups} must be in 1..{ng} and inner_iter={inner_iter} "
            f"must be ceil(ng/n_groups)={-(-ng // n_groups)}"
        )
    # Only the last round can run off the end, and only when the split is ragged.
    ragged = n_groups * inner_iter != ng
    # Fold the online-softmax rescale into the PV MFMA. See the loop body. A
    # single tile has nothing to rescale, so it keeps the plain form.
    seeded = inner_iter > 1
    if fuse_combine and n_groups < 2:
        raise ValueError("fuse_combine needs at least two partial records")
    # Reduction geometry. A token owns `4 * n_groups` waves, and how its H heads
    # are cut into 64-lane slots depends only on that count, so the schedule --
    # and therefore the round count -- is static.
    combine_units = PARTIAL_WAVES * n_groups
    combine_plan = _fused_combine_plan(n_groups) if fuse_combine else []
    scalar_lse = n_groups <= 16

    @fx.struct
    class PartialStorage:
        vlds: fx.Array[fx.Uint8, BLOCK_I * PITCH, 16]
        rmax: fx.Array[fx.Float32, PARTIAL_WAVES * H, 16]
        rsum: fx.Array[fx.Float32, PARTIAL_WAVES * H, 16]
        plds: fx.Array[fx.Uint8, BLOCK_I * H, 16]
        ilds: fx.Array[fx.Int32, BLOCK_I, 16]
        qlds: fx.Array[fx.Uint8, 64 * 144, 16]

    attrs = {"rocdl.waves_per_eu": int(waves_per_eu)}

    @flyc.kernel(
        name=(
            f"flydsl_sparse_mla_partial_ng{ng}_ii{inner_iter}_xor_partner_w128"
            + f"_pf{_XPF_DEPTH}_primed_qflat"
            + ("_fusedrescale" if seeded else "")
            + ("_split_major" if split_major else "")
            + ("_buf" if use_buffer else "")
            + (f"_prime{xpf_prime}" if xpf_prime > _XPF_DEPTH else "")
            + (f"_ixpf{ixpf_depth}" if ixpf_depth else "")
            + ("_ixro" if ixpf_depth and ixpf_rows_only else "")
            + (f"_ng{n_groups}" if ragged else "")
            + (
                "_coop_"
                + "_".join(f"{dv}x{slots}" for dv, slots, _ in combine_plan)
                + "_dev_staged"
                if fuse_combine
                else ""
            )
        ),
        known_block_size=[PARTIAL_THREADS, 1, 1],
    )
    def kernel(
        q_ptr: fx.Pointer,
        kv_ptr: fx.Pointer,
        index_ptr: fx.Pointer,
        partial_ptr: fx.Pointer,
        lse_ptr: fx.Pointer,
        out_ptr: fx.Pointer,
        bar_ptr: fx.Pointer,
        scale_log2e: fx.Float32,
        seq: fx.Int32,
    ):
        v4u8_t = fx.Vector.make_type(16, fx.Uint8)
        v2i32_t = fx.Vector.make_type(2, fx.Int32)
        tid = fx.Int32(fx.thread_idx.x)
        wave = tid // fx.Int32(64)
        lane = tid % fx.Int32(64)
        group = lane // fx.Int32(16)
        head = lane % fx.Int32(16)
        owner = fx.Int32(fx.block_idx.x)
        if fx.const_expr(split_major):
            split = owner // seq
            tok = owner % seq
        else:
            tok = owner // fx.Int32(n_groups)
            split = owner % fx.Int32(n_groups)
        lds = fx.SharedAllocator().allocate(PartialStorage).peek()

        def load16(ptr, offset):
            return fx.ptr_load(ptr + fx.Int64(offset), result_type=v4u8_t).bitcast(
                fx.Int32
            )

        if fx.const_expr(use_buffer):
            # 32-bit buffer offsets: one VGPR per address instead of a 64-bit
            # pair, so the gather drops its per-lane v_add_co/v_addc chain.
            kv_rsrc = buffer_ops.create_buffer_resource_from_addr(
                fx.Int64(fx.ptrtoint(kv_ptr))
            )

            def kvload16(row, dword):
                return fx.Vector(
                    buffer_ops.buffer_load(
                        kv_rsrc,
                        row * fx.Int32(DIM // 4) + dword,
                        vec_width=4,
                        dtype=fx.Int32,
                    )
                )

        else:

            def kvload16(row, dword):
                return load16(kv_ptr, fx.Int64(row) * DIM + fx.Int64(dword) * 4)

        def join8(lo, hi):
            return fx.Vector.from_elements(
                [lo[i] for i in fx.range_constexpr(4)]
                + [hi[i] for i in fx.range_constexpr(4)],
                fx.Int32,
            )

        # Keep this as a compile-time region. A runtime guard here makes the
        # FlyDSL rewriter capture LDS handles as branch state.
        if fx.const_expr(True):
            # QK-to-PV lane permutation for one 64-key tile.
            slot = (
                fx.Int32(32) * (wave // fx.Int32(2))
                + fx.Int32(8) * (head // fx.Int32(4))
                + fx.Int32(4) * (wave % fx.Int32(2))
                + head % fx.Int32(4)
            )
            # Gather view of the same 16 rows: eight lanes share one row and
            # cover 128 contiguous bytes of it, so a request spans a full pair
            # of cache lines instead of 64 B. Lanes 0..7 of every group serve
            # the wave's rows `head` 0..7 and lanes 8..15 serve `head` 8..15,
            # whose slots sit exactly 16 apart.
            wide_slot = (
                fx.Int32(32) * (wave // fx.Int32(2))
                + fx.Int32(8) * (lane // fx.Int32(32))
                + fx.Int32(4) * (wave % fx.Int32(2))
                + (lane // fx.Int32(8)) % fx.Int32(4)
            )
            wide_dword = (lane % fx.Int32(8)) * fx.Int32(4)
            wide_low_base = (
                lds.vlds.ptr + wide_slot * fx.Int32(PITCH) + wide_dword * fx.Int32(4)
            )
            wide_high_base = wide_low_base + fx.Int32(16 * PITCH)
            # QK wants row `head` in lane `head`, which no wide gather can
            # produce, so the operand is read back from the LDS copy the PV
            # stage needs anyway. Writer and reader are the same wave, so this
            # adds no barrier.
            key_base = lds.vlds.ptr + slot * fx.Int32(PITCH) + group * fx.Int32(16)
            out_record = (fx.Int64(tok) * n_groups + fx.Int64(split)) * H + fx.Int64(
                head
            )
            # The fused reducer sits on another CU, and for a token whose splits
            # do not all land on one L2 domain on another domain too, so the
            # records it will read are published past every cache. Both sides
            # carry the policy on the instruction; see `_CPOL_DEVICE`.
            if fx.const_expr(fuse_combine):
                partial_rsrc = buffer_ops.create_buffer_resource_from_addr(
                    fx.Int64(fx.ptrtoint(partial_ptr))
                )
                lse_rsrc = buffer_ops.create_buffer_resource_from_addr(
                    fx.Int64(fx.ptrtoint(lse_ptr))
                )
                out_elem = (
                    (tok * fx.Int32(n_groups) + split) * fx.Int32(H) + head
                ) * fx.Int32(DV)

            def store_partial(vec, out_col):
                if fx.const_expr(fuse_combine):
                    buffer_ops.buffer_store(
                        vec,
                        partial_rsrc,
                        out_elem + out_col,
                        cache_modifier=_CPOL_DEVICE,
                    )
                else:
                    fx.ptr_store(
                        vec, partial_ptr + out_record * DV + fx.Int64(out_col)
                    )

            running_max = fx.Float32(float("-inf"))
            running_denom = fx.Float32(0.0)
            running_acc = [
                fx.Vector.filled(4, 0.0, fx.Float32) for _ in fx.range_constexpr(8)
            ]

            def tile_of(k_i):
                """The 64-key tile this CTA's round `k_i` owns."""
                # Preserve the direct reducer's XOR tree: for ng=32 and
                # inner_iter=2, merge (0,16), (1,17), ... rather than adjacent
                # rows.  This removes one real partial row per pair while
                # matching the reducer's first active shuffle level.
                tile = split + fx.Int32(k_i * n_groups)
                # A ragged split leaves the last round short. Rather than a
                # second kernel body, the short groups re-read tile `split` and
                # publish a negative index row, which the padding mask below
                # scores as `-inf`: probabilities and denominator come out zero,
                # alpha comes out one, so the tile leaves the accumulator, the
                # running maximum and the LSE bit for bit unchanged.
                if fx.const_expr(ragged and k_i + 1 == inner_iter):
                    return tile, tile < fx.Int32(ng)
                return tile, None

            def issue_index(k_i, fields="all"):
                """Put one tile's three index rows on the wire.

                Split from the gather because the two stages are priced on
                different axes: a KV stage in flight is 36 live registers,
                which is what `_XPF_DEPTH` is about, while an index stage is
                three dwords. Nothing here consumes the loads -- the clamp and
                the padding mask live in `issue_gather` -- so the wait for them
                sinks to the gather whose addresses they are.

                Left joined, the body issues these three and then drains them to
                `vmcnt(0)` before the nine `buffer_load_dwordx4` they address
                can issue, with eight `ds_write_b128` and six MFMA to cover the
                round trip. Lifted, the body's deepest wait is `vmcnt(1)` -- the
                KV arrival its LDS writes have to wait for regardless.
                """
                tile, in_range = tile_of(k_i)
                if fx.const_expr(in_range is not None):
                    tile = in_range.select(tile, split)
                index_base = fx.Int64(tok) * (ng * BLOCK_I) + fx.Int64(
                    tile * fx.Int32(BLOCK_I)
                )
                wide = (wide_slot, wide_slot + fx.Int32(16))
                # `fields` picks which of the three rows this call puts on the
                # wire. "wide" is the pair that addresses eight of the nine
                # gather loads; "narrow" is the one the rope tail and the LDS
                # write need. Splitting them lets the prologue lift only the
                # pair, at two registers a tile instead of three.
                sl = {
                    "all": (slot,) + wide,
                    "wide": wide,
                    "narrow": (slot,),
                }[fields]
                return [
                    fx.Int32(fx.ptr_load(index_ptr + index_base + fx.Int64(x)))
                    for x in sl
                ]

            def issue_gather(k_i):
                """Issue one tile's KV gather; nothing here touches LDS."""
                if fx.const_expr(ixpf_depth and ixpf_rows_only):
                    raw_low, raw_high = rows_held[k_i]
                    (row,) = issue_index(k_i, "narrow")
                elif fx.const_expr(ixpf_depth):
                    row, raw_low, raw_high = rows_held[k_i]
                else:
                    row, raw_low, raw_high = issue_index(k_i)
                _, in_range = tile_of(k_i)
                if fx.const_expr(in_range is not None):
                    row = in_range.select(row, fx.Int32(-1))

                def gather_row(raw):
                    return (raw >= fx.Int32(0)).select(raw, fx.Int32(0))

                low_row = gather_row(raw_low)
                high_row = gather_row(raw_high)
                low = [
                    kvload16(low_row, wide_dword + fx.Int32(cc * 32))
                    for cc in fx.range_constexpr(4)
                ]
                high = [
                    kvload16(high_row, wide_dword + fx.Int32(cc * 32))
                    for cc in fx.range_constexpr(4)
                ]
                # The 64 B rope tail is one line per row either way, so it keeps
                # the narrow mapping and stays a register operand.
                safe_row = (row >= fx.Int32(0)).select(row, fx.Int32(0))
                rope = kvload16(safe_row, fx.Int32(DV // 4) + group * fx.Int32(4))
                return row, low, high, rope

            # Gathers in flight at once. The prologue issues the first ones
            # and every loop body issues the one it will consume `_XPF_DEPTH`
            # tiles later, so the fetch latency hides behind softmax and PV.
            #
            # Ordering below is load-bearing. A tile's rows come from the index
            # array alone, so the gather can be in flight during the Q publish
            # and its barrier, which every wave would otherwise wait out with no
            # traffic of its own. The Q loads are still issued first because
            # vmcnt retires in order: the publishing stores then drain just
            # those loads instead of the whole primed tile.
            #
            # The 16x576 B Q block is published once per CTA as 576 sixteen-byte
            # chunks spread over the 256 threads. Consecutive threads take
            # consecutive chunks, so a load is one contiguous 4 KB run (32 line
            # requests) instead of 16 rows x 64 B, and the block base stays
            # 128 B aligned: 72 requests for the block against 144 for the
            # per-head mapping.
            q_block = fx.Int64(tok) * (H * DIM)
            qlane = lds.qlds.ptr + lane * fx.Int32(144)

            def q_chunk_dst(chunk):
                """LDS address of a flat 16 B chunk of the Q block."""
                row = chunk // fx.Int32(DIM // 16)
                col = chunk % fx.Int32(DIM // 16)
                return (
                    lds.qlds.ptr
                    + ((col % fx.Int32(4)) * fx.Int32(H) + row) * fx.Int32(144)
                    + (col // fx.Int32(4)) * fx.Int32(16)
                )

            q_held = [
                (load16(q_ptr, q_block + fx.Int64(chunk) * 16), q_chunk_dst(chunk))
                for chunk in [
                    tid + fx.Int32(c * PARTIAL_THREADS) for c in fx.range_constexpr(2)
                ]
            ]
            # The prologue may run deeper than the steady state: the first two
            # tiles have nothing to hide behind, so issuing both before the Q
            # publish folds their two serial index->KV chases into one burst.
            kv_prime = min(max(xpf_prime, _XPF_DEPTH), inner_iter)
            index_ahead = (
                min(max(ixpf_depth, kv_prime), inner_iter) if ixpf_depth else 0
            )
            _ixf = "wide" if ixpf_rows_only else "all"
            rows_held = [
                issue_index(k, _ixf) for k in fx.range_constexpr(index_ahead)
            ]
            pipeline = [issue_gather(k) for k in fx.range_constexpr(kv_prime)]
            for q_data, q_dst in q_held:
                fx.ptr_store(q_data.bitcast(fx.Uint8), q_dst)
            # 576 = 2 x 256 + 64, so one wave publishes the remainder. Its load
            # stays here even though it forces the publish onto `vmcnt(0)`:
            # issuing it with the other two only saves a wait the CTA has to do
            # a few instructions later anyway, and the out-of-range chunks the
            # other three waves would then have to clamp cost more than that
            # (measured +0.0 to +0.4% on the producer at seq 8, 48, 60 and 84,
            # and +1.5 to +2.2% again at seq 48 and 60 once the index column
            # moved to the prologue and left the burst that much tighter).
            q_tail_chunk = tid + fx.Int32(2 * PARTIAL_THREADS)
            with _if_then(
                scf.IfOp(
                    arith.cmpi(
                        CmpIPredicate.eq, _raw(wave), arith.constant(0, type=T.i32)
                    )
                )
            ):
                fx.ptr_store(
                    load16(q_ptr, q_block + fx.Int64(q_tail_chunk) * 16).bitcast(
                        fx.Uint8
                    ),
                    q_chunk_dst(q_tail_chunk),
                )

            # Every wave reads the whole published block back, one head per lane.
            fx.gpu.barrier()
            bq = [None] * 5
            for cc in fx.range_constexpr(4):
                lo = fx.ptr_load(qlane + fx.Int32(cc * 32), result_type=v4u8_t).bitcast(
                    fx.Int32
                )
                hi = fx.ptr_load(
                    qlane + fx.Int32(cc * 32 + 16), result_type=v4u8_t
                ).bitcast(fx.Int32)
                bq[cc] = join8(lo, hi)
            q_tail = fx.ptr_load(qlane + fx.Int32(128), result_type=v4u8_t).bitcast(
                fx.Int32
            )
            bq[4] = fx.Vector.from_elements(
                [q_tail[i] for i in fx.range_constexpr(4)] + [fx.Int32(0)] * 4,
                fx.Int32,
            )

            for k_i in fx.range_constexpr(inner_iter):
                # Keep the index stage `index_ahead` tiles in front of the
                # gather it feeds. Issued at the top of the body, so the rows
                # a later body's gather needs have already retired behind the
                # KV wait this body's LDS writes do anyway.
                if fx.const_expr(
                    index_ahead
                    and len(rows_held) < min(inner_iter, k_i + 1 + index_ahead)
                ):
                    rows_held.append(issue_index(len(rows_held), _ixf))
                row, low, high, rope = pipeline[k_i]
                lds.ilds[slot] = row
                for cc in fx.range_constexpr(4):
                    fx.ptr_store(
                        low[cc].bitcast(fx.Uint8), wide_low_base + fx.Int32(cc * 128)
                    )
                    fx.ptr_store(
                        high[cc].bitcast(fx.Uint8), wide_high_base + fx.Int32(cc * 128)
                    )
                # The next tile's rows are independent of everything below, and
                # the registers just published to LDS are free, so issue that
                # gather now and let the softmax and PV work hide its latency.
                if fx.const_expr(
                    len(pipeline) < inner_iter
                    and len(pipeline) - (k_i + 1) < _XPF_DEPTH
                ):
                    pipeline.append(issue_gather(len(pipeline)))

                def key_operand(cc):
                    klo = fx.ptr_load(
                        key_base + fx.Int32(cc * 128), result_type=v4u8_t
                    ).bitcast(fx.Int32)
                    khi = fx.ptr_load(
                        key_base + fx.Int32(cc * 128 + 64), result_type=v4u8_t
                    ).bitcast(fx.Int32)
                    return join8(klo, khi)

                score = _mfma128(
                    fx.Vector.from_elements(
                        [rope[i] for i in fx.range_constexpr(4)] + [fx.Int32(0)] * 4,
                        fx.Int32,
                    ),
                    bq[4],
                    fx.Vector.filled(4, 0.0, fx.Float32),
                )
                for cc in fx.range_constexpr(4):
                    score = _mfma128(key_operand(cc), bq[cc], score)

                ids = fx.ptr_load(
                    lds.ilds.ptr
                    + fx.Int32(32) * (wave // fx.Int32(2))
                    + fx.Int32(8) * group
                    + fx.Int32(4) * (wave % fx.Int32(2)),
                    result_type=fx.Vector.make_type(4, fx.Int32),
                )
                qk = [None] * 4
                for r in fx.range_constexpr(4):
                    qk[r] = (ids[r] >= fx.Int32(0)).select(
                        fx.Float32(score[r]) * scale_log2e,
                        fx.Float32(float("-inf")),
                    )
                local_max = fx.Float32(float("-inf"))
                for r in fx.range_constexpr(4):
                    local_max = local_max.maximumf(qk[r])
                local_max = local_max.maximumf(
                    local_max.shuffle_xor(fx.Int32(16), fx.Int32(64))
                )
                local_max = local_max.maximumf(
                    local_max.shuffle_xor(fx.Int32(32), fx.Int32(64))
                )
                with _if_then(
                    scf.IfOp(
                        arith.cmpi(
                            CmpIPredicate.slt,
                            _raw(lane),
                            arith.constant(16, type=T.i32),
                        )
                    )
                ):
                    lds.rmax[wave * fx.Int32(H) + head] = local_max
                fx.gpu.barrier()

                tile_max = fx.Float32(float("-inf"))
                for ww in fx.range_constexpr(PARTIAL_WAVES):
                    tile_max = tile_max.maximumf(fx.Float32(lds.rmax[ww * H + head]))
                # Rescaling P by `beta` before the fp8 pack instead of rescaling
                # the PV result after it: both put the tile's contribution on the
                # running maximum's scale, but this way the factor rides along in
                # the exponent that is computed anyway. The fp8 grid then covers
                # `beta * P` rather than `P`, so a tile below the running maximum
                # keeps fewer mantissa bits, in proportion to how little it
                # contributes; measured end to end this lowers the sparse-MLA
                # relative L2 against the fp32 reference from 0.0262 to 0.0249.
                if fx.const_expr(seeded):
                    scale_max = running_max.maximumf(tile_max)
                    # Folding beta into P left alpha depending only on
                    # loop-carried state, so it no longer has to wait for the
                    # rsum rendezvous below and can issue alongside it.
                    alpha = (running_denom == fx.Float32(0.0)).select(
                        fx.Float32(0.0), _exp2(running_max - scale_max)
                    )
                else:
                    scale_max = tile_max
                max_safe = (scale_max == fx.Float32(float("-inf"))).select(
                    fx.Float32(0.0), scale_max
                )
                probs = [None] * 4
                prob_sum = fx.Float32(0.0)
                for r in fx.range_constexpr(4):
                    probs[r] = _exp2(qk[r] - max_safe)
                    prob_sum = prob_sum + probs[r]
                packed = fx.rocdl.cvt_pk_fp8_f32(
                    T.i32,
                    probs[0] * fx.Float32(FP8_MAX),
                    probs[1] * fx.Float32(FP8_MAX),
                    fx.Int32(0),
                    False,
                )
                packed = fx.rocdl.cvt_pk_fp8_f32(
                    T.i32,
                    probs[2] * fx.Float32(FP8_MAX),
                    probs[3] * fx.Float32(FP8_MAX),
                    packed,
                    True,
                )
                fx.ptr_store(
                    fx.Vector.from_elements([packed], fx.Int32).bitcast(fx.Uint8),
                    lds.plds.ptr + lane * fx.Int32(H) + wave * fx.Int32(4),
                )
                prob_sum = prob_sum + prob_sum.shuffle_xor(
                    fx.Int32(16), fx.Int32(64)
                )
                prob_sum = prob_sum + prob_sum.shuffle_xor(
                    fx.Int32(32), fx.Int32(64)
                )
                with _if_then(
                    scf.IfOp(
                        arith.cmpi(
                            CmpIPredicate.slt,
                            _raw(lane),
                            arith.constant(16, type=T.i32),
                        )
                    )
                ):
                    lds.rsum[wave * fx.Int32(H) + head] = prob_sum
                fx.gpu.barrier()

                # The four per-wave sums are folded here, once per tile, rather
                # than carried per wave and folded in the last tile. Deferring
                # them is exact -- the fold is linear in the rescale and the
                # `denom == 0` sentinels equal the wave-uniform `max == -inf` --
                # but it is slower: the four ds_read_b32 below are the first LDS
                # traffic after the barrier and they cover the latency of the
                # plds read that feeds PV. Dropping them cost 1.2% at seq 48,
                # 2.0% at seq 60 and 5.9% at seq 84.
                tile_denom = fx.Float32(0.0)
                for ww in fx.range_constexpr(PARTIAL_WAVES):
                    tile_denom = tile_denom + fx.Float32(lds.rsum[ww * H + head])
                if fx.const_expr(inner_iter == 1):
                    output_scale = (tile_denom == fx.Float32(0.0)).select(
                        fx.Float32(0.0),
                        fx.Float32(
                            fx.rocdl.rcp(
                                T.f32,
                                (tile_denom * fx.Float32(FP8_MAX)).ir_value(),
                            )
                        ),
                    )
                else:
                    # `probs` already carry beta, so `tile_denom` does too and the
                    # accumulator stays in units of FP8_MAX until the epilogue.
                    next_denom = running_denom * alpha + tile_denom
                    if fx.const_expr(k_i + 1 == inner_iter):
                        final_inv_denom = (next_denom == fx.Float32(0.0)).select(
                            fx.Float32(0.0),
                            fx.Float32(
                                fx.rocdl.rcp(
                                    T.f32,
                                    (next_denom * fx.Float32(FP8_MAX)).ir_value(),
                                )
                            ),
                        )

                p4 = fx.ptr_load(
                    lds.plds.ptr + lane * fx.Int32(H),
                    result_type=fx.Vector.make_type(16, fx.Uint8),
                ).bitcast(fx.Int32)
                pvec = fx.Vector.from_elements(
                    [p4[i] for i in fx.range_constexpr(4)] + [fx.Int32(0)] * 4,
                    fx.Int32,
                )
                trbase = (fx.Int32(8) * group + head // fx.Int32(2)) * PITCH + fx.Int32(
                    8
                ) * (head % fx.Int32(2))
                for j in fx.range_constexpr(8):
                    dv_base = (wave * fx.Int32(8) + fx.Int32(j)) * 16
                    # Seeding the MFMA's C operand with the rescaled running
                    # accumulator makes the online-softmax update the same
                    # instruction as the product: the 16 `v_pk_mul_f32` that
                    # scaled this tile's PV result and the 16 `v_pk_add_f32`
                    # that merged it are gone, leaving only alpha's 16 per tile
                    # per wave. That block was the kernel's largest.
                    if fx.const_expr(seeded):
                        acc = fx.Vector(running_acc[j]) * alpha
                    else:
                        acc = fx.Vector.filled(4, 0.0, fx.Float32)
                    for half in fx.range_constexpr(2):
                        vptr = (
                            lds.vlds.ptr
                            + trbase
                            + dv_base
                            + fx.Int32(half * 32 * PITCH)
                        )
                        vaddr = fx.Int64(fx.ptrtoint(vptr))
                        llvm_vptr = llvm.inttoptr(
                            ir.Type.parse("!llvm.ptr<3>"), _raw(vaddr)
                        )
                        avec = fx.Vector(
                            fx.rocdl.ds_read_tr8_b64(v2i32_t, llvm_vptr).result
                        )
                        bvec = fx.Vector.from_elements(
                            [pvec[2 * half], pvec[2 * half + 1]], fx.Int32
                        )
                        acc = _mfma32(avec, bvec, acc)
                    if fx.const_expr(inner_iter == 1):
                        store_partial(
                            (fx.Vector(acc) * output_scale).to(fx.BFloat16),
                            dv_base + fx.Int32(4) * group,
                        )
                    else:
                        if fx.const_expr(k_i + 1 == inner_iter):
                            store_partial(
                                (fx.Vector(acc) * final_inv_denom).to(fx.BFloat16),
                                dv_base + fx.Int32(4) * group,
                            )
                        else:
                            running_acc[j] = fx.Vector(acc)

                if fx.const_expr(inner_iter > 1):
                    running_max = scale_max
                    running_denom = next_denom
                    if fx.const_expr(k_i + 1 < inner_iter):
                        fx.gpu.barrier()

            with _if_then(
                scf.IfOp(
                    arith.andi(
                        arith.cmpi(
                            CmpIPredicate.eq,
                            _raw(wave),
                            arith.constant(0, type=T.i32),
                        ),
                        arith.cmpi(
                            CmpIPredicate.slt,
                            _raw(lane),
                            arith.constant(16, type=T.i32),
                        ),
                    )
                )
            ):
                if fx.const_expr(inner_iter == 1):
                    lse = (tile_denom == fx.Float32(0.0)).select(
                        fx.Float32(-(2**30)), fly_math.log2(tile_denom) + tile_max
                    )
                else:
                    lse = (running_denom == fx.Float32(0.0)).select(
                        fx.Float32(-(2**30)),
                        fly_math.log2(running_denom) + running_max,
                    )
                if fx.const_expr(fuse_combine):
                    buffer_ops.buffer_store(
                        lse,
                        lse_rsrc,
                        (tok * fx.Int32(n_groups) + split) * fx.Int32(H) + head,
                        cache_modifier=_CPOL_DEVICE,
                    )
                else:
                    fx.ptr_store(lse, lse_ptr + out_record)

            if fx.const_expr(fuse_combine):
                # Publish, rendezvous, reduce. Every wave drains its own stores
                # -- which are already write-through, so retiring them is all
                # the release this needs -- then one thread per CTA joins the
                # token's rendezvous while the rest wait on the cheap
                # workgroup barrier.
                fx.rocdl.s_waitcnt(vmcnt=0)
                fx.gpu.barrier()
                with _if_then(
                    scf.IfOp(
                        arith.cmpi(
                            CmpIPredicate.eq, _raw(tid), arith.constant(0, type=T.i32)
                        )
                    )
                ):
                    _arrive_and_wait(
                        bar_ptr + fx.Int64(tok) * BARRIER_STRIDE,
                        n_groups,
                        split == fx.Int32(0),
                    )
                fx.gpu.barrier()

                # A token's reducers are the very CTAs that produced it, and
                # its work is spread over all of them: the waves number
                # themselves `split * 4 + wave` and take slots round robin, so
                # a round's slots are contiguous and no CTA is left holding the
                # whole reduction.
                final_buf = _pointer_buffer_tensor(
                    out_ptr, fx.BFloat16, (seq, H, DV), (H * DV, DV, 1)
                )
                lse_row = tok * fx.Int32(n_groups * H)
                unit = split * fx.Int32(PARTIAL_WAVES) + _uniform_i32(wave)

                def load_lse(s, rhead):
                    return fx.Float32(
                        buffer_ops.buffer_load(
                            lse_rsrc,
                            lse_row + s * fx.Int32(H) + rhead,
                            vec_width=1,
                            dtype=fx.Float32,
                            cache_modifier=_CPOL_DEVICE,
                        )
                    )

                def load_partial(s, rhead, out_lane, vals):
                    # bf16 pairs ride as i32 so a slice is still one dword
                    # transaction; a copy atom would carry the default policy.
                    words = vals // 2
                    row = tok * fx.Int32(n_groups) + fx.Int32(s)
                    raw = buffer_ops.buffer_load(
                        partial_rsrc,
                        (row * fx.Int32(H) + rhead) * fx.Int32(DV // 2)
                        + out_lane * fx.Int32(words),
                        vec_width=words,
                        dtype=fx.Int32,
                        cache_modifier=_CPOL_DEVICE,
                    )
                    if fx.const_expr(words == 1):
                        vec = fx.Vector.from_elements([fx.Int32(raw)], fx.Int32)
                    else:
                        vec = fx.Vector(raw, shape=words, dtype=fx.Int32)
                    vec = vec.bitcast(fx.BFloat16)
                    return [vec[i].to(fx.Float32) for i in fx.range_constexpr(vals)]

                def round_reads(dv, slots, head_base):
                    """Put one round's whole column on the wire.

                    A short round reads with its unit clamped instead of under
                    the store's guard: the address stays in bounds, the result
                    is thrown away, and the loads leave the branch so they can
                    be issued alongside the other rounds'.
                    """
                    vals = DV // (64 * dv)
                    u = unit
                    if fx.const_expr(slots < combine_units):
                        u = (unit < fx.Int32(slots)).select(unit, fx.Int32(0))
                    if fx.const_expr(dv == 1):
                        rhead, out_lane = fx.Int32(head_base) + u, lane
                    else:
                        rhead = fx.Int32(head_base) + u // fx.Int32(dv)
                        out_lane = (u % fx.Int32(dv)) * fx.Int32(64) + lane
                    return (
                        rhead,
                        out_lane,
                        vals,
                        _lse_reads(
                            lambda s: load_lse(s, rhead), n_groups, scalar_lse
                        ),
                        [
                            load_partial(s, rhead, out_lane, vals)
                            for s in fx.range_constexpr(n_groups)
                        ],
                    )

                def round_combine(reads):
                    rhead, out_lane, vals, lse_vals, parts = reads
                    scales = _split_weights(lse_vals, n_groups, scalar_lse)
                    acc = [fx.Float32(0.0) for _ in fx.range_constexpr(vals)]
                    for s in fx.range_constexpr(n_groups):
                        acc = [
                            acc[i] + parts[s][i] * scales[s]
                            for i in fx.range_constexpr(vals)
                        ]
                    _store_final_out(
                        final_buf, tok, rhead, out_lane, acc, vals, fx.BFloat16
                    )

                def emit_round(slots, reads):
                    if fx.const_expr(slots == combine_units):
                        round_combine(reads)
                    else:
                        with _if_then(
                            scf.IfOp(
                                arith.cmpi(
                                    CmpIPredicate.slt,
                                    _raw(unit),
                                    arith.constant(slots, type=T.i32),
                                )
                            )
                        ):
                            round_combine(reads)

                # Every round's column goes on the wire before the first one is
                # consumed. `round_reads` already lifts its loads out of a short
                # round's guard for this reason; carrying it across rounds is
                # what keeps round r+1's partials from queueing behind round r's
                # store, which they otherwise do -- one in-order vmcnt covers
                # both. Worth 3.2% of the seq 84 call, bit identical, and a
                # compile-time identity for the single-round plans.
                staged = [
                    (slots, round_reads(dv, slots, head_base))
                    for dv, slots, head_base in combine_plan
                ]
                for slots, reads in staged:
                    emit_round(slots, reads)

    @flyc.jit
    def launch(
        q_ptr: fx.Pointer,
        kv_ptr: fx.Pointer,
        index_ptr: fx.Pointer,
        partial_ptr: fx.Pointer,
        lse_ptr: fx.Pointer,
        out_ptr: fx.Pointer,
        bar_ptr: fx.Pointer,
        scale_log2e: fx.Float32,
        seq: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(
            q_ptr,
            kv_ptr,
            index_ptr,
            partial_ptr,
            lse_ptr,
            out_ptr,
            bar_ptr,
            scale_log2e,
            seq,
        ).launch(
            grid=(seq * fx.Int32(n_groups), 1, 1),
            block=(PARTIAL_THREADS, 1, 1),
            stream=stream,
            value_attrs=attrs,
        )

    return launch
