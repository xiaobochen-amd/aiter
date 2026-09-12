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

H = 16
DV = 512
DT = 64
DIM = DV + DT
BLOCK_I = 64
FP8_MAX = 448.0
PARTIAL_THREADS = 256
PARTIAL_WAVES = 4
PITCH = DV + 16
# Tiles whose gather is in flight at once inside a producer CTA.
_XPF_DEPTH = 1


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
):
    """Compile the 64-key BF16-partial, log2-LSE producer."""
    if not 1 <= ng <= 33:
        raise ValueError(f"sparse MLA decode needs 1..33 splits, got {ng}")
    if inner_iter < 1 or inner_iter & (inner_iter - 1) or ng % inner_iter != 0:
        raise ValueError(
            f"inner_iter={inner_iter} must be a power-of-two divisor of ng={ng}"
        )
    n_groups = ng // inner_iter
    # Fold the online-softmax rescale into the PV MFMA. See the loop body. A
    # single tile has nothing to rescale, so it keeps the plain form.
    seeded = inner_iter > 1

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
        ),
        known_block_size=[PARTIAL_THREADS, 1, 1],
    )
    def kernel(
        q_ptr: fx.Pointer,
        kv_ptr: fx.Pointer,
        index_ptr: fx.Pointer,
        partial_ptr: fx.Pointer,
        lse_ptr: fx.Pointer,
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
            running_max = fx.Float32(float("-inf"))
            running_denom = fx.Float32(0.0)
            running_acc = [
                fx.Vector.filled(4, 0.0, fx.Float32) for _ in fx.range_constexpr(8)
            ]

            def issue_gather(k_i):
                """Issue one tile's gather; nothing here touches LDS."""
                # Preserve the direct reducer's XOR tree: for ng=32 and
                # inner_iter=2, merge (0,16), (1,17), ... rather than adjacent
                # rows.  This removes one real partial row per pair while
                # matching the reducer's first active shuffle level.
                tile = split + fx.Int32(k_i * n_groups)
                index_base = fx.Int64(tok) * (ng * BLOCK_I) + fx.Int64(
                    tile * fx.Int32(BLOCK_I)
                )

                def gather_row(gather_slot):
                    raw = fx.Int32(
                        fx.ptr_load(index_ptr + index_base + fx.Int64(gather_slot))
                    )
                    return (raw >= fx.Int32(0)).select(raw, fx.Int32(0))

                row = fx.Int32(fx.ptr_load(index_ptr + index_base + fx.Int64(slot)))
                low_row = gather_row(wide_slot)
                high_row = gather_row(wide_slot + fx.Int32(16))
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
            pipeline = [
                issue_gather(k)
                for k in fx.range_constexpr(
                    min(max(xpf_prime, _XPF_DEPTH), inner_iter)
                )
            ]
            for q_data, q_dst in q_held:
                fx.ptr_store(q_data.bitcast(fx.Uint8), q_dst)
            # 576 = 2 x 256 + 64, so one wave publishes the remainder.
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
                prob_sum = prob_sum + prob_sum.shuffle_xor(fx.Int32(16), fx.Int32(64))
                prob_sum = prob_sum + prob_sum.shuffle_xor(fx.Int32(32), fx.Int32(64))
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
                        out_col = dv_base + fx.Int32(4) * group
                        fx.ptr_store(
                            (fx.Vector(acc) * output_scale).to(fx.BFloat16),
                            partial_ptr + out_record * DV + fx.Int64(out_col),
                        )
                    else:
                        if fx.const_expr(k_i + 1 == inner_iter):
                            out_col = dv_base + fx.Int32(4) * group
                            fx.ptr_store(
                                (fx.Vector(acc) * final_inv_denom).to(fx.BFloat16),
                                partial_ptr + out_record * DV + fx.Int64(out_col),
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
                fx.ptr_store(lse, lse_ptr + out_record)

    @flyc.jit
    def launch(
        q_ptr: fx.Pointer,
        kv_ptr: fx.Pointer,
        index_ptr: fx.Pointer,
        partial_ptr: fx.Pointer,
        lse_ptr: fx.Pointer,
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
            scale_log2e,
            seq,
        ).launch(
            grid=(seq * fx.Int32(n_groups), 1, 1),
            block=(PARTIAL_THREADS, 1, 1),
            stream=stream,
            value_attrs=attrs,
        )

    return launch
