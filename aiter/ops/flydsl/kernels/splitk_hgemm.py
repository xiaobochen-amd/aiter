# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import contextlib
import functools
from abc import ABC, abstractmethod

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm, scf
from flydsl.expr import arith, const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import Int32, T
from flydsl.runtime.device import get_rocm_arch

from aiter.ops.flydsl.kernels import buffer_ops, vector

from .tensor_shim import GTensor, get_dtype_in_kernel

# Cross-XCD partials need agent scope; system scope evicts past the last level.
SPLIT_K_CPOL_COHERENT = 0x10

SPLIT_K_CPOL_XCD_LOCAL = 0x1

# One line per tile counter; packed dwords would serialise a launch's arrivals.
SPLIT_K_SEMAPHORE_STRIDE = 32

NUM_XCD = 8

# B's cache policy is a per-config choice: the default keeps both levels, and a
# long read-once stream opts into skipping them.
B_CPOL_OPTIONS = ("", "nt sc0 sc1")


def _pairwise_sum(parts):
    """Sum a Python list of vectors as a balanced tree.

    Module level because the kernel AST rewriter would turn this ``while`` into
    an ``scf.while``; it is trace-time metaprogramming over a Python list. The
    tree also lets every split-K plane load issue before the first add.
    """
    while len(parts) > 1:
        nxt = [lhs + rhs for lhs, rhs in zip(parts[0::2], parts[1::2])]
        if len(parts) % 2:
            nxt.append(parts[-1])
        parts = nxt
    return parts[0]


@contextlib.contextmanager
def _if_then(cond):
    """Emit the body under ``cond``, or unguarded when ``cond`` is None."""
    if cond is None:
        yield
        return
    if_op = scf.IfOp(cond, results_=[], has_else=False)
    with ir.InsertionPoint(if_op.then_block):
        yield
        scf.YieldOp([])


def swizzle_xor16(row, col_in_bytes, k_blocks16):
    return col_in_bytes ^ ((row % k_blocks16) * 16)


class WmmaHalfBase(ABC):
    @abstractmethod
    def __init__(self, dtype: str):
        pass

    @abstractmethod
    def __call__(self, a_frag, b_frag, c_frag):
        pass


class WmmaHalf_m16n16k16(WmmaHalfBase):
    WMMA_M = 16
    WMMA_N = 16
    WMMA_K = 16
    WMMA_A_FRAG_VALUES = 4
    WMMA_B_FRAG_VALUES = 4
    WMMA_C_FRAG_VALUES = 4

    def __init__(self, dtype: str):
        self.dtype = dtype

    def __call__(self, a_frag, b_frag, c_frag):
        if self.dtype == "bf16":
            a_frag_vi16 = vector.bitcast(T.vec(self.WMMA_A_FRAG_VALUES, T.i16), a_frag)
            b_frag_vi16 = vector.bitcast(T.vec(self.WMMA_B_FRAG_VALUES, T.i16), b_frag)
            return rocdl.mfma_f32_16x16x16bf16_1k(
                T.f32x4, [a_frag_vi16, b_frag_vi16, c_frag, 0, 0, 0]
            )
        return rocdl.mfma_f32_16x16x16f16(
            T.vec(self.WMMA_C_FRAG_VALUES, T.f32), [a_frag, b_frag, c_frag, 0, 0, 0]
        )


class WmmaHalf_m16n16k32(WmmaHalfBase):
    WMMA_M = 16
    WMMA_N = 16
    WMMA_K = 32
    WMMA_A_FRAG_VALUES = 8
    WMMA_B_FRAG_VALUES = 8
    WMMA_C_FRAG_VALUES = 4

    def __init__(self, dtype: str):
        self.dtype = dtype

    def __call__(self, a_frag, b_frag, c_frag):
        if self.dtype == "bf16":
            return rocdl.mfma_f32_16x16x32_bf16(
                T.vec(self.WMMA_C_FRAG_VALUES, T.f32), [a_frag, b_frag, c_frag, 0, 0, 0]
            )
        return rocdl.mfma_f32_16x16x32_f16(
            T.vec(self.WMMA_C_FRAG_VALUES, T.f32), [a_frag, b_frag, c_frag, 0, 0, 0]
        )


class OnlineScheduler:
    def __init__(self, total_signals: int, init_count: int = 0):
        self.total_signals = total_signals
        self.current_signal_id = init_count
        self.remaining = init_count

    def release(self, count: int):
        count = min(count, self.total_signals - self.current_signal_id)
        self.current_signal_id += count
        self.remaining += count

    def consume(self, count: int):
        count = min(count, self.remaining)
        self.remaining -= count
        return count


@functools.lru_cache(maxsize=16384)
def compile_hgemm_kernel(
    dtype: str,
    n: int,
    k: int,
    TILE_M: int = 128,
    TILE_N: int = 128,
    TILE_K: int = 64,
    STAGES: int = 2,
    SPLIT_K: int = 1,
    BLOCK_M_WARPS: int = 2,
    BLOCK_N_WARPS: int = 2,
    BLOCK_K_WARPS: int = 1,
    B_TO_LDS: bool = False,
    HAS_BIAS: bool = False,
    XCD_BAND: int = 1,
    K_ROT: int = 0,
    M_ROWS: int = 0,
    B_CPOL: int = 0,
):
    assert 0 <= B_CPOL < len(B_CPOL_OPTIONS)
    assert BLOCK_M_WARPS * BLOCK_N_WARPS * BLOCK_K_WARPS <= 16
    assert TILE_M * TILE_N * TILE_K <= 256 * 256 * 64
    if (TILE_M == 256) and (TILE_N == 256):
        assert (TILE_K == 64) and (SPLIT_K == 1) and (STAGES == 2)
    assert STAGES >= 2
    # Clamping and predicating the last N tile frees tile_n from N's divisors.
    N_BLOCKS = -(-n // TILE_N)
    assert N_BLOCKS >= 1
    assert XCD_BAND >= 1
    assert (K_ROT == 0) or B_TO_LDS
    XCD_SPAN_FULL = N_BLOCKS // (NUM_XCD * XCD_BAND) * (NUM_XCD * XCD_BAND)
    XCD_SPAN_TAIL = N_BLOCKS - XCD_SPAN_FULL
    assert (XCD_BAND == 1) or (XCD_SPAN_FULL > 0)
    IS_SPLIT_K = SPLIT_K > 1
    # Split s of tile t is workgroup t + s * grid_x, so a grid_x that is a
    # multiple of NUM_XCD keeps every split of a tile on one XCD.
    SPLIT_K_CPOL = (
        SPLIT_K_CPOL_XCD_LOCAL if (N_BLOCKS % NUM_XCD) == 0 else SPLIT_K_CPOL_COHERENT
    )
    IS_SLICE_K = BLOCK_K_WARPS > 1
    BLOCK_K = TILE_K
    assert (k % SPLIT_K == 0) and (k // SPLIT_K >= 1)
    ks = k // SPLIT_K
    assert (ks % BLOCK_K == 0) and (ks // BLOCK_K >= 1)
    assert BLOCK_K >= 32
    GPU_ARCH = get_rocm_arch()
    if GPU_ARCH == "gfx942":
        WMMA_IMPL = WmmaHalf_m16n16k16(dtype)
        DMA_BYTES = 4
        MFMA_PER_WARP_K = 2
        ASYNC_COPY = True
    else:
        WMMA_IMPL = WmmaHalf_m16n16k32(dtype)
        DMA_BYTES = 16
        MFMA_PER_WARP_K = 1
        ASYNC_COPY = True

    # Fixed parameters:
    WARP_SIZE = 64
    DTYPE_BYTES = 2
    LDG_VEC_SIZE = 8

    # Propagated parameters:
    WMMA_M = WMMA_IMPL.WMMA_M
    WMMA_N = WMMA_IMPL.WMMA_N
    WMMA_K = WMMA_IMPL.WMMA_K
    WMMA_A_FRAG_VALUES = WMMA_IMPL.WMMA_A_FRAG_VALUES
    WMMA_B_FRAG_VALUES = WMMA_IMPL.WMMA_B_FRAG_VALUES
    WMMA_C_FRAG_VALUES = WMMA_IMPL.WMMA_C_FRAG_VALUES
    WARP_ATOM_M = WMMA_M
    WARP_ATOM_N = WMMA_N
    WARP_ATOM_K = WMMA_K * MFMA_PER_WARP_K
    BLOCK_K_LOOPS = ks // BLOCK_K
    assert BLOCK_K_LOOPS >= STAGES
    WARP_GROUP_K = BLOCK_K_WARPS * WARP_ATOM_K
    WARP_K_STEPS = BLOCK_K // WARP_GROUP_K
    assert (BLOCK_K % WARP_GROUP_K == 0) and (WARP_K_STEPS >= 1)
    K_SLICE = BLOCK_K // BLOCK_K_WARPS
    assert K_SLICE % WARP_ATOM_K == 0
    BLOCK_THREADS = BLOCK_M_WARPS * BLOCK_N_WARPS * BLOCK_K_WARPS * WARP_SIZE
    BLOCK_MN_WARPS = BLOCK_M_WARPS * BLOCK_N_WARPS
    WARP_M_STEPS = TILE_M // BLOCK_M_WARPS // WARP_ATOM_M
    WARP_N_STEPS = TILE_N // BLOCK_N_WARPS // WARP_ATOM_N
    assert (WARP_M_STEPS >= 1) and (WARP_N_STEPS >= 1)
    assert TILE_M % (BLOCK_M_WARPS * WARP_ATOM_M) == 0
    assert TILE_N % (BLOCK_N_WARPS * WARP_ATOM_N) == 0
    WARP_M = WARP_M_STEPS * WARP_ATOM_M
    WARP_N = WARP_N_STEPS * WARP_ATOM_N
    BLOCK_M = BLOCK_M_WARPS * WARP_M
    BLOCK_N = BLOCK_N_WARPS * WARP_N
    assert n >= BLOCK_N
    N_TAIL_PREDICATE = (n % BLOCK_N) != 0
    assert (not N_TAIL_PREDICATE) or (B_TO_LDS and (n % LDG_VEC_SIZE == 0))
    BLOCK_MK_SIZE = BLOCK_M * BLOCK_K
    BLOCK_NK_SIZE = BLOCK_N * BLOCK_K
    BLOCK_MN_SIZE = BLOCK_M * BLOCK_N
    LDG_A_X_THREADS = BLOCK_K // LDG_VEC_SIZE
    # LDG_B_X_THREADS = BLOCK_K // LDG_VEC_SIZE
    LDG_C_X_THREADS = BLOCK_N // LDG_VEC_SIZE
    BLOCK_VECS = LDG_VEC_SIZE * BLOCK_THREADS
    LDG_REG_A_COUNT = BLOCK_MK_SIZE // BLOCK_VECS
    if not B_TO_LDS:
        # This path stages A through registers, which has no tail predicate.
        assert BLOCK_MK_SIZE % BLOCK_VECS == 0
        assert BLOCK_NK_SIZE % BLOCK_VECS == 0
    # Divisibility here would forbid the many-warp narrow tile that narrow N needs.
    LDG_C_VECS = BLOCK_MN_SIZE // LDG_VEC_SIZE
    LDG_REG_C_COUNT = -(-LDG_C_VECS // BLOCK_THREADS)
    C_TAIL_PREDICATE = (LDG_C_VECS % BLOCK_THREADS) != 0
    BLOCK_K_BYTES = BLOCK_K * DTYPE_BYTES

    # LDS parameters: the A/B pipeline and C scratch are live at disjoint times,
    # so model their storage overlap explicitly with a union. SharedAllocator
    # emits one static LDS symbol for a union, preserving the old contiguous
    # A-then-B layout without relying on separate struct leaves being adjacent.
    AS_ELEMS = STAGES * BLOCK_M * BLOCK_K
    BS_ELEMS = STAGES * BLOCK_N * BLOCK_K
    CMN_ELEMS = BLOCK_K_WARPS * BLOCK_M * BLOCK_N
    fx_dtype = fx.Float16 if dtype == "f16" else fx.BFloat16
    if B_TO_LDS:
        assert ASYNC_COPY

        @fx.struct
        class PipelineStorage:
            a_lds: fx.Array[fx_dtype, AS_ELEMS, 16]
            b_lds: fx.Array[fx_dtype, BS_ELEMS, 16]

    else:

        @fx.struct
        class PipelineStorage:
            a_lds: fx.Array[fx_dtype, AS_ELEMS, 16]

    @fx.union
    class TileStorage:
        pipeline: PipelineStorage
        c_lds: fx.Array[fx_dtype, CMN_ELEMS, 16]

    @fx.struct
    class SharedStorage:
        tile: TileStorage
        if IS_SPLIT_K:
            split_flag: fx.Array[Int32, 1, 4]

    LDG_ASYNC_VEC_SIZE = DMA_BYTES // DTYPE_BYTES
    LDG_A_X_THREADS_AS = BLOCK_K // LDG_ASYNC_VEC_SIZE
    LDG_B_X_THREADS_AS = BLOCK_K // LDG_ASYNC_VEC_SIZE

    def split_g2s_rounds(tile_vecs):
        """Spread ``tile_vecs`` DMA chunks over the block's threads.

        Either the block covers the tile in whole rounds, or the tile is
        smaller than one round and a single round carries it with the trailing
        waves predicated off. The predicate is wave uniform, so those waves
        branch over the copy instead of running it under a partial exec mask,
        which would scatter the LDS destinations the readers expect.
        """
        if tile_vecs % BLOCK_THREADS == 0:
            return tile_vecs // BLOCK_THREADS, False
        assert (tile_vecs < BLOCK_THREADS) and (tile_vecs % WARP_SIZE == 0)
        return 1, True

    # The MFMA tile is 16 rows wide, so a caller with fewer rows spends part of
    # every A g2s round on padding; fetch only the rows the caller can use and
    # let the rest of the LDS tile feed accumulators the epilogue drops.
    A_ROWS = BLOCK_M if M_ROWS <= 0 else min(M_ROWS, BLOCK_M)
    assert A_ROWS >= 1
    LDG_A_VECS_AS = A_ROWS * BLOCK_K // LDG_ASYNC_VEC_SIZE
    LDG_REG_A_COUNT_AS, LDG_A_PREDICATE = split_g2s_rounds(LDG_A_VECS_AS)
    # A wave that issues no load shortens the graded wait for every wave, so a
    # row count that turns a whole-block A stream into a predicated one is not
    # a valid trade and is rejected rather than silently shallowing the pipe.
    assert LDG_A_PREDICATE == split_g2s_rounds(BLOCK_MK_SIZE // LDG_ASYNC_VEC_SIZE)[1]
    LDG_REG_B_COUNT_AS, LDG_B_PREDICATE = split_g2s_rounds(
        BLOCK_NK_SIZE // LDG_ASYNC_VEC_SIZE
    )
    LDG_B_VECS_AS = BLOCK_NK_SIZE // LDG_ASYNC_VEC_SIZE
    # The barrier counter is per wave, so it must hold for a wave issuing none.
    LDG_WAIT_COUNT = (0 if LDG_B_PREDICATE else LDG_REG_B_COUNT_AS) + (
        0 if LDG_A_PREDICATE else LDG_REG_A_COUNT_AS
    )
    assert ((STAGES - 2) * LDG_WAIT_COUNT) < 63

    USE_8WAVE_PIPE = (
        ASYNC_COPY
        and B_TO_LDS
        and BLOCK_M == 256
        and BLOCK_N == 256
        and BLOCK_K == 64
        and LDG_REG_A_COUNT_AS == 4
        and LDG_REG_B_COUNT_AS == 4
        and MFMA_PER_WARP_K == 1
    )
    USE_8WAVE_PIPE = (
        USE_8WAVE_PIPE
        and BLOCK_M_WARPS == 2
        and BLOCK_N_WARPS == 4
        and BLOCK_K_WARPS == 1
    )

    KERNEL_NAME = f"hgemm_{dtype}_{BLOCK_M}x{BLOCK_N}x{BLOCK_K}x{STAGES}_SPK{SPLIT_K}_W{BLOCK_M_WARPS}x{BLOCK_N_WARPS}x{BLOCK_K_WARPS}_BLDS{int(B_TO_LDS)}_TN"
    KERNEL_NAME += "_AS0" if not ASYNC_COPY else "_AS1"
    if XCD_BAND > 1:
        KERNEL_NAME += f"_XB{XCD_BAND}"
    if K_ROT:
        KERNEL_NAME += f"_KR{K_ROT}"
    if A_ROWS != BLOCK_M:
        KERNEL_NAME += f"_MR{A_ROWS}"
    if B_CPOL:
        KERNEL_NAME += f"_CP{B_CPOL}"
    if HAS_BIAS:
        KERNEL_NAME += "_BIAS"

    @flyc.kernel(known_block_size=[BLOCK_THREADS, 1, 1])
    def hgemm_kernel(
        C: fx.Pointer,
        A: fx.Pointer,
        B: fx.Pointer,
        BIAS: fx.Pointer,
        m: fx.Int32,
        semaphore: fx.Pointer,
        workspace: fx.Pointer,
    ):
        dtype_ = get_dtype_in_kernel(dtype)
        acc_init = arith.constant_vector(0.0, T.vec(WMMA_C_FRAG_VALUES, T.f32))

        A_ = GTensor(A, dtype=dtype_, shape=(-1, k))
        B_ = GTensor(B, dtype=dtype_, shape=(n, k))
        C_ = GTensor(C, dtype=dtype_, shape=(-1, n))
        if const_expr(HAS_BIAS):
            BIAS_ = GTensor(BIAS, dtype=dtype_, shape=(n,))
        lds = fx.SharedAllocator().allocate(SharedStorage)
        a_lds_ptr = lds.tile.pipeline.a_lds.peek().ptr
        c_lds_ptr = lds.tile.c_lds.peek().ptr
        a_lds_i64 = fx.Int64(fx.ptrtoint(a_lds_ptr))
        if const_expr(B_TO_LDS):
            b_lds_ptr = lds.tile.pipeline.b_lds.peek().ptr
            b_lds_i64 = fx.Int64(fx.ptrtoint(b_lds_ptr))

        def _lds_a3_ptr(base_i64, elem_off):
            off_i64 = arith.index_cast(
                T.i64, fx.Index(elem_off) * fx.Index(DTYPE_BYTES)
            )
            return buffer_ops.create_llvm_ptr(
                base_i64 + fx.Int64(off_i64), address_space=3
            )

        # LDS accessors: linear element offsets mirroring the old STensor shapes
        # as_/bs_ = (stage, row, col) over (STAGES, BLOCK*, BLOCK_K);
        # cs_ = (k_slice, row, col) over (BLOCK_K_WARPS, BLOCK_M, BLOCK_N),
        # using the C variant of the pipeline/C union.
        def as_store(stage, row, col, value):
            elem_off = (
                fx.Int64(stage) * (BLOCK_M * BLOCK_K)
                + fx.Int64(row) * BLOCK_K
                + fx.Int64(col)
            )
            fx.ptr_store(value, a_lds_ptr + elem_off)

        def as_load(stage, row, col, vec_size):
            elem_off = (
                fx.Int64(stage) * (BLOCK_M * BLOCK_K)
                + fx.Int64(row) * BLOCK_K
                + fx.Int64(col)
            )
            return fx.ptr_load(
                a_lds_ptr + elem_off,
                result_type=fx.Vector.make_type(vec_size, fx_dtype),
            )

        def bs_store(stage, row, col, value):
            elem_off = (
                fx.Int64(stage) * (BLOCK_N * BLOCK_K)
                + fx.Int64(row) * BLOCK_K
                + fx.Int64(col)
            )
            fx.ptr_store(value, b_lds_ptr + elem_off)

        def bs_load(stage, row, col, vec_size):
            elem_off = (
                fx.Int64(stage) * (BLOCK_N * BLOCK_K)
                + fx.Int64(row) * BLOCK_K
                + fx.Int64(col)
            )
            return fx.ptr_load(
                b_lds_ptr + elem_off,
                result_type=fx.Vector.make_type(vec_size, fx_dtype),
            )

        def cs_store_scalar(k_slice, row, col, value):
            elem_off = (
                fx.Int64(k_slice) * (BLOCK_M * BLOCK_N)
                + fx.Int64(row) * BLOCK_N
                + fx.Int64(col)
            )
            fx.ptr_store(value, c_lds_ptr + elem_off)

        def cs_load_vec(k_slice, row, col, vec_size):
            elem_off = (
                fx.Int64(k_slice) * (BLOCK_M * BLOCK_N)
                + fx.Int64(row) * BLOCK_N
                + fx.Int64(col)
            )
            return fx.ptr_load(
                c_lds_ptr + elem_off,
                result_type=fx.Vector.make_type(vec_size, fx_dtype),
            )

        if const_expr(IS_SPLIT_K):
            # Tile-major, so the reducing block reads consecutive lines.
            WS_ = GTensor(
                workspace,
                dtype=dtype_,
                shape=(-1,),
                cache_modifier=SPLIT_K_CPOL,
            )
            split_flag_ptr = lds.split_flag.peek().ptr
            tile_idx = fx.Int32(fx.block_idx.x)

        tid = fx.thread_idx.x
        wid = tid // WARP_SIZE
        wid_mn = wid % BLOCK_MN_WARPS
        wid_k = wid // BLOCK_MN_WARPS
        w_tid = tid % WARP_SIZE

        def swizzle_for_cache_reuse(pid):
            # n-fastest crowds one XCD's B rows into a few L2 sets; use runs.
            block_n_idx = pid % N_BLOCKS
            if const_expr(XCD_BAND > 1):
                xcd = block_n_idx % NUM_XCD
                pos = block_n_idx // NUM_XCD
                banded = (pos // XCD_BAND * NUM_XCD + xcd) * XCD_BAND + (pos % XCD_BAND)
                # Band whole spans only; the leftover tail stays identity so the
                # map is a bijection for any N_BLOCKS, not just exact multiples.
                if const_expr(XCD_SPAN_TAIL):
                    banded = (block_n_idx < XCD_SPAN_FULL).select(banded, block_n_idx)
                block_n_idx = banded
            return pid // N_BLOCKS, block_n_idx

        block_m_idx, block_n_idx = swizzle_for_cache_reuse(fx.block_idx.x)
        ks_idx = fx.Index(fx.block_idx.y)
        ks_begin = arith.index_cast(T.i32, ks_idx * ks)
        if const_expr(K_ROT):
            # Every block otherwise streams the same k window at the same time,
            # so their B bursts crowd one set of DRAM pages; stagger the starts.
            k_rot_tiles = (block_n_idx * K_ROT) % BLOCK_K_LOOPS
            ks_first = arith.index_cast(T.i32, ks_idx * ks + k_rot_tiles * BLOCK_K)
            ks_wrap = ks_begin + (ks - BLOCK_K)
        else:
            ks_first = ks_begin

        def k_next(k_offset):
            nxt = k_offset + fx.Int32(BLOCK_K)
            if const_expr(K_ROT):
                # Wrap, so a rotated start still visits every k tile exactly once.
                nxt = (k_offset < ks_wrap).select(nxt, ks_begin)
            return nxt

        m_offset = fx.Index(block_m_idx * BLOCK_M)
        n_offset = fx.Index(block_n_idx * BLOCK_N)
        k_blocks16 = fx.Int32(BLOCK_K_BYTES // 16)

        warp_m_idx = wid_mn // BLOCK_N_WARPS * WARP_M
        warp_n_idx = wid_mn % BLOCK_N_WARPS * WARP_N
        ldmatrix_a_m_idx = w_tid % WMMA_M
        ldmatrix_a_k_vec_idx = w_tid // WMMA_M * WMMA_A_FRAG_VALUES * MFMA_PER_WARP_K
        ldmatrix_b_n_idx = w_tid % WMMA_N
        ldmatrix_b_k_vec_idx = w_tid // WMMA_N * WMMA_B_FRAG_VALUES * MFMA_PER_WARP_K
        warp_k_slice_base = wid_k * K_SLICE
        C_FRAGS_LEN = WARP_M_STEPS * WARP_N_STEPS
        c_frags = [acc_init] * C_FRAGS_LEN

        def __barrier(vmcnt=0, use_s_barrier=True):
            if const_expr(use_s_barrier):
                asm = f"s_waitcnt vmcnt({vmcnt})\n\ts_barrier"
            else:
                asm = f"s_waitcnt vmcnt({vmcnt})"
            llvm.InlineAsmOp(None, [], asm, "", has_side_effects=True)

        def get_llvm_ptr(
            ptr,
            offset,
            dtype_bytes,
            ptr_type=ir.Type.parse("!llvm.ptr<1>"),  # noqa: B008
        ):
            base_ptr = arith.index_cast(T.i64, fx.ptrtoint(ptr))
            byte_offset = arith.index_cast(
                T.i64, fx.Index(offset) * fx.Index(dtype_bytes)
            )
            llvm_ptr = llvm.AddOp(
                base_ptr, byte_offset, llvm.IntegerOverflowFlags(0)
            ).result
            llvm_ptr = llvm.IntToPtrOp(ptr_type, llvm_ptr).result
            ptr_v = (
                llvm_ptr._value if const_expr(hasattr(llvm_ptr, "_value")) else llvm_ptr
            )
            return ptr_v

        def split_k_arrive():
            """Publish this block's partial and read back this tile's arrival count.

            The partial store carries sc0|sc1 and is waited on, which is the whole
            release: the block that sees ``SPLIT_K - 1`` prior arrivals reduces.
            No block waits for a designated one, so there is no spin loop and C is
            never pre-zeroed.
            """
            rocdl.s_waitcnt(0)
            gpu.barrier()
            is_t0_cond = arith.cmpi(arith.CmpIPredicate.eq, fx.Index(tid), fx.Index(0))
            is_t0_cond_if = scf.IfOp(is_t0_cond, results_=[], has_else=False)
            with ir.InsertionPoint(is_t0_cond_if.then_block):
                semaphore_ptr = get_llvm_ptr(
                    semaphore, tile_idx * SPLIT_K_SEMAPHORE_STRIDE, 4
                )
                arrive_idx = llvm.AtomicRMWOp(
                    llvm.AtomicBinOp.add,
                    semaphore_ptr,
                    arith.constant(1, type=T.i32),
                    llvm.AtomicOrdering.monotonic,
                    syncscope="agent",
                    alignment=4,
                ).result
                fx.ptr_store(arrive_idx, split_flag_ptr)
                scf.YieldOp([])
            gpu.barrier()
            return fx.Index(fx.ptr_load(split_flag_ptr))

        def split_k_release():
            # Undo with the same atomic, so a next-launch increment survives.
            is_t0_cond = arith.cmpi(arith.CmpIPredicate.eq, fx.Index(tid), fx.Index(0))
            is_t0_cond_if = scf.IfOp(is_t0_cond, results_=[], has_else=False)
            with ir.InsertionPoint(is_t0_cond_if.then_block):
                llvm.AtomicRMWOp(
                    llvm.AtomicBinOp.add,
                    get_llvm_ptr(semaphore, tile_idx * SPLIT_K_SEMAPHORE_STRIDE, 4),
                    arith.constant(-SPLIT_K, type=T.i32),
                    llvm.AtomicOrdering.monotonic,
                    syncscope="agent",
                    alignment=4,
                )
                scf.YieldOp([])

        def ldg_a(k_offset):
            vecs = []
            for i in range_constexpr(LDG_REG_A_COUNT):
                global_tid = BLOCK_THREADS * i + tid
                m_local_idx = global_tid // LDG_A_X_THREADS
                k_local_idx = global_tid % LDG_A_X_THREADS * LDG_VEC_SIZE
                row_idx = m_offset + fx.Index(m_local_idx)
                safe_row_idx = arith.select(
                    arith.cmpi(arith.CmpIPredicate.ult, row_idx, fx.Index(m)),
                    row_idx,
                    fx.Index(0),
                )
                col_idx = fx.Index(k_offset + k_local_idx)
                vec = A_.vec_load((safe_row_idx, col_idx), LDG_VEC_SIZE)
                vecs.append(vec)
            return vecs

        def sts_a(vecs, lds_stage):
            for i in range_constexpr(LDG_REG_A_COUNT):
                global_tid = BLOCK_THREADS * i + tid
                m_local_idx = global_tid // LDG_A_X_THREADS
                k_local_idx = global_tid % LDG_A_X_THREADS * LDG_VEC_SIZE
                col_in_bytes = k_local_idx * DTYPE_BYTES
                col_in_bytes = swizzle_xor16(m_local_idx, col_in_bytes, k_blocks16)
                as_store(lds_stage, m_local_idx, col_in_bytes // DTYPE_BYTES, vecs[i])

        def get_dma_copy_warp_offset():
            warp_offset = rocdl.readfirstlane(
                T.i64,
                arith.index_cast(
                    T.i64,
                    fx.Index(wid) * arith.constant(WARP_SIZE * DMA_BYTES, index=True),
                ),
            )
            return warp_offset

        def buffer_load_lds_inline(rsrc, lds_ptr, global_offset, cpol="sc0"):
            if const_expr(DMA_BYTES == 16):
                op = "buffer_load_dwordx4"
            elif const_expr(DMA_BYTES == 8):
                op = "buffer_load_dwordx2"
            elif const_expr(DMA_BYTES == 4):
                op = "buffer_load_dword"
            else:
                raise NotImplementedError(f"DMA_BYTES={DMA_BYTES} not supported")
            asm = f"s_mov_b32 m0, $0\n\t{op} $1, $2, 0 offen {cpol} lds".replace(
                "  ", " "
            )
            llvm.InlineAsmOp(
                None,
                [lds_ptr, global_offset, rsrc],
                asm,
                "s,v,s",
                has_side_effects=True,
            )

        def ldg_sts_a_async_one(ii, k_offset, write_stage, lds_ptr=None):
            global_tid = BLOCK_THREADS * ii + tid
            m_local_idx = global_tid // LDG_A_X_THREADS_AS
            k_local_idx = global_tid % LDG_A_X_THREADS_AS * LDG_ASYNC_VEC_SIZE
            col_in_bytes = k_local_idx * DTYPE_BYTES
            col_in_bytes = swizzle_xor16(m_local_idx, col_in_bytes, k_blocks16)
            row_idx = m_offset + fx.Index(m_local_idx)
            safe_row_idx = arith.select(
                arith.cmpi(arith.CmpIPredicate.ult, row_idx, fx.Index(m)),
                row_idx,
                fx.Index(0),
            )
            col_idx = fx.Index(k_offset + col_in_bytes // DTYPE_BYTES)
            global_offset = A_.linear_offset((safe_row_idx, col_idx)) * DTYPE_BYTES
            global_offset = arith.index_cast(T.i32, global_offset)
            if const_expr(lds_ptr is None):
                lds_ptr_base = _lds_a3_ptr(
                    a_lds_i64, fx.Index(write_stage) * (BLOCK_M * BLOCK_K)
                )
                lds_ptr = buffer_ops.get_element_ptr(lds_ptr_base, warp_offset)
            else:
                lds_ptr = buffer_ops.get_element_ptr(
                    lds_ptr, static_byte_offset=BLOCK_THREADS * DMA_BYTES
                )
            buffer_load_lds_inline(A_.rsrc, lds_ptr, global_offset)
            return lds_ptr

        def g2s_wave_active(tile_vecs):
            return arith.cmpi(
                arith.CmpIPredicate.ult, fx.Index(tid), fx.Index(tile_vecs)
            )

        def ldg_sts_a_async(k_offset, lds_stage):
            cond = g2s_wave_active(LDG_A_VECS_AS) if LDG_A_PREDICATE else None
            with _if_then(cond):
                lds_ptr = None
                for i in range_constexpr(LDG_REG_A_COUNT_AS):
                    lds_ptr = ldg_sts_a_async_one(
                        i, k_offset, lds_stage, lds_ptr if i > 0 else None
                    )

        def ldg_sts_b_async_one(ii, k_offset, write_stage, lds_ptr=None):
            global_tid = BLOCK_THREADS * ii + tid
            n_local_idx = global_tid // LDG_B_X_THREADS_AS
            k_local_idx = global_tid % LDG_B_X_THREADS_AS * LDG_ASYNC_VEC_SIZE
            col_in_bytes = k_local_idx * DTYPE_BYTES
            col_in_bytes = swizzle_xor16(n_local_idx, col_in_bytes, k_blocks16)
            row_idx = n_offset + fx.Index(n_local_idx)
            safe_row_idx = arith.select(
                arith.cmpi(arith.CmpIPredicate.ult, row_idx, fx.Index(n)),
                row_idx,
                fx.Index(0),
            )
            col_idx = fx.Index(k_offset + col_in_bytes // DTYPE_BYTES)
            global_offset = B_.linear_offset((safe_row_idx, col_idx)) * DTYPE_BYTES
            global_offset = arith.index_cast(T.i32, global_offset)
            if const_expr(lds_ptr is None):
                lds_ptr_base = _lds_a3_ptr(
                    b_lds_i64, fx.Index(write_stage) * (BLOCK_N * BLOCK_K)
                )
                lds_ptr = buffer_ops.get_element_ptr(lds_ptr_base, warp_offset)
            else:
                lds_ptr = buffer_ops.get_element_ptr(
                    lds_ptr, static_byte_offset=BLOCK_THREADS * DMA_BYTES
                )
            buffer_load_lds_inline(
                B_.rsrc, lds_ptr, global_offset, cpol=B_CPOL_OPTIONS[B_CPOL]
            )
            return lds_ptr

        def ldg_sts_b_async(k_offset, lds_stage):
            cond = g2s_wave_active(LDG_B_VECS_AS) if LDG_B_PREDICATE else None
            with _if_then(cond):
                lds_ptr = None
                for i in range_constexpr(LDG_REG_B_COUNT_AS):
                    lds_ptr = ldg_sts_b_async_one(
                        i, k_offset, lds_stage, lds_ptr if i > 0 else None
                    )

        def ldg_matrix_b(k_offset):
            vecs = []
            for kk in range_constexpr(WARP_K_STEPS):
                for ii in range_constexpr(WARP_N_STEPS):
                    warp_atom_n_idx = warp_n_idx + ii * WARP_ATOM_N
                    warp_atom_k_idx = warp_k_slice_base + kk * WARP_ATOM_K
                    n_idx = n_offset + warp_atom_n_idx + ldmatrix_b_n_idx
                    k_idx = k_offset + warp_atom_k_idx + ldmatrix_b_k_vec_idx
                    vec = B_.vec_load(
                        (n_idx, k_idx), WMMA_B_FRAG_VALUES * MFMA_PER_WARP_K
                    )
                    vecs.append(vec)
            return vecs

        def ldmatrix_compute_tile_streaming(lds_stage, c_frags, initial_b_frags=None):
            s = fx.Index(lds_stage)
            c_frags_new = [cx for cx in c_frags]
            for kk in range_constexpr(WARP_K_STEPS):
                warp_atom_k_idx = warp_k_slice_base + kk * WARP_ATOM_K
                if const_expr(initial_b_frags is None):
                    b_frags = [0] * WARP_N_STEPS
                    for ii in range_constexpr(WARP_N_STEPS):
                        warp_atom_n_idx = warp_n_idx + ii * WARP_ATOM_N
                        row = warp_atom_n_idx + ldmatrix_b_n_idx
                        col_in_bytes = (
                            warp_atom_k_idx + ldmatrix_b_k_vec_idx
                        ) * DTYPE_BYTES
                        col_in_bytes = swizzle_xor16(row, col_in_bytes, k_blocks16)
                        vec = bs_load(
                            s,
                            row,
                            col_in_bytes // DTYPE_BYTES,
                            WMMA_B_FRAG_VALUES * MFMA_PER_WARP_K,
                        )
                        b_frags[ii] = vec
                else:
                    b_frags = [
                        initial_b_frags[i]
                        for i in range_constexpr(
                            kk * WARP_N_STEPS, (kk + 1) * WARP_N_STEPS
                        )
                    ]
                a_frags = [0] * WARP_M_STEPS
                for ii in range_constexpr(WARP_M_STEPS):
                    warp_atom_m_idx = warp_m_idx + ii * WARP_ATOM_M
                    row = warp_atom_m_idx + ldmatrix_a_m_idx
                    col_in_bytes = (
                        warp_atom_k_idx + ldmatrix_a_k_vec_idx
                    ) * DTYPE_BYTES
                    col_in_bytes = swizzle_xor16(row, col_in_bytes, k_blocks16)
                    vec = as_load(
                        s,
                        row,
                        col_in_bytes // DTYPE_BYTES,
                        WMMA_A_FRAG_VALUES * MFMA_PER_WARP_K,
                    )
                    a_frags[ii] = vec
                rocdl.sched_barrier(0)
                for ii in range_constexpr(WARP_M_STEPS):
                    a_frag = a_frags[ii]
                    for jj in range_constexpr(WARP_N_STEPS):
                        b_frag = b_frags[jj]
                        if const_expr(MFMA_PER_WARP_K == 2):
                            # split a
                            a_i64x2 = vector.bitcast(T.i64x2, a_frag)
                            a0_i64 = vector.extract(
                                a_i64x2, static_position=[0], dynamic_position=[]
                            )
                            a1_i64 = vector.extract(
                                a_i64x2, static_position=[1], dynamic_position=[]
                            )
                            a_v0 = vector.bitcast(
                                T.f16x4, vector.from_elements(T.vec(1, T.i64), [a0_i64])
                            )
                            a_v1 = vector.bitcast(
                                T.f16x4, vector.from_elements(T.vec(1, T.i64), [a1_i64])
                            )
                            # split b
                            b_i64x2 = vector.bitcast(T.i64x2, b_frag)
                            b0_i64 = vector.extract(
                                b_i64x2, static_position=[0], dynamic_position=[]
                            )
                            b1_i64 = vector.extract(
                                b_i64x2, static_position=[1], dynamic_position=[]
                            )
                            b_v0 = vector.bitcast(
                                T.f16x4, vector.from_elements(T.vec(1, T.i64), [b0_i64])
                            )
                            b_v1 = vector.bitcast(
                                T.f16x4, vector.from_elements(T.vec(1, T.i64), [b1_i64])
                            )
                            # wmma
                            c_idx = ii * WARP_N_STEPS + jj
                            acc_in = c_frags_new[c_idx]
                            acc_mid = WMMA_IMPL(a_v0, b_v0, acc_in)
                            c_frags_new[c_idx] = WMMA_IMPL(a_v1, b_v1, acc_mid)
                        elif const_expr(MFMA_PER_WARP_K == 1):
                            c_idx = ii * WARP_N_STEPS + jj
                            c_frags_new[c_idx] = WMMA_IMPL(
                                a_frag, b_frag, c_frags_new[c_idx]
                            )
                        else:
                            raise NotImplementedError(
                                f"MFMA_PER_WARP_K={MFMA_PER_WARP_K} not supported"
                            )
            return c_frags_new

        def async_copy_ldmatrix_compute_tile_streaming(
            lds_stage,
            c_frags,
            k_offset_next_tile,
            write_stage,
        ):
            assert LDG_REG_A_COUNT_AS == 4
            assert LDG_REG_B_COUNT_AS == 4
            assert MFMA_PER_WARP_K == 1
            assert WARP_M_STEPS % 2 == 0
            assert WARP_N_STEPS % 2 == 0

            M_HALF_STEPS = WARP_M_STEPS // 2
            N_HALF_STEPS = WARP_N_STEPS // 2

            s = fx.Index(lds_stage)
            c_frags_new = [cx for cx in c_frags]
            lds_ptr_a = None
            lds_ptr_b = None

            def load_b_frag(n_step, warp_atom_k_idx):
                warp_atom_n_idx = warp_n_idx + n_step * WARP_ATOM_N
                row = warp_atom_n_idx + ldmatrix_b_n_idx
                col_in_bytes = (warp_atom_k_idx + ldmatrix_b_k_vec_idx) * DTYPE_BYTES
                col_in_bytes = swizzle_xor16(row, col_in_bytes, k_blocks16)
                return bs_load(
                    s,
                    row,
                    col_in_bytes // DTYPE_BYTES,
                    WMMA_B_FRAG_VALUES * MFMA_PER_WARP_K,
                )

            def load_a_frag(m_step, warp_atom_k_idx):
                warp_atom_m_idx = warp_m_idx + m_step * WARP_ATOM_M
                row = warp_atom_m_idx + ldmatrix_a_m_idx
                col_in_bytes = (warp_atom_k_idx + ldmatrix_a_k_vec_idx) * DTYPE_BYTES
                col_in_bytes = swizzle_xor16(row, col_in_bytes, k_blocks16)
                return as_load(
                    s,
                    row,
                    col_in_bytes // DTYPE_BYTES,
                    WMMA_A_FRAG_VALUES * MFMA_PER_WARP_K,
                )

            for kk in range_constexpr(WARP_K_STEPS):
                warp_atom_k_idx = kk * WARP_ATOM_K
                b0_frags = [0] * N_HALF_STEPS  # 2
                b1_frags = [0] * N_HALF_STEPS
                a0_frags = [0] * M_HALF_STEPS  # 4
                a1_frags = [0] * M_HALF_STEPS
                if const_expr(kk == 0):
                    lds_ptr_b = ldg_sts_b_async_one(
                        0, k_offset_next_tile, write_stage, lds_ptr_b
                    )
                    lds_ptr_b = ldg_sts_b_async_one(
                        1, k_offset_next_tile, write_stage, lds_ptr_b
                    )
                for ni in range_constexpr(N_HALF_STEPS):
                    b0_frags[ni] = load_b_frag(ni, warp_atom_k_idx)
                for mi in range_constexpr(M_HALF_STEPS):
                    a0_frags[mi] = load_a_frag(mi, warp_atom_k_idx)
                if const_expr(kk == 0):
                    rocdl.s_setprio(1)
                for mi in range_constexpr(M_HALF_STEPS):
                    for ni in range_constexpr(N_HALF_STEPS):
                        c_idx = mi * WARP_N_STEPS + ni
                        c_frags_new[c_idx] = WMMA_IMPL(
                            a0_frags[mi],
                            b0_frags[ni],
                            c_frags_new[c_idx],
                        )
                if const_expr(kk == 0):
                    rocdl.s_setprio(0)
                if const_expr(kk == 0):
                    lds_ptr_a = ldg_sts_a_async_one(
                        0, k_offset_next_tile, write_stage, lds_ptr_a
                    )
                    lds_ptr_a = ldg_sts_a_async_one(
                        1, k_offset_next_tile, write_stage, lds_ptr_a
                    )
                for ni in range_constexpr(N_HALF_STEPS):
                    b1_frags[ni] = load_b_frag(N_HALF_STEPS + ni, warp_atom_k_idx)
                if const_expr(kk == 0):
                    rocdl.s_setprio(1)
                for mi in range_constexpr(M_HALF_STEPS):
                    for ni in range_constexpr(N_HALF_STEPS):
                        c_idx = mi * WARP_N_STEPS + N_HALF_STEPS + ni
                        c_frags_new[c_idx] = WMMA_IMPL(
                            a0_frags[mi],
                            b1_frags[ni],
                            c_frags_new[c_idx],
                        )
                if const_expr(kk == 0):
                    rocdl.s_setprio(0)
                    lds_ptr_b = ldg_sts_b_async_one(
                        2, k_offset_next_tile, write_stage, lds_ptr_b
                    )
                    lds_ptr_b = ldg_sts_b_async_one(
                        3, k_offset_next_tile, write_stage, lds_ptr_b
                    )
                for mi in range_constexpr(M_HALF_STEPS):
                    a1_frags[mi] = load_a_frag(M_HALF_STEPS + mi, warp_atom_k_idx)
                if const_expr(kk == 0):
                    rocdl.s_setprio(1)
                for mi in range_constexpr(M_HALF_STEPS):
                    for ni in range_constexpr(N_HALF_STEPS):
                        c_idx = (M_HALF_STEPS + mi) * WARP_N_STEPS + ni
                        c_frags_new[c_idx] = WMMA_IMPL(
                            a1_frags[mi],
                            b0_frags[ni],
                            c_frags_new[c_idx],
                        )
                if const_expr(kk == 0):
                    rocdl.s_setprio(0)
                    lds_ptr_a = ldg_sts_a_async_one(
                        2, k_offset_next_tile, write_stage, lds_ptr_a
                    )
                    lds_ptr_a = ldg_sts_a_async_one(
                        3, k_offset_next_tile, write_stage, lds_ptr_a
                    )
                    rocdl.s_setprio(1)
                for mi in range_constexpr(M_HALF_STEPS):
                    for ni in range_constexpr(N_HALF_STEPS):
                        c_idx = (M_HALF_STEPS + mi) * WARP_N_STEPS + N_HALF_STEPS + ni
                        c_frags_new[c_idx] = WMMA_IMPL(
                            a1_frags[mi],
                            b1_frags[ni],
                            c_frags_new[c_idx],
                        )
                if const_expr(kk == 0):
                    rocdl.s_setprio(0)
            return c_frags_new

        warp_offset = get_dma_copy_warp_offset()

        if const_expr(B_TO_LDS):
            k_load = ks_first
            for s in range_constexpr(STAGES - 1):
                ldg_sts_b_async(k_load, s)
                ldg_sts_a_async(k_load, s)
                k_load = k_next(k_load)
            rocdl.sched_barrier(0)

            def hot_loop_scheduler():
                # ================ Ordered ================
                if const_expr(USE_8WAVE_PIPE):
                    for ki in range_constexpr(WARP_K_STEPS):
                        if const_expr(ki == 0):
                            rocdl.sched_vmem(2)
                        rocdl.sched_dsrd(2)
                        rocdl.sched_dsrd(4)
                        rocdl.sched_mfma(8)
                        if const_expr(ki == 0):
                            rocdl.sched_vmem(2)
                        rocdl.sched_dsrd(2)
                        rocdl.sched_mfma(8)
                        if const_expr(ki == 0):
                            rocdl.sched_vmem(2)
                        rocdl.sched_dsrd(4)
                        rocdl.sched_mfma(8)
                        if const_expr(ki == 0):
                            rocdl.sched_vmem(2)
                        rocdl.sched_mfma(8)
                else:
                    for i in range_constexpr(LDG_REG_B_COUNT_AS):
                        rocdl.sched_vmem(1)  # ldg_sts_b_async next
                    for i in range_constexpr(LDG_REG_A_COUNT_AS):
                        rocdl.sched_vmem(1)  # ldg_sts_a_async next
                    for ki in range_constexpr(WARP_K_STEPS):
                        for i in range_constexpr(WARP_N_STEPS):
                            rocdl.sched_dsrd(1)  # lds_matrix_b current
                        for i in range_constexpr(WARP_M_STEPS):
                            rocdl.sched_dsrd(1)  # lds_matrix_a current
                        for i in range_constexpr(WARP_M_STEPS):
                            rocdl.sched_mfma(WARP_N_STEPS)
                # ================ Reordered ================
                rocdl.sched_barrier(0)

            init_state = [k_load, arith.constant(0, index=True)] + c_frags
            for bki, state in range(
                0, BLOCK_K_LOOPS - (STAGES - 1), 1, init=init_state
            ):
                k_load = state[0]
                current_stage = fx.Index(state[1])
                c_frags = state[2:]
                next_stage = (current_stage + 1) % STAGES
                write_stage = (current_stage + STAGES - 1) % STAGES
                __barrier((STAGES - 2) * LDG_WAIT_COUNT)
                if const_expr(USE_8WAVE_PIPE):
                    c_frags_new = async_copy_ldmatrix_compute_tile_streaming(
                        current_stage,
                        c_frags,
                        k_load,
                        write_stage,
                    )
                else:
                    ldg_sts_b_async(k_load, write_stage)
                    ldg_sts_a_async(k_load, write_stage)
                    c_frags_new = ldmatrix_compute_tile_streaming(
                        current_stage, c_frags
                    )
                k_load_next = k_next(k_load)
                hot_loop_scheduler()
                results = yield [k_load_next, next_stage] + c_frags_new
            current_stage = fx.Index(results[1])
            c_frags = results[2:]
            for s in range_constexpr(0, STAGES - 1):
                __barrier((STAGES - 2 - s) * LDG_WAIT_COUNT)
                c_frags = ldmatrix_compute_tile_streaming(current_stage, c_frags)
                current_stage = (current_stage + 1) % STAGES

        else:
            assert STAGES == 2
            sts_a(ldg_a(ks_begin), 0)
            b_frags_next = ldg_matrix_b(ks_begin)
            rocdl.sched_barrier(0)
            __barrier()

            def hot_loop_scheduler():
                LDG_REG_A_COUNT_ = (
                    LDG_REG_A_COUNT_AS if const_expr(ASYNC_COPY) else LDG_REG_A_COUNT
                )
                LDG_TOTAL = LDG_REG_A_COUNT_ + WARP_K_STEPS * WARP_N_STEPS
                # ================ Ordered ================
                for i in range_constexpr(LDG_TOTAL):
                    rocdl.sched_vmem(1)
                for ki in range_constexpr(WARP_K_STEPS):
                    for i in range_constexpr(WARP_M_STEPS):
                        rocdl.sched_dsrd(1)
                    for i in range_constexpr(WARP_M_STEPS):
                        rocdl.sched_mfma(WARP_N_STEPS)
                # ================ Reordered ================
                rocdl.sched_barrier(0)

            init_state = (
                [ks_begin, arith.constant(0, index=True)] + c_frags + b_frags_next
            )
            for bki, state in range(1, BLOCK_K_LOOPS, init=init_state):
                k_offset = state[0]
                current_stage = fx.Index(state[1])
                next_stage = 1 - current_stage
                c_frags = state[2 : 2 + C_FRAGS_LEN]
                b_frags = state[2 + C_FRAGS_LEN :]
                if const_expr(ASYNC_COPY):
                    ldg_sts_a_async(k_offset + BLOCK_K, next_stage)
                else:
                    a_regs_next = ldg_a(k_offset + BLOCK_K)
                b_frags_next = ldg_matrix_b(k_offset + BLOCK_K)
                c_frags_new = ldmatrix_compute_tile_streaming(
                    current_stage, c_frags, b_frags
                )
                if const_expr(not ASYNC_COPY):
                    sts_a(a_regs_next, next_stage)
                k_offset = k_offset + fx.Int32(BLOCK_K)
                hot_loop_scheduler()
                __barrier()
                results = yield [k_offset, next_stage] + c_frags_new + b_frags_next
            current_stage = fx.Index(results[1])
            c_frags = results[2 : 2 + C_FRAGS_LEN]
            b_frags = results[2 + C_FRAGS_LEN :]
            c_frags = ldmatrix_compute_tile_streaming(current_stage, c_frags, b_frags)

        # write to lds
        stmatrix_c_m_vec_idx = w_tid // WMMA_N * WMMA_C_FRAG_VALUES
        stmatrix_c_n_idx = w_tid % WMMA_N
        gpu.barrier()
        for ii in range_constexpr(WARP_M_STEPS):
            warp_atom_m_idx = warp_m_idx + ii * WARP_ATOM_M
            for jj in range_constexpr(WARP_N_STEPS):
                warp_atom_n_idx = warp_n_idx + jj * WARP_ATOM_N
                for kk in range_constexpr(WMMA_C_FRAG_VALUES):
                    lds_m_idx = fx.Index(warp_atom_m_idx + stmatrix_c_m_vec_idx + kk)
                    lds_n_idx = fx.Index(warp_atom_n_idx + stmatrix_c_n_idx)
                    val = vector.extract(
                        c_frags[ii * WARP_N_STEPS + jj],
                        static_position=[kk],
                        dynamic_position=[],
                    )
                    val = val.truncf(dtype_)
                    if const_expr(IS_SLICE_K):
                        cs_store_scalar(wid_k, lds_m_idx, lds_n_idx, val)
                    else:
                        cs_store_scalar(0, lds_m_idx, lds_n_idx, val)

        def c_lane_active(global_tid, cond=None):
            if const_expr(C_TAIL_PREDICATE):
                lane_cond = arith.cmpi(
                    arith.CmpIPredicate.ult,
                    fx.Index(global_tid),
                    fx.Index(LDG_C_VECS),
                )
                cond = lane_cond if cond is None else arith.andi(cond, lane_cond)
            return cond

        def c_col_active(n_local_idx, cond=None):
            if const_expr(N_TAIL_PREDICATE):
                col_cond = arith.cmpi(
                    arith.CmpIPredicate.ult, n_offset + n_local_idx, fx.Index(n)
                )
                cond = col_cond if cond is None else arith.andi(cond, col_cond)
            return cond

        # write back to global
        if const_expr(IS_SPLIT_K):
            gpu.barrier()
            # Reduce in f32 rather than rounding at every tree level.
            acc_vec_type = fx.Vector.make_type(LDG_VEC_SIZE, fx.Float32)
            out_vec_type = fx.Vector.make_type(LDG_VEC_SIZE, fx_dtype)
            # Whole tile including rows past M; the reduction drops them.
            ws_tile_base = (
                tile_idx * (SPLIT_K * BLOCK_MN_SIZE)
                + arith.index_cast(T.i32, ks_idx) * BLOCK_MN_SIZE
            )
            for i in range_constexpr(LDG_REG_C_COUNT):
                global_tid = BLOCK_THREADS * i + tid
                with _if_then(c_lane_active(global_tid)):
                    m_local_idx = fx.Index(global_tid // LDG_C_X_THREADS)
                    n_local_idx = fx.Index(global_tid % LDG_C_X_THREADS * LDG_VEC_SIZE)
                    pk_val = cs_load_vec(0, m_local_idx, n_local_idx, LDG_VEC_SIZE)
                    for ksi in range_constexpr(1, BLOCK_K_WARPS):
                        pk_val += cs_load_vec(
                            ksi, m_local_idx, n_local_idx, LDG_VEC_SIZE
                        )
                    WS_.vec_store(
                        (ws_tile_base + global_tid * LDG_VEC_SIZE,),
                        pk_val,
                        LDG_VEC_SIZE,
                    )

            is_last = arith.cmpi(
                arith.CmpIPredicate.eq, split_k_arrive(), fx.Index(SPLIT_K - 1)
            )
            is_last_if = scf.IfOp(is_last, results_=[], has_else=False)
            with ir.InsertionPoint(is_last_if.then_block):
                for i in range_constexpr(LDG_REG_C_COUNT):
                    global_tid = BLOCK_THREADS * i + tid
                    m_local_idx = fx.Index(global_tid // LDG_C_X_THREADS)
                    n_local_idx = fx.Index(global_tid % LDG_C_X_THREADS * LDG_VEC_SIZE)
                    m_global_idx = m_offset + m_local_idx
                    cond_boundary = arith.cmpi(
                        arith.CmpIPredicate.ult, m_global_idx, fx.Index(m)
                    )
                    cond_boundary = c_lane_active(global_tid, cond_boundary)
                    cond_boundary = c_col_active(n_local_idx, cond_boundary)
                    cond_boundary_if = scf.IfOp(
                        cond_boundary, results_=[], has_else=False
                    )
                    with ir.InsertionPoint(cond_boundary_if.then_block):
                        ws_off = (
                            tile_idx * (SPLIT_K * BLOCK_MN_SIZE)
                            + global_tid * LDG_VEC_SIZE
                        )
                        planes = [
                            WS_.vec_load(
                                (ws_off + s * BLOCK_MN_SIZE,), LDG_VEC_SIZE
                            ).extf(acc_vec_type)
                            for s in range_constexpr(SPLIT_K)
                        ]
                        vec = _pairwise_sum(planes)
                        if const_expr(HAS_BIAS):
                            vec = vec + BIAS_.vec_load(
                                (n_offset + n_local_idx,), LDG_VEC_SIZE
                            ).extf(acc_vec_type)
                        vec = vec.truncf(out_vec_type)
                        C_.vec_store(
                            (m_global_idx, n_offset + n_local_idx), vec, LDG_VEC_SIZE
                        )
                        scf.YieldOp([])
                split_k_release()
                scf.YieldOp([])
        else:
            gpu.barrier()
            for i in range_constexpr(LDG_REG_C_COUNT):
                global_tid = BLOCK_THREADS * i + tid
                m_local_idx = fx.Index(global_tid // LDG_C_X_THREADS)
                n_local_idx = fx.Index(global_tid % LDG_C_X_THREADS * LDG_VEC_SIZE)
                m_global_idx = m_offset + m_local_idx
                cond_boundary = arith.cmpi(
                    arith.CmpIPredicate.ult, m_global_idx, fx.Index(m)
                )
                cond_boundary = c_lane_active(global_tid, cond_boundary)
                cond_boundary = c_col_active(n_local_idx, cond_boundary)
                cond_boundary_if = scf.IfOp(cond_boundary, results_=[], has_else=False)
                with ir.InsertionPoint(cond_boundary_if.then_block):
                    vec = cs_load_vec(0, m_local_idx, n_local_idx, LDG_VEC_SIZE)
                    for ksi in range_constexpr(1, BLOCK_K_WARPS):
                        vec += cs_load_vec(ksi, m_local_idx, n_local_idx, LDG_VEC_SIZE)
                    if const_expr(HAS_BIAS):
                        bias_vec = BIAS_.vec_load(
                            (n_offset + n_local_idx,), LDG_VEC_SIZE
                        )
                        vec = vec + bias_vec
                    C_.vec_store(
                        (m_global_idx, n_offset + n_local_idx), vec, LDG_VEC_SIZE
                    )
                    scf.YieldOp([])

    @flyc.jit
    def launch_hgemm_kernel(
        C: fx.Pointer,
        A: fx.Pointer,
        B: fx.Pointer,
        BIAS: fx.Pointer,
        m: fx.Int32,
        semaphore: fx.Pointer,
        workspace: fx.Pointer,
        stream: fx.Stream,
    ):
        bm = (m + BLOCK_M - 1) // BLOCK_M
        hgemm_kernel._func.__name__ = KERNEL_NAME
        value_attrs = (
            {
                "rocdl.waves_per_eu": 2,
                "rocdl.flat_work_group_size": f"{BLOCK_THREADS},{BLOCK_THREADS}",
            }
            if USE_8WAVE_PIPE
            else None
        )
        # Preloading the leading kernarg dwords into user SGPRs removes the
        # scalar round trip every block serialises before its first B load.
        preload = [{"llvm.inreg": ir.UnitAttr.get()}] * 5 + [{}] * 2
        hgemm_kernel(
            C,
            A,
            B,
            BIAS,
            m,
            semaphore,
            workspace,
            value_attrs={"arg_attrs": preload},
        ).launch(
            grid=(bm * N_BLOCKS, SPLIT_K, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
            value_attrs=value_attrs,
        )

    return launch_hgemm_kernel
