# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import triton
import triton.language as tl

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr
from aiter.ops.triton.utils.gemm_config_utils import get_gemm_config

_batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant_repr = make_kernel_repr(
    "_batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant_kernel",
    [
        "HAS_BIAS",
        "BLOCK_SIZE_M",
        "BLOCK_SIZE_N",
        "BLOCK_SIZE_K",
        "GROUP_SIZE_M",
        "EVEN_K",
        "EVEN_MN",
        "cache_modifier",
    ],
)


@triton.jit
def _reciprocal(x):
    """1 / x to within about half an ulp, for a strictly positive x.

    The backend expands `1.0 / x` into the IEEE sequence -- two v_div_scale, a
    v_rcp, three refinement FMAs, v_div_fmas and v_div_fixup -- and neither
    tl.fdiv(..., ieee_rounding=False) nor the fast-math flags relax it on
    gfx950. v_rcp_f32 carries at most one ulp and the Newton step halves that.
    """
    r = tl.inline_asm_elementwise(
        "v_rcp_f32_e32 $0, $1",
        "=v,v",
        [x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    return r * (2.0 - x * r)


@triton.heuristics(
    {
        "EVEN_K": lambda args: args["K"] % args["BLOCK_SIZE_K"] == 0,
        "EVEN_MN": lambda args: (args["M"] % args["BLOCK_SIZE_M"] == 0)
        and (args["N"] % args["BLOCK_SIZE_N"] == 0),
    }
)
@triton.jit(
    repr=_batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant_repr,
    do_not_specialize=["M", "N"],
)
def _batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant_kernel(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    b_scale_ptr,
    bias_ptr,
    # Matrix dimensions
    M,
    N,
    K,
    # The stride variables represent how much to increase the ptr by when
    # moving by 1 element in a particular dimension. E.g. `stride_am` is
    # how much to increase `a_ptr` by to get the element one row down
    # (A has M rows).
    stride_in_ab,
    stride_in_am,
    stride_in_ak,
    stride_in_bb,
    stride_in_bk,
    stride_in_bn,
    stride_in_cb,
    stride_in_cm,
    stride_in_cn,
    stride_in_biasb,
    # Meta-parameters
    HAS_BIAS: tl.constexpr,
    DTYPE_MAX: tl.constexpr,
    DTYPE_MIN: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    EVEN_K: tl.constexpr,
    EVEN_MN: tl.constexpr,
    cache_modifier: tl.constexpr,
):
    """
    Note: this is Triton jited function and not meant to be called directly. Call batched_gemm_a8w8 function
    below

    Computes the matmul C[i] = A[i] x B[i] and applies a conversion scale for every i in a given batch.
    Optionally, adds a bias to each result.

    The conversion scale for each matmul is received in the form of two 1D tensors that are multiplied to form a
    2D one before being applied.

    Key parameters:
    - A: Batch tensor A with shape (B, M, K).
    - B: Batch tensor B with shape (B, K, N).
    - C: Batch tensor C with shape (B, M, N).
    - A_scale: First scale batch tensor with shape (B, M, 1).
    - B_scale: Second scale batch tensor with shape (B, 1, N).
    - Bias: Bias batch tensor with shape (B, 1, N).
    """

    stride_ab = tl.cast(stride_in_ab, tl.int64)
    stride_am = tl.cast(stride_in_am, tl.int64)
    stride_ak = tl.cast(stride_in_ak, tl.int64)
    stride_bb = tl.cast(stride_in_bb, tl.int64)
    stride_bk = tl.cast(stride_in_bk, tl.int64)
    stride_bn = tl.cast(stride_in_bn, tl.int64)
    stride_cb = tl.cast(stride_in_cb, tl.int64)
    stride_cm = tl.cast(stride_in_cm, tl.int64)
    stride_cn = tl.cast(stride_in_cn, tl.int64)
    stride_biasb = tl.cast(stride_in_biasb, tl.int64)

    tl.assume(stride_ab > 0)
    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bb > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_cb > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)
    tl.assume(stride_biasb > 0)

    # -----------------------------------------------------------
    # Get batch program id
    batch_id = tl.program_id(axis=0)
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    pid = tl.program_id(axis=1)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    if GROUP_SIZE_M == 1:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n
    else:
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

    batch_id = tl.cast(batch_id, tl.int64)
    pid_m = tl.cast(pid_m, tl.int64)
    pid_n = tl.cast(pid_n, tl.int64)

    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)
    tl.assume(batch_id >= 0)

    offs_k = tl.arange(0, BLOCK_SIZE_K)
    if EVEN_MN:
        offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    else:
        offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    a_ptrs = a_ptr + (
        batch_id * stride_ab
        + offs_am[:, None] * stride_am
        + offs_k[None, :] * stride_ak
    )
    b_ptrs = b_ptr + (
        batch_id * stride_bb
        + offs_k[:, None] * stride_bk
        + offs_bn[None, :] * stride_bn
    )
    one_over_DTYPE_MAX = 1.0 / DTYPE_MAX
    b_scale = tl.load(b_scale_ptr)

    acc_dtype = tl.float32 if c_ptr.type.element_ty != tl.int8 else tl.int32
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)

    # BLOCK_SIZE_K is the quantisation group size, so a K that is not a whole
    # number of groups runs its short last group at full width: at K=192 against
    # group_size 128 a quarter of the arithmetic is spent on a block of zeros,
    # and the mask rides along on the full-width groups that never needed one.
    # Peeling that group to its own power-of-two width is bit-exact (verified on
    # eleven shapes by zero-padding K) but not reachable through this codegen:
    # a second dot of a different K makes Triton stage the narrow B tile through
    # LDS one byte at a time (M=48: 673 -> 1026 instructions, VGPR 82 -> 118,
    # 32 buffer_load_ubyte + 35 ds_write_b8) and re-materialise the 64-bit index
    # math (M=84: 673 -> 3513, 2388 of them SALU). Measured +43..+55% on uk,
    # unchanged on uv, with the tail loads either in place or hoisted above the
    # loop. Do not re-walk it without first checking what the narrow dot lowers to.
    # The loop order is load-quantise-dot per group and it has to stay that
    # way. Moving the gathers of several groups above the dot chain makes this
    # kernel wrong: Triton stages each group's activation scale through LDS to
    # carry it from the reduction layout into the MFMA output layout, and with
    # more than one group in flight it emits the extra ds_write without the
    # matching fence -- 8 ds_write against 11 s_barrier where a separated form
    # needs 16 -- so one group's scale write races the previous group's read.
    # Measured relL2 0.059..0.070 against 0.0017 for this loop in 8 of 192
    # (M, K, tile, num_warps, num_stages) combinations, non-deterministic at
    # M=200, and one combination changed verdict between two processes. The
    # hazard is not reachable from here: an explicit tl.debug_barrier() on
    # either side of the quantise still diverges on a heavy-tailed input, and
    # hoisting is not worth repairing anyway -- two or four groups ahead
    # measure +1.0 / +2.2% at (N=256, K=512) and +0.7 / -0.4% at (N=512, K=192)
    # against this loop, and the barriers themselves add up to +9.7%. Hoisting
    # only B is fenced correctly and bit-exact but costs +8.8..+16.8% at
    # (N=256, K=512): the latency that matters is the activation gather feeding
    # the prequant, not the weights. num_stages cannot substitute either -- 4 /
    # 5 / 6 measure +16.6 / +31.1 / +47.4% against 3 at (M=48, N=512, K=192).
    # Nor can the compile options: kpack=2 is -0.0%, schedule_hint=attention
    # -0.2..-0.0%, memory-bound-attention +0.2/+1.4%, enable_fp_fusion=False
    # +0.9/+1.1% and no longer bit-exact. num_warps below 4 removes the scale's
    # cross-warp broadcast outright and still loses: 2 costs +8.5..+22% and 1
    # costs +11..+57% over BN in {32, 64, 128, 256}, all bit-exact.
    for k in range(tl.cdiv(K, BLOCK_SIZE_K)):
        if EVEN_K:
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs, cache_modifier=cache_modifier)
        else:
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)

        # Widen A once and feed both the group amax and the rescale from the
        # f32 copy: reading `a` twice makes the backend widen it twice, once
        # fused into the abs (v_and_b32_sdwa per element) and again for the
        # multiply. The clamp is dead by construction -- a_scale is
        # max|a| / DTYPE_MAX, so |a * (1 / a_scale)| <= DTYPE_MAX for every
        # element, including the max|a| < 1e-10 floor -- and it costs a
        # v_med3_f32 per element. Together that is 10 of the 117 instructions
        # in the K-loop body at (N=256, K=512) and 11 of 138 at (N=512, K=192),
        # bit-identical on both shapes at M = 8, 42, 48, 60, 84.
        # The reciprocal is the other ten: a_scale is one value per row, so an
        # IEEE-correct 1/x buys nothing the fp8 rounding can see, and the
        # refined v_rcp_f32 is bit-identical to it on every shape and input
        # distribution measured. It cannot ride on the convert's own scale
        # operand, which the backend ties off at 1.0 -- an fp8-typed inline asm
        # result fails register allocation on this LLVM whatever the pack width.
        af = a.to(tl.float32)
        m = tl.maximum(tl.max(tl.abs(af), axis=-1), 1e-10)[:, None]
        a_scale = m * one_over_DTYPE_MAX
        a = (af * _reciprocal(a_scale)).to(b_ptr.dtype.element_ty)

        accumulator += tl.dot(a, b) * a_scale

        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    accumulator *= b_scale

    if HAS_BIAS:
        if EVEN_MN:
            offs_bias = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        else:
            offs_bias = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        bias = tl.load(bias_ptr + batch_id * stride_biasb + offs_bias)
        accumulator = accumulator.to(bias_ptr.type.element_ty) + bias[None, :]

    c = accumulator.to(c_ptr.type.element_ty)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = (
        c_ptr
        + stride_cb * batch_id
        + stride_cm * offs_cm[:, None]
        + stride_cn * offs_cn[None, :]
    )
    if EVEN_MN:
        tl.store(c_ptrs, c)
    else:
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(c_ptrs, c, mask=c_mask)


# Absorbed-MLA decode runs this kernel at M = 6 * concurrency, so the rows that
# matter sit just under the standard 64 bound and a single M_LEQ_64 entry has to
# serve both ends of it. The extra bound splits them; tables without an
# M_LEQ_48 entry fall through to M_LEQ_64 exactly as before.
#
# What the two ends disagree about is num_stages, and the split runs along
# whether the M tile is ragged rather than along M itself. Repeated-launch
# pricing of the B=16 decode shapes puts the whole body in the latency regime
# -- a second dot chain per iteration costs 0.008 us on (N=512, K=192) and
# -0.004 us on (N=256, K=512), the prequant prologue is 2.5% and 16% of the
# body, and 46-54% of the call is the graph node itself (1.47-1.54 us, flat in
# grid, workgroup size and kernarg count, and unchanged when consecutive nodes
# write disjoint buffers) -- so the depth of the load pipeline is the one thing
# left that moves it. Deeper only pays where ceil(M / BLOCK_SIZE_M) is exact:
# 2 -> 3 measures -1.9% at M=48 and -1.3% at M=64 on (N=256, K=512) but +0.3%
# at M=60, and -1.1% / -2.3% at M=60 / M=64 on (N=512, K=192). Past 3 it turns
# over hard on both (+3.4 / +3.7 / +15.1% at 4 / 5 / 6 on uv48, +16.3 / +31.0 /
# +47.2% on uk48), and the tile axis is not an alternative: BLOCK_SIZE_M 32 is
# +27..+45% even where it buys both more CTAs and fewer bytes than the shipped
# 16, and GROUP_SIZE_M above 1 is +1.9..+6.1% everywhere. All of it bit-exact.
_M_BOUNDS = (1, 4, 8, 16, 32, 48, 64, 128, 256, 512, 1024, 2048, 4096, 8192)


def _get_config(
    M: int,
    N: int,
    K: int,
    B: int | None = None,
):
    # B is a factor of the launch width -- the grid is
    # (B, cdiv(M, BLOCK_SIZE_M) * cdiv(N, BLOCK_SIZE_N)) -- so the tile that
    # fills the CU array depends on it. Pass it through so a B-specialized
    # table is picked when one exists; callers that omit B keep the
    # (N, K)-keyed behaviour.
    return get_gemm_config(
        "BATCHED_GEMM-A8W8-A_PER_TOKEN_GROUP_PREQUANT_W_PER_BATCHED_TENSOR_QUANT",
        M,
        N,
        K,
        bounds=_M_BOUNDS,
        B=B,
    )
