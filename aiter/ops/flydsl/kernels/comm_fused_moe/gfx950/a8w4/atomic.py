# SPDX-License-Identifier: Apache-2.0
"""Multi-launch atomic BF16 and MXFP8 communication family."""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, ptrtoint, range_constexpr
from flydsl.expr import math as fmath
from flydsl.expr.typing import ReductionOp, T

from .collectives import (
    buffer_tensor_from_addr,
    decode_scaled_fp8_f32,
    e8m0_scale,
    emit_block_rendezvous,
    load_bf16,
    load_buffer,
    load_e8m0_scale,
    load_fp8_words,
    pack_fp8_words,
    peer_base,
    store_bf16,
    store_buffer,
    store_fp8_words,
)
from .config import BLOCK, REDUCE_BLOCK, RENDEZVOUS_FLAG_SLOTS, AtomicConfig

VECTOR_WIDTH = 16
QUANT_BLOCK = 256


@flyc.jit
def _emit_quantize_chunk(config, local, shared, partial, token, column, zero_local):
    """Sum one VECTOR_WIDTH chunk with the shared partial and store it MXFP8."""
    shape = config.shape
    h = shape.model_dim
    groups_per_row = h // 32
    local_row = buffer_tensor_from_addr(
        fx.Int64(ptrtoint(local)) + fx.Int64(token) * fx.Int64(h * 2),
        fx.BFloat16,
        h * 2,
    )
    shared_row = buffer_tensor_from_addr(
        fx.Int64(ptrtoint(shared)) + fx.Int64(token) * fx.Int64(h * 2),
        fx.BFloat16,
        h * 2,
    )
    values = []
    for chunk in range_constexpr(VECTOR_WIDTH // 8):
        chunk_offset = column + fx.Int32(chunk * 8)
        loaded = load_bf16(local_row, chunk_offset, 8, 2).to(fx.Float32)
        shared_values = load_bf16(shared_row, chunk_offset, 8, 2).to(fx.Float32)
        # Round through BF16 so the sum matches an accumulator that was
        # seeded with the shared partial before the GEMM ran.
        loaded = (loaded + shared_values).to(fx.BFloat16).to(fx.Float32)
        values.extend(loaded[element] for element in range_constexpr(8))

    if const_expr(zero_local):
        # The GEMM scatters into `local` with atomics, so it needs a zeroed
        # target. Clearing the row here, once it has been read in full, keeps
        # that invariant without a host-side copy per call.
        store_bf16(
            local_row,
            column,
            [fx.Float32(0.0) for _ in range_constexpr(VECTOR_WIDTH)],
            VECTOR_WIDTH,
        )

    vector = fx.Vector.from_elements(values, fx.Float32)
    local_max = fx.Float32(1e-10).maximumf(fmath.absf(vector).reduce(ReductionOp.MAX))
    lane = fx.Int32(gpu.thread_id("x")) & fx.Int32(63)
    remote_bits = fx.rocdl.ds_bpermute(
        T.i32,
        (lane ^ fx.Int32(1)) * fx.Int32(4),
        local_max.bitcast(fx.Int32),
    )
    local_max = local_max.maximumf(fx.Int32(remote_bits).bitcast(fx.Float32))
    e8m0, quant_scale = e8m0_scale(local_max)
    packed = pack_fp8_words(vector, quant_scale, VECTOR_WIDTH // 4)

    payload_row = buffer_tensor_from_addr(
        fx.Int64(ptrtoint(partial)) + fx.Int64(token) * fx.Int64(h),
        fx.Int32,
        h,
    )
    store_fp8_words(
        payload_row,
        column,
        packed,
        VECTOR_WIDTH // 4,
    )
    if lane % fx.Int32(32 // VECTOR_WIDTH) == fx.Int32(0):
        scale_row = buffer_tensor_from_addr(
            fx.Int64(ptrtoint(partial))
            + fx.Int64(config.m * h)
            + fx.Int64(token) * fx.Int64(groups_per_row),
            fx.Int8,
            groups_per_row,
        )
        store_buffer(
            scale_row,
            column // fx.Int32(32),
            e8m0.to(fx.Int8),
            fx.Int8,
        )


@flyc.jit
def _emit_reduce_chunk(config, partial_base, output, token, group, column):
    """Sum one VECTOR_WIDTH chunk across every TP peer into the BF16 output."""
    shape = config.shape
    h = shape.model_dim
    groups_per_row = h // 32
    acc = fx.Vector.filled(VECTOR_WIDTH, 0.0, fx.Float32)
    for source_round in range_constexpr(shape.tp_size):
        # Stagger the peer order by token so the ranks do not all hit the same
        # peer's window in the same cycle.
        source = (token + fx.Int32(source_round)) % fx.Int32(shape.tp_size)
        source_base = peer_base(partial_base, source)
        source_row = buffer_tensor_from_addr(
            source_base + fx.Int64(token) * fx.Int64(h),
            fx.Int32,
            h,
        )
        words = load_fp8_words(
            source_row,
            column // fx.Int32(4),
            word_count=VECTOR_WIDTH // 4,
            load_width=4,
            cache_modifier=0,
        )
        scale_row = buffer_tensor_from_addr(
            source_base
            + fx.Int64(config.m * h)
            + fx.Int64(token) * fx.Int64(groups_per_row),
            fx.Int8,
            groups_per_row,
        )
        values = decode_scaled_fp8_f32(words, load_e8m0_scale(scale_row, group, 0))
        acc = acc + fx.Vector.from_elements(values, fx.Float32)

    output_row = buffer_tensor_from_addr(
        fx.Int64(ptrtoint(output)) + fx.Int64(token) * fx.Int64(h * 2),
        fx.BFloat16,
        h * 2,
    )
    store_bf16(output_row, column, acc, VECTOR_WIDTH)


@functools.cache
def compile_quantize(config: AtomicConfig, zero_local: bool):
    """Add the shared BF16 output, quantize to MXFP8, optionally clear local.

    One block owns one column tile of one row, which keeps the row base -- and
    with it the buffer descriptor every chunk addresses through -- uniform
    across the block. Walking the rows end to end instead packs the blocks
    fuller, but makes the base per-lane and measures slower for it.
    """
    m = config.m
    shape = config.shape
    column_tiles = (shape.model_dim + QUANT_BLOCK * VECTOR_WIDTH - 1) // (
        QUANT_BLOCK * VECTOR_WIDTH
    )

    @flyc.kernel(
        name=(
            f"gemm2_tp_atomic_pipeline_{shape.tag}_m{m}_quant_add_shared"
            f"{'_zero_local' if zero_local else ''}_v16_b256"
        ),
        known_block_size=[QUANT_BLOCK, 1, 1],
    )
    def kernel(local: fx.Pointer, shared: fx.Pointer, partial: fx.Pointer):
        block = fx.Int32(gpu.block_id("x"))
        token = block // fx.Int32(column_tiles)
        tile = block - token * fx.Int32(column_tiles)
        column = tile * fx.Int32(QUANT_BLOCK * VECTOR_WIDTH) + fx.Int32(
            gpu.thread_id("x")
        ) * fx.Int32(VECTOR_WIDTH)
        if column < fx.Int32(shape.model_dim):
            _emit_quantize_chunk(
                config, local, shared, partial, token, column, zero_local
            )

    @flyc.jit
    def launch(local, shared, partial, stream):
        kernel(local, shared, partial).launch(
            grid=(m * column_tiles, 1, 1),
            block=(QUANT_BLOCK, 1, 1),
            stream=stream,
        )

    return launch


@functools.cache
def compile_full_reduce(config: AtomicConfig):
    """Reduce every row from every TP peer straight into the BF16 output.

    One launch instead of reduce-scatter + barrier + all-gather. It reads
    ``tp-1`` times more peer bytes than the sharded pair, which only pays off
    while the launches dominate the byte traffic -- see ``use_full_reduce``.

    The kernel opens with the per-block rendezvous that replaces the epoch
    barrier launch. The quantize launch that produced the partials has already
    released them, so this rendezvous only has to exchange flags.
    """
    shape = config.shape
    groups_per_row = shape.model_dim // 32
    packs_per_group = 32 // VECTOR_WIDTH
    grid = _rendezvous_grid(config, REDUCE_BLOCK)
    flag_offset = config.block_flag_offset(0)

    @flyc.kernel(
        name=(
            f"gemm2_tp_atomic_pipeline_{shape.tag}_m{config.m}"
            f"_full_reduce_rendezvous_g{grid}_b{REDUCE_BLOCK}"
        ),
        known_block_size=[REDUCE_BLOCK, 1, 1],
    )
    def kernel(partial: fx.Pointer, partial_base: fx.Int64, output: fx.Pointer):
        emit_block_rendezvous(partial, partial_base, flag_offset, shape.tp_size)
        item = fx.Int32(gpu.block_id("x")) * fx.Int32(REDUCE_BLOCK) + fx.Int32(
            gpu.thread_id("x")
        )
        group_item = item // fx.Int32(packs_per_group)
        pack_in_group = item - group_item * fx.Int32(packs_per_group)
        token = group_item // fx.Int32(groups_per_row)
        group = group_item - token * fx.Int32(groups_per_row)
        column = group * fx.Int32(32) + pack_in_group * fx.Int32(VECTOR_WIDTH)
        _emit_reduce_chunk(config, partial_base, output, token, group, column)

    @flyc.jit
    def launch(partial, partial_base, output, stream):
        kernel(partial, partial_base, output).launch(
            grid=(grid, 1, 1),
            block=(REDUCE_BLOCK, 1, 1),
            stream=stream,
        )

    return launch


def _rendezvous_grid(config: AtomicConfig, block: int) -> int:
    """Blocks for a rendezvous kernel: one work item per thread, no tail.

    A pairwise rendezvous pairs block ``b`` with block ``b`` of the peers, so
    the grid has to cover the work exactly and stay inside the reserved flags.
    """
    groups_per_row = config.shape.model_dim // 32
    packs_per_group = 32 // VECTOR_WIDTH
    work_items = config.m * groups_per_row * packs_per_group
    if work_items % block:
        raise ValueError(
            f"{work_items} work items for m={config.m} do not fill whole "
            f"blocks of {block}"
        )
    grid = work_items // block
    if grid > RENDEZVOUS_FLAG_SLOTS:
        raise ValueError(
            f"m={config.m} needs {grid} rendezvous flags, only "
            f"{RENDEZVOUS_FLAG_SLOTS} are reserved"
        )
    return grid


@functools.cache
def compile_reduce_scatter(config: AtomicConfig):
    """Reduce this rank's shard of every peer's partial, re-quantized.

    The kernel opens with the rendezvous that used to be a separate epoch
    barrier launch: entering it already means this rank's quantize kernel
    retired, so the flag exchange is all the peers need.
    """
    shape = config.shape
    h = shape.model_dim
    groups_per_row = h // 32
    packs_per_group = 32 // VECTOR_WIDTH
    work_items = config.shard_rows * groups_per_row * packs_per_group
    flag_offset = config.block_flag_offset(0)

    @flyc.kernel(
        name=(
            f"gemm2_tp_atomic_pipeline_{shape.tag}_m{config.m}"
            f"_rs_rendezvous_g{config.reduce_scatter_grid}"
        ),
        known_block_size=[BLOCK, 1, 1],
    )
    def kernel(
        partial: fx.Pointer,
        partial_base: fx.Int64,
        output: fx.Pointer,
        payload: fx.Pointer,
        scales: fx.Pointer,
        rank: fx.Int32,
    ):
        emit_block_rendezvous(partial, partial_base, flag_offset, shape.tp_size)
        start = fx.Int32(gpu.block_id("x")) * fx.Int32(BLOCK) + fx.Int32(
            gpu.thread_id("x")
        )
        for item in range(
            start,
            fx.Int32(work_items),
            fx.Int32(config.reduce_scatter_grid * BLOCK),
        ):
            group_item = item // fx.Int32(packs_per_group)
            pack_in_group = item - group_item * fx.Int32(packs_per_group)
            local_token = group_item // fx.Int32(groups_per_row)
            group = group_item - local_token * fx.Int32(groups_per_row)
            column = group * fx.Int32(32) + pack_in_group * fx.Int32(VECTOR_WIDTH)
            global_token = rank * fx.Int32(config.shard_rows) + local_token
            lane = fx.Int32(gpu.thread_id("x")) & fx.Int32(63)
            acc = fx.Vector.filled(VECTOR_WIDTH, 0.0, fx.Float32)
            for source_round in range_constexpr(shape.tp_size):
                source = (rank + local_token + fx.Int32(source_round)) % fx.Int32(
                    shape.tp_size
                )
                source_base = peer_base(partial_base, source)
                source_row = buffer_tensor_from_addr(
                    source_base + fx.Int64(global_token) * fx.Int64(h),
                    fx.Int32,
                    h,
                )
                words = load_fp8_words(
                    source_row,
                    column // fx.Int32(4),
                    word_count=VECTOR_WIDTH // 4,
                    load_width=4,
                    cache_modifier=2,
                )
                scale_row = buffer_tensor_from_addr(
                    source_base
                    + fx.Int64(config.m * h)
                    + fx.Int64(global_token) * fx.Int64(groups_per_row),
                    fx.Int8,
                    groups_per_row,
                )
                values = decode_scaled_fp8_f32(
                    words, load_e8m0_scale(scale_row, group, 2)
                )
                acc = acc + fx.Vector.from_elements(values, fx.Float32)

            local_max = fx.Float32(1e-10).maximumf(
                fmath.absf(acc).reduce(ReductionOp.MAX)
            )
            max_bits = local_max.bitcast(fx.Int32)
            for xor_lane in range_constexpr(1, 2):
                remote_bits = fx.rocdl.ds_bpermute(
                    T.i32,
                    (lane ^ fx.Int32(xor_lane)) * fx.Int32(4),
                    max_bits,
                )
                local_max = local_max.maximumf(
                    fx.Int32(remote_bits).bitcast(fx.Float32)
                )
                max_bits = local_max.bitcast(fx.Int32)
            e8m0, quant_scale = e8m0_scale(local_max)
            packed = pack_fp8_words(acc, quant_scale, VECTOR_WIDTH // 4)

            payload_row = buffer_tensor_from_addr(
                fx.Int64(ptrtoint(payload)) + fx.Int64(local_token) * fx.Int64(h),
                fx.Int32,
                h,
            )
            store_fp8_words(
                payload_row,
                column,
                packed,
                4,
            )
            if pack_in_group == fx.Int32(0):
                scale_row = buffer_tensor_from_addr(
                    fx.Int64(ptrtoint(scales))
                    + fx.Int64(local_token) * fx.Int64(groups_per_row),
                    fx.Int8,
                    groups_per_row,
                )
                store_buffer(
                    scale_row,
                    group,
                    e8m0.to(fx.Int8),
                    fx.Int8,
                )

            decoded = decode_scaled_fp8_f32(
                packed,
                (fx.Uint32(e8m0) << fx.Uint32(23)).bitcast(fx.Float32),
            )
            output_row = buffer_tensor_from_addr(
                fx.Int64(ptrtoint(output)) + fx.Int64(local_token) * fx.Int64(h * 2),
                fx.BFloat16,
                h * 2,
            )
            store_bf16(output_row, column, decoded, VECTOR_WIDTH)

    @flyc.jit
    def launch(partial, partial_base, output, payload, scales, rank, stream):
        kernel(partial, partial_base, output, payload, scales, rank).launch(
            grid=(config.reduce_scatter_grid, 1, 1),
            block=(BLOCK, 1, 1),
            stream=stream,
        )

    return launch


@functools.cache
def compile_all_gather(config: AtomicConfig):
    """Broadcast every peer's reduced shard into the BF16 output.

    Like the reduce-scatter, it opens with the rendezvous that replaces the
    epoch barrier launch: a peer only reaches this kernel once its own
    reduce-scatter retired.
    """
    shape = config.shape
    h = shape.model_dim
    groups_per_row = h // 32
    packs_per_group = 32 // VECTOR_WIDTH
    work_items = config.shard_rows * groups_per_row * packs_per_group
    source_count = shape.tp_size - 1
    workers_per_source = config.all_gather_grid // source_count
    flag_offset = config.block_flag_offset(1)

    @flyc.kernel(
        name=(
            f"gemm2_tp_atomic_pipeline_{shape.tag}_m{config.m}"
            f"_ag_rendezvous_g{config.all_gather_grid}"
        ),
        known_block_size=[BLOCK, 1, 1],
    )
    def kernel(
        partial: fx.Pointer,
        partial_base: fx.Int64,
        payload_base: fx.Int64,
        scale_base: fx.Int64,
        output: fx.Pointer,
        rank: fx.Int32,
    ):
        emit_block_rendezvous(partial, partial_base, flag_offset, shape.tp_size)
        worker = fx.Int32(gpu.block_id("x"))
        source_slot = worker % fx.Int32(source_count)
        source_block = worker // fx.Int32(source_count)
        source = (rank + source_slot + fx.Int32(1)) % fx.Int32(shape.tp_size)
        payload = buffer_tensor_from_addr(
            peer_base(payload_base, source),
            fx.Int32,
            config.shard_rows * h,
        )
        scales = buffer_tensor_from_addr(
            peer_base(scale_base, source),
            fx.Int8,
            config.shard_rows * groups_per_row,
        )
        output_resource = buffer_tensor_from_addr(
            fx.Int64(ptrtoint(output)),
            fx.BFloat16,
            shape.tp_size * config.shard_rows * h * 2,
        )
        start = source_block * fx.Int32(BLOCK) + fx.Int32(gpu.thread_id("x"))
        for item in range(
            start,
            fx.Int32(work_items),
            fx.Int32(workers_per_source * BLOCK),
        ):
            group_item = item // fx.Int32(packs_per_group)
            pack_in_group = item - group_item * fx.Int32(packs_per_group)
            words = load_fp8_words(
                payload,
                item * fx.Int32(VECTOR_WIDTH // 4),
                word_count=VECTOR_WIDTH // 4,
                load_width=4,
                cache_modifier=1,
            )
            loaded = fx.Uint32(
                fx.Uint8(load_buffer(scales, group_item, fx.Int8, cache_modifier=1))
            )
            scale = (loaded << fx.Uint32(23)).bitcast(fx.Float32)
            values = decode_scaled_fp8_f32(words, scale)
            shard_row = group_item // fx.Int32(groups_per_row)
            group = group_item - shard_row * fx.Int32(groups_per_row)
            output_column = (
                source * fx.Int32(config.shard_rows * h)
                + shard_row * fx.Int32(h)
                + group * fx.Int32(32)
                + pack_in_group * fx.Int32(VECTOR_WIDTH)
            )
            store_bf16(output_resource, output_column, values, VECTOR_WIDTH)

    @flyc.jit
    def launch(partial, partial_base, payload_base, scale_base, output, rank, stream):
        kernel(partial, partial_base, payload_base, scale_base, output, rank).launch(
            grid=(config.all_gather_grid, 1, 1),
            block=(BLOCK, 1, 1),
            stream=stream,
        )

    return launch
