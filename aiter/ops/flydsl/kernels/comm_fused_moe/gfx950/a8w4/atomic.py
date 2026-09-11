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
    emit_rendezvous_acquire,
    emit_rendezvous_collect,
    emit_rendezvous_epoch,
    emit_rendezvous_mark,
    emit_rendezvous_peer_signal,
    emit_rendezvous_signal,
    emit_rendezvous_wide_epoch,
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
from .config import (
    BLOCK,
    REDUCE_BLOCK,
    REDUCE_BLOCKS,
    RENDEZVOUS_FLAG_SLOTS,
    AtomicConfig,
)

VECTOR_WIDTH = 16
QUANT_BLOCK = 256

# Column chunks one row slot of a shard-rendezvous block may own, narrowest
# first. One chunk per thread makes the block TP slots wide, so a slot is a
# whole wave at the narrowest choice: the reduce phase, which only the owning
# shard runs, then costs no divergence, and a slot's payload footprint is a
# whole number of cache lines. Splitting below a wave measured 2 us worse at
# m=64 even though it doubles the grid.
SHARD_CHUNKS_CHOICES = (64, 128)

# Blocks a shard rendezvous spreads over. Widening it past this shortens the
# contiguous run each block pulls out of a peer's window, and a shard
# rendezvous pays more for the shorter burst than it gains from the extra
# blocks -- measured still true once the flag traffic was taken out of it.
SHARD_RENDEZVOUS_BLOCKS = 96

# Cache policy for the stores a TP peer reads. The value is the gfx950 CPol
# field: 1 is sc0, 2 is nt, 16 is sc1, so 17 is the `sc0 sc1` write-through
# that puts a store at the system coherence point without an L2 writeback.
PUBLISH_CACHE_MODIFIER = 17


def _emit_local_row(pointer, token, h):
    return buffer_tensor_from_addr(
        fx.Int64(ptrtoint(pointer)) + fx.Int64(token) * fx.Int64(h * 2),
        fx.BFloat16,
        h * 2,
    )


def _emit_clear_local(config, local, token, column):
    """Reseed the GEMM's atomic-scatter accumulator, once it has been read.

    The GEMM adds into `local`, so it needs a zeroed target; clearing it from
    the kernel that consumes it saves a host-side copy per call.
    """
    store_bf16(
        _emit_local_row(local, token, config.shape.model_dim),
        column,
        [fx.Float32(0.0) for _ in range_constexpr(VECTOR_WIDTH)],
        VECTOR_WIDTH,
    )


def _emit_quantize_vector(vector):
    """MXFP8 one VECTOR_WIDTH chunk, sharing one exponent across its group.

    A chunk is half an MXFP8 group, and the two halves of a group always land
    on adjacent lanes, so one ``ds_bpermute`` against ``lane ^ 1`` is the whole
    group reduction.
    """
    local_max = fx.Float32(1e-10).maximumf(fmath.absf(vector).reduce(ReductionOp.MAX))
    lane = fx.Int32(gpu.thread_id("x")) & fx.Int32(63)
    remote_bits = fx.rocdl.ds_bpermute(
        T.i32,
        (lane ^ fx.Int32(1)) * fx.Int32(4),
        local_max.bitcast(fx.Int32),
    )
    local_max = local_max.maximumf(fx.Int32(remote_bits).bitcast(fx.Float32))
    e8m0, quant_scale = e8m0_scale(local_max)
    return e8m0, pack_fp8_words(vector, quant_scale, VECTOR_WIDTH // 4)


def _emit_quantize_values(config, local, shared, token, column, zero_local):
    """Sum one VECTOR_WIDTH chunk with the shared partial, MXFP8 it."""
    h = config.shape.model_dim
    local_row = _emit_local_row(local, token, h)
    shared_row = _emit_local_row(shared, token, h)
    values = []
    for chunk in range_constexpr(VECTOR_WIDTH // 8):
        chunk_offset = column + fx.Int32(chunk * 8)
        loaded = load_bf16(local_row, chunk_offset, 8, 2).to(fx.Float32)
        shared_values = load_bf16(shared_row, chunk_offset, 8, 2).to(fx.Float32)
        # Round through BF16 so the sum matches an accumulator that was
        # seeded with the shared partial before the GEMM ran.
        loaded = (loaded + shared_values).to(fx.BFloat16).to(fx.Float32)
        values.extend(loaded[element] for element in range_constexpr(8))

    if zero_local:
        _emit_clear_local(config, local, token, column)

    return _emit_quantize_vector(fx.Vector.from_elements(values, fx.Float32))


def _emit_publish_payload(config, partial, token, column, packed, cache_modifier):
    """Store one quantized chunk into this rank's symmetric window."""
    h = config.shape.model_dim
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
        cache_modifier=cache_modifier,
    )


def _emit_publish_scale(config, partial, token, column, e8m0, cache_modifier):
    """Store one MXFP8 group's shared exponent, once per group."""
    h = config.shape.model_dim
    groups_per_row = h // 32
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
        cache_modifier=cache_modifier,
    )


def _emit_decoded_chunk(e8m0, packed):
    """Rebuild the values a peer will read back out of a quantized chunk."""
    scale = (fx.Uint32(e8m0) << fx.Uint32(23)).bitcast(fx.Float32)
    return fx.Vector.from_elements(
        decode_scaled_fp8_f32(packed, scale), fx.Float32
    )


@flyc.jit
def _emit_quantize_chunk(config, local, shared, partial, token, column, zero_local):
    """Sum one VECTOR_WIDTH chunk with the shared partial and store it MXFP8."""
    e8m0, packed = _emit_quantize_values(
        config, local, shared, token, column, zero_local
    )
    _emit_publish_payload(config, partial, token, column, packed, 0)
    lane = fx.Int32(gpu.thread_id("x")) & fx.Int32(63)
    if lane % fx.Int32(32 // VECTOR_WIDTH) == fx.Int32(0):
        _emit_publish_scale(config, partial, token, column, e8m0, 0)


def _emit_peer_rotation(shape, rank, row, offset):
    """Name one peer, rotated by row, by rank, and by ``offset``.

    Rotating by row keeps the rows spread over the peers and rotating by rank
    on top keeps the ranks from all starting on the same peer. ``offset`` is
    the round, plus whatever else the caller wants to rotate by; it is folded
    modulo the peer count, so it need not already be in range.
    """
    peers = shape.tp_size - 1
    slot = (row + fx.Int32(offset)) % fx.Int32(peers)
    source = rank + fx.Int32(1) + slot
    return (source >= fx.Int32(shape.tp_size)).select(
        source - fx.Int32(shape.tp_size), source
    )


def _emit_source_chunk(config, source_base, row, group, column, cache_modifier):
    """Decode one VECTOR_WIDTH chunk of one row out of one rank's window."""
    h = config.shape.model_dim
    groups_per_row = h // 32
    source_row = buffer_tensor_from_addr(
        source_base + fx.Int64(row) * fx.Int64(h),
        fx.Int32,
        h,
    )
    words = load_fp8_words(
        source_row,
        column // fx.Int32(4),
        word_count=VECTOR_WIDTH // 4,
        load_width=4,
        cache_modifier=cache_modifier,
    )
    scale_row = buffer_tensor_from_addr(
        source_base + fx.Int64(config.m * h) + fx.Int64(row) * fx.Int64(groups_per_row),
        fx.Int8,
        groups_per_row,
    )
    values = decode_scaled_fp8_f32(
        words, load_e8m0_scale(scale_row, group, cache_modifier)
    )
    return fx.Vector.from_elements(values, fx.Float32)


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
            f"{'_zero_local' if zero_local else ''}"
            f"_v{VECTOR_WIDTH}_b{QUANT_BLOCK}"
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
def compile_fused_reduce(config: AtomicConfig, zero_local: bool):
    """Quantize, publish, and reduce every TP peer in a single launch.

    The split pair spends a whole dispatch -- measured at 1.8-2.4 us on this
    pipeline, wherever in the chain it sits -- on handing the partial from one
    kernel to the next, and the second kernel then reads back the quarter of
    the payload it just wrote. Both go away once the two share a block: the
    per-block rendezvous already covers exactly the work items block ``b``
    owns, so block ``b`` can publish as soon as its own chunk is stored and
    keep its own contribution in registers.

    What made this expensive before is the release. A system-scope release
    fence lowers to an L2 writeback, and one per block costs 10 us at m=64 --
    the whole reason a launch boundary looked cheaper. Writing the payload
    through the L2 instead (``sc0 sc1``) removes the need for it: the stores
    land at the coherence point on their own, so retiring them is the whole
    release. The clear of the GEMM accumulator is deferred past the publish,
    where it fills the peers' remaining skew the way the local decode used to.
    """
    shape = config.shape
    h = shape.model_dim
    groups_per_row = h // 32
    packs_per_group = 32 // VECTOR_WIDTH
    block, grid = _rendezvous_geometry(config.m * groups_per_row * packs_per_group)
    flag_offset = config.block_flag_offset(0)

    @flyc.kernel(
        name=(
            f"gemm2_tp_atomic_pipeline_{shape.tag}_m{config.m}"
            f"_fused_reduce_rendezvous_g{grid}_b{block}"
            f"{'_zero_local' if zero_local else ''}"
        ),
        known_block_size=[block, 1, 1],
    )
    def kernel(
        local: fx.Pointer,
        shared: fx.Pointer,
        partial: fx.Pointer,
        partial_base: fx.Int64,
        output: fx.Pointer,
        rank: fx.Int32,
    ):
        item = fx.Int32(gpu.block_id("x")) * fx.Int32(block) + fx.Int32(
            gpu.thread_id("x")
        )
        group_item = item // fx.Int32(packs_per_group)
        pack_in_group = item - group_item * fx.Int32(packs_per_group)
        token = group_item // fx.Int32(groups_per_row)
        group = group_item - token * fx.Int32(groups_per_row)
        column = group * fx.Int32(32) + pack_in_group * fx.Int32(VECTOR_WIDTH)

        # Nothing orders this read, so issuing it up front hides the round
        # trip to the flag behind the quantize below.
        epoch = emit_rendezvous_epoch(partial, flag_offset, shape.tp_size)
        e8m0, packed = _emit_quantize_values(
            config, local, shared, token, column, False
        )
        _emit_publish_payload(
            config, partial, token, column, packed, PUBLISH_CACHE_MODIFIER
        )
        if pack_in_group == fx.Int32(0):
            _emit_publish_scale(
                config, partial, token, column, e8m0, PUBLISH_CACHE_MODIFIER
            )
        acc = _emit_decoded_chunk(e8m0, packed)

        # Write-through stores are at the coherence point once they retire, so
        # draining them is the whole release the peers need.
        fx.rocdl.s_waitcnt(0)
        gpu.barrier()
        emit_rendezvous_mark(partial, flag_offset, epoch)

        # Reseeding the accumulator is the one thing left that no peer gates,
        # so it goes in the window where the peers are still catching up.
        if const_expr(zero_local):
            _emit_clear_local(config, local, token, column)
        emit_rendezvous_acquire(partial_base, flag_offset, epoch, shape.tp_size)

        for source_round in range_constexpr(shape.tp_size - 1):
            source = _emit_peer_rotation(shape, rank, token, source_round)
            acc = acc + _emit_source_chunk(
                config, peer_base(partial_base, source), token, group, column, 0
            )

        output_row = buffer_tensor_from_addr(
            fx.Int64(ptrtoint(output)) + fx.Int64(token) * fx.Int64(h * 2),
            fx.BFloat16,
            h * 2,
        )
        store_bf16(output_row, column, acc, VECTOR_WIDTH)

    @flyc.jit
    def launch(local, shared, partial, partial_base, output, rank, stream):
        kernel(local, shared, partial, partial_base, output, rank).launch(
            grid=(grid, 1, 1),
            block=(block, 1, 1),
            stream=stream,
        )

    return launch


def _rendezvous_geometry(work_items: int) -> tuple[int, int]:
    """Pick the (block, grid) shape of a rendezvous collective.

    A pairwise rendezvous pairs block ``b`` with block ``b`` of the peers, so
    the grid has to cover the work exactly and stay inside the reserved flags.
    Covering it exactly is also what lets a caller split the wait around its
    rank-local share, which needs the accumulator to live in registers across
    the wait. Peer flag traffic scales with the grid while the payload reads
    do not, so the widest block that still leaves ``REDUCE_BLOCKS`` blocks
    wins, one item per thread.
    """
    for block in range(REDUCE_BLOCK, 0, -64):
        if work_items % block == 0 and work_items // block >= REDUCE_BLOCKS:
            break
    else:
        raise ValueError(
            f"no block of at most {REDUCE_BLOCK} threads splits {work_items} "
            f"work items into {REDUCE_BLOCKS} whole blocks"
        )
    grid = work_items // block
    if grid > RENDEZVOUS_FLAG_SLOTS:
        raise ValueError(
            f"{work_items} work items need {grid} rendezvous flags, only "
            f"{RENDEZVOUS_FLAG_SLOTS} are reserved"
        )
    return block, grid


def _shard_rendezvous_geometry(config: AtomicConfig) -> tuple[int, int, int, int]:
    """Pick the (block, grid, column groups, chunks) shape of a shard rendezvous.

    A block owns one column group of one row per shard, so the rows it reduces
    are a quarter of the rows it quantizes and the work stays balanced across
    the grid however the shards are split. That leaves the column split as the
    only free dimension: the narrowest one keeps a shard slot down to a single
    wave, which is what the reduce phase wants, but it also shortens the run
    each block pulls out of a peer's window, so the split only stays narrow
    while the grid does. Measured (fused us): m=64 grid 96 89.2-89.6 against
    grid 48 90.3, m=128 grid 96 94.0-94.4 against grid 192 95.3-97.1.

    Widening a group by giving each lane several chunks instead, so that m=128
    could hold its grid at 96 with a slot of one wave, measured 94.7-95.2
    against 93.3-93.8 -- with m=64's geometry untouched at 88.66-88.71 in the
    same reads. Halving the threads doubles the reduce phase's serial chain of
    peer read, sum, requantize and publish, and that costs more than aligning
    the slot to a wave saves. So a chunk stays one per thread.
    """
    chunks_per_row = config.shape.model_dim // VECTOR_WIDTH
    splits = [
        (config.shard_rows * (chunks_per_row // shard_chunks), shard_chunks)
        for shard_chunks in SHARD_CHUNKS_CHOICES
        if chunks_per_row % shard_chunks == 0
    ]
    splits = [split for split in splits if split[0] <= RENDEZVOUS_FLAG_SLOTS]
    if not splits:
        raise ValueError(
            f"m={config.m} model_dim={config.shape.model_dim} has no column "
            f"split that fits {RENDEZVOUS_FLAG_SLOTS} rendezvous flags"
        )
    fitting = [split for split in splits if split[0] <= SHARD_RENDEZVOUS_BLOCKS]
    grid, shard_chunks = max(fitting) if fitting else min(splits)
    return (
        config.shape.tp_size * shard_chunks,
        grid,
        chunks_per_row // shard_chunks,
        shard_chunks,
    )


def _emit_reduced_chunk(config, payload_base, scale_base, row, group, column):
    """Decode one VECTOR_WIDTH chunk of one row of a rank's reduced shard."""
    h = config.shape.model_dim
    groups_per_row = h // 32
    payload_row = buffer_tensor_from_addr(
        payload_base + fx.Int64(row) * fx.Int64(h),
        fx.Int32,
        h,
    )
    words = load_fp8_words(
        payload_row,
        column // fx.Int32(4),
        word_count=VECTOR_WIDTH // 4,
        load_width=4,
        cache_modifier=0,
    )
    scale_row = buffer_tensor_from_addr(
        scale_base + fx.Int64(row) * fx.Int64(groups_per_row),
        fx.Int8,
        groups_per_row,
    )
    values = decode_scaled_fp8_f32(words, load_e8m0_scale(scale_row, group, 0))
    return fx.Vector.from_elements(values, fx.Float32)


@flyc.jit
def _emit_publish_reduced(config, payload, scales, row, group, column, packed, e8m0):
    """Store one re-quantized chunk of this rank's shard for its peers."""
    h = config.shape.model_dim
    groups_per_row = h // 32
    payload_row = buffer_tensor_from_addr(
        fx.Int64(ptrtoint(payload)) + fx.Int64(row) * fx.Int64(h),
        fx.Int32,
        h,
    )
    store_fp8_words(
        payload_row,
        column,
        packed,
        VECTOR_WIDTH // 4,
        cache_modifier=PUBLISH_CACHE_MODIFIER,
    )
    if fx.Int32(gpu.thread_id("x")) % fx.Int32(32 // VECTOR_WIDTH) == fx.Int32(0):
        scale_row = buffer_tensor_from_addr(
            fx.Int64(ptrtoint(scales)) + fx.Int64(row) * fx.Int64(groups_per_row),
            fx.Int8,
            groups_per_row,
        )
        store_buffer(
            scale_row,
            group,
            e8m0.to(fx.Int8),
            fx.Int8,
            cache_modifier=PUBLISH_CACHE_MODIFIER,
        )


@functools.cache
def compile_fused_rsag(config: AtomicConfig, zero_local: bool):
    """Quantize, reduce-scatter, and all-gather in a single launch.

    The full reduce buys its single dispatch by having every rank read every
    peer's whole payload -- three times the bytes a reduce-scatter moves. The
    split pair keeps those bytes but pays three dispatches, measured at 1.8-2.4
    us each, and hands the payload between them through memory twice.

    A block can have both if it owns a column group of one row per shard: the
    rows it reduces are exactly the rows of its own shard, and every peer's
    block of the same index published exactly those rows, so the per-block
    rendezvous that already covers the quantize covers the reduce as well. The
    same holds one phase later -- peer ``p``'s block reduced the row this
    block gathers from ``p`` -- so a second flag finishes the chain. Peer
    bytes stay at the reduce-scatter optimum, the payload never leaves
    registers between phases, and the whole tail is one dispatch.
    """
    shape = config.shape
    h = shape.model_dim
    tp = shape.tp_size
    shard_rows = config.shard_rows
    block, grid, column_groups, shard_chunks = _shard_rendezvous_geometry(config)
    publish_flag = config.block_flag_offset(0)
    reduced_flag = config.block_flag_offset(1)
    # A slot that is exactly one wave owns a whole phase of the chain on its
    # own, which lets the release drop to wave scope.
    wave_slot = shard_chunks == 64

    @flyc.kernel(
        name=(
            f"gemm2_tp_atomic_pipeline_{shape.tag}_m{config.m}"
            f"_fused_rsag_rendezvous_g{grid}_b{block}"
            f"{'_zero_local' if zero_local else ''}"
        ),
        known_block_size=[block, 1, 1],
    )
    def kernel(
        local: fx.Pointer,
        shared: fx.Pointer,
        partial: fx.Pointer,
        partial_base: fx.Int64,
        reduced_payload: fx.Pointer,
        reduced_payload_base: fx.Int64,
        reduced_scale: fx.Pointer,
        reduced_scale_base: fx.Int64,
        output: fx.Pointer,
        rank: fx.Int32,
    ):
        thread = fx.Int32(gpu.thread_id("x"))
        slot = thread // fx.Int32(shard_chunks)
        lane = thread - slot * fx.Int32(shard_chunks)
        worker = fx.Int32(gpu.block_id("x"))
        shard_row = worker // fx.Int32(column_groups)
        column_group = worker - shard_row * fx.Int32(column_groups)
        column = (
            column_group * fx.Int32(shard_chunks) + lane
        ) * fx.Int32(VECTOR_WIDTH)
        group = column // fx.Int32(32)
        token = slot * fx.Int32(shard_rows) + shard_row

        # Neither flag read is ordered, so issuing both up front hides their
        # round trip behind the quantize below.
        publish_epoch = emit_rendezvous_wide_epoch(partial, publish_flag, rank)
        reduced_epoch = emit_rendezvous_wide_epoch(partial, reduced_flag, rank)

        e8m0, packed = _emit_quantize_values(
            config, local, shared, token, column, False
        )
        _emit_publish_payload(
            config, partial, token, column, packed, PUBLISH_CACHE_MODIFIER
        )
        if lane % fx.Int32(32 // VECTOR_WIDTH) == fx.Int32(0):
            _emit_publish_scale(
                config, partial, token, column, e8m0, PUBLISH_CACHE_MODIFIER
            )

        # Write-through stores are at the coherence point once they retire, so
        # draining them is the whole release the peers need. A slot publishes
        # exactly the row the peer of the same index reads, so once a slot is a
        # whole wave it can drain by itself and release that one peer, with
        # nothing here waiting on the rest of the block. A wider slot still
        # lines up first: giving each of its waves its own flag word so that it
        # would not have to measured no better.
        fx.rocdl.s_waitcnt(0)
        if const_expr(not wave_slot):
            gpu.barrier()
        if lane == fx.Int32(0):
            emit_rendezvous_peer_signal(
                partial_base, publish_flag, publish_epoch, rank, slot
            )

        # Reseeding the accumulator is the one thing left that no peer gates,
        # so it goes in the window where the peers are still catching up.
        if const_expr(zero_local):
            _emit_clear_local(config, local, token, column)

        # The wait stays block wide even though only the reduce slot needs it:
        # the slots do identical work and reach the flag together, so a barrier
        # between them is nearly free, and dropping it to wave scope measured a
        # wash at m=64 (88.68-88.82 against 88.52-88.74). The release above is
        # the asymmetric half of the pair, which is why only it went per wave.
        emit_rendezvous_collect(partial, publish_flag, publish_epoch, tp)

        if slot == rank:
            acc = _emit_decoded_chunk(e8m0, packed)
            for source_round in range_constexpr(tp - 1):
                source = _emit_peer_rotation(shape, rank, shard_row, source_round)
                acc = acc + _emit_source_chunk(
                    config, peer_base(partial_base, source), token, group, column, 0
                )
            reduced_e8m0, reduced_packed = _emit_quantize_vector(acc)
            _emit_publish_reduced(
                config,
                reduced_payload,
                reduced_scale,
                shard_row,
                group,
                column,
                reduced_packed,
                reduced_e8m0,
            )
            store_bf16(
                _emit_local_row(output, token, h),
                column,
                _emit_decoded_chunk(reduced_e8m0, reduced_packed),
                VECTOR_WIDTH,
            )
            # The reduce slot holds the only stores a peer reads out of this
            # phase, so when it is a whole wave it drains and releases all of
            # them, again without the rest of the block having to arrive.
            if const_expr(wave_slot):
                fx.rocdl.s_waitcnt(0)
                if lane < fx.Int32(tp):
                    emit_rendezvous_peer_signal(
                        partial_base, reduced_flag, reduced_epoch, rank, lane
                    )

        if const_expr(not wave_slot):
            # Handing these four to the slot leaders instead measured worse:
            # nobody is released early once the block has to line up anyway,
            # and one wave issues the four stores back to back.
            fx.rocdl.s_waitcnt(0)
            gpu.barrier()
            emit_rendezvous_signal(partial_base, reduced_flag, reduced_epoch, rank, tp)
        emit_rendezvous_collect(partial, reduced_flag, reduced_epoch, tp)

        if slot != rank:
            gathered = _emit_reduced_chunk(
                config,
                peer_base(reduced_payload_base, slot),
                peer_base(reduced_scale_base, slot),
                shard_row,
                group,
                column,
            )
            store_bf16(_emit_local_row(output, token, h), column, gathered, VECTOR_WIDTH)

    @flyc.jit
    def launch(
        local,
        shared,
        partial,
        partial_base,
        reduced_payload,
        reduced_payload_base,
        reduced_scale,
        reduced_scale_base,
        output,
        rank,
        stream,
    ):
        kernel(
            local,
            shared,
            partial,
            partial_base,
            reduced_payload,
            reduced_payload_base,
            reduced_scale,
            reduced_scale_base,
            output,
            rank,
        ).launch(grid=(grid, 1, 1), block=(block, 1, 1), stream=stream)

    return launch


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
                acc = acc + _emit_source_chunk(
                    config,
                    peer_base(partial_base, source),
                    global_token,
                    group,
                    column,
                    2,
                )

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
