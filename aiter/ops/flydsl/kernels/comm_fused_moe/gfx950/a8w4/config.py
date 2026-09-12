# SPDX-License-Identifier: Apache-2.0
"""Configuration and workspace layouts for gfx950 A8W4 comm-fused MoE."""

from dataclasses import dataclass

BLOCK = 256
SLOTS = 2
PRODUCER_COUNTER_STRIDE = 64

# Full-reduce geometry: the widest block up to REDUCE_BLOCK that still splits
# the work into whole blocks, at least REDUCE_BLOCKS of them, one item per
# thread. Wider blocks mean fewer rendezvous participants and, because a row
# of the payload is 384 items, blocks that cover whole rows -- which is what
# makes a peer's window arrive as one contiguous burst. Measured at m=64
# (fused us, one item per thread throughout): 384 threads/64 blocks 93.0,
# 768/32 93.3, 1024/24 93.5, 256/96 93.4, 128/192 93.9, 512/48 94.3; halving
# the thread count to 2 items per thread costs 2 us on top.
REDUCE_BLOCK = 384
REDUCE_BLOCKS = 24

# One in-kernel rendezvous flag per block per collective phase, spread over
# separate cache lines so peers spinning on them do not share a line with a
# neighbour. Two phases cover the longest pipeline (reduce-scatter, then
# all-gather); the flags are epoch counters, so each phase needs its own.
RENDEZVOUS_FLAG_SLOTS = 256
RENDEZVOUS_FLAG_STRIDE = PRODUCER_COUNTER_STRIDE
RENDEZVOUS_PHASES = 2

# Smallest bucket the single-launch shard rendezvous serves. It halves a full
# reduce's peer bytes but adds a second in-kernel rendezvous and the round
# trip behind it, and below this the bytes it saves stop paying for them.
FUSED_RSAG_MIN_M = 32

SUPPORTED_TP_SIZES = (2, 4, 8)


@dataclass(frozen=True, slots=True)
class Shape:
    """Model dimensions specialized into one generated kernel."""

    model_dim: int
    inter_dim: int
    experts: int
    topk: int
    tp_size: int = 8

    def __post_init__(self):
        for name, value in (
            ("model_dim", self.model_dim),
            ("inter_dim", self.inter_dim),
            ("experts", self.experts),
            ("topk", self.topk),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.model_dim % 32:
            raise ValueError(
                "model_dim must be divisible by the 32-column MXFP8 scale "
                f"group, got {self.model_dim}"
            )
        if self.inter_dim % 128:
            raise ValueError(
                "inter_dim must be divisible by the 128-element shuffled "
                f"A8W4 K block, got {self.inter_dim}"
            )
        if self.tp_size not in SUPPORTED_TP_SIZES:
            raise ValueError(
                "gfx950 A8W4 GEMM2/TP currently requires TP in "
                f"{SUPPORTED_TP_SIZES}, "
                f"got {self.tp_size}"
            )

    @property
    def tag(self) -> str:
        return (
            f"h{self.model_dim}_i{self.inter_dim}_e{self.experts}"
            f"_k{self.topk}_tp{self.tp_size}"
        )


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _mxfp8_scale_bytes(payload_bytes: int, vector_width: int) -> int:
    if vector_width == 8:
        return payload_bytes // 32
    return payload_bytes * 4 // vector_width


@dataclass(frozen=True)
class MegakernelConfig:
    """Configuration for the single-launch GEMM2 + TP collective kernel."""

    shape: Shape
    m: int
    tile_m: int = 16
    tile_n: int = 256
    tile_k: int = 128
    sort_block_m: int = 32
    compute_groups: int = 96
    block_threads: int = BLOCK
    vector_width: int = 16
    waves_per_eu: int = 0
    b_cache_modifier: int = 0
    local_load_cache_modifier: int = 1
    remote_load_cache_modifier: int = 1
    gather_load_cache_modifier: int = -1
    remote_store_cache_modifier: int = 0
    n_tile_cohort: int = 0
    collective: str = "direct"
    service_groups: int = 1
    service_tile_group: int = 1
    producer_mode: str = "routes"
    flat_producer_grid: bool = False
    # Activation precision of the producer GEMM. The underlying port
    # (compile_gemm2_a4w4_port) accepts ("fp4","fp4") and ("fp8","fp4"); this
    # directory was written against fp8 and hard-coded it, which is why the
    # fused path looked a8w4-only. GLM-5.2 dispatches a4w4, so it needs "fp4".
    # Default stays "fp8" so existing tuned rows keep their meaning.
    a_dtype: str = "fp8"

    def __post_init__(self):
        if self.m <= 0:
            raise ValueError(f"m must be positive, got {self.m}")
        if self.vector_width not in (8, 16):
            raise ValueError(
                "production GEMM2 TP megakernel requires vector_width 8 or 16"
            )
        if self.sort_block_m <= 0:
            raise ValueError(f"sort_block_m must be positive, got {self.sort_block_m}")
        if self.tile_m <= 0 or self.tile_n <= 0 or self.tile_k <= 0:
            raise ValueError(
                "tile sizes must be positive, got "
                f"{(self.tile_m, self.tile_n, self.tile_k)}"
            )
        if self.tile_m not in (16, 32, 64, 128):
            raise ValueError(f"unsupported tile_m={self.tile_m}")
        if self.tile_n not in (128, 256, 512):
            raise ValueError(f"unsupported tile_n={self.tile_n}")
        if self.tile_k not in (128, 256):
            raise ValueError(f"unsupported tile_k={self.tile_k}")
        if self.shape.model_dim % self.tile_n:
            raise ValueError(
                f"model_dim={self.shape.model_dim} must be divisible by "
                f"tile_n={self.tile_n}"
            )
        if self.shape.inter_dim % self.tile_k:
            raise ValueError(
                f"inter_dim={self.shape.inter_dim} must be divisible by "
                f"tile_k={self.tile_k}"
            )
        if self.sort_block_m % self.tile_m:
            raise ValueError(
                "sort_block_m must be divisible by tile_m, got "
                f"sort_block_m={self.sort_block_m}, tile_m={self.tile_m}"
            )
        if self.tile_n % self.vector_width:
            raise ValueError(
                "tile_n must be divisible by vector_width, got "
                f"tile_n={self.tile_n}, vector_width={self.vector_width}"
            )
        if self.b_cache_modifier not in (0, 2):
            raise ValueError(
                "MXMoE GEMM2 producer requires b_cache_modifier 0 or 2, got "
                f"{self.b_cache_modifier}"
            )
        if not 0 <= self.waves_per_eu <= 10:
            raise ValueError(
                f"waves_per_eu must be in [0, 10], got {self.waves_per_eu}"
            )
        if self.compute_groups <= 0:
            raise ValueError(
                f"compute_groups must be positive, got {self.compute_groups}"
            )
        if self.block_threads != BLOCK:
            raise ValueError(
                f"MXMoE GEMM2 producer requires block_threads={BLOCK}, "
                f"got {self.block_threads}"
            )
        num_waves = self.block_threads // 64
        if self.tile_n % (num_waves * 16):
            raise ValueError(
                "tile_n must provide an integral number of 16-column MFMA "
                "tiles per wave, got "
                f"tile_n={self.tile_n}, block_threads={self.block_threads}"
            )
        if self.local_load_cache_modifier not in (0, 1, 2, 3):
            raise ValueError("local_load_cache_modifier must be in [0, 3]")
        if self.remote_load_cache_modifier not in (0, 1, 2, 3):
            raise ValueError("remote_load_cache_modifier must be in [0, 3]")
        if self.gather_load_cache_modifier not in (-1, 0, 1, 2, 3):
            raise ValueError("gather_load_cache_modifier must be -1 or in [0, 3]")
        if self.remote_store_cache_modifier not in (0, 1, 2, 3):
            raise ValueError("remote_store_cache_modifier must be in [0, 3]")
        if self.n_tile_cohort < 0:
            raise ValueError(
                f"n_tile_cohort must be non-negative, got {self.n_tile_cohort}"
            )
        if self.n_tile_cohort and self.n_tiles % self.n_tile_cohort:
            raise ValueError(
                "n_tile_cohort must divide n_tiles, got "
                f"n_tile_cohort={self.n_tile_cohort}, n_tiles={self.n_tiles}"
            )
        if self.collective not in (
            "direct",
            "rsag",
            "rs_broadcast",
        ):
            raise ValueError(
                "collective must be 'direct', 'rsag', or 'rs_broadcast', got "
                f"{self.collective!r}"
            )
        if not 1 <= self.service_groups <= self.compute_groups:
            raise ValueError(
                "service_groups must be in [1, compute_groups], got "
                f"service_groups={self.service_groups}, "
                f"compute_groups={self.compute_groups}"
            )
        if self.collective != "rsag" and self.service_groups != 1:
            raise ValueError(
                "direct and rs_broadcast collectives require service_groups=1"
            )
        if self.collective == "rsag" and (
            self.service_groups not in (1, 2, 4, 8)
            or self.service_groups > self.shape.tp_size
            or self.shape.tp_size % self.service_groups
        ):
            raise ValueError(
                "rsag service_groups must be a supported divisor of TP, got "
                f"service_groups={self.service_groups}, "
                f"TP={self.shape.tp_size}"
            )
        if self.service_tile_group <= 0 or self.n_tiles % self.service_tile_group:
            raise ValueError(
                "service_tile_group must be a positive divisor of n_tiles, got "
                f"service_tile_group={self.service_tile_group}, "
                f"n_tiles={self.n_tiles}"
            )
        if self.service_tile_group > 1 and (
            self.collective != "rsag" or self.service_groups == 1
        ):
            raise ValueError(
                "grouped service synchronization requires collective='rsag' "
                "and service_groups > 1"
            )
        if self.producer_mode not in ("routes", "atomic_shared"):
            raise ValueError(
                "producer_mode must be 'routes' or 'atomic_shared', got "
                f"{self.producer_mode!r}"
            )
        if self.flat_producer_grid and self.collective == "direct":
            raise ValueError("flat_producer_grid requires a dynamic collective path")
        if self.flat_producer_grid and self.n_tile_cohort:
            raise ValueError(
                "flat_producer_grid and n_tile_cohort are mutually exclusive"
            )
        if self.collective == "direct" and self.producer_mode != "routes":
            raise ValueError("direct production path requires producer_mode='routes'")
        if self.collective != "rsag" and self.gather_load_cache_modifier != -1:
            raise ValueError("gather_load_cache_modifier only applies to rsag")
        if self.collective == "direct" and self.remote_store_cache_modifier != 0:
            raise ValueError("remote_store_cache_modifier does not apply to direct")
        if self.producer_mode == "atomic_shared" and self.collective != "rs_broadcast":
            raise ValueError(
                "atomic_shared production requires collective='rs_broadcast'"
            )
        if self.uses_rsag and self.m % self.shape.tp_size:
            raise ValueError(
                f"m={self.m} must be divisible by TP={self.shape.tp_size} "
                f"for collective={self.collective!r}"
            )
        if (
            self.uses_rsag
            and (self.m * self.tile_n // self.vector_width) % self.shape.tp_size
        ):
            raise ValueError("rsag vector items must divide evenly across TP ranks")

    @property
    def n_tiles(self) -> int:
        return self.shape.model_dim // self.tile_n

    @property
    def uses_rsag(self) -> bool:
        return self.collective in ("rsag", "rs_broadcast")

    @property
    def shared_bf16_partials(self) -> bool:
        return self.collective == "rs_broadcast"

    @property
    def wide_partial_scales(self) -> bool:
        return (
            self.collective == "direct"
            and self.tile_n == 256
            and (
                self.m == 4
                or (self.m in (1, 2) and self.vector_width == 8)
                or (self.m == 8 and self.vector_width == 16)
            )
        )

    @property
    def single_pass_direct(self) -> bool:
        return bool(
            self.collective == "direct"
            and self.m * self.tile_n // self.vector_width <= self.block_threads
        )

    @property
    def producer_rows(self) -> int:
        route_rows = self.m * self.shape.topk
        # Sorting pads each non-empty expert independently to sort_block_m.
        max_sort_blocks = (
            route_rows
            if route_rows <= self.shape.experts
            else self.shape.experts
            + (route_rows - self.shape.experts) // self.sort_block_m
        )
        return max_sort_blocks * self.sort_block_m // self.tile_m

    @property
    def payload_bytes(self) -> int:
        return self.m * self.shape.model_dim * 2

    @property
    def partial_bytes(self) -> int:
        return _align_up(self.partial_payload_bytes + self.partial_scale_bytes, 16)

    @property
    def partial_payload_bytes(self) -> int:
        return self.m * self.shape.model_dim

    @property
    def partial_scale_bytes(self) -> int:
        if self.shared_bf16_partials:
            return 0
        if self.wide_partial_scales:
            return self.partial_payload_bytes // 8
        return _mxfp8_scale_bytes(self.partial_payload_bytes, self.vector_width)

    @property
    def reduced_payload_bytes(self) -> int:
        if not self.uses_rsag:
            return 0
        element_bytes = 2 if self.shared_bf16_partials else 1
        return self.m * self.shape.model_dim * element_bytes // self.shape.tp_size

    @property
    def reduced_scale_bytes(self) -> int:
        if not self.uses_rsag or self.shared_bf16_partials:
            return 0
        return _mxfp8_scale_bytes(self.reduced_payload_bytes, self.vector_width)

    @property
    def reduced_shard_bytes(self) -> int:
        return _align_up(self.reduced_payload_bytes + self.reduced_scale_bytes, 16)

    @property
    def reduced_offset(self) -> int:
        return SLOTS * self.partial_bytes

    @property
    def route_offset(self) -> int:
        return self.reduced_offset + SLOTS * self.reduced_shard_bytes

    @property
    def route_bytes(self) -> int:
        if self.producer_mode == "routes":
            return self.m * self.shape.topk * self.shape.model_dim * 2
        return SLOTS * self.payload_bytes

    @property
    def output_offset(self) -> int:
        return self.route_offset + self.route_bytes

    @property
    def producer_done_offset(self) -> int:
        return self.output_offset + self.payload_bytes

    @property
    def producer_counter_slots(self) -> int:
        return SLOTS if self.producer_mode == "atomic_shared" else 1

    @property
    def epoch_offset(self) -> int:
        return self.gather_service_done_offset + self.n_tiles * 8

    @property
    def service_done_offset(self) -> int:
        return (
            self.producer_done_offset
            + self.n_tiles * self.producer_counter_slots * PRODUCER_COUNTER_STRIDE
        )

    @property
    def reduce_done_offset(self) -> int:
        return self.service_done_offset + self.n_tiles * 8

    @property
    def gather_service_done_offset(self) -> int:
        return self.reduce_done_offset + self.n_tiles * 8

    @property
    def rank_ready_offset(self) -> int:
        return self.epoch_offset + self.n_tiles * 8

    @property
    def flat_base_offset(self) -> int:
        return _align_up(
            self.gather_done_offset
            + (self.n_tiles * self.shape.tp_size * 4 if self.uses_rsag else 0),
            8,
        )

    @property
    def gather_done_offset(self) -> int:
        return self.owner_ready_offset + (
            self.n_tiles * self.shape.tp_size * 4
            if self.uses_rsag
            else self.n_tiles * 4
        )

    @property
    def owner_ready_offset(self) -> int:
        return self.reduced_collective_ready_offset + self.n_tiles * 4

    @property
    def reduced_collective_ready_offset(self) -> int:
        return self.collective_ready_offset + self.n_tiles * 4

    @property
    def collective_ready_offset(self) -> int:
        return self.rank_ready_offset + self.n_tiles * self.shape.tp_size * 4

    @property
    def workspace_bytes(self) -> int:
        return _align_up(self.flat_base_offset + 8, 256)


@dataclass(frozen=True)
class AtomicConfig:
    shape: Shape
    m: int
    reduce_scatter_grid: int
    all_gather_grid: int

    def __post_init__(self):
        if self.m <= 0 or self.m % self.shape.tp_size:
            raise ValueError(
                f"m={self.m} must be a positive multiple of " f"TP={self.shape.tp_size}"
            )
        if not 0 < self.reduce_scatter_grid <= RENDEZVOUS_FLAG_SLOTS:
            raise ValueError(
                "reduce_scatter_grid must be positive and at most "
                f"{RENDEZVOUS_FLAG_SLOTS}, got {self.reduce_scatter_grid}"
            )
        source_count = self.shape.tp_size - 1
        if (
            self.all_gather_grid < source_count
            or self.all_gather_grid % source_count
            or self.all_gather_grid > RENDEZVOUS_FLAG_SLOTS
        ):
            raise ValueError(
                "all_gather_grid must be a multiple of TP-1 in "
                f"(0, {RENDEZVOUS_FLAG_SLOTS}], got "
                f"all_gather_grid={self.all_gather_grid}, TP={self.shape.tp_size}"
            )

    @property
    def shard_rows(self) -> int:
        return self.m // self.shape.tp_size

    @property
    def use_fused_rsag(self) -> bool:
        """Collapse quantize + reduce-scatter + all-gather into one launch.

        Sharding the rendezvous by row lets one block own a column group of
        one row per shard, which moves the reduce-scatter's peer bytes in the
        full reduce's single dispatch, so where it applies it supersedes both
        of the paths below. Only the smallest buckets stay behind, where the
        bytes it saves no longer pay for its second rendezvous.
        """
        return self.m >= FUSED_RSAG_MIN_M

    @property
    def use_full_reduce(self) -> bool:
        """Collapse reduce-scatter + all-gather into one full-reduce launch.

        The pair halves the peer bytes but costs a second launch and a second
        rendezvous. Measured on GLM-5.2 TP4/EP4 the two are within half a
        microsecond at m=64, where the single kernel is still ahead, and the
        pair pulls clear from m=96 up as the bytes start to dominate.
        """
        return self.m <= 64

    @property
    def ready_offset(self) -> int:
        """Byte offset of the epoch flag that follows the MXFP8 partial."""
        model_dim = self.shape.model_dim
        return self.m * (model_dim + model_dim // 32)

    def block_flag_offset(self, phase: int) -> int:
        """Byte offset of one collective phase's per-block rendezvous flags."""
        if not 0 <= phase < RENDEZVOUS_PHASES:
            raise ValueError(f"rendezvous phase must be in [0, 2), got {phase}")
        base = _align_up(self.ready_offset + 8, RENDEZVOUS_FLAG_STRIDE)
        return base + phase * RENDEZVOUS_FLAG_SLOTS * RENDEZVOUS_FLAG_STRIDE

    @property
    def partial_bytes(self) -> int:
        return _align_up(
            self.block_flag_offset(RENDEZVOUS_PHASES - 1)
            + RENDEZVOUS_FLAG_SLOTS * RENDEZVOUS_FLAG_STRIDE,
            256,
        )


@dataclass(frozen=True)
class WindowConfig:
    shape: Shape
    m: int
    tile_m: int
    tile_n: int
    tile_k: int
    sort_block_m: int
    window: int
    local_workers: int
    reduce_scatter_grid: int
    all_gather_grid: int

    def __post_init__(self):
        if self.m <= 0:
            raise ValueError(f"m must be positive, got {self.m}")
        for name, value in (
            ("tile_m", self.tile_m),
            ("tile_n", self.tile_n),
            ("tile_k", self.tile_k),
            ("sort_block_m", self.sort_block_m),
            ("window", self.window),
            ("local_workers", self.local_workers),
            ("reduce_scatter_grid", self.reduce_scatter_grid),
            ("all_gather_grid", self.all_gather_grid),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.tile_m not in (16, 32, 64, 128):
            raise ValueError(f"unsupported tile_m={self.tile_m}")
        if self.tile_n not in (128, 256, 512):
            raise ValueError(f"unsupported tile_n={self.tile_n}")
        if self.tile_k not in (128, 256):
            raise ValueError(f"unsupported tile_k={self.tile_k}")
        divisibility = (
            ("m", self.m, "tp_size", self.shape.tp_size),
            ("model_dim", self.shape.model_dim, "window", self.window),
            ("window", self.window, "tile_n", self.tile_n),
            ("window", self.window, "MXFP8 group", 32),
            ("inter_dim", self.shape.inter_dim, "tile_k", self.tile_k),
            ("sort_block_m", self.sort_block_m, "tile_m", self.tile_m),
        )
        for value_name, value, divisor_name, divisor in divisibility:
            if divisor <= 0 or value % divisor:
                raise ValueError(
                    f"{value_name}={value} must be divisible by "
                    f"{divisor_name}={divisor}"
                )
        if self.shape.model_dim // self.window < 3:
            raise ValueError(
                "window pipeline requires at least three phases, got "
                f"model_dim={self.shape.model_dim}, window={self.window}"
            )
        source_count = self.shape.tp_size - 1
        if self.all_gather_grid % source_count:
            raise ValueError(
                "all_gather_grid must be a multiple of TP-1, got "
                f"all_gather_grid={self.all_gather_grid}, TP={self.shape.tp_size}"
            )
        if self.local_workers > self.compute_workers:
            raise ValueError(
                "local_workers cannot exceed the compute grid, got "
                f"local_workers={self.local_workers}, "
                f"compute_workers={self.compute_workers}"
            )
        service_grid = max(self.reduce_scatter_grid, self.all_gather_grid)
        if self.compute_workers < 2 * service_grid:
            raise ValueError(
                "paired window services require at least two compute workers "
                "per service worker, got "
                f"compute_workers={self.compute_workers}, service_grid={service_grid}"
            )

    @property
    def shard_rows(self) -> int:
        return self.m // self.shape.tp_size

    @property
    def tiles_per_window(self) -> int:
        return self.window // self.tile_n

    @property
    def groups_per_row(self) -> int:
        return self.window // 32

    @property
    def compute_workers(self) -> int:
        max_sorted = (
            self.m * self.shape.topk
            + self.shape.experts * self.sort_block_m
            - self.shape.topk
        )
        sort_blocks = (max_sorted + self.sort_block_m - 1) // self.sort_block_m
        producer_rows = sort_blocks * self.sort_block_m // self.tile_m
        return producer_rows * self.tiles_per_window


PipelineConfig = MegakernelConfig | AtomicConfig | WindowConfig
