# SPDX-License-Identifier: MIT

from pathlib import Path

from aiter.ops.flydsl.kernels.communication_ops_utils import GeometryTuningTable
from aiter.ops.flydsl.kernels.mega_moe.mega_moe_config import (
    select_mega_moe_config,
)


def test_glm52_ep8_decode_specialization():
    expected = {
        1: (224, 256, 4),
        4: (128, 512, 8),
        6: (192, 512, 8),
        12: (128, 512, 8),
        24: (128, 512, 8),
        48: (160, 512, 8),
        96: (96, 512, 8),
    }
    for tokens, stage1_values in expected.items():
        config = select_mega_moe_config(
            tokens,
            256,
            experts_per_rank=32,
            model_dim=6144,
            inter_dim=2048,
        )
        assert (
            config.stage1.num_dispatch_cu,
            config.stage1.tile_n,
            config.stage1.num_waves,
        ) == stage1_values
        assert config.stage2.block_n == 128
        assert config.stage2.persist
        assert config.stage2.persist_cu == 192


def test_glm52_ep4_keeps_generic_policy():
    config = select_mega_moe_config(
        24,
        256,
        experts_per_rank=64,
        model_dim=6144,
        inter_dim=2048,
    )
    assert config.stage1.num_dispatch_cu == 64
    assert config.stage2.block_n == 256
    assert config.stage2.persist_cu == 240


def test_other_ep8_shape_keeps_generic_policy():
    config = select_mega_moe_config(
        24,
        256,
        experts_per_rank=32,
        model_dim=7168,
        inter_dim=3072,
    )
    assert config.stage1.num_dispatch_cu == 64
    assert config.stage2.block_n == 256
    assert config.stage2.persist_cu == 240


def test_glm52_ep8_combine_geometry_table():
    root = Path(__file__).resolve().parents[2]
    table = GeometryTuningTable.from_tuning_file(
        root
        / "aiter/ops/flydsl/kernels/mega_moe_tuning_config"
        / "flydsl_gfx950_mi355x_IntraNode_ep8.json",
        dtype="fp8_ocp",
        hidden_dim=6144,
        zero_copy=False,
        topk=8,
        local_expert_num=32,
        combine_dtype="bf16",
    )
    assert table.dispatch == {}
    assert table.lookup("combine", 1) == (64, 4)
    assert table.lookup("combine", 6) == (64, 8)
    assert table.lookup("combine", 48) == (96, 16)
    assert table.lookup("combine", 96) == (96, 16)
