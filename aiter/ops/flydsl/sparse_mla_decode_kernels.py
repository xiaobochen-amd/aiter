# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Public launcher for gfx950 FlyDSL sparse MLA decode."""

from __future__ import annotations

import math

import flydsl.expr as fx
import torch

from .kernels.sparse_mla_decode import BLOCK_I, DIM, DV, H, compile_sparse_mla_partial
from .kernels.tensor_shim import _run_compiled, ptr_arg
from .mla_reduce_kernels import _flydsl_sparse_mla_decode_combine


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
    # seq is a runtime scalar (passed as int(q.shape[0]) to the launch); the
    # kernel is compiled per split count `ng`, not per seq. Measured on MI355X
    # against the TileLang decode at width=2048, all correct at 32.0-32.2 dB:
    #   seq   1     2     4     6     8    12    24    48
    #   ratio 0.60x 0.62x 0.64x 0.66x 0.65x 0.71x 0.75x 1.05x   (FlyDSL/TileLang)
    # The crossover is between 24 and 48, so cap at 24 rather than admitting a
    # regression. b (batch) is untested above 1 and stays pinned in the reducer.
    if not 1 <= seq <= 24:
        raise ValueError(f"supported sparse decode scope is 1<=seq<=24, got {seq}")
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
    ng: int,
    device: torch.device,
) -> None:
    _require_cuda_tensor("partial_output", partial_output, dtype=torch.bfloat16)
    _require_cuda_tensor("partial_lse", partial_lse, dtype=torch.float32)
    if partial_output.device != device or partial_lse.device != device:
        raise ValueError("sparse decode workspace must share the decode device")
    if tuple(partial_output.shape) != (seq, ng, H, DV):
        raise ValueError(
            f"partial_output must have shape [{seq},{ng},{H},{DV}], got "
            f"{tuple(partial_output.shape)}"
        )
    if tuple(partial_lse.shape) != (seq, ng, H):
        raise ValueError(
            f"partial_lse must have shape [{seq},{ng},{H}], got "
            f"{tuple(partial_lse.shape)}"
        )


def _launch_partial(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    partial_output: torch.Tensor,
    partial_lse: torch.Tensor,
    sm_scale: float,
    *,
    ng: int,
) -> None:
    launch = compile_sparse_mla_partial(ng)
    _run_compiled(
        launch,
        ptr_arg(q, fx.Uint8),
        ptr_arg(kv.reshape(-1, DIM), fx.Uint8),
        ptr_arg(indices, fx.Int32),
        ptr_arg(partial_output, fx.BFloat16),
        ptr_arg(partial_lse, fx.Float32),
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
    if (partial_output is None) != (partial_lse is None):
        raise ValueError(
            "partial_output and partial_lse must be provided together for "
            "graph-safe scratch reuse"
        )
    if partial_output is None:
        partial_output = torch.empty(
            (seq, ng, H, DV), device=q.device, dtype=torch.bfloat16
        )
        partial_lse = torch.empty((seq, ng, H), device=q.device, dtype=torch.float32)
    else:
        _validate_workspace(
            partial_output,
            partial_lse,
            seq=seq,
            ng=ng,
            device=q.device,
        )

    _launch_partial(q, kv, indices, partial_output, partial_lse, sm_scale, ng=ng)
    _flydsl_sparse_mla_decode_combine(
        partial_output.unsqueeze(0),
        partial_lse.unsqueeze(0),
        out.unsqueeze(0),
    )
    return out


__all__ = ["flydsl_sparse_mla_decode"]
