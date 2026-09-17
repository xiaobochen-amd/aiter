#!/usr/bin/env python3
"""Check real BF16 split-K GEMMs after buffer growth and allocator pressure."""

import argparse
import json
from pathlib import Path

import torch

from aiter.ops.flydsl import gemm_kernels as kernels

parser = argparse.ArgumentParser()
parser.add_argument("--output", required=True)
args = parser.parse_args()
torch.cuda.set_device(0)
torch.manual_seed(1729)
records = []
for m, n, k in [(32, 2624, 6144), (24, 4096, 2048), (12, 6144, 2048)]:
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
        out = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)

        # Called only in this iteration. Capture records device work immediately;
        # graph replay does not retain or invoke this Python closure.
        def gemm():
            kernels.flydsl_hgemm(
                a,  # noqa: B023
                b,  # noqa: B023
                out=out,  # noqa: B023
                tile_m=16,
                tile_n=64,
                tile_k=64,
                split_k=4,
                block_m_warps=1,
                block_n_warps=2,
                block_k_warps=1,
                stages=3,
                async_copy=True,
                b_to_lds=True,
            )

        for _ in range(3):
            gemm()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            gemm()
        key = kernels._stream_cache_key(stream)
        old_s = kernels.SPLIT_K_GLOBAL_SEMAPHORE[key].numel()
        old_w = kernels.SPLIT_K_GLOBAL_WORKSPACE[key].numel()
        kernels._get_split_k_buffers(a.device, stream, old_s * 2, old_w * 2)
        pressure = [
            torch.full((old_w,), 119, dtype=torch.uint8, device="cuda")
            for _ in range(8)
        ]
        pressure += [
            torch.full((old_s,), 119, dtype=torch.int32, device="cuda")
            for _ in range(8)
        ]
        # A stale output can no longer pass by repeating the previous result.
        a.add_(0.25)
        for _ in range(5):
            graph.replay()
        torch.cuda.synchronize()
        reference = a.float() @ b.float().t()
        error = out.float() - reference
        snr = float(10 * torch.log10(reference.square().mean() / error.square().mean()))
        row = {
            "m": m,
            "n": n,
            "k": k,
            "snr_db": snr,
            "finite": bool(torch.isfinite(out).all()),
            "passed": snr > 35 and bool(torch.isfinite(out).all()),
        }
        records.append(row)
        print(json.dumps(row), flush=True)
        Path(args.output).write_text(json.dumps(records, indent=2) + "\n")
raise SystemExit(0 if all(r["passed"] for r in records) else 1)
