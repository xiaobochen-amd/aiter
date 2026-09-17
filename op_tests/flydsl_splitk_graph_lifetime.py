#!/usr/bin/env python3
"""Numerical regression: cached split-K inputs must survive a later cache growth.

A captured graph stores raw device pointers, so dropping the last Tensor owner
can make a later allocation overwrite its semaphore/workspace. No model needed.
"""

import argparse
import json
import weakref
from pathlib import Path

import torch

from aiter.ops.flydsl import gemm_kernels as kernels

parser = argparse.ArgumentParser()
parser.add_argument("--output", required=True)
args = parser.parse_args()
torch.cuda.set_device(0)
stream = torch.cuda.current_stream()
semaphore, workspace = kernels._get_split_k_buffers(
    torch.device("cuda:0"), stream, 512, 1 << 20
)
semaphore.fill_(17)
workspace.fill_(3)
old_sem_ptr, old_ws_ptr = semaphore.data_ptr(), workspace.data_ptr()
sem_ref, ws_ref = weakref.ref(semaphore), weakref.ref(workspace)
out_sem = torch.empty(512, dtype=torch.int32, device="cuda")
out_ws = torch.empty(1024, dtype=torch.uint8, device="cuda")
torch.cuda.synchronize()
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    out_sem.copy_(semaphore[:512])
    out_ws.copy_(workspace[:1024])
graph.replay()
torch.cuda.synchronize()
assert bool((out_sem == 17).all()) and bool((out_ws == 3).all())
del semaphore, workspace
new_sem, new_ws = kernels._get_split_k_buffers(
    torch.device("cuda:0"), stream, 4096, 4 << 20
)
new_sem.fill_(9)
new_ws.fill_(9)
# Reuse free blocks with live allocations, making dangling graph pointers
# observable without relying on freed memory becoming unmapped.
clobber = []
for _ in range(16):
    a = torch.full((1024,), 119, dtype=torch.int32, device="cuda")
    b = torch.full((1 << 20,), 119, dtype=torch.uint8, device="cuda")
    clobber.extend((a, b))
graph.replay()
torch.cuda.synchronize()
record = {
    "semaphore_owner_alive": sem_ref() is not None,
    "workspace_owner_alive": ws_ref() is not None,
    "old_semaphore_address_reused": any(x.data_ptr() == old_sem_ptr for x in clobber),
    "old_workspace_address_reused": any(x.data_ptr() == old_ws_ptr for x in clobber),
    "semaphore_values_preserved": bool((out_sem == 17).all()),
    "workspace_values_preserved": bool((out_ws == 3).all()),
    "observed_semaphore": int(out_sem[0]),
    "observed_workspace": int(out_ws[0]),
}
record["passed"] = (
    record["semaphore_values_preserved"] and record["workspace_values_preserved"]
)
Path(args.output).write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record), flush=True)
raise SystemExit(0 if record["passed"] else 1)
