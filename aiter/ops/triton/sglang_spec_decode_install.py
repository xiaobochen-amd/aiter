# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Idempotent installer for aiter's sglang speculative-decode call sites.

``aiter.ops.triton.draft_sample`` implements the EAGLE draft proposal and the
vocab-parallel greedy pick, but neither runs unless sglang calls it. The call
sites live in sglang's tree, so they are shipped here as a unified diff
(``sglang_spec_decode.patch``) and applied to the installed sglang on import.

Applying at import is early enough: on ROCm sglang pulls ``aiter`` in before any
of ``layers/logits_processor``, ``speculative/spec_utils`` or
``speculative/eagle_worker_v2``, so the rewritten sources are what the server
process actually loads.

The installer is a no-op once applied -- it looks for the names the diff
introduces before touching anything -- and it never raises: if sglang is absent,
read-only, or has moved on, the deployment keeps its stock behaviour and the
aiter ops stay unused.

Env switches (read by the installed call sites, not by this module):

  AITER_GREEDY_SPEC_GATE=0    restore sglang's forced rejection sampling
  AITER_FUSED_DRAFT_SAMPLE=0  restore the torch draft proposal chain
  AITER_FUSED_VERIFY_PROBS=0  restore the torch verify softmax

Set ``AITER_SGLANG_SPEC_INSTALL=0`` to skip the install itself.
"""

import importlib.util
import logging
import os
import shutil
import subprocess
import sys

logger = logging.getLogger("aiter")

PATCH_FILE = os.path.join(os.path.dirname(__file__), "sglang_spec_decode.patch")

# The diff is rooted at the package dir, one level above ``sglang/``.
_PATCH_STRIP = 1
_PROBE = "python/sglang/srt/speculative/spec_utils.py"

# One name the patch introduces per file it touches. These decide whether the
# call sites are in place: ``patch -R --dry-run`` cannot, because on an unpatched
# tree it reports "Unreversed patch detected! Ignoring -R" and still exits 0.
# Requiring all of them also catches a half-reverted tree, which is otherwise
# silent -- the server starts and simply never enters the aiter path.
_MARKERS = {
    "python/sglang/srt/layers/logits_processor.py": "aiter_vocab_parallel",
    "python/sglang/srt/arg_groups/speculative_hook.py": "AITER_GREEDY_SPEC_GATE",
    "python/sglang/srt/speculative/spec_utils.py": "greedy_draft_pick",
    "python/sglang/srt/speculative/eagle_utils.py": "fused_verify_probs",
    "python/sglang/srt/speculative/eagle_worker_v2.py": "_init_vocab_parallel_greedy",
}


def _missing_markers(root):
    missing = []
    for rel, marker in _MARKERS.items():
        try:
            with open(os.path.join(root, rel), errors="replace") as f:
                if marker not in f.read():
                    missing.append(rel)
        except OSError:
            missing.append(rel)
    return missing


def _sglang_roots():
    """Candidate directories the patch's ``python/sglang/...`` paths hang off.

    Located without executing sglang: this runs while sglang is importing aiter,
    so the package is in ``sys.modules`` but only half built, and forcing a fresh
    import of it here would be both circular and expensive.
    """
    roots = []
    explicit = os.environ.get("AITER_SGLANG_ROOT")
    if explicit:
        roots.append(explicit)
    origins = []
    mod = sys.modules.get("sglang")
    if mod is not None and getattr(mod, "__file__", None):
        origins.append(mod.__file__)
    else:
        try:
            spec = importlib.util.find_spec("sglang")
            if spec is not None and spec.origin:
                origins.append(spec.origin)
        except Exception:
            pass
    # <root>/python/sglang/__init__.py -> <root>
    roots += [
        os.path.dirname(os.path.dirname(os.path.dirname(o))) for o in origins
    ]
    seen = []
    for root in roots:
        if os.path.isfile(os.path.join(root, _PROBE)) and root not in seen:
            seen.append(root)
    return seen


def _patch(root, args):
    return subprocess.run(
        ["patch", "-p%d" % _PATCH_STRIP, "--batch", "--no-backup-if-mismatch"]
        + args
        + ["-i", PATCH_FILE],
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def install(root=None):
    """Apply the call sites to ``root``. Returns "present", "installed" or a reason."""
    if not os.path.isfile(PATCH_FILE):
        return "no-patch-file"
    if shutil.which("patch") is None:
        return "no-patch-tool"
    roots = [root] if root else _sglang_roots()
    if not roots:
        return "no-sglang"
    root = roots[0]
    missing = _missing_markers(root)
    if not missing:
        return "present"
    if len(missing) != len(_MARKERS):
        # Some files carry the call sites and some do not. Patching from here
        # would fail half-way and leave a worse tree than either end state.
        return "partially applied, missing: " + ", ".join(missing)
    probe = _patch(root, ["--forward", "--dry-run"])
    if probe.returncode != 0:
        return "does-not-apply: " + probe.stdout.decode(errors="replace").strip()[-300:]
    applied = _patch(root, ["--forward"])
    if applied.returncode != 0:
        return "failed: " + applied.stdout.decode(errors="replace").strip()[-300:]
    missing = _missing_markers(root)
    if missing:
        return "applied but incomplete, missing: " + ", ".join(missing)
    return "installed"


def maybe_install():
    """Best-effort install driven by ``AITER_SGLANG_SPEC_INSTALL`` (default on)."""
    if os.environ.get("AITER_SGLANG_SPEC_INSTALL", "1") != "1":
        return "disabled"
    try:
        status = install()
    except Exception as e:  # never let the install break an import
        status = "error: %r" % (e,)
    if status == "installed":
        logger.info("aiter installed its sglang speculative-decode call sites")
    elif status not in ("present", "no-sglang"):
        logger.warning("aiter sglang speculative-decode call sites: %s", status)
    return status


if __name__ == "__main__":
    print(install())
