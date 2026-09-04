#!/usr/bin/env python3
"""Campaign scorer: FlyDSL against the 23 GLM-5.2 decode points it currently loses.

Runs the a16w16 tuner restricted to --libtype flydsl over exactly those points,
then scores geomean(frozen_champion_us / flydsl_us).  <1 means FlyDSL is still
behind; 1.0 is parity with whatever backend owns the row today; >1 means FlyDSL
has taken the row.

The champion times are FROZEN CONSTANTS measured from the tuned table -- they are
torch / triton / skinny rows, none of which this repo's FlyDSL changes can move,
so the ruler cannot drift.  Only the FlyDSL side is re-measured each round.

Prints one JSON object on the last stdout line: {"ok": bool, "gm_ratio": float, ...}
"""
import csv
import json
import math
import os
import subprocess
import sys
import tempfile

# (M, N, K) -> frozen champion.  See campaign goal for how these were measured.
BASELINE = {
    "1,32,6144": {
        "champ": "skinny",
        "us": 3.4721
    },
    "2,32,6144": {
        "champ": "skinny",
        "us": 4.7249
    },
    "4,32,6144": {
        "champ": "skinny",
        "us": 5.8368
    },
    "8,32,6144": {
        "champ": "torch",
        "us": 8.9937
    },
    "48,32,6144": {
        "champ": "torch",
        "us": 8.0695
    },
    "1,160,6144": {
        "champ": "skinny",
        "us": 4.1738
    },
    "1,256,6144": {
        "champ": "skinny",
        "us": 4.7363
    },
    "2,256,6144": {
        "champ": "skinny",
        "us": 4.6129
    },
    "24,19360,6144": {
        "champ": "torch",
        "us": 49.4333
    },
    "24,38720,6144": {
        "champ": "triton",
        "us": 89.2077
    },
    "1,128,6144": {
        "champ": "skinny",
        "us": 3.5904
    },
    "12,7168,512": {
        "champ": "triton",
        "us": 3.9882
    },
    "24,6144,2048": {
        "champ": "triton",
        "us": 7.9751
    },
    "24,6144,3072": {
        "champ": "triton",
        "us": 10.2498
    },
    "24,6144,4096": {
        "champ": "triton",
        "us": 12.4426
    },
    "24,6144,6144": {
        "champ": "triton",
        "us": 18.7201
    },
    "10,19360,6144": {
        "champ": "triton",
        "us": 42.7869
    },
    "84,19360,6144": {
        "champ": "torch",
        "us": 58.2707
    },
    "10,38720,6144": {
        "champ": "triton",
        "us": 81.2794
    },
    "12,38720,6144": {
        "champ": "triton",
        "us": 83.2529
    },
    "14,38720,6144": {
        "champ": "triton",
        "us": 86.8387
    },
    "60,38720,6144": {
        "champ": "torch",
        "us": 102.3144
    },
    "84,38720,6144": {
        "champ": "torch",
        "us": 101.9431
    }
}

REPO = os.path.dirname(os.path.abspath(__file__))
TUNER = os.path.join(REPO, "csrc", "gemm_a16w16", "gemm_tuner.py")
GPUS = os.environ.get("HIP_VISIBLE_DEVICES", "0,1,4,5")
MP = len([g for g in GPUS.split(",") if g])


def main():
    tmp = tempfile.mkdtemp(prefix="camp23_")
    inp = os.path.join(tmp, "in.csv")
    out = os.path.join(tmp, "tuned.csv")
    with open(inp, "w") as f:
        f.write("M,N,K,bias,dtype,outdtype,scaleAB,bpreshuffle\n")
        for key in BASELINE:
            m, n, k = key.split(",")
            f.write("%s,%s,%s,False,torch.bfloat16,torch.bfloat16,False,False\n" % (m, n, k))

    env = dict(os.environ, PYTHONPATH=REPO, HIP_VISIBLE_DEVICES=GPUS)
    cmd = [sys.executable, "-u", TUNER, "--input_file", inp, "--tuned_file", out,
           "--libtype", "flydsl", "--mp", str(MP)]
    print("running:", " ".join(cmd), flush=True)
    p = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True)
    if p.returncode != 0:
        sys.stderr.write(p.stdout[-4000:] + p.stderr[-4000:])
        print(json.dumps({"ok": False, "gm_ratio": 0.0, "reason": "tuner rc=%d" % p.returncode}))
        return

    got = {}
    if os.path.exists(out):
        for r in csv.DictReader(open(out)):
            us = float(r["us"])
            if r["libtype"] != "flydsl" or us <= 0:
                continue
            err = float(r.get("err_ratio") or 0.0)
            if err > 0.02:                      # tuner's own correctness gate
                continue
            got["%s,%s,%s" % (r["M"], r["N"], r["K"])] = us

    ratios, missing, per = [], [], {}
    for key, ref in BASELINE.items():
        f = got.get(key)
        if f is None:
            missing.append(key)
            continue
        ratio = ref["us"] / f
        ratios.append(ratio)
        per[key] = round(ratio, 4)

    for key in sorted(per, key=lambda k: per[k]):
        ref = BASELINE[key]
        print("  %-16s champ=%-7s %8.4f us  flydsl %8.4f us  ratio %.4f"
              % (key, ref["champ"], ref["us"], got[key], per[key]))
    for key in missing:
        print("  %-16s champ=%-7s %8.4f us  flydsl   NO CANDIDATE  ratio 0"
              % (key, BASELINE[key]["champ"], BASELINE[key]["us"]))

    # A point with no FlyDSL candidate scores 0 rather than being dropped: dropping
    # it would let a change that kills candidates look like an improvement.
    for _ in missing:
        ratios.append(0.0)

    if not ratios or all(r == 0.0 for r in ratios):
        print(json.dumps({"ok": False, "gm_ratio": 0.0, "reason": "no FlyDSL result"}))
        return

    # geomean over a set containing zeros would collapse, so score those as a
    # small floor instead -- still a heavy penalty, but keeps the ruler ordered.
    adj = [r if r > 0 else 0.05 for r in ratios]
    gm = math.exp(sum(math.log(r) for r in adj) / len(adj))
    print(json.dumps({
        "ok": True,
        "gm_ratio": round(gm, 6),
        "points": len(BASELINE),
        "with_flydsl": len(per),
        "missing": len(missing),
        "wins": sum(1 for r in per.values() if r > 1.0),
        "worst": min(per.values()) if per else 0.0,
    }))


if __name__ == "__main__":
    main()
