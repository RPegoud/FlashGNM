"""Export the numbers shown in clip 2 (naive n^2 sampling) from the committed benchmark.

Source: ../results/26-11-21_14-40-12_sim_baseline/results.jsonl, rows impl == "toolbox",
n == 1000 (Griffa scale500, T = 10785). The toolbox rows are extrapolated from
--probe-steps timed steps (meta.json args.probe_steps).

usage (from presentation/): python3 scripts/clip2_naive_export.py
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN = HERE.parent.parent / "results" / "26-11-21_14-40-12_sim_baseline"
OUT = HERE.parent / "src" / "data" / "clip2_naive.json"

rows = [json.loads(line) for line in (RUN / "results.jsonl").read_text().splitlines() if line.strip()]
meta = json.loads((RUN / "meta.json").read_text())
gpu = meta["gpu"]
probe = meta["args"]["probe_steps"]


def pick(B):
    (r,) = [r for r in rows if r["impl"] == "toolbox" and r["n"] == 1000 and r["B"] == B]
    assert r["status"] == "ok" and r["extrapolated"], r
    return {
        "B": B,
        "ms_per_step": r["ms_per_step"],
        "run_s": r["run_s"],
        "total_s": r["total_s"],
    }


b1, b1024 = pick(1), pick(1024)
(r0,) = [r for r in rows if r["impl"] == "toolbox" and r["n"] == 1000 and r["B"] == 1]
out = {
    "source": "results/26-11-21_14-40-12_sim_baseline/results.jsonl",
    "gpu": gpu,
    "probe_steps": probe,
    "n": r0["n"],
    "T": r0["T"],
    "dist": r0["dist"],
    "toolbox": [b1, b1024],
}
OUT.write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out, indent=2))
