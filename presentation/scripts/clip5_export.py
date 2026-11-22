"""Export the prototype vs v2 measurements used by clip 5 (kernel optimisations).

Source: results/26-11-21_15-10-58_sim_v2/results.jsonl (both kernels in the same run).
usage: python scripts/clip5_export.py  (from presentation/)
"""
import json
from pathlib import Path

SRC = Path("../results/26-11-21_15-10-58_sim_v2/results.jsonl")
OUT = Path("src/data/clip5_bench.json")
IMPLS = {"triton_proto": "proto", "src.kernel:TritonGNM": "v2"}

rows = [json.loads(l) for l in SRC.read_text().splitlines() if l.strip()]
out = {"source": str(SRC), "configs": []}
for n, B in [(1000, 1024), (2000, 1024)]:
    cfg = {"n": n, "B": B}
    for r in rows:
        if r["n"] == n and r["B"] == B and r["impl"] in IMPLS:
            k = IMPLS[r["impl"]]
            cfg["T"] = r["T"]
            cfg[k] = {key: r[key] for key in ("peak_mem_gb", "run_s", "setup_s", "sims_per_s")}
    out["configs"].append(cfg)
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(out, indent=2) + "\n")
print(OUT.read_text())
