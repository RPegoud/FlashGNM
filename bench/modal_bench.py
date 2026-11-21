"""Run bench/bench_sim.py on a Modal GPU and write the results directory locally.

    modal run bench/modal_bench.py                                   # full default grid, H100
    modal run bench/modal_bench.py --args "--impls triton_proto --n 68 1000 --B 1024"
    modal run bench/modal_bench.py --gpu "B200" --args "..." --tag b200

Results land in results/<YY-MM-DD_HH-MM-SS>_sim[_tag]/ (meta.json, results.jsonl,
summary.md), same as a local run. Commit them.
"""

import subprocess
from pathlib import Path, PurePosixPath

import modal

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent if modal.is_local() else Path("/workspace")
REMOTE = PurePosixPath("/workspace")
TRITON_CACHE = "/cache/triton"


def _git_sha() -> str:
    if not modal.is_local():
        return ""
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
        return sha + ("-dirty" if dirty.strip() else "")
    except Exception:
        return ""


image = (
    modal.Image.from_dockerfile(ROOT / "Dockerfile", context_dir=ROOT)
    .entrypoint([])
    .env({"TRITON_CACHE_DIR": TRITON_CACHE, "PYTHONPATH": f"{REMOTE}:{REMOTE}/prep/kernel",
          "GIT_SHA": _git_sha()})
    # add_local_dir must come last: files are attached at container start, not built in
    .add_local_dir(
        ROOT,
        remote_path=str(REMOTE),
        ignore=[".git", ".git/**", ".claude", ".claude/**", ".venv", ".venv/**", ".env",
                "papers", "papers/**", "data/mica", "data/mica/**", "results", "results/**",
                "**/__pycache__", "**/__pycache__/**"],
    )
)
app = modal.App("FlashGNM_bench_sim", image=image)
triton_vol = modal.Volume.from_name("triton_kernels", create_if_missing=True)


@app.function(volumes={TRITON_CACHE: triton_vol}, timeout=6 * 3600)
def bench_remote(argv: list[str]) -> tuple[str, dict[str, str]]:
    import sys

    sys.path[:0] = [str(REMOTE), str(REMOTE / "prep" / "kernel")]
    sys.path.insert(0, str(REMOTE / "bench"))
    import bench_sim

    out = bench_sim.main(argv)
    triton_vol.commit()
    return out.name, {p.name: p.read_text() for p in out.iterdir() if p.is_file()}


@app.local_entrypoint()
def run(gpu: str = "H100!", args: str = "", tag: str = ""):
    import shlex

    argv = shlex.split(args) + (["--tag", tag] if tag else [])
    name, files = bench_remote.with_options(gpu=gpu).remote(argv)
    out = ROOT / "results" / name
    out.mkdir(parents=True, exist_ok=True)
    for fname, text in files.items():
        (out / fname).write_text(text)
    print(files.get("summary.md", ""))
    print(f"wrote {out}")
