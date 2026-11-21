"""Run tests/ (the harness against src/kernel.py) on a Modal GPU.

    modal run tests/modal_tests.py                               # H100, -v -rs
    modal run tests/modal_tests.py --args "tests -v -rs --slow"
"""

from pathlib import Path, PurePosixPath

import modal

ROOT = Path(__file__).resolve().parents[1] if modal.is_local() else Path("/workspace")
REMOTE = PurePosixPath("/workspace")
TRITON_CACHE = "/cache/triton"

image = (
    modal.Image.from_dockerfile(ROOT / "Dockerfile", context_dir=ROOT)
    .entrypoint([])
    .env({"TRITON_CACHE_DIR": TRITON_CACHE})
    .add_local_dir(
        ROOT,
        remote_path=str(REMOTE),
        ignore=[".git", ".git/**", ".claude", ".claude/**", ".venv", ".venv/**", ".env",
                "papers", "papers/**", "data", "data/**", "results", "results/**",
                "**/__pycache__", "**/__pycache__/**"],
    )
)
app = modal.App("FlashGNM_tests", image=image)
triton_vol = modal.Volume.from_name("triton_kernels", create_if_missing=True)


@app.function(volumes={TRITON_CACHE: triton_vol}, timeout=3600)
def pytest_remote(args: list[str]) -> tuple[int, str]:
    import subprocess
    import sys

    p = subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", *args],
        cwd=REMOTE, capture_output=True, text=True, check=False,
    )
    return p.returncode, p.stdout + p.stderr


@app.local_entrypoint()
def run_tests(gpu: str = "H100!", args: str = "tests -v -rs"):
    import shlex

    code, out = pytest_remote.with_options(gpu=gpu).remote(shlex.split(args))
    print(out)
    if code:
        raise SystemExit(code)
