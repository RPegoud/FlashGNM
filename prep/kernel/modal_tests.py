"""Run the kernel tests on a Modal GPU, using the repo's Dockerfile as the image.

modal run prep/kernel/modal_tests.py::run_tests [--gpu H100!] [--args "test_kernel.py -v -rs --slow"]
"""

from pathlib import Path, PurePosixPath

import modal

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1] if modal.is_local() else Path("/workspace")
REMOTE = PurePosixPath("/workspace/prep/kernel")
TRITON_CACHE = "/cache/triton"

image = (
    modal.Image.from_dockerfile(ROOT / "Dockerfile", context_dir=ROOT)
    .entrypoint([])
    .env({"TRITON_CACHE_DIR": TRITON_CACHE, "PYTHONPATH": str(REMOTE)})
    # add_local_dir must come last: files are attached at container start, not built in
    .add_local_dir(HERE, remote_path=str(REMOTE), ignore=["**/__pycache__"])
)
app = modal.App("FlashGNM_kernel_tests", image=image)
triton_vol = modal.Volume.from_name("triton_kernels", create_if_missing=True)


@app.function(volumes={TRITON_CACHE: triton_vol}, timeout=3600)
def pytest_remote(args: list[str]) -> tuple[int, str]:
    import subprocess
    import sys

    p = subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", *args],
        cwd=REMOTE,
        capture_output=True,
        text=True,
        check=False,
    )
    return p.returncode, p.stdout + p.stderr


@app.local_entrypoint()
def run_tests(gpu: str = "H100!", args: str = "test_kernel.py -v -rs"):
    import shlex

    code, out = pytest_remote.with_options(gpu=gpu).remote(shlex.split(args))
    print(out)
    if code:
        raise SystemExit(code)
