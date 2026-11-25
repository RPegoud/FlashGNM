"""Run the toolbox PR's tests and benchmark on a Modal GPU.

The image is this repo's Dockerfile (torch 2.8, triton 3.4, the toolbox at c6eabe1);
the patched toolbox source is mounted and put first on PYTHONPATH. wandb is stubbed
(the toolbox imports it at module level; it doesn't build on Python 3.12 here).

    python toolbox_pr/modal_check.py --toolbox /path/to/patched/GenerativeNetworkModels
    python toolbox_pr/modal_check.py --toolbox ... --pytest "tests/test_flash_kernel.py -v" --bench ""
"""

import argparse
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
TRITON_CACHE = "/cache/triton"

app = modal.App("FlashGNM_toolbox_pr")
triton_vol = modal.Volume.from_name("triton_kernels", create_if_missing=True)


def image(toolbox: str) -> modal.Image:
    return (
        modal.Image.from_dockerfile(ROOT / "Dockerfile", context_dir=ROOT)
        .entrypoint([])
        .env({"TRITON_CACHE_DIR": TRITON_CACHE, "PYTHONPATH": "/toolbox/src:/stubs"})
        .run_commands(
            "mkdir -p /stubs/wandb && echo '# stub: tests never log' > /stubs/wandb/__init__.py"
        )
        .add_local_dir(toolbox, remote_path="/toolbox",
                       ignore=["**/__pycache__", "**/__pycache__/**", ".git", ".git/**"])
    )


def run(pytest_args: str, bench_args: str) -> tuple[int, str]:
    import shlex
    import subprocess
    import sys

    out, code = [], 0
    if pytest_args:
        p = subprocess.run([sys.executable, "-m", "pytest", "-p", "no:cacheprovider",
                            *shlex.split(pytest_args)],
                           cwd="/toolbox", capture_output=True, text=True)
        out.append(p.stdout + p.stderr)
        code |= p.returncode
    for args in filter(None, bench_args.split(";")):
        p = subprocess.run([sys.executable, "scripts/benchmark_flash.py", *shlex.split(args)],
                           cwd="/toolbox", capture_output=True, text=True)
        out.append(p.stdout + p.stderr)
        code |= p.returncode
    return code, "\n".join(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--toolbox", required=True, help="patched GenerativeNetworkModels checkout")
    p.add_argument("--gpu", default="H100!")
    p.add_argument("--pytest", default="tests/test_flash_kernel.py tests/test_flash_sweep.py "
                   "tests/test_clamp_summary.py -v -rs")
    p.add_argument("--no-slow", action="store_true", help="skip the slow statistical test")
    p.add_argument("--bench", default="--n 90 --steps 400 --sims 64 1024;"
                   "--n 1000 --steps 10000 --sims 64 1024 --toolbox-steps 10",
                   help="';'-separated argument lists for scripts/benchmark_flash.py")
    a = p.parse_args()
    fn = app.function(image=image(a.toolbox), gpu=a.gpu, timeout=3600,
                      volumes={TRITON_CACHE: triton_vol},
                      env={} if a.no_slow else {"GNM_RUN_SLOW": "1"},
                      serialized=True, name="run")(run)
    with modal.enable_output(), app.run():
        code, out = fn.remote(a.pytest, a.bench)
    print(out)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
