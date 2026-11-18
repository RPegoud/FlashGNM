# FlashGNM: fast generative network model fitting on GPU

Event repository for the SoTA Scientific Parallelism Hackathon (21–22 Nov 2026, London).

- `SPEC.md`: what we build, how we test it, the hypotheses, and the schedule (§12).
- `PRE_EVENT.md`: what existed before the event, and the rules for marking event work.
- `data/`: processed connectome targets (Griffa et al., MICA-MICs). See `data/README.md`.
- `prep/`: pre-event work. `prep/kernel/` holds the prototype Triton kernel and its tests
  (reference only); the other scripts produced `data/`.

The model: a binary GNM grows a network one edge at a time with wiring probability
`P_ij ∝ D_ij^eta · K_ij^gamma` (matching index K), and a fit compares simulated networks with
a target connectome by the max of four KS statistics. Built on the
[GenerativeNetworkModels](https://github.com/EdwardJamesYoung/GenerativeNetworkModels)
toolbox (Young, Mills, Poli; MIT).

## Setup

The environment is CUDA 12.8, Python 3.12, torch 2.8 and triton 3.4 (linux/x86_64 only), with
the toolbox pinned to upstream commit `c6eabe1`. Managed with [uv](https://docs.astral.sh/uv/).

**On a GPU machine (e.g. the event's):**

```bash
uv sync --frozen
uv run python -c "import torch, triton, gnm; print(torch.__version__, triton.__version__, torch.cuda.is_available())"
cd prep/kernel && uv run pytest test_kernel.py -v -rs     # smoke test of the prototype kernel
```

**Devcontainer / Docker:** `Dockerfile` builds the same environment into `/opt/venv`
(`docker buildx build --platform linux/amd64 -t FlashGNM .`); `.devcontainer/` uses it,
with the repo mounted at `/workspace`. It reads secrets (e.g. Modal tokens) from `.env`,
which is git-ignored.

**On Modal** (uses the `Dockerfile` as the image; needs `modal token new` once):

```bash
uv run modal run prep/kernel/modal_tests.py::run_tests                       # H100 by default
uv run modal run prep/kernel/modal_tests.py::run_tests --args "test_kernel.py -v -rs --slow"
```

**Without a GPU:** `cd prep/kernel && TRITON_INTERPRET=1 uv run pytest test_kernel.py -k "12-3"`
runs the kernel in Triton's interpreter (slow: use small cases).

Note: the toolbox pins `wandb==0.15.12`, which doesn't build on Python 3.12. `pyproject.toml`
overrides it away; `import gnm` doesn't need it.

## Loading a target

```python
import numpy as np

z = np.load("data/griffa/scale033.npz")          # or data/mica/schaefer400.npz
n, D = int(z["n"]), z["dist_mm"]                   # D: n x n, symmetric, zero diagonal
s = 0                                              # subject index
m = z["sub"] == s
A = np.zeros((n, n), bool)
A[z["i"][m], z["j"][m]] = True
A |= A.T                                           # binary target; weights in z["w"][m]
```

For dense MICA targets, keep a subject's strongest k% of pairs by `w` (SPEC §7.1).
