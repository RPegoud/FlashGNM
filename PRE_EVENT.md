# What existed before the event

SoTA Scientific Parallelism Hackathon, 21–22 Nov 2026, London. The organisers allow
pre-existing code as long as it is disclosed clearly. This file is that disclosure. It was
written on 2026-11-18, before the event.

**Summary.** A working prototype existed before the event: an O(n)-per-step GNM simulator
as a fused Triton kernel with its test harness and prepared data. The prototype kernel and its tests are in `prep/kernel/`
as a reference. At the event we rebuild the kernel in this repository from `SPEC.md` with
the changes in SPEC §4.0, build the evaluation and landscape pipeline, and run the science in
SPEC §12.

## Where the prototype lives

- `prep/kernel/` in this repo: the Triton kernel, its test harness, the PyTorch reference
  (`RowSampled`) and the dense baseline (`LeanDense`), copied from the fork with import paths
  adjusted.
- The fork of `EdwardJamesYoung/GenerativeNetworkModels` at
  `github.com/RPegoud/GenerativeNetworkModels`, prototype code under `bench/`. Pre-event
  commits, with git author dates:

  | Commit | Date | Content |
  | --- | --- | --- |
  | `1df2d88` | 2026-10-28 | CPU benchmarks of the toolbox, incremental C and an O(n) step (`bench_gnm.py`, `bench_lean.py`), Modal runner skeleton |
  | `83d9edb` | 2026-11-01 | Triton kernel `TritonGNM`, its test harness, `row_sampled.py` and `lean_dense.py` in their current form |
  | `e98948c` (branch `dense-targets`) | 2026-11-18 | Spec: dense probabilistic targets (MICA-MICs) |
  
## Inventory

### Simulation (in `prep/kernel/`)

| Component | File | State before the event |
| --- | --- | --- |
| Fused Triton kernel, one program per simulation, persistent over T steps | `kernel.py` (`TritonGNM`) | Passes its tests on an H100 (38 passed, 2026-11-02) |
| Test harness: teacher forcing, shared-uniform trajectories, completion, chi-square sampling, drift, KS against the toolbox | `test_kernel.py`, `conftest.py`, `modal_tests.py` | n ≤ 130 |
| O(n)/step PyTorch reference (incremental C, maintained P and row sums, two-stage sampling, batched over (eta, gamma)) | `row_sampled.py`, `bench_gnm.py` | CPU and GPU, eager or with CUDA graphs |
| Dense baseline (the toolbox's maths without input checks or host syncs) | `lean_dense.py` | The fair baseline |
| Toolbox comparison helpers | `assess.py`, `AAL_DISTANCES.pt` (from the toolbox, MIT) | |

Headline numbers (H100, simulation only, fork `bench/results/26-11-02_08-10-19`, median of 5):
- n = 1000, B = 1024, T = 5,000: 0.181 s per batch, ~5,660 sims/s.
- n = 1000, B = 1024, T = 11,000 (Griffa density): 0.509 s, ~2,010 sims/s.
- n = 2000, B = 1024, T = 5,000: 0.457 s, 37 GB peak.
- Baselines, extrapolated from 200 steps: dense baseline ~2 sims/s and toolbox ~0.13 sims/s
  at n = 1000, B = 1024, T = 5,000.

### Not prototyped before the event

- The additive wiring rule with maintained maxima (SPEC §4.1, H9), and the exponential
  distance term.
- The kernel changes in SPEC §4.0 (no stored P, int16 C, shared log D, T per simulation).
- Tiled rows for large n.

### Data (in this repo, prepared before the event)

- `data/griffa/`: Griffa et al. (2019), 70 subjects, Lausanne 68–1000 cortical regions.
  Converted on 2026-11-02 (`prep/griffa/`), packaged and checked on 2026-11-18
  (`prep/griffa_package.py`).
- `data/mica/`: MICA-MICs, 50 subjects, Schaefer 100–1000, probabilistic tractography.
  Built and checked on 2026-11-18 (`prep/mica_build.py`).

### Setup

- `pyproject.toml`, `uv.lock`, `Dockerfile`, `.devcontainer/`: the environment (CUDA 12.8,
  torch 2.8, triton 3.4, the toolbox pinned to upstream commit `c6eabe1`), adapted from the
  fork.

### Documents

- `SPEC.md` (2026-11-10, extended 2026-11-18): design, tests, hypotheses, schedule.

## Shared with others before the event

- **Toolbox maintainers** (E. Young, W. Mills, F. Poli), email of 2026-11-13: described the
  kernel's approach (O(n) steps, fused Triton kernel, batching over parameters), the
  speedups, and the planned experiments. The code itself was not sent; we offered a pull
  request after the event. F. Poli replied, suggesting probabilistic (dense) targets and
  per-subject variance estimates. Both are in the plan (H8, H5), credited to him.
- **Organisers:** asked about pre-built code; reply: allowed if disclosed clearly.

## Rules for event work

- Everything outside `prep/` is written at the event. `prep/` holds pre-event work: data
  preparation and the prototype kernel with its tests.
- Code taken from `prep/kernel/` into the event code is committed with a message starting
  `port:` that names the source file. Everything else is new.
- Benchmarks and results in the final presentation come from runs made at the event, with
  their result directories committed.
