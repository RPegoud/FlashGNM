# Pull request to the GNM toolbox: flash backend

A pull request for
[EdwardJamesYoung/GenerativeNetworkModels](https://github.com/EdwardJamesYoung/GenerativeNetworkModels),
based on upstream `056221c` (master, 2026-11-25). It adds this repository's kernel to the
toolbox as an optional backend: `perform_sweep(..., backend="flash")`.

- `flash-backend.patch`: the change, as a unified diff against `056221c`.
- `files/`: the new and changed files, at their paths in the toolbox, for reading.
- `PR.md`: the pull request description.
- `modal_check.py`: runs the PR's tests and benchmark on a Modal GPU.

## Opening the pull request

1. Fork the toolbox on GitHub (the old fork `RPegoud/GenerativeNetworkModels` no longer
   exists).
2. Apply the patch on a branch:

   ```bash
   git clone https://github.com/<you>/GenerativeNetworkModels && cd GenerativeNetworkModels
   git checkout -b flash-backend 056221c
   git apply /path/to/FlashGNM/toolbox_pr/flash-backend.patch
   git add -A && git commit -m "Add flash backend: fused GPU kernel for binary matching-index GNMs"
   git push -u origin flash-backend
   ```

3. Open the pull request against `EdwardJamesYoung/GenerativeNetworkModels:master` with
   the text of `PR.md`.

## Checking it

On a CUDA machine with the toolbox environment, from the toolbox checkout:

```bash
pytest tests/test_flash_kernel.py tests/test_flash_sweep.py -v
GNM_RUN_SLOW=1 pytest tests/test_flash_kernel.py -v -k toolbox
python scripts/benchmark_flash.py --n 1000 --steps 10000 --sims 64 1024
```

From this repository, on Modal (H100):

```bash
python toolbox_pr/modal_check.py --toolbox /path/to/patched/GenerativeNetworkModels
```

Without a GPU: `TRITON_INTERPRET=1 pytest tests/test_flash_*.py` (slow; the comparison
against `GenerativeNetworkModel` is skipped).

## Relation to `src/kernel.py`

`src/gnm/fast/kernel.py` is `src/kernel.py` with these changes:

- `prob_offset` is a parameter;
- inf/nan weights are clamped as in the toolbox (`nan -> 0`, `inf -> 1e30`), with a count
  per simulation;
- `0 ** 0 = 1` for zero distances with `eta = 0`;
- seed networks;
- the distance diagonal is handled internally;
- `run()` raises if asked for more edges than are missing.

The results in `results/` were produced with `src/kernel.py`.
