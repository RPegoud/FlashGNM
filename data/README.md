# Data

Processed, validated structural connectome targets, prepared before the event (see
`../PRE_EVENT.md`). Checksums in `SHA256SUMS` (`sha256sum -c data/SHA256SUMS` from the repo root).

## Common format

Each `.npz` holds one parcellation scale for all subjects, as a sparse strict upper triangle:

| Field | Type | Meaning |
| --- | --- | --- |
| `n` | int | number of cortical regions |
| `subjects` | array | subject ids (Griffa: 0–69; MICA: `sub-HC001`, ...) |
| `i`, `j` | int16 | region indices, `i < j` |
| `sub` | uint8 | subject index into `subjects` |
| `w` | float32 | edge weight, > 0 (Griffa: fibre density; MICA: SIFT2 weight) |
| `dist_mm` | float32, n × n | Euclidean distance between template centroids (MNI), symmetric, zero diagonal |
| `hemi` | uint8, n | 0 = left, 1 = right |
| `names` | str, n | region labels |

Self-loops and non-cortical regions are removed. Binarise with `w > 0`, or threshold by `w`
(dense targets, SPEC §7.1).

## Griffa et al. (`griffa/scale{033,060,125,250,500}.npz`)

- **Source:** Griffa, Alemán-Gómez & Hagmann, "Structural and functional connectome from 70
  young healthy adults", Zenodo 2872624, **CC BY 4.0** (attribution required).
  `Individual_Connectomes.mat`, md5 `87ef63debc0cbe44afc1e6cf9ab9717e`.
- 70 subjects; Lausanne (Cammoun 2012) scales with n = 68 / 114 / 219 / 448 / 1000 cortical
  regions; deterministic tractography. Region order: atlas id order, **right hemisphere
  first**.
- Extra fields: `xyz` (centroids, mm), `yeo7`, `von_economo`.
- **Distances:** centroids of the netneurotools Cammoun2012 MNI152NLin2009aSym volumes
  (template, not per subject).
- Edges per subject: 611 ± 42 (n = 68, density 27%) to 10,786 ± 721 (n = 1000, density 2.2%).
- **Validation** (`griffa/validation.json`): source exactly symmetric; diagonal dropped;
  strict upper triangle, positive weights, no duplicates; distances symmetric with zero
  diagonal. Region order checked without labels: Spearman(distance, fraction of subjects
  with the edge) is −0.33 to −0.45, against −0.11 to −0.16 (sd ≤ 0.023) under
  within-hemisphere permutations. At n ≤ 448 every region connects (in ≥ 50% of subjects) to
  one of its 5 nearest centroids; 96% at n = 1000. A more thorough check (within-gyrus
  reassignment keeps 983/1000 regions in place) is in `prep/griffa/check_order.py`.
- **Caveat:** uniform consensus thresholding removes long-range edges at high resolution
  (SPEC §7). Fit individual subjects.
- Pipeline (`prep/griffa/`): `convert_modal.py` (the .mat needs > 1.5 GB to load, so it is
  converted on a Modal CPU container), `centroids.py`, `build_distances.py`,
  `check_order.py`, then `prep/griffa_package.py`.

## MICA-MICs (`mica/schaefer{100,200,400,1000}.npz` and `_length.npz`)

- **Source:** Royer et al. (2022), "An open MRI dataset for multiscale neuroscience", Sci
  Data; OSF `j532r`, `micapipe_MICs_v1.1.zip`, **CC0** (please cite the paper).
- 50 subjects; Schaefer 2018, 7-network order, left hemisphere first; micapipe v0.1.1, MRtrix
  iFOD2 probabilistic tractography (40M streamlines) with SIFT2 weights.
- `_length.npz` holds `length` (mean streamline length, mm) aligned with `i`, `j`, `sub` of
  the main file. It is split off to keep each file under 100 MB.
- **Distances:** CBIG Schaefer2018 MNI centroid tables (`..._7Networks_order_FSLMNI152_1mm.
  Centroid_RAS.csv`, MIT licence, © 2016 Computational Brain Imaging Group).
- Raw density: 99.5% (Schaefer 100), 96% (200), 86% (400), 59% (1000; min 38%).
- **Validation** (`mica/validation.json`): medial-wall rows are zero; tract length exists
  exactly where the weight does. Region order checked without labels: centroid distance vs
  each pair's mean tract length, Spearman 0.79–0.84, against 0.23–0.26 under
  within-hemisphere permutations. The 17-network order scores at chance (0.36), so the
  7-network order is right.
- Pipeline: `prep/mica_build.py`; density and thresholding analysis: `prep/mica_density.py`.

## Not included

- Raw downloads (Griffa .mat, 701 MB; MICA zip, 2.7 GB). The prep scripts document how to
  get them.
- Other probabilistic datasets surveyed (Rosen & Halgren, HCP-MMP 360; braingraph.org,
  Lausanne): SPEC §7.1.
