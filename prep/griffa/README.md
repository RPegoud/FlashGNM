# Griffa connectome preparation

Targets for fitting GNMs at 68 to 1000 cortical regions, from Griffa, Alemán-Gómez and Hagmann,
"Structural and functional connectome from 70 young healthy adults" (Zenodo 2872624, CC BY 4.0).
Work done 2026-11-02, before the event.

## Findings

**The data.** `Individual_Connectomes.mat` (701 MB, md5 `87ef63debc0cbe44afc1e6cf9ab9717e`) holds
`connMatrices.SC` and `.FC`. Each has 5 Lausanne scales of shape n × n × 70 subjects.

- The matrices are **cortex only**, with n = 68 / 114 / 219 / 448 / 1000. The atlas's subcortical regions are not included.
- SC is exactly symmetric. Its diagonal is nonzero (self-loops, max ~0.11), so drop the diagonal before using it as a GNM target.
- Weights are fibre density: streamline counts normalised by length and region area.

| n    | edges / subject (mean ± sd) | density | edges in ≥ 50% of subjects |
| ---- | --------------------------- | ------- | -------------------------- |
| 68   | 611 ± 42                    | 26.8%   | 562                        |
| 114  | 1228 ± 80                   | 19.1%   | 1067                       |
| 219  | 2640 ± 166                  | 11.1%   | 2002                       |
| 448  | 5324 ± 334                  | 5.3%    | 3359                       |
| 1000 | 10786 ± 721                 | 2.16%   | 4824                       |

**Distances.** Distances are Euclidean, in mm, between region centroids of the Lausanne (Cammoun 2012) atlas in MNI152NLin2009aSym
space, as hosted by netneurotools. These are template centroids, not per-subject ones.

**Region order.** Griffa's order is the atlas id order with subcortex removed: R cortex first, then L. This was checked using only
distance against connection probability, so it doesn't rely on the region labels:

- **Global:** Spearman(distance, fraction of subjects with the edge) is −0.33 to −0.45 for the atlas order. For random
  within-hemisphere permutations it is −0.11 to −0.17 (sd ≤ 0.02). At n = 1000, connected pairs are 43 mm apart on
  average, against 75 mm under permutation. R-first beats L-first at every scale where the hemisphere sizes differ.
- **Local:** at n ≤ 448, every region is connected (in ≥ 50% of subjects) to at least one of its 5 nearest centroids.
  Deliberately mislabelled regions mostly drop to 0 or 1, so the test can detect mislabels.
- **Within gyrus:** a Hungarian reassignment of labels within each gyrus keeps 983/1000 regions in place at n = 1000 and
  446/448 at n = 448, improving the fit by only 0.13% and 0.02%. The few swaps are on the medial wall (cingulate, cuneus,
  lingual). At n = 1000, 37 regions miss all 5 nearest neighbours; most have very few edges at the ≥ 50% threshold.

**Target construction.** At high resolution, uniform consensus thresholding removes long-range edges. Share of edges longer than 90 mm:

| n    | all pairs | individual | consensus ≥ 0.3 | ≥ 0.5 | ≥ 0.75 |
| ---- | --------- | ---------- | --------------- | ----- | ------ |
| 68   | 37.6%     | 14.6%      | 14.8%           | 10.1% | 5.3%   |
| 219  | 44.8%     | 13.8%      | 11.2%           | 8.5%  | 4.2%   |
| 448  | 43.7%     | 11.8%      | 8.5%            | 4.7%  | 1.2%   |
| 1000 | 43.0%     | 9.6%       | 3.9%            | 1.1%  | 0.2%   |

Small parcels make long edges land in different parcels from subject to subject, so a consensus filter removes them. With a
uniform-consensus target, "resolution doesn't improve long-range recovery" could be an artefact of thresholding. Use
individual subjects, or a distance-dependent consensus (Betzel et al. 2019) that keeps the edge-length distribution. The
decision is still open.

**A second dataset: MICA-MICs** (Royer et al. 2022, Sci Data; full text read via PMC9477866).
- 50 healthy adults, with per-subject SIFT2 SC *and* edge-length matrices for Schaefer 100 to 1000, Glasser 360 and DK-based 100 to 400.
- Hosted on OSF `j532r` and CONP.
- Rows are L then R, each starting with a medial-wall entry to drop. Subcortical nodes are added to SC/FC.
- It would give an independent replication at 1000 regions, with tract lengths. Its data terms are not checked, and it is not downloaded.

## Pipeline

Data lives outside the repo, in `/workspace/data/` (the default paths).

```
modal run bench/griffa/convert_modal.py --out-dir /workspace/data/griffa   # download + sparse npz (loadmat OOMs locally)
python bench/griffa/centroids.py /workspace/data/lausanne_mni/MNI152NLin2009aSym /workspace/data/lausanne_mni  # needs nibabel
python bench/griffa/build_distances.py    # distance matrices + edge-count table
python bench/griffa/check_order.py        # region-order tests
python bench/griffa/long_range.py         # long-edge fractions by target type
```

The atlas for `centroids.py` is netneurotools' `atl-cammoun2012` MNI152NLin2009aSym package:
`https://osf.io/download/67326efaee12e85a3662f7c6/`, md5 `7cc281d6916baf78bb53b12806d28454`.

Outputs:
- `data/griffa/griffa_sc_scale{033,060,125,250,500}_sparse.npz`: strict upper triangle only, so self-loops are
  dropped. Fields are `i`, `j` (int16), `subject` (int8) and `w` (float32), plus `n` and `n_subjects`.
- `data/lausanne_mni/lausanne_scale*_centroids_mni.csv`: centroids for every atlas region, including subcortex.
- `data/lausanne_mni/griffa_order_scale*_cortex_dist_mm.npy`: float64, symmetric, zero diagonal (what the toolbox's
  checks require), in Griffa order.
