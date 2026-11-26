# Changelog

## Branch `3D_reconstruction_lars`

### 1. Joint rigid alignment: rotation scan + optimal-transport refinement (replaces voxel matching + RANSAC)

New aligner `manta.align.rigid_joint(source, target, ...)`, an alternative to `manta.align.rigid` for 2D sections.

**Why.** `manta.align.rigid` first links every source voxel to its most similar target voxel by expression alone and
then lets RANSAC find the largest group of links that agree on one transform. When many voxels look alike (e.g.
ectoderm all around a section) most links are wrong and RANSAC can lock onto a wrong pose; for small sections there
are often too few matches and MANTA silently returns the identity.

**How it works** (`src/manta/matching/_joint_rigid.py`):
- `_scan_features`: the expression of both sets is unit-normalised and projected on `n_scan_features` shared
  uncentred principal axes, so inner products approximate cosine similarities.
- `_rotation_scan` (2D): for every angle in `angles` (default 0, 2, ..., 358 deg) the source is rotated about its
  centroid, both sets are rasterised into blurred expression images (pixel `scan_pixel_size`, Gaussian
  `scan_sigma`), and ALL translations are scored at once by FFT (phase-space) cross-correlation. The best
  (angle, shift) on that grid is the start pose: every cell votes, no hard links, no RANSAC. `peak_ratio` = best
  score more than 30 deg away / chosen score (near 1 = ambiguous rotation, e.g. a small round section).
- `_ot_refine` (any dimension): alternates an unbalanced entropic optimal-transport coupling
  (`_dense_unbalanced_sinkhorn`, log domain, warm-started; cost = |T(x) - y|^2 / sigma^2 + alpha (1 - cosine);
  `rho` sets how much mass may stay unmatched, `epsilon` the entropic blur) with a weighted Kabsch(-Umeyama) fit of
  all coupled pairs, the length scale annealing over `sigmas` (default 40, 25, 15, 10, 6). `max_points` subsamples
  each set. `allow_scaling` optional.
- `init=(R, t, s)` skips the scan and starts the refinement there (required in 3D).

**API / outputs** (`src/manta/api/_alignment.py`, `src/manta/align.py`):
- `manta.align.rigid_joint` stores its result like `rigid`: `source.uns['rigid_alignment']` with `R`, `t`, `s`
  (x -> s R x + t), `aligned_pts`, `history` (per sigma: rotation, transported mass, median residual, coupled cosine),
  `method='joint'` and `scan` (angles, scores, best angle, `peak_ratio`); `source.obsm['rigid']`.
- `manta.plot.rigid_joint_history(adata)` (`src/manta/plotting/_history.py`, exported in `src/manta/plot.py`): the
  scan score per angle and the refinement per sigma; `manta.plot.rigid_history` dispatches to it for
  `method='joint'` results.

### 2. 3D reconstruction from serial sections (`manta.reconstruction`)

New subpackage `src/manta/reconstruction/` with one entry point,
`manta.reconstruction.reconstruct(adata, section_key, z_key, spatial_key, ...)`, which places all cells of a serially
sectioned specimen in one 3D frame and writes `adata.obsm['spatial_manta_3d']`. Full description:
`docs/development/3D_reconstruction.md`.

Pipeline:
1. **Embedding** (`_features.py`): one PCA of the z-scored expression of all cells, shared by every section (or an
   existing embedding via `embedding_key`).
2. **Chain** (`_chain.py`): every section onto its neighbour with `manta.align.rigid_joint`; the centroid overlay is
   kept as a fallback candidate, chosen by label agreement (`label_key`) or expression agreement; fits composed
   outward from the anchor (largest section in the middle half).
3. **Joint rigid refinement** (`_joint.py`): all poses refined together against all section pairs within `max_dz`
   (default 2.05 x section spacing; weight spacing / dz): unbalanced entropic OT couplings between the cells of each
   pair (warm-started Sinkhorn from `matching/_joint_rigid.py`) alternated with Gauss-Newton steps on all poses,
   sigma 25 -> 6 um, anchor fixed.
4. **End sections** (`_ends.py`): the end zones (from each end inward, the run of sections with fewer than 0.4 x the
   robust maximum cell count) are re-placed from the inside outward against a prediction: the expression +
   occupancy images of the inner neighbours extrapolated one step along z (`weight` 0.5, `n_predict` 2), full
   rotation x translation scan by FFT normalised cross-correlation. Ambiguous rotations default to the common imaging
   orientation of the large sections (`orient_margin` 0.1: the half turn is used only with clear evidence). Move
   rules: score gain > 0.02 and pose change > 5 um / 5 deg, plain neighbour match may drop by at most 0.05; then a
   short joint rigid polish. Per-section log in `Reconstruction.end_log`.
5. **Warp** (`_warp.py`): one smooth bilinear displacement field per section (20 um grid), all sections in one sparse
   solve, second-order data term (pull to the z-weighted interpolation of the partners below and above, so a taper
   is kept), bending + membrane + shrink regularisers, common scale removed afterwards.

API: `reconstruct`, `Reconstruction` (stacks of every stage, `final`, `end_log`, `sections()`, `xyz()`),
`SectionStack` / `Similarity` (per-section poses, `xy`, `xyz`, `table`, `save` / `load_transforms`), and the steps
`chain`, `fit_pair`, `refine_rigid`, `end_zone`, `predict_and_place`, `place_end_sections`, `refine_warp`,
`WarpedStack`. `manta/__init__.py` imports the subpackage.
