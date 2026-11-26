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
