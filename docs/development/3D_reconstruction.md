# 3D reconstruction from serial sections (`manta.reconstruction`)

`manta.reconstruction.reconstruct(adata)` places the cells of a serially sectioned specimen in one 3D frame. Every
section keeps its own shape up to one rigid transform (rotation + translation) and a smooth, small displacement
field; z comes from the section order and thickness, so the sections are never reordered or tilted.

```python
import manta
res = manta.reconstruction.reconstruct(adata, section_key="section_id", z_key="section_z_raw_um",
                                       spatial_key="spatial_raw", label_key="cell_type")
adata.obsm["spatial_manta_3d"]          # (n_cells, 3) coordinates in um
res.sections()                          # per section: pose, end zone, moved
res.end_log                             # per end section: scores, shift, rotation, decision
res.stacks["rigid"], res.stacks["ends"], res.stacks["warp"]   # every stage
```

## Input

| field | meaning |
|---|---|
| `obs[section_key]` | section of each cell |
| `obs[z_key]` | z of the section in um (one value per section); gives the order and the spacing |
| `obsm[spatial_key]` | slide coordinates in um; an optional 3rd column is the depth of the cell inside its section (added to z, clipped to `[0, depth_clip]`) |
| `X` or `obsm[embedding_key]` | expression; one embedding shared by all sections |
| `obs[label_key]` (optional) | labels such as cell types, only used to choose between two candidate poses in the chain |

The embedding (step 1) is a PCA of the z-scored expression of all cells (`n_comps=30`, optionally a gene subset
`genes`). Any shared embedding can be passed instead with `embedding_key`.

## Conventions

- A `SectionStack` holds one `Similarity` per section, `T_k(x) = s R x + t` (in practice `s = 1`), mapping the slide
  coordinates of section `k` into the common stack frame. Points are rows.
- Sections are indexed by their stack position `k = 0 .. K-1` in order of z.
- The **anchor** is the section with most cells in the middle half of the stack. It is centred at the origin and
  held fixed by every joint refinement, which fixes the gauge of the whole stack.
- **Section pairs** `(k, l, w)`: every pair with `0 < z_l - z_k <= max_dz`, weight `w = spacing / (z_l - z_k)`.
  Default `max_dz = 2.05 x spacing` (spacing = median distance between neighbouring sections): each section is
  tied to its neighbour (weight 1) and its next neighbour (weight 0.5); across a lost section the pair that spans
  the gap remains.

## Step 2: chain of pairwise fits

Every section `k + 1` is fitted onto section `k` with `manta.align.rigid_joint`:

1. **Rotation scan.** Both sections are rasterised into blurred expression images (features: the unit-normalised
   embedding projected on 10 shared axes). For every rotation (2 deg steps) all translations are scored at once by
   FFT cross-correlation; the best (angle, shift) is the global optimum on that grid. Every cell votes; there are
   no hard matches and no RANSAC.
2. **Optimal-transport refinement.** An unbalanced entropic OT coupling with cost
   `|T(x) - y|^2 / sigma^2 + alpha (1 - cos(e_x, e_y))` is alternated with a weighted Kabsch fit, sigma annealing
   from coarse to fine.

The plain centroid overlay is kept as a second candidate; the candidate with the higher agreement (same label of the
nearest cell within 8 um when `label_key` is given, otherwise the mean expression cosine of the nearest cells) is
used. The pair fits are composed outward from the anchor.

## Step 3: joint rigid refinement

A chain fits each pair on its own, so one poor pair displaces every section beyond it. Here all poses are refined
together against **all** section pairs:

```
E = sum_(k,l,w)  w  sum_ij  P_ij |T_k(x_i) - T_l(y_j)|^2
```

`P` is the unbalanced entropic OT coupling between the cells of sections `k` and `l` (same cost as above, `alpha = 3`,
`epsilon = 0.05`, `rho = 0.5`; at most `max_points = 1200` random cells per section, fixed during the fit so the
Sinkhorn potentials can be warm-started). The fit alternates

1. the couplings of all pairs at the current poses, and
2. Gauss-Newton steps on all `3K` pose parameters at once. Each coupled cell is paired with its P-weighted mean
   partner, in both directions, so neither section of a pair is favoured. The rotation update turns a section about
   its own centroid.

The length scale `sigma` goes 25, 15, 10, 6 um (2 coupling rounds per sigma, 3 Gauss-Newton steps each). The
anchor is held fixed.

## Step 4: end sections

### The problem

The outermost sections are grazing cuts through the surface of the specimen and consist mostly of the outer cell
layer. The neighbour also has that layer, but only along its rim. Matched by expression, an end section therefore
fits best onto the neighbour's rim and slides off to the side (tens of um), and small end sections are free to turn.

### End zone

From each end of the stack inward, the contiguous run of sections with fewer than `zone_frac x` (robust maximum
cell count) is the end zone; the robust maximum is the median of the three largest sections and `zone_frac = 0.4`.
The first section at or above the threshold, and everything inside it, is never re-placed by this step. Consequences:

- no fixed number of end sections: the zone holds as many sections as the stack tapers over;
- a narrowing in the middle of the stack (tissue that gets smaller and then larger again) is not reached, because
  the walk stops at the first large section;
- a damaged interior section that lost part of its cells usually stays above the threshold.

### Prediction

Sections are processed from the inside outward (first the zone section next to the large sections, then the next one
out), so each prediction is built from sections that are already decided. For section `k` with inner neighbours
`N1` (nearest) and `N2`, both at their current pose:

- each neighbour is rasterised into an image `R` with the 10 feature channels plus one occupancy channel (cells
  present), pixel 4 um, Gaussian blur 10 um, on a grid centred on `N1`;
- the prediction extrapolates the images one step outward along z:

```
T = R1 + w (trend(z_k) - R1)        trend = least-squares line through the inner images along z
                                    (two sections: trend(z_k) = R1 + f (R1 - R2),  f = (z_k - z_1)/(z_1 - z_2))
```

with `w = weight = 0.5` and the extrapolation distance capped at 2 section spacings. `n_predict` (default 2) sets
how many inner sections the trend is fitted to. `w = 0` would be the plain neighbour match (which carries the rim
bias); `w = 1` the full linear trend. Where the tissue disappears outward the prediction falls towards zero (or
below) at the rim, so a placement hanging over the rim is penalised; where the tissue grows the prediction grows
with it. No shape model is involved.

### Search

The section is rasterised the same way and every rotation (3 deg steps) and every translation is scored by the
normalised cross-correlation with `T` (one FFT per rotation):

```
J = <S, T> / (|S| |T|)
```

The section's centroid may move at most one rms radius of `N1` away from `N1`'s centroid. The best pose has score
`J_new`; the current pose has `J_current`.

### Orientation default

Sections are mounted and imaged in roughly the same orientation, so the rotation of a section's stack pose relative
to its slide coordinates is similar for all sections. The common angle `theta0` is the cell-weighted circular mean of
the rotations of the sections **outside** the end zones (recomputed on those within 90 deg of the first estimate).

Each end-zone section is scanned twice: with the rotations within 90 deg of `theta0` ("upright") and with those
within 90 deg of `theta0 + 180` ("flipped"). The flipped half is used only when its best score beats the upright one
by more than `orient_margin = 0.1`. A small end section looks nearly the same after a half turn and its two scores
differ by a few hundredths, so it takes the common orientation; a section that sits flipped is turned back. A
genuinely flipped section with clear expression evidence can still be placed flipped. `orient_margin=None` switches
this off (all rotations in one scan).

### Decision

- **Orientation correction** (the current pose lies in the other half than the chosen one): the section moves to
  the best pose of the chosen half, unless that pose scores more than `orient_margin` below the current one.
- **Otherwise** the section moves when `J_new > J_current + tol` (`tol = 0.02`) and the pose changes by more than
  `min_move_um = 5` um or 5 deg.
- **Data guard** (both cases): the move is refused when the plain match with `N1` (the same score against the raw
  image `R1` instead of the prediction) drops by more than `max_data_loss = 0.05`; the prediction may overrule a small
  loss of direct agreement with the neighbour, not a large one.
- Sections with fewer than 10 cells are skipped.

If any section moved, a short joint rigid refinement of the whole stack follows (sigma 10, then 6 um) so the moved
sections settle against their partners; interior sections change by about a micrometre in that polish.

`res.end_log` has one row per examined section: `k`, `n_cells`, `score_current`, `score_new`, `score_upright`,
`score_flipped`, `orientation`, `orientation_correction`, `plain_current`, `plain_new`, `shift_um`, `rotation_deg`,
`moved`. `res.stacks["ends"].zone` lists the end-zone positions and `.theta0` the common angle.

## Step 5: warp

Every section gets a smooth 2D displacement field `D_k`, bilinear on a regular grid with node spacing 20 um, so a
cell at rigid position `p` moves to `p + D_k(p)`. Data term (second order): each cell of section `k` is pulled to
the z-weighted interpolation of its partners in the closest sections below (`l`) and above (`u`):

```
p + D_k(p)  ~  (1 - f) (y_l + D_l(y_l)) + f (y_u + D_u(y_u)),     f = (z_k - z_l) / (z_u - z_l)
```

where `y_l`, `y_u` are the P-weighted mean partners from the OT couplings (also with the next-nearest sections, half
weight). A steady taper or drift along the stack costs nothing; only kinks between sections are removed. The end
sections have partners on one side only and are not warped by the data term. Regularisers per section, scaled by the
mean data weight per node: bending energy of the node values (`bend = 0.3`; zero for any affine field), a membrane
term (`stretch = 0.3`; resists a uniform stretch or squeeze, so a section next to a partial one is not shrunk to its
size) and a small pull to zero (`shrink = 0.01`). For fixed couplings the optimum over all sections is one sparse
linear solve; couplings are recomputed at sigma 15, 10, 6 um (at most 2000 cells per section). Finally the scale
common to all sections (barycentric partners at the tissue edge lie slightly inward, which shrinks every section by
~1-2 %) is removed by rescaling each section about its centroid; size differences between sections are kept.

## Parameters

| step | argument (`*_kw`) | default | effect |
|---|---|---|---|
| all | `max_dz` | 2.05 x spacing | which section pairs are tied together |
| 1 | `n_comps`, `genes`, `embedding_key` | 30, all, None | shared expression embedding |
| 2 | `chain_kw`: `alpha`, `radius`, rigid_joint keywords | 3.0, 8 um | pairwise fits |
| 3 | `rigid_kw`: `sigmas`, `alpha`, `epsilon`, `rho`, `max_points` | (25, 15, 10, 6), 3.0, 0.05, 0.5, 1200 | joint rigid refinement |
| 4 | `end_kw`: `zone_frac` | 0.4 | how far inward the end zone may reach |
| 4 | `end_kw`: `weight` | 0.5 | prediction (1) vs plain neighbour (0) |
| 4 | `end_kw`: `n_predict` | 2 | inner sections the prediction trend is fitted to |
| 4 | `end_kw`: `orient_margin` | 0.1 | evidence needed to place an end section against the common orientation (None = off) |
| 4 | `end_kw`: `tol`, `min_move_um`, `max_data_loss` | 0.02, 5, 0.05 | move rules |
| 5 | `warp_kw`: `spacing`, `bend`, `stretch`, `shrink`, `max_points` | 20, 0.3, 0.3, 0.01, 2000 | warp stiffness |
| | `end_sections`, `warp` | True, True | switch steps off |
| | `init` | None | start from a given `SectionStack` instead of the chain |

## Outputs

- `adata.obsm[key_added]` (default `spatial_manta_3d`): 3D coordinates (um), z centred on the median.
- `adata.uns[key_added]`: `sections` (per-section table), `end_log`, `params`.
- `Reconstruction`: `stacks` (`chain`, `rigid`, `ends`, `warp`), `final`, `pairs`, `end_log`, `sections()`, `xyz()`.
  `SectionStack.save(path)` / `load_transforms(path)` store the rigid poses; `WarpedStack.save(path)` also the warp.

## Assumptions and limits

- Section order and z are correct; no section is mirrored; sections may be missing (the pair across the gap
  remains).
- Expression is comparable between sections (one shared embedding).
- The orientation default assumes sections were mounted in roughly the same orientation. Without that, set
  `orient_margin=None`.
- An end section next to a damaged or partial section is predicted from that damaged section and can still be
  placed off. Damaged interior sections are not detected or excluded.
- The warp corrects kinks, not a wrong rigid start (e.g. a section turned by 180 deg in the middle of the stack).

## Code

| module | content |
|---|---|
| `_stack.py` | `Similarity`, `SectionStack`, `section_pairs` |
| `_features.py` | PCA embedding, scan features, blurred expression images |
| `_chain.py` | `fit_pair`, `chain` (step 2) |
| `_joint.py` | OT couplings, `refine_rigid` (step 3) |
| `_ends.py` | `end_zone`, `predict_and_place`, `place_end_sections` (step 4) |
| `_warp.py` | `Grid`, `WarpedStack`, `refine_warp` (step 5) |
| `_pipeline.py` | `reconstruct`, `Reconstruction` |
| `../matching/_joint_rigid.py` | rotation scan, unbalanced Sinkhorn, OT refinement used by `rigid_joint` and here |
