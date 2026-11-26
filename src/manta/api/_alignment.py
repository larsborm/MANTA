import anndata as ad
import numpy as np
import torch

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple

from ..old.alignment._rigid import (
    _aggregate,
    _match_voxels,
    _apply_transform,
    _ransac
)
from ..matching._joint_rigid import (
    _scan_features,
    _rotation_scan,
    _ot_refine
)
from ..old.alignment._non_rigid import (
    _match,
    _ProbabilisticRegistration
)
from ..utils._progress import (
    _get_progress,
    _update_progress,
)
from ..utils._tensor_utils import (
    _get_device,
    _as_tensor,
    _from_tensor
)


def rigid(
    source: ad.AnnData,
    target: ad.AnnData,
    voxel_scales: Optional[List[int]] = None,
    min_points_per_voxel: int = 25,
    min_similarity: float = 0.0,
    mutual_matching: bool = True,
    ransac_iters: int = 2000,
    ransac_treshold_frac: float = 0.5,
    allow_scaling: bool = False,
    weight_by_voxel_count: bool = True,
    seed: int = 42,
    spatial_key: str = "spatial_manta",
    expression_key: str | None = None,
):
    generator = np.random.default_rng(seed)
    
    if voxel_scales is None:
        pts = np.asarray(target.obsm.get(spatial_key), dtype=float)
        extent = (pts.max(axis=1) - pts.min(axis=1)).max()
        voxel_scales = [extent / f for f in [5, 10, 20, 40]]

    d = np.asarray(source.obsm.get(spatial_key)).shape[1]
    R_total = np.eye(d)
    t_total = np.zeros(d)
    s_total = 1.0

    history: List[Dict[str, Any]] = []
    last_inlier_pair = None

    source.obsm["rigid"] = source.obsm.get(spatial_key)
    target.obsm["rigid"] = target.obsm.get(spatial_key)

    progress, _ = _get_progress(
        steps=len(voxel_scales) + 1,
        desc="Rigid Alignment"
    )

    for bin_size in sorted(voxel_scales, reverse=True):
        _update_progress(
            progress=progress, 
            message=f"Bin Size: {bin_size}"
        )

        # Revoxelize using the CURRENT aggregated transform
        src_transformed = _apply_transform(source.obsm.get(spatial_key), R_total, t_total, s_total)

        source_tmp = source.copy()
        source_tmp.obsm["rigid"] = src_transformed

        _aggregate(source_tmp, bin_size, min_points_per_voxel, spatial_key="rigid", expression_key=expression_key)
        _aggregate(target, bin_size, min_points_per_voxel, spatial_key="rigid", expression_key=expression_key)

        voxel_key = f"voxel_{bin_size}"
        src_vox = source_tmp.uns.get(voxel_key)
        tgt_vox = target.uns.get(voxel_key)

        src_idx, tgt_idx, sims = _match_voxels(
            src_vox["expr"],
            tgt_vox["expr"],
            mutual=mutual_matching,
            min_similarity=min_similarity
        )

        if len(src_idx) < d + 1:
            history.append(
                {
                    "bin_size": bin_size,
                    "status": "skipped; too few matches",
                    "n_matches": int(len(src_idx))
                }
            )

            continue


        src_pts = src_vox["centroid"][src_idx]
        tgt_pts = tgt_vox["centroid"][tgt_idx]
        match_weights = None
        if weight_by_voxel_count:
            match_weights = np.minimum(src_vox["counts"][src_idx], tgt_vox["counts"][tgt_idx])

        try:
            result = _ransac(
                source_pts=src_pts,
                target_pts=tgt_pts,
                weights=match_weights,
                n_iters=ransac_iters,
                threshold=ransac_treshold_frac * bin_size,
                allow_scaling=allow_scaling,
                generator=generator
            )
        except RuntimeError as e:
            history.append(
                {
                    "bin_size": bin_size,
                    "status": f"failed: {e}",
                    "n_matches": int(len(src_idx))
                }
            )

            continue

        # Compose delta (fit on already-transformed points) onto the running total
        R_total, t_total, s_total = (
            result.R @ R_total,
            result.s * (result.R @ t_total) + result.t,
            result.s * s_total
        )

        last_inlier_pair = (src_pts[result.inliers], tgt_pts[result.inliers])
        history.append(
            {
                "bin_size": bin_size,
                "status": "ok",
                "n_matches": int(len(src_idx)),
                "n_inliers": int(result.inliers.sum()),
                "inlier_ratio": float(result.inliers.mean()),
                "mean_cosine_sim_inliers": float(sims[result.inliers].mean())
            }
        )

    aligned_pts = _apply_transform(
        np.asarray(source.obsm.get(spatial_key), dtype=float), 
        R_total, 
        t_total, 
        s_total
    )

    source.obsm["rigid"] = aligned_pts
    target.obsm["rigid"] = target.obsm.get(spatial_key)

    source.uns[f"rigid_alignment"] = {
        "R": R_total,
        "t": t_total,
        "s": s_total,
        "aligned_pts": aligned_pts,
        "history": history,
        "last_inlier_pair": last_inlier_pair
    }

    _update_progress(
        progress=progress, 
        message="Finished"
    )


def rigid_joint(
    source: ad.AnnData,
    target: ad.AnnData,
    angles: Optional[List[float]] = None,
    scan_pixel_size: float = 5.0,
    scan_sigma: float = 10.0,
    n_scan_features: int = 10,
    refine: bool = True,
    sigmas: Tuple[float, ...] = (40, 25, 15, 10, 6),
    alpha: float = 3.0,
    epsilon: float = 0.05,
    rho: float = 0.5,
    n_inner: int = 3,
    n_sinkhorn_iters: int = 30,
    max_points: int = 1500,
    allow_scaling: bool = False,
    init: Optional[Tuple[np.ndarray, np.ndarray, float]] = None,
    seed: int = 42,
    spatial_key: str = "spatial_manta",
    expression_key: str | None = None,
    progress: bool = True,
):
    """Rigid alignment of `source` onto `target` that optimises correspondences and transform jointly
    (alternative to `rigid`, which matches voxels by expression first and filters the matches with RANSAC).

    1. Rotation scan (2D): every angle in `angles` (default 0, 2, ..., 358 deg) and every translation is scored by
       the FFT cross-correlation of blurred expression images (pixel `scan_pixel_size`, blur `scan_sigma`, in
       coordinate units; expression reduced to `n_scan_features` axes). Global optimum on that grid; no RANSAC.
    2. OT refinement (`refine`, any dimension): alternate an unbalanced entropic OT coupling with cost
       |T(x) - y|^2 / sigma^2 + alpha (1 - cosine(expression)) and a weighted Kabsch fit, sigma annealing over
       `sigmas`. `rho` sets how strongly every point must be matched (smaller = more partial overlap allowed),
       `epsilon` the entropic blur of the coupling, `max_points` the subsample size per set.

    `init` = (R, t, s) skips the scan and starts the refinement there (required for 3D).
    `scan['peak_ratio']` measures how ambiguous the pose is: the best score more than 30 deg away from the chosen
    angle relative to the chosen one (near 1 = ambiguous; restrict `angles` if a prior on the rotation exists).
    The result is stored like `rigid`: `source.uns['rigid_alignment']` with R, t, s (x -> s R x + t), aligned_pts,
    history (one entry per sigma), method='joint' and the scan (angles, scores); `source.obsm['rigid']`.
    """
    generator = np.random.default_rng(seed)

    def _expr(adata):
        X = adata.X if expression_key is None else adata.obsm.get(expression_key)
        X = X.toarray() if hasattr(X, "toarray") else X
        return np.asarray(X, dtype=float)

    src_pts = np.asarray(source.obsm.get(spatial_key), dtype=float)
    tgt_pts = np.asarray(target.obsm.get(spatial_key), dtype=float)
    src_expr, tgt_expr = _expr(source), _expr(target)
    d = src_pts.shape[1]

    if angles is None:
        angles = list(np.arange(0.0, 360.0, 2.0))

    bar = None
    if progress:
        bar, _ = _get_progress(steps=2 + (len(sigmas) if refine else 0), desc="Joint Rigid Alignment")

    scan = None
    if init is None:
        if d != 2:
            raise ValueError("rigid_joint: the rotation scan needs 2D coordinates; pass init=(R, t, s) for 3D")
        fs, ft = _scan_features(src_expr, tgt_expr, n_scan_features, seed=seed)
        scan = _rotation_scan(src_pts, fs, tgt_pts, ft, angles, pixel_size=scan_pixel_size, sigma=scan_sigma)
        R, t, s = scan["R"], scan["t"], scan["s"]
    else:
        R, t, s = np.asarray(init[0], float), np.asarray(init[1], float), float(init[2])
    _update_progress(progress=bar, message="Rotation scan" if scan is not None else "Init")

    history: List[Dict[str, Any]] = []
    if refine:
        ref = _ot_refine(
            src_pts, src_expr, tgt_pts, tgt_expr, R, t, s,
            sigmas=sigmas, alpha=alpha, epsilon=epsilon, rho=rho, n_inner=n_inner,
            n_sinkhorn_iters=n_sinkhorn_iters, max_points=max_points, allow_scaling=allow_scaling,
            generator=generator, progress=bar
        )
        R, t, s, history = ref["R"], ref["t"], ref["s"], ref["history"]

    aligned_pts = _apply_transform(src_pts, R, t, s)
    source.obsm["rigid"] = aligned_pts
    target.obsm["rigid"] = target.obsm.get(spatial_key)

    source.uns["rigid_alignment"] = {
        "method": "joint",
        "R": R,
        "t": t,
        "s": s,
        "aligned_pts": aligned_pts,
        "history": history,
        "scan": None if scan is None else {
            "R": scan["R"], "t": scan["t"], "angle": scan["angle"],
            "angles": scan["angles"], "scores": scan["scores"], "peak_ratio": scan["peak_ratio"]
        },
    }

    if bar is not None:
        _update_progress(progress=bar, message="Finished")


def non_rigid(
    source: ad.AnnData,
    target: ad.AnnData,

    spatial_key: str = "spatial_manta",

    embedding_key: str | None = None,
    clustering_key: str | None = None,

    top_n_clusters: int = 5,
    top_k_matches: int = 5,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 1.0,
    temperature: float = 1.0,

    # OT Hyperparameters
    epsilon: float = 0.05,
    rho_src: float = 1.0,
    rho_tgt: float = 1.0,
    num_sinkhorn_iters: int = 50,

    # Probabilistic registration parameters
    scales: list[int] = [4, 8, 16],
    regularisation_shape: Literal["bending", "membrane", "combined"] = "bending",

    pi0_init: float = 0.8,
    sigma_in_init: float | None = None,
    alpha_init: float = 1.0,
    alpha_max: float = 1e6,
    
    barrier_alpha: float = 1e-3,
    barrier_lr: float = 0.5,
    barrier_steps: int = 3,

    tolerance: float = 1e-4,
    patience: int = 5,
    min_iters: int = 5,
    n_iters: int = 20,

    discontinuity_aware: bool = False,
    allow_tears: bool = True,
    allow_folds: bool = True,
    kappa_tear: float | None = None,
    kappa_fold: float | None = None,
    fold_barrier_suppression: bool = True
):
    device = _get_device()

    # Extract embedding
    src_embedding_dict = source.uns.get(embedding_key)
    tgt_embedding_dict = target.uns.get(embedding_key)
    
    if src_embedding_dict is None:
        raise ValueError(
            f"expected embedding to be of type `dict`, got `None`"
        )
    if tgt_embedding_dict is None:
        raise ValueError(
            f"expected embedding to be of type `dict`, got `None`"
        )

    # Extract clustering
    src_clustering_dict = source.uns.get(clustering_key)
    tgt_clustering_dict = target.uns.get(clustering_key)

    if src_clustering_dict is None:
        raise ValueError(
            f"expected clustering to be of type `dict`, got `None`"
        )
    if tgt_clustering_dict is None:
        raise ValueError(
            f"expected clustering to be of type `dict`, got `None`"
        )

    internal_spatial_key = spatial_key

    for scale in scales:
        transport_dict = _match(
            source=source,
            target=target,

            src_embedding_dict=src_embedding_dict,
            tgt_embedding_dict=tgt_embedding_dict,

            src_clustering_dict=src_clustering_dict,
            tgt_clustering_dict=tgt_clustering_dict,

            spatial_key=internal_spatial_key,
            
            top_n_clusters=top_n_clusters,
            top_k_matches=top_k_matches,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            temperature=temperature,
            epsilon=epsilon,
            rho_src=rho_src,
            rho_tgt=rho_tgt,
            num_sinkhorn_iters=num_sinkhorn_iters
        )

        # Set matching dict for potential plotting
        source.uns["matching"] = transport_dict
        target.uns["matching"] = transport_dict

        src_anchors = src_embedding_dict["pts"]
        tgt_indices = transport_dict["target_idx"]
        tgt_scores  = transport_dict["scores"]
        tgt_anchors = _as_tensor(target.obsm[spatial_key], dtype=torch.float32, device=device)[tgt_indices]

        registration = _ProbabilisticRegistration(
            l_x=scale,
            l_y=scale,
            regularisation_shape=regularisation_shape,

            pi0_init=pi0_init,
            sigma_in_init=sigma_in_init,
            alpha_init=alpha_init,
            alpha_max=alpha_max,

            barrier_alpha=barrier_alpha,
            barrier_lr=barrier_lr,
            barrier_steps=barrier_steps,

            tolerance=tolerance,
            patience=patience,
            min_iters=min_iters,
            n_iters=n_iters,

            discontinuity_aware=discontinuity_aware,
            allow_tears=allow_tears,
            allow_folds=allow_folds,
            kappa_tear=kappa_tear,
            kappa_fold=kappa_fold,
            fold_barrier_suppression=fold_barrier_suppression
        )

        result = registration.fit(
            src_x=src_anchors,
            tgt_x=tgt_anchors,
            tgt_scores=tgt_scores,
            use_softmax=True
        )

        src_untransformed = _as_tensor(source.obsm[internal_spatial_key], dtype=torch.float32, device=device)
        src_transformed = registration.apply_deformation(
            x=src_untransformed, 
            result=result
        )

        internal_spatial_key = f"nonrigid_{scale}"

        source.obsm[internal_spatial_key] = _from_tensor(src_transformed)
        target.obsm[internal_spatial_key] = target.obsm[spatial_key]    # This doesn't need the internal key

        source.uns[internal_spatial_key] = result
        target.uns[internal_spatial_key] = result



from ..matching._prestack import _prealign_stack
from ..models._encoder import ExpressionEncoder
from ..models._field import CanonicalField
from ..models._ffd import MultiscaleFFD
from ..training._cavi import MantaModelTrainer, SliceData


@dataclass
class MantaResult:
    registered_x:       List[torch.Tensor]
    embeddings:         List[torch.Tensor]
    inlier_rates:       List[torch.Tensor]
    field:              CanonicalField
    encoder:            ExpressionEncoder
    deformation:        MultiscaleFFD
    origin:             torch.Tensor
    h:                  float
    elbo_hist:          List[float]

    def apply_deformation(
        self,
        x: torch.Tensor,
        slice_id: int
    ) -> torch.Tensor:
        return self.deformation.apply_full(
            x=x,
            slice_id=slice_id,
            origin=self.origin,
            h=self.h
        )

    @torch.no_grad()
    def query_field(
        self,
        x_canonical: torch.Tensor
    ) -> torch.Tensor:
        return self.field(x_canonical)


class MantaRegistration:
    def __init__(
        self,
        n_scales: int = 3,
        l_init: int = 8,
        l_final: int = 32,
        latent_dim: int = 64,
        inter_slice_distance: Optional[float] = None,
        **trainer_kwargs
    ):
        self.device = _get_device()

        self.n_scales = n_scales
        self.l_init = l_init
        self.l_final = l_final
        self.latent_dim = latent_dim
        self.inter_slice_distance = inter_slice_distance
        self.trainer_kwargs = trainer_kwargs

    def _make_grid_shapes(self, D: int) -> List[Tuple[int,...]]:
        shapes = []
        for i in range(self.n_scales):
            t = i / max(self.n_scales - 1, 1)
            L = int(round(self.l_init + t * (self.l_final - self.l_init)))
            shapes.append(tuple([L] * D))
        return shapes

    def _make_slice_data(
        self, 
        slices: List[ad.AnnData],
        spatial_key: str,
        expression_key: str | None = None,
    ) -> List[SliceData]:
        if spatial_key is None:
            raise ValueError(f"spatial_key must be provided to align slices")
        return [
            SliceData(
                x=_as_tensor(slice.obsm[spatial_key], dtype=torch.float32, device=self.device),
                expr=_as_tensor(slice.X, dtype=torch.float32, device=self.device) \
                    if expression_key is None else _as_tensor(slice.obsm[expression_key], dtype=torch.float32, device=self.device) 
            )
            for slice in slices
        ]

    def fit(
        self,
        slices: List[ad.AnnData],
        spatial_key: str,
        expression_key: str | None = None,
        verbose: bool = True
    ) -> MantaResult:
        slices = self._make_slice_data(
            slices=slices,
            spatial_key=spatial_key,
            expression_key=expression_key
        )

        D_in = slices[0].x.shape[1]
        if D_in == 2:
            spacing = self.inter_slice_distance

            if spacing is None:
                # Heuristic - mean NN spacing along x
                x = slices[0].x
                dd = torch.cdist(x[:1000], x[:1000])
                dd.fill_diagonal_(float("inf"))
                spacing = float(dd.min(dim=1).values.median().item())

                if verbose:
                    print(f"Using inter-slice spacing: {spacing}")
            paired = [(s.x, s.expr) for s in slices]
            stacked = _prealign_stack(paired, inter_slice_distance=spacing)
            slices = [SliceData(x=stacked[k], expr=slices[k].expr) for k in range(len(slices))]

            D = 3
        else:
            D = D_in

        G = slices[0].expr.shape[1]
        grid_shapes = self._make_grid_shapes(D=D)

        trainer = MantaModelTrainer(
            D=D,
            G=G,
            n_slices=len(slices),
            grid_shapes=grid_shapes,
            latent_dim=self.latent_dim,
            **self.trainer_kwargs
        )
        out = trainer.fit(slices=slices, verbose=True)

        return MantaResult(
            registered_x=out["canonical"],
            embeddings=out["embeddings"],
            inlier_rates=out["inlier_rates"],
            field=out["field"],
            encoder=out["encoder"],
            deformation=out["deformation"],
            origin=out["origin"],
            h=out["h"],
            elbo_hist=out["elbo_hist"]
        )


def register(
    slices: List[ad.AnnData],
    spatial_key: str,
    expression_key: str | None = None,

    n_scales: int = 3,
    l_init: int = 8,
    l_final: int = 32,
    latent_dim: int = 64,
    inter_slice_distance: Optional[float] = None,
    verbose: bool = True,

    **trainer_kwargs
) -> MantaResult:
    registration = MantaRegistration(
        n_scales=n_scales,
        l_init=l_init,
        l_final=l_final,
        latent_dim=latent_dim,
        inter_slice_distance=inter_slice_distance,
        **trainer_kwargs
    )
    return registration.fit(
        slices=slices,
        spatial_key=spatial_key,
        expression_key=expression_key,
        verbose=verbose
    )