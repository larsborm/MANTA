"""Rigid alignment that optimises correspondences and transform jointly.

The voxel method (`manta.align.rigid`) first links every source voxel to its most similar target voxel by expression
alone, then asks RANSAC to find the largest group of links that agree on one transform. When many voxels look alike
(e.g. ectoderm all around a section) most links are wrong and RANSAC can lock onto a wrong pose.

Here no hard links are made:

1. `_rotation_scan` (2D only): exhaustive search over rotation angles; for every angle all translations are scored at
   once with an FFT cross-correlation of blurred expression images. The score of a pose is the summed expression
   similarity of all overlapping cells, so every cell votes and the result is the global optimum on the angle /
   pixel grid.
2. `_ot_refine` (any dimension): alternate an unbalanced entropic optimal-transport coupling, with a cost that fuses
   spatial distance (in the current pose) and expression dissimilarity, with a weighted Kabsch fit on all coupled
   pairs. The spatial length scale anneals from coarse to fine.
"""
import numpy as np

from scipy import ndimage
from typing import Optional, Sequence

from ..old.alignment._rigid import _kabsch_umeyama, _apply_transform


def _unit(X: np.ndarray) -> np.ndarray:
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def _rotation_2d(deg: float) -> np.ndarray:
    a = np.deg2rad(deg)
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


def _angle_deg(R: np.ndarray) -> float:
    """Signed angle (2D) or axis-angle magnitude (3D) of a rotation matrix, in degrees."""
    if R.shape[0] == 2:
        return float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def _scan_features(
    src_expr: np.ndarray,
    tgt_expr: np.ndarray,
    n_features: int,
    max_fit: int = 20000,
    seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Unit-normalise the expression of both sets and project onto the top `n_features` principal axes of the
    pooled unit vectors. Inner products of the projections approximate the cosine similarity of the cells."""
    fs, ft = _unit(np.asarray(src_expr, float)), _unit(np.asarray(tgt_expr, float))
    if fs.shape[1] <= n_features:
        return fs, ft

    pooled = np.concatenate([fs, ft])
    rng = np.random.default_rng(seed)
    if len(pooled) > max_fit:
        pooled = pooled[rng.choice(len(pooled), max_fit, replace=False)]

    # uncentred: we want to preserve inner products, not variance around the mean
    _, _, Vt = np.linalg.svd(pooled, full_matrices=False)
    V = Vt[:n_features].T
    return fs @ V, ft @ V


def _raster(
    pts: np.ndarray,
    feats: np.ndarray,
    origin: np.ndarray,
    n: int,
    px: float,
    sigma_px: float
) -> np.ndarray:
    """Sum per-point features into a (C, n, n) image with pixel size `px`, then Gaussian blur."""
    ij = np.floor((pts - origin) / px).astype(np.int64)
    ok = (ij >= 0).all(axis=1) & (ij < n).all(axis=1)
    flat = ij[ok, 0] * n + ij[ok, 1]

    img = np.zeros((feats.shape[1], n * n), np.float32)
    for c in range(feats.shape[1]):
        img[c] = np.bincount(flat, weights=feats[ok, c], minlength=n * n)

    return ndimage.gaussian_filter(img.reshape(-1, n, n), (0, sigma_px, sigma_px))


def _rotation_scan(
    source_pts: np.ndarray,
    source_feats: np.ndarray,
    target_pts: np.ndarray,
    target_feats: np.ndarray,
    angles: Sequence[float],
    pixel_size: float = 5.0,
    sigma: float = 10.0,
    peak_separation: float = 30.0
) -> dict:
    """Global 2D rigid search. For every angle the source is rotated about its centroid; the cross-correlation
    cc[d] = sum_p <S(p), T(p + d)> of the blurred feature images is computed for all shifts d by FFT.
    Returns R, t (x -> R x + t), the best angle, the best score per angle and `peak_ratio`: the best score at
    angles more than `peak_separation` deg away from the best angle, divided by the best score. Near 1 = a
    competing pose scores almost as well (ambiguous, e.g. a small round end section); low = one clear pose."""
    src = np.asarray(source_pts, float)
    tgt = np.asarray(target_pts, float)
    if src.shape[1] != 2:
        raise ValueError("the rotation scan is implemented for 2D coordinates only; pass `init` for 3D")

    cs, ct = src.mean(axis=0), tgt.mean(axis=0)
    radius = max(np.linalg.norm(src - cs, axis=1).max(), np.linalg.norm(tgt - ct, axis=1).max()) + 3 * sigma

    # image side: room for both clouds and every shift between them without wrap-around
    n = int(2 ** np.ceil(np.log2(4 * radius / pixel_size)))
    origin = -np.full(2, n * pixel_size / 2)
    sigma_px = sigma / pixel_size

    tgt_fft = np.fft.rfft2(_raster(tgt - ct, target_feats, origin, n, pixel_size, sigma_px))

    angles = np.asarray(angles, float)
    scores = np.empty(len(angles))
    shifts = np.empty((len(angles), 2))
    for a_i, ang in enumerate(angles):
        R = _rotation_2d(ang)
        src_fft = np.fft.rfft2(_raster((src - cs) @ R.T, source_feats, origin, n, pixel_size, sigma_px))
        cc = np.fft.irfft2((np.conj(src_fft) * tgt_fft).sum(axis=0), s=(n, n))

        k = np.unravel_index(np.argmax(cc), cc.shape)
        scores[a_i] = cc[k]
        shifts[a_i] = ((np.array(k) + n // 2) % n - n // 2) * pixel_size  # signed shift

    best = int(np.argmax(scores))
    R = _rotation_2d(angles[best])
    t = ct + shifts[best] - R @ cs  # x -> R (x - cs) + ct + shift

    away = np.abs((angles - angles[best] + 180.0) % 360.0 - 180.0) > peak_separation
    peak_ratio = float(scores[away].max() / scores[best]) if away.any() and scores[best] > 0 else np.nan

    return {"R": R, "t": t, "s": 1.0, "angle": float(angles[best]), "angles": angles, "scores": scores,
            "peak_ratio": peak_ratio}


def _lse(M: np.ndarray, axis: int) -> np.ndarray:
    m = M.max(axis=axis, keepdims=True)
    return (m + np.log(np.exp(M - m).sum(axis=axis, keepdims=True))).squeeze(axis)


def _dense_unbalanced_sinkhorn(
    cost: np.ndarray,
    epsilon: float,
    rho: float,
    n_iters: int = 30,
    potentials: Optional[tuple[np.ndarray, np.ndarray]] = None
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """Entropic unbalanced OT with uniform marginals and KL marginal relaxation `rho` (log domain, float32).
    A small `rho` lets points without a partner stay unmatched. `potentials` warm-starts the dual variables.
    Returns the coupling P (n, m) and the potentials (f, g)."""
    n, m = cost.shape
    la, lb = np.float32(-np.log(n)), np.float32(-np.log(m))
    if potentials is None:
        f, g = np.zeros(n, np.float32), np.zeros(m, np.float32)
    else:
        f, g = potentials

    eps = np.float32(epsilon)
    k = np.float32(rho / (rho + epsilon))
    C = (cost / epsilon).astype(np.float32)

    for _ in range(n_iters):
        f = -k * eps * _lse(g[None] / eps - C + lb, axis=1)
        g = -k * eps * _lse(f[:, None] / eps - C + la, axis=0)

    P = np.exp(f[:, None] / eps + g[None] / eps - C + la + lb)
    return P, (f, g)


def _ot_refine(
    source_pts: np.ndarray,
    source_expr: np.ndarray,
    target_pts: np.ndarray,
    target_expr: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    s: float = 1.0,
    sigmas: Sequence[float] = (40, 25, 15, 10, 6),
    alpha: float = 3.0,
    epsilon: float = 0.05,
    rho: float = 0.5,
    n_inner: int = 3,
    n_sinkhorn_iters: int = 30,
    max_points: int = 1500,
    allow_scaling: bool = False,
    generator: Optional[np.random.Generator] = None,
    progress=None
) -> dict:
    """Alternate (1) the coupling P = unbalanced OT with cost |T(x_i) - y_j|^2 / sigma^2 + alpha (1 - cos(e_i, e_j))
    and (2) the weighted Kabsch fit of T on all pairs (i, j) with weight P_ij, for sigma in `sigmas` (coarse to fine).
    Up to `max_points` points per set are used (random subset; the dense coupling is max_points^2)."""
    generator = np.random.default_rng() if generator is None else generator
    src, tgt = np.asarray(source_pts, float), np.asarray(target_pts, float)
    i_s = generator.choice(len(src), min(len(src), max_points), replace=False)
    i_t = generator.choice(len(tgt), min(len(tgt), max_points), replace=False)

    X, Y = src[i_s], tgt[i_t]
    cos = _unit(np.asarray(source_expr, float)[i_s]) @ _unit(np.asarray(target_expr, float)[i_t]).T
    cost_expr = alpha * (1.0 - cos)

    potentials, history = None, []
    for sigma in sigmas:
        for _ in range(n_inner):
            TX = _apply_transform(X, R, t, s)
            cost = ((TX[:, None] - Y[None]) ** 2).sum(axis=-1) / sigma ** 2 + cost_expr
            P, potentials = _dense_unbalanced_sinkhorn(cost, epsilon, rho, n_sinkhorn_iters, potentials)

            # for a rigid fit, all (i, j, P_ij) pairs are equivalent to each source point paired with its
            # P-weighted mean partner, weighted by its transported mass
            w = P.sum(axis=1).astype(float)
            Y_bar = (P @ Y) / (w[:, None] + 1e-30)
            R, t, s = _kabsch_umeyama(X, Y_bar, weights=w + 1e-12, allow_scaling=allow_scaling)

        mass = float(P.sum())
        history.append({
            "sigma": float(sigma),
            "rotation_deg": _angle_deg(R),
            "transported_mass": mass,
            "median_residual": float(np.median(np.linalg.norm(Y_bar - _apply_transform(X, R, t, s), axis=1))),
            "mean_cosine_sim": float((P * cos).sum() / max(mass, 1e-30)),
        })
        if progress is not None:
            progress(f"OT sigma {sigma:g}")

    return {"R": R, "t": t, "s": s, "history": history}
