"""Expression features shared by all sections, and blurred expression images for FFT scans."""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy import ndimage


def pca_embedding(adata, n_comps: int = 30, genes=None, max_fit: int = 40000, seed: int = 0) -> np.ndarray:
    """One PCA of the z-scored expression (adata.X) of all cells, so every section lives in the same embedding.
    genes: optional subset of var_names. Returns (n_cells, n_comps) float32."""
    g = list(adata.var_names) if genes is None else list(genes)
    X = adata[:, g].X
    X = X.toarray() if sp.issparse(X) else np.asarray(X, float)
    Z = (X - X.mean(0)) / (X.std(0) + 1e-6)
    rng = np.random.default_rng(seed)
    sub = Z[rng.choice(len(Z), min(len(Z), max_fit), replace=False)]
    m0 = sub.mean(0)
    _, _, Vt = np.linalg.svd(sub - m0, full_matrices=False)
    return ((Z - m0) @ Vt[:n_comps].T).astype(np.float32)


def unit(E) -> np.ndarray:
    E = np.asarray(E, np.float32)
    return E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)


def scan_features(E, n_feat: int = 10, max_fit: int = 40000, seed: int = 0) -> np.ndarray:
    """Unit-normalised embedding projected on its top n_feat uncentred principal axes (shared by all sections), so
    inner products of the features approximate cosine similarities of the cells."""
    U = unit(E)
    rng = np.random.default_rng(seed)
    sub = U[rng.choice(len(U), min(len(U), max_fit), replace=False)]
    _, _, Vt = np.linalg.svd(sub, full_matrices=False)
    return (U @ Vt[:n_feat].T).astype(np.float32)


def raster(P, F, origin, n: int, px: float, sigma_px: float) -> np.ndarray:
    """(C, n, n) image: per feature channel the sum of F over the cells in each pixel (pixel size px, grid origin
    `origin`), blurred with a Gaussian of sigma_px pixels."""
    ij = np.floor((np.asarray(P) - origin) / px).astype(np.int64)
    ok = (ij >= 0).all(1) & (ij < n).all(1)
    flat = ij[ok, 0] * n + ij[ok, 1]
    img = np.zeros((F.shape[1], n * n), np.float32)
    for c in range(F.shape[1]):
        img[c] = np.bincount(flat, weights=F[ok, c], minlength=n * n)
    return ndimage.gaussian_filter(img.reshape(-1, n, n), (0, sigma_px, sigma_px))
