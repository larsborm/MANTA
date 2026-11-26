"""Non-rigid step: one smooth displacement field per section, all sections solved together.

Every section k gets a 2D displacement field D_k, bilinear on a regular grid (node spacing `spacing` um). A cell at
rigid stack position p moves to p + D_k(p). Data term (second order): each cell of section k is pulled to the
z-weighted interpolation of its partners in the closest sections below (l) and above (u),

    p + D_k(p)  ~  (1 - f) (y_l + D_l(y_l)) + f (y_u + D_u(y_u)),    f = (z_k - z_l) / (z_u - z_l),

with partners from the same optimal-transport couplings as the rigid refinement. A steady taper or drift along the
stack costs nothing; only kinks between sections are removed. Regularisers per section: bending energy of the node
values (bend), a membrane term that resists uniform stretching (stretch), and a small pull to zero (shrink). For
fixed couplings the optimum over all sections is one sparse linear solve. Finally the scale that the warp applies to
all sections alike (a small inward bias of barycentric partners at the tissue edge) is removed.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

from ._joint import Couplings
from ._stack import SectionStack, Similarity


class Grid:
    """Regular 2D grid of nodes (stack frame) for bilinear displacement fields."""

    def __init__(self, xy, spacing: float, pad: int = 1):
        lo = xy.min(0) - pad * spacing
        self.shape = tuple((np.ceil((xy.max(0) + pad * spacing - lo) / spacing)).astype(int) + 1)
        self.origin, self.spacing = lo, float(spacing)
        self.m = self.shape[0] * self.shape[1]

    def nodes(self) -> np.ndarray:
        ii, jj = np.meshgrid(np.arange(self.shape[0]), np.arange(self.shape[1]), indexing="ij")
        return self.origin + self.spacing * np.stack([ii.ravel(), jj.ravel()], 1)

    def basis(self, p) -> sp.csr_matrix:
        """Sparse (n, m) bilinear interpolation weights of the points p (n, 2)."""
        u = (np.asarray(p, float) - self.origin) / self.spacing
        u = np.clip(u, 0, np.array(self.shape) - 1 - 1e-9)
        i0 = np.floor(u).astype(int)
        f = u - i0
        n = len(p)
        rows, cols, vals = [], [], []
        for di in (0, 1):
            for dj in (0, 1):
                rows.append(np.arange(n))
                cols.append((i0[:, 0] + di) * self.shape[1] + i0[:, 1] + dj)
                vals.append((f[:, 0] if di else 1 - f[:, 0]) * (f[:, 1] if dj else 1 - f[:, 1]))
        return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), (n, self.m))

    def bending(self) -> sp.csr_matrix:
        """Second differences along x and y and the mixed difference (zero for any affine field)."""
        nx, ny = self.shape
        idx = np.arange(self.m).reshape(nx, ny)
        rows, cols, vals, r = [], [], [], 0
        for a, b, c in [(idx[:-2], idx[1:-1], idx[2:]), (idx[:, :-2], idx[:, 1:-1], idx[:, 2:])]:
            for x, y, w in zip(a.ravel(), b.ravel(), c.ravel()):
                rows += [r, r, r]; cols += [x, y, w]; vals += [1.0, -2.0, 1.0]; r += 1
        s2 = np.sqrt(2.0)
        for x00, x01, x10, x11 in zip(idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel(), idx[1:, :-1].ravel(),
                                      idx[1:, 1:].ravel()):
            rows += [r] * 4; cols += [x00, x01, x10, x11]; vals += [s2, -s2, -s2, s2]; r += 1
        return sp.csr_matrix((vals, (rows, cols)), (r, self.m))

    def membrane(self) -> sp.csr_matrix:
        """First differences along x and y of the node values."""
        nx, ny = self.shape
        idx = np.arange(self.m).reshape(nx, ny)
        a = np.concatenate([idx[:-1].ravel(), idx[:, :-1].ravel()])
        b = np.concatenate([idx[1:].ravel(), idx[:, 1:].ravel()])
        r = np.arange(len(a))
        return sp.csr_matrix((np.r_[np.ones(len(a)), -np.ones(len(a))], (np.r_[r, r], np.r_[a, b])), (len(a), self.m))


class WarpedStack(SectionStack):
    """A rigid SectionStack plus one displacement field per section (node values C[k], (m, 2), on `grid`)."""

    def __init__(self, rigid: SectionStack, grid: Grid, C):
        self.__dict__.update(rigid.__dict__)
        self.rigid, self.grid, self.C = rigid, grid, np.asarray(C)

    def displacement(self, sec) -> np.ndarray:
        s = self.key(sec)
        return self.grid.basis(self.rigid.xy(s)) @ self.C[self.ids.index(s)]

    def xy(self, sec) -> np.ndarray:
        return self.rigid.xy(sec) + self.displacement(sec)

    def save(self, path) -> None:
        """Rigid transforms, grid and node displacements in one .npz file."""
        R = self.rigid
        np.savez(path, ids=np.array(R.ids, dtype=object), R=np.stack([R.T[s].R for s in R.ids]),
                 t=np.stack([R.T[s].t for s in R.ids]), s=np.array([R.T[s].s for s in R.ids]), anchor=str(R.anchor),
                 grid_origin=self.grid.origin, grid_shape=np.array(self.grid.shape), grid_spacing=self.grid.spacing,
                 C=self.C)


def _triplets(z, pairs):
    """For every section k: (l, k, u, weight, f) with (l, k) and (k, u) in pairs, closest partners first (the next
    ones with half the weight); f = (z_k - z_l) / (z_u - z_l)."""
    pw = {(k, l): w for k, l, w in pairs}
    K = len(z)
    out = []
    for k in range(K):
        lows = sorted([l for l in range(k) if (l, k) in pw], key=lambda l: z[k] - z[l])
        ups = sorted([u for u in range(k + 1, K) if (k, u) in pw], key=lambda u: z[u] - z[k])
        for rank, (l, u) in enumerate(zip(lows, ups)):
            out.append((l, k, u, pw[(l, k)] * pw[(k, u)] * (0.5 ** rank), (z[k] - z[l]) / (z[u] - z[l])))
    return out


def remove_common_scale(ws: WarpedStack, min_cells: int = 300) -> WarpedStack:
    """Rescale every section about its centroid by 1 / (cell-weighted geometric mean of the per-section scales of
    the warp); relative size differences between sections are kept."""
    nodes = ws.grid.nodes()
    sc, wt = [], []
    for s in ws.ids:
        p, q = ws.rigid.xy(s), ws.xy(s)
        if len(p) >= min_cells:
            sc.append(np.log(Similarity.fit(p, q, allow_scaling=True).s))
            wt.append(len(p))
    if not sc:
        return ws
    sbar = float(np.exp(np.average(sc, weights=wt)))
    C = ws.C.copy()
    for k, s in enumerate(ws.ids):
        c = ws.rigid.xy(s).mean(0)
        C[k] = ws.C[k] / sbar + (1.0 / sbar - 1.0) * (nodes - c)
    out = WarpedStack(ws.rigid, ws.grid, C)
    out.common_scale_removed = sbar
    return out


def refine_warp(stack: SectionStack, emb, pairs, spacing: float = 20.0, bend: float = 0.3, stretch: float = 0.3,
                shrink: float = 0.01, sigmas=(15, 10, 6), alpha: float = 3.0, epsilon: float = 0.05, rho: float = 0.5,
                n_inner: int = 2, n_sinkhorn_iters: int = 30, max_points: int = 2000, seed: int = 0,
                keep_scale: bool = True, verbose: bool = False) -> WarpedStack:
    """Joint non-rigid fit on top of the rigid `stack` (see the module docstring). Returns a WarpedStack."""
    cp = Couplings(stack, emb, pairs, max_points, seed)
    K = cp.K
    rigid_pos = [stack.T[s](cp.raw[k]) for k, s in enumerate(cp.ids)]
    grid = Grid(np.concatenate([stack.xy(s) for s in stack.ids]), spacing)
    m = grid.m
    Phi = [grid.basis(p) for p in rigid_pos]
    D = grid.bending()
    reg = bend * (D.T @ D) + shrink * sp.identity(m)
    if stretch > 0:
        G = grid.membrane()
        reg = reg + stretch * (G.T @ G)
    trip = _triplets(stack.z_sections, pairs)
    C = np.zeros((K, m, 2))
    for sigma in sigmas:
        for _ in range(n_inner):
            pos = [rigid_pos[k] + Phi[k] @ C[k] for k in range(K)]
            coup, _ = cp.couple(pos, sigma, alpha, epsilon, rho, n_sinkhorn_iters)
            Pd = {(k, l): P for k, l, _, P in coup}
            rows, cols, vals, rhs, wts = [], [], [], [], []
            r0 = 0
            for l, k, u, wt, f in trip:
                P1, P2 = Pd[(l, k)], Pd[(k, u)]
                nk = len(rigid_pos[k])
                w1, w2 = P1.sum(0), P2.sum(1)
                ym = (P1.T @ rigid_pos[l]) / (w1[:, None] + 1e-30)
                yp = (P2 @ rigid_pos[u]) / (w2[:, None] + 1e-30)
                p = rigid_pos[k]
                Bp, Bm, Bq = grid.basis(p).tocoo(), grid.basis(ym).tocoo(), grid.basis(yp).tocoo()
                rows += [Bp.row + r0, Bm.row + r0, Bq.row + r0]
                cols += [Bp.col + k * m, Bm.col + l * m, Bq.col + u * m]
                vals += [Bp.data, -(1 - f) * Bm.data, -f * Bq.data]
                rhs.append(p - ((1 - f) * ym + f * yp))
                wts.append(wt * nk * np.minimum(w1, w2))
                r0 += nk
            if not trip:
                break
            M = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), (r0, K * m))
            dvec, w = np.concatenate(rhs), np.concatenate(wts)
            A = (M.T @ sp.diags(w) @ M).tocsc()
            dg = A.diagonal()
            A = A + sp.block_diag([np.mean(dg[dg > 0]) * reg] * K, format="csc")
            b = -(M.T @ (w[:, None] * dvec))
            C = splu(A.tocsc()).solve(b).reshape(K, m, 2)
        if verbose:
            disp = np.concatenate([np.linalg.norm(Phi[k] @ C[k], axis=1) for k in range(K)])
            print(f"  warp sigma {sigma:g}: median displacement {np.median(disp):.1f} um", flush=True)
    ws = WarpedStack(stack, grid, C)
    return remove_common_scale(ws) if keep_scale else ws
