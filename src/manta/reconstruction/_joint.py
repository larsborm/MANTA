"""Joint rigid refinement of all sections at once.

Every section's pose (rotation + translation) is a free parameter of one objective summed over all section pairs
(k, l, w) from `section_pairs`:

    E = sum_pairs  w  sum_ij  P_ij |T_k(x_i) - T_l(y_j)|^2

P is a soft correspondence between the cells of the two sections: unbalanced entropic optimal transport with cost
|T_k(x) - T_l(y)|^2 / sigma^2 + alpha (1 - cosine of the expression); cells without a counterpart may stay
unmatched (rho). The fit alternates couplings for all pairs (warm-started Sinkhorn) and one Gauss-Newton solve for
all poses together, while sigma shrinks from coarse to fine. The anchor section is held fixed.
"""
from __future__ import annotations

import numpy as np

from ..matching._joint_rigid import _dense_unbalanced_sinkhorn
from ._features import unit
from ._stack import SectionStack, Similarity

_J = np.array([[0.0, -1.0], [1.0, 0.0]])        # d/dtheta of a rotation at 0: R'(0) x = J x


def _rot(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


class Couplings:
    """Per section a fixed random subset of cells (slide xy, unit embedding), the section pairs, and the Sinkhorn
    potentials of every pair (kept between calls for warm starts)."""

    def __init__(self, stack: SectionStack, emb, pairs, max_points: int = 1200, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.stack, self.ids, self.K = stack, stack.ids, stack.K
        self.rows, self.raw, self.E = [], [], []
        for s in self.ids:
            r = stack.rows[s]
            r = np.sort(rng.choice(r, min(len(r), max_points), replace=False))
            self.rows.append(r)
            self.raw.append(stack.raw[r])
            self.E.append(unit(emb[r]))
        self.pairs = [(int(k), int(l), float(w)) for k, l, w in pairs]
        self.potentials = {}

    def couple(self, pos, sigma, alpha, epsilon, rho, n_iters):
        """Couplings of all pairs at the current positions pos[k] (stack xy of the sampled cells of section k).
        Returns [(k, l, w, P)] and a summary per pair (transported mass, mean coupled cosine)."""
        out, info = [], []
        for k, l, wp in self.pairs:
            cos = self.E[k] @ self.E[l].T
            cost = ((pos[k][:, None] - pos[l][None]) ** 2).sum(-1) / sigma ** 2 + alpha * (1.0 - cos)
            P, self.potentials[(k, l)] = _dense_unbalanced_sinkhorn(cost, epsilon, rho, n_iters,
                                                                    self.potentials.get((k, l)))
            out.append((k, l, wp, P))
            m = P.sum()
            info.append({"k": k, "l": l, "mass": float(m), "cos": float((P * cos).sum() / max(m, 1e-30))})
        return out, info


def correspondences(wp, P, A, B):
    """Two-sided barycentric pairs of one coupling: each cell of k with its P-weighted mean partner in l and each
    cell of l with its mean partner in k. Returns (a, b, w); w in matched cells, times the pair weight, halved for
    the two directions."""
    nk, nl = P.shape
    wf, wb = P.sum(1), P.sum(0)
    Ab = (P.T @ A) / (wb[:, None] + 1e-30)
    Bf = (P @ B) / (wf[:, None] + 1e-30)
    return np.concatenate([A, Ab]), np.concatenate([Bf, B]), 0.5 * wp * np.concatenate([wf * nk, wb * nl])


def _normal_equations(corr, R, t, c, K):
    """Gauss-Newton normal equations of sum w |T_k(a) - T_l(b)|^2 in the update (dtheta_k, dt_k) of every section;
    the rotation update turns section k about its stack centroid c[k]."""
    H, g, cost = np.zeros((3 * K, 3 * K)), np.zeros(3 * K), 0.0
    for k, l, a, b, w in corr:
        pa, pb = a @ R[k].T + t[k], b @ R[l].T + t[l]
        r = pa - pb
        Jk = np.zeros((len(a), 2, 3))
        Jk[:, :, 0] = (pa - c[k]) @ _J.T
        Jk[:, :, 1:] = np.eye(2)
        Jl = np.zeros_like(Jk)
        Jl[:, :, 0] = -((pb - c[l]) @ _J.T)
        Jl[:, :, 1:] = -np.eye(2)
        Jr = np.concatenate([Jk, Jl], axis=2)
        idx = np.r_[3 * k:3 * k + 3, 3 * l:3 * l + 3]
        H[np.ix_(idx, idx)] += np.einsum("n,nri,nrj->ij", w, Jr, Jr)
        g[idx] += np.einsum("n,nri,nr->i", w, Jr, r)
        cost += float((w * (r ** 2).sum(1)).sum())
    return H, g, cost


def refine_rigid(stack: SectionStack, emb, pairs, sigmas=(25, 15, 10, 6), alpha: float = 3.0, epsilon: float = 0.05,
                 rho: float = 0.5, n_inner: int = 2, n_gn: int = 3, n_sinkhorn_iters: int = 30, max_points: int = 1200,
                 seed: int = 0, hold=(), verbose: bool = False) -> SectionStack:
    """Joint rigid fit of all sections, starting from the transforms of `stack`.
    sigmas: OT length scales (um), coarse to fine; n_inner coupling rounds per sigma, each followed by n_gn
    Gauss-Newton steps. hold: stack positions kept fixed in addition to the anchor.
    Returns a new SectionStack (.history: rms residual per sigma)."""
    cp = Couplings(stack, emb, pairs, max_points, seed)
    K = cp.K
    R = [stack.T[s].R.copy() for s in cp.ids]
    t = [stack.T[s].t.copy() for s in cp.ids]
    fixed = {cp.ids.index(stack.anchor)} | {int(h) for h in hold}
    free = np.array([i for k in range(K) if k not in fixed for i in range(3 * k, 3 * k + 3)], int)
    history = []
    for sigma in sigmas:
        for _ in range(n_inner):
            pos = [cp.raw[k] @ R[k].T + t[k] for k in range(K)]
            coup, info = cp.couple(pos, sigma, alpha, epsilon, rho, n_sinkhorn_iters)
            corr = [(k, l, *correspondences(wp, P, cp.raw[k], cp.raw[l])) for k, l, wp, P in coup]
            for _ in range(n_gn):
                c = [cp.raw[k].mean(0) @ R[k].T + t[k] for k in range(K)]
                H, g, cost = _normal_equations(corr, R, t, c, K)
                if len(free) == 0:
                    break
                Hf = H[np.ix_(free, free)]
                d = np.zeros(3 * K)
                d[free] = np.linalg.solve(Hf + 1e-6 * np.diag(np.diag(Hf) + 1e-12), -g[free])
                for k in range(K):
                    Q = _rot(d[3 * k])
                    R[k] = Q @ R[k]
                    t[k] = Q @ (t[k] - c[k]) + c[k] + d[3 * k + 1:3 * k + 3]
        wsum = sum(x[4].sum() for x in corr)
        history.append({"sigma": sigma, "rms_residual_um": float(np.sqrt(cost / max(wsum, 1e-30)))})
        if verbose:
            print(f"  rigid sigma {sigma:g}: rms residual {history[-1]['rms_residual_um']:.2f} um", flush=True)
    out = stack.with_transforms({s: Similarity(R[k], t[k]) for k, s in enumerate(cp.ids)})
    out.history = history
    return out
