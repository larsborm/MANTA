"""Start of the reconstruction: a chain of pairwise rigid fits of neighbouring sections."""
from __future__ import annotations

import anndata as ad
import numpy as np
from scipy.spatial import cKDTree

from ._stack import SectionStack, Similarity


def _pair_adata(P, E):
    a = ad.AnnData(X=np.zeros((len(P), 1), np.float32))
    a.obsm["spatial_manta"] = np.asarray(P, float)
    a.obsm["X_emb"] = np.asarray(E, float)
    return a


def label_agreement(src_xy, src_lab, tgt_xy, tgt_lab, radius: float = 8.0) -> float:
    """Fraction of source cells whose nearest target cell lies within `radius` and has the same label."""
    d, i = cKDTree(tgt_xy).query(src_xy)
    return float(np.mean((d < radius) & (np.asarray(tgt_lab)[i] == np.asarray(src_lab))))


def expression_agreement(src_xy, src_emb, tgt_xy, tgt_emb, radius: float = 8.0) -> float:
    """Mean cosine similarity between each source cell and its nearest target cell (0 beyond `radius`)."""
    d, i = cKDTree(tgt_xy).query(src_xy)
    a = src_emb / (np.linalg.norm(src_emb, axis=1, keepdims=True) + 1e-12)
    b = tgt_emb[i] / (np.linalg.norm(tgt_emb[i], axis=1, keepdims=True) + 1e-12)
    return float(np.mean(np.where(d < radius, (a * b).sum(1), 0.0)))


def fit_pair(src_xy, src_emb, tgt_xy, tgt_emb, src_lab=None, tgt_lab=None, radius: float = 8.0, alpha: float = 3.0,
             seed: int = 1, **joint_kw):
    """Pose of a source section on a target section: `manta.align.rigid_joint` (rotation scan + optimal-transport
    refinement) or, if that is worse, the plain centroid overlay. The two candidates are compared by label
    agreement (when labels are given) or by expression agreement. Returns (Similarity, name of the chosen candidate)."""
    from ..api._alignment import rigid_joint

    src_xy, tgt_xy = np.asarray(src_xy, float), np.asarray(tgt_xy, float)
    cand = {"centroid": Similarity(np.eye(2), tgt_xy.mean(0) - src_xy.mean(0))}
    try:
        S, T = _pair_adata(src_xy, src_emb), _pair_adata(tgt_xy, tgt_emb)
        rigid_joint(S, T, spatial_key="spatial_manta", expression_key="X_emb", progress=False, alpha=alpha,
                    seed=seed, **joint_kw)
        r = S.uns["rigid_alignment"]
        fit = Similarity(r["R"], r["t"], r["s"])
        if np.isfinite(fit.R).all() and np.isfinite(fit.t).all():
            cand["joint"] = fit
    except (RuntimeError, ValueError, np.linalg.LinAlgError):
        pass

    def score(T_):
        P = T_(src_xy)
        if src_lab is not None and tgt_lab is not None:
            return label_agreement(P, src_lab, tgt_xy, tgt_lab, radius)
        return expression_agreement(P, src_emb, tgt_xy, tgt_emb, radius)

    best = max(cand, key=lambda name: score(cand[name]))
    return cand[best], best


def chain(stack: SectionStack, emb, labels=None, verbose: bool = False, **pair_kw) -> SectionStack:
    """Fit every section k + 1 onto section k (`fit_pair`) and compose the fits outward from the anchor, which is
    centred at the origin. labels: optional per-cell labels (e.g. cell types) used to choose between candidates."""
    K = stack.K
    lab = None if labels is None else np.asarray(labels).astype(str)
    fits, chosen = [], []
    for k in range(K - 1):
        rs, rt_ = stack.rows[stack.ids[k + 1]], stack.rows[stack.ids[k]]
        T_, name = fit_pair(stack.raw[rs], emb[rs], stack.raw[rt_], emb[rt_],
                            None if lab is None else lab[rs], None if lab is None else lab[rt_], **pair_kw)
        fits.append(T_)
        chosen.append(name)
        if verbose:
            print(f"  chain {k + 1} -> {k}: {name}", flush=True)
    a = stack.ids.index(stack.anchor)
    G = [None] * K
    G[a] = Similarity(np.eye(2), -stack.raw[stack.rows[stack.anchor]].mean(0))
    for k in range(a + 1, K):
        G[k] = G[k - 1] @ fits[k - 1]
    for k in range(a - 1, -1, -1):
        G[k] = G[k + 1] @ fits[k].inv()
    out = stack.with_transforms(dict(zip(stack.ids, G)))
    out.chain_choice = chosen
    return out
