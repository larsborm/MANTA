"""Section stack: one similarity transform per section from slide coordinates to the common stack frame."""
from __future__ import annotations

from dataclasses import dataclass

import anndata as ad
import numpy as np
import pandas as pd


@dataclass
class Similarity:
    """2D (or 3D) similarity transform x -> s R x + t; points as rows (P -> s P R^T + t).
    `A @ B` is A after B, `T.inv()` the inverse, `T(P)` applies it."""
    R: np.ndarray
    t: np.ndarray
    s: float = 1.0

    def __post_init__(self):
        self.R = np.asarray(self.R, float)
        self.t = np.asarray(self.t, float).reshape(-1)
        self.s = float(self.s)

    @classmethod
    def identity(cls, d: int = 2) -> "Similarity":
        return cls(np.eye(d), np.zeros(d), 1.0)

    @classmethod
    def fit(cls, P, Q, weights=None, allow_scaling: bool = False) -> "Similarity":
        """Least-squares (Kabsch-Umeyama) transform with T(P) ~ Q for matched rows."""
        P, Q = np.asarray(P, float), np.asarray(Q, float)
        w = np.ones(len(P)) if weights is None else np.asarray(weights, float)
        w = w / w.sum()
        mp, mq = w @ P, w @ Q
        Pc, Qc = P - mp, Q - mq
        U, S, Vt = np.linalg.svd((Qc * w[:, None]).T @ Pc)
        D = np.eye(P.shape[1])
        D[-1, -1] = np.sign(np.linalg.det(U @ Vt))
        R = U @ D @ Vt
        s = (S * np.diag(D)).sum() / (w @ (Pc ** 2).sum(1)) if allow_scaling else 1.0
        return cls(R, mq - s * R @ mp, s)

    def __call__(self, P) -> np.ndarray:
        return self.s * np.asarray(P, float) @ self.R.T + self.t

    def __matmul__(self, other: "Similarity") -> "Similarity":
        return Similarity(self.R @ other.R, self.s * self.R @ other.t + self.t, self.s * other.s)

    def inv(self) -> "Similarity":
        return Similarity(self.R.T, -(self.R.T / self.s) @ self.t, 1.0 / self.s)

    @property
    def angle_deg(self) -> float:
        """Signed rotation angle (2D) or axis-angle magnitude (3D), degrees."""
        if self.R.shape[0] == 2:
            return float(np.degrees(np.arctan2(self.R[1, 0], self.R[0, 0])))
        return float(np.degrees(np.arccos(np.clip((np.trace(self.R) - 1) / 2, -1, 1))))


def rotation(deg: float) -> np.ndarray:
    r = np.radians(deg)
    return np.array([[np.cos(r), -np.sin(r)], [np.sin(r), np.cos(r)]])


class SectionStack:
    """The cells of one specimen, cut into serial sections, with one 2D similarity transform per section.

    Sections are ordered by their z (`adata.obs[z_key]`, one value per section, e.g. the cutting position in um).
    `raw` are the slide coordinates of every cell (`adata.obsm[spatial_key][:, :2]`); a third column, if present,
    is the depth of the cell inside its section and is added to the section z (clipped to [0, depth_clip]).
    `T[section_id]` maps the slide coordinates of that section into the common stack frame. `anchor` is the
    section that defines the frame (kept fixed by the joint refinements).
    """

    def __init__(self, adata: ad.AnnData, section_key: str = "section_id", z_key: str = "section_z_raw_um",
                 spatial_key: str = "spatial_raw", transforms=None, anchor=None, use_depth: bool = True,
                 depth_clip: float = 10.0):
        self.adata = adata
        self.section_key, self.z_key, self.spatial_key = section_key, z_key, spatial_key
        sec = adata.obs[section_key].astype(str).to_numpy()
        zc = adata.obs[z_key].to_numpy(float)
        tab = pd.DataFrame({"section": sec, "z": zc}).groupby("section").agg(z=("z", "first"), n_cells=("z", "size"))
        tab = tab.sort_values("z")
        self.ids = [str(s) for s in tab.index]
        self.z_sections = tab["z"].to_numpy(float)
        self.n_cells = tab["n_cells"].to_numpy(int)
        self.sec = sec
        self.rows = {s: np.flatnonzero(sec == s) for s in self.ids}
        coords = np.asarray(adata.obsm[spatial_key], float)
        self.raw = coords[:, :2]
        self.depth = coords[:, 2] if (use_depth and coords.shape[1] > 2) else None
        self.depth_clip = depth_clip
        self.z_cells = zc
        self.T = {s: Similarity.identity(2) for s in self.ids} if transforms is None else dict(transforms)
        self.anchor = anchor if anchor is not None else self.default_anchor()

    @property
    def K(self) -> int:
        return len(self.ids)

    def default_anchor(self) -> str:
        """The section with most cells in the middle half of the stack."""
        K = len(self.ids)
        cand = [k for k in range(K) if abs(k - (K - 1) / 2) <= K / 4] or list(range(K))
        return self.ids[max(cand, key=lambda k: self.n_cells[k])]

    def with_transforms(self, transforms, anchor=None) -> "SectionStack":
        """Same specimen, other transforms (shares the AnnData and the indexing)."""
        out = SectionStack.__new__(SectionStack)
        out.__dict__.update(self.__dict__)
        out.T = dict(transforms)
        out.anchor = self.anchor if anchor is None else anchor
        return out

    def copy(self) -> "SectionStack":
        return self.with_transforms(self.T)

    def key(self, sec) -> str:
        return self.ids[sec] if isinstance(sec, (int, np.integer)) else str(sec)

    def xy(self, sec) -> np.ndarray:
        """Stack-frame xy of the cells of one section (section id or stack position)."""
        s = self.key(sec)
        return self.T[s](self.raw[self.rows[s]])

    def z(self) -> np.ndarray:
        """z (um) of every cell: section z plus its depth inside the section, centred on the median."""
        z = self.z_cells.copy()
        if self.depth is not None:
            z = z + np.clip(self.depth, 0, self.depth_clip)
        return z - np.median(z)

    def xyz(self) -> np.ndarray:
        """3D coordinates (um) of all cells, rows as in adata."""
        out = np.zeros((self.adata.n_obs, 3))
        for s in self.ids:
            out[self.rows[s], :2] = self.xy(s)
        out[:, 2] = self.z()
        return out

    def table(self) -> pd.DataFrame:
        """Per section: stack position k, z, cells, rotation (deg) and translation of its transform."""
        return pd.DataFrame({"k": np.arange(self.K), "z": self.z_sections, "n_cells": self.n_cells,
                             "rotation_deg": [self.T[s].angle_deg for s in self.ids],
                             "tx": [self.T[s].t[0] for s in self.ids], "ty": [self.T[s].t[1] for s in self.ids],
                             "anchor": [s == self.anchor for s in self.ids]}, index=pd.Index(self.ids, name="section"))

    def save(self, path) -> None:
        """Write the transforms (R, t, s per section) and the anchor to a .npz file."""
        np.savez(path, ids=np.array(self.ids, dtype=object), R=np.stack([self.T[s].R for s in self.ids]),
                 t=np.stack([self.T[s].t for s in self.ids]), s=np.array([self.T[s].s for s in self.ids]),
                 anchor=str(self.anchor))

    def load_transforms(self, path) -> "SectionStack":
        """This stack with the transforms read from a file written by `save`."""
        f = np.load(path, allow_pickle=True)
        T = {str(s): Similarity(R, t, sc) for s, R, t, sc in zip(f["ids"], f["R"], f["t"], f["s"])}
        if set(T) != set(self.ids):
            raise ValueError("the saved sections do not match this specimen")
        return self.with_transforms(T, anchor=str(f["anchor"]) or None)


def section_pairs(z_sections: np.ndarray, max_dz: float | None = None, spacing: float | None = None):
    """All section pairs (k, l, weight) with 0 < z_l - z_k <= max_dz, weight = spacing / dz.
    spacing: the nominal section thickness (default: median distance between neighbouring sections);
    max_dz: default 2.05 x spacing (neighbour and next neighbour; across a lost section the pair spanning it)."""
    z = np.asarray(z_sections, float)
    if spacing is None:
        dz = np.diff(z)
        spacing = float(np.median(dz[dz > 0])) if (dz > 0).any() else 1.0
    if max_dz is None:
        max_dz = 2.05 * spacing
    out = []
    for k in range(len(z)):
        for l in range(k + 1, len(z)):
            d = z[l] - z[k]
            if 0 < d <= max_dz:
                out.append((k, l, spacing / d))
    return out
