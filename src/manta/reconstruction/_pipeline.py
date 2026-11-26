"""3D reconstruction of a specimen from its serial sections."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import anndata as ad
import numpy as np
import pandas as pd

from ._chain import chain
from ._ends import place_end_sections
from ._features import pca_embedding
from ._joint import refine_rigid
from ._stack import SectionStack, section_pairs
from ._warp import refine_warp


@dataclass
class Reconstruction:
    """Result of `reconstruct`. `stacks` holds every stage ('chain', 'rigid', 'ends', 'warp'); `final` the last
    one. `end_log` lists every examined end section (scores, shift, rotation, whether it moved)."""
    stacks: dict
    final: SectionStack
    pairs: list
    end_log: pd.DataFrame = field(default_factory=pd.DataFrame)
    params: dict = field(default_factory=dict)
    seconds: float = 0.0

    def xyz(self) -> np.ndarray:
        return self.final.xyz()

    def sections(self) -> pd.DataFrame:
        """Per section: stack position, z, cells, the final rigid pose, and whether it is in an end zone / moved."""
        rig = getattr(self.final, "rigid", self.final)
        tab = rig.table()
        ends = self.stacks.get("ends")
        zone = set(getattr(ends, "zone", []) or [])
        tab["end_zone"] = [k in zone for k in tab["k"]]
        moved = set(self.end_log.loc[self.end_log["moved"], "k"]) if len(self.end_log) else set()
        tab["end_moved"] = [k in moved for k in tab["k"]]
        return tab


def reconstruct(adata: ad.AnnData, section_key: str = "section_id", z_key: str = "section_z_raw_um",
                spatial_key: str = "spatial_raw", embedding_key: str | None = None, genes=None, n_comps: int = 30,
                label_key: str | None = None, key_added: str = "spatial_manta_3d", max_dz: float | None = None,
                end_sections: bool = True, warp: bool = True, init: SectionStack | None = None,
                chain_kw: dict | None = None, rigid_kw: dict | None = None, end_kw: dict | None = None,
                warp_kw: dict | None = None, use_depth: bool = True, depth_clip: float = 10.0,
                verbose: bool = True) -> Reconstruction:
    """Reconstruct the 3D positions of all cells of a serially sectioned specimen.

    Input: one AnnData with all cells of the specimen.
      obs[section_key]   section of each cell
      obs[z_key]         z of the section (um; one value per section, gives the order and spacing)
      obsm[spatial_key]  slide coordinates (um); a 3rd column = depth of the cell inside its section (optional)
      expression         adata.X, or an existing embedding in obsm[embedding_key] (shared by all sections)

    Steps (docs/development/3D_reconstruction.md):
      1. embedding: PCA (n_comps) of the z-scored expression of `genes` (default all), unless embedding_key is given
      2. chain: every section onto its neighbour with `manta.align.rigid_joint` (or the centroid overlay if that
         agrees worse), composed outward from the anchor section (largest in the middle half)
      3. joint rigid refinement of all sections against all section pairs within max_dz (default 2.05 x the
         section spacing)
      4. end sections: the small sections at both ends are placed against a prediction extrapolated from their inner
         neighbours, defaulting to the common imaging orientation when the rotation is ambiguous
      5. warp: one smooth displacement field per section removes kinks between sections

    label_key: optional per-cell labels (e.g. cell types) used in step 2 to choose between candidate poses.
    init: a SectionStack to start from instead of step 2. *_kw: keyword arguments of the steps (`chain`,
    `refine_rigid`, `place_end_sections`, `refine_warp`).
    Writes adata.obsm[key_added] (3D coordinates, um) and adata.uns[key_added] (per-section table, end-section log,
    parameters). Returns a Reconstruction."""
    t0 = time.time()

    def log(msg):
        if verbose:
            print(f"{msg}  ({time.time() - t0:.0f} s)", flush=True)

    emb = (np.asarray(adata.obsm[embedding_key], float) if embedding_key is not None
           else pca_embedding(adata, n_comps=n_comps, genes=genes))
    base = SectionStack(adata, section_key, z_key, spatial_key, use_depth=use_depth, depth_clip=depth_clip)
    pairs = section_pairs(base.z_sections, max_dz)
    log(f"{base.K} sections, {adata.n_obs} cells, {len(pairs)} section pairs")
    stacks = {}
    if init is None:
        labels = None if label_key is None else adata.obs[label_key].to_numpy()
        st = chain(base, emb, labels=labels, **(chain_kw or {}))
        stacks["chain"] = st
        log("chain of pairwise fits")
    else:
        st = init
    st = refine_rigid(st, emb, pairs, verbose=verbose, **(rigid_kw or {}))
    stacks["rigid"] = st
    log("joint rigid refinement")
    end_log = pd.DataFrame()
    if end_sections:
        st = place_end_sections(st, emb, pairs, verbose=verbose, **(end_kw or {}))
        stacks["ends"] = st
        end_log = st.end_log
        log(f"end sections: {int(end_log['moved'].sum()) if len(end_log) else 0} of {len(st.zone)} placed anew")
    if warp:
        st = refine_warp(st, emb, pairs, verbose=verbose, **(warp_kw or {}))
        stacks["warp"] = st
        log("warp")
    res = Reconstruction(stacks=stacks, final=st, pairs=pairs, end_log=end_log,
                         params=dict(section_key=section_key, z_key=z_key, spatial_key=spatial_key,
                                     embedding_key=embedding_key, n_comps=n_comps, label_key=label_key,
                                     max_dz=max_dz, end_sections=end_sections, warp=warp, chain_kw=chain_kw,
                                     rigid_kw=rigid_kw, end_kw=end_kw, warp_kw=warp_kw),
                         seconds=time.time() - t0)
    adata.obsm[key_added] = res.xyz()
    adata.uns[key_added] = {"sections": res.sections().reset_index(), "end_log": end_log,
                            "params": {k: str(v) for k, v in res.params.items()}}
    log("done")
    return res
