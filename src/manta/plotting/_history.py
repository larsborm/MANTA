import anndata as ad
import matplotlib.pyplot as plt
import numpy as np


def rigid_history(adata: ad.AnnData):   
    if adata.uns["rigid_alignment"].get("method") == "joint":
        return rigid_joint_history(adata)

    history = adata.uns["rigid_alignment"]["history"]
    voxel_scales = np.array([h["bin_size"] for h in history])

    n_matches = np.array([
        h.get("n_matches", np.nan)
        for h in history
    ], dtype=float)

    n_inliers = np.array([
        h.get("n_inliers", np.nan)
        for h in history
    ], dtype=float)

    inlier_ratio = np.array([
        h.get("inlier_ratio", np.nan)
        for h in history
    ], dtype=float)

    mean_cosine_sim = np.array([
        h.get("mean_cosine_sim_inliers", np.nan)
        for h in history
    ], dtype=float)

    metrics = [
        (n_matches, "Number of matches", "Matches"),
        (n_inliers, "Number of inliers", "Inliers"),
        (inlier_ratio, "Inlier ratio", "Inlier ratio"),
        (mean_cosine_sim, "Mean cosine similarity", "Cosine similarity"),
    ]

    fig, axes = plt.subplots(
        1, 4,
        figsize=(18, 6),
        sharey=True,
        constrained_layout=True
    )
    
    for ax, (values, title, xlabel) in zip(axes, metrics):
        ax.plot(values, voxel_scales, marker="o")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.grid(True, alpha=0.3)
    
    axes[0].set_ylabel("Bin size")
    
    # Metrics with natural [0, 1] range
    axes[2].set_xlim(0, 1)
    axes[3].set_xlim(0, 1)
    
    fig.suptitle("Rigid Alignment Across Voxel Scales", fontsize=14)
    plt.show()


def rigid_joint_history(adata: ad.AnnData):
    """Diagnostics of `manta.align.rigid_joint`: the rotation-scan score per angle (one sharp peak = a confident
    angle; several similar peaks = ambiguous pose) and, per OT length scale, the rotation, the transported mass,
    the median residual and the coupling-weighted cosine similarity."""
    res = adata.uns["rigid_alignment"]
    scan, history = res.get("scan"), res["history"]

    fig, axes = plt.subplots(1, 5, figsize=(22, 4.5), constrained_layout=True)

    ax = axes[0]
    if scan is not None:
        ax.plot(scan["angles"], scan["scores"])
        ax.axvline(scan["angle"], color="k", ls="--", lw=0.8)
        ax.set_title(f"Rotation scan (best {scan['angle']:g} deg, peak ratio {scan.get('peak_ratio', np.nan):.2f})")
        ax.set_xlabel("Rotation (deg)")
        ax.set_ylabel("Cross-correlation score")
    else:
        ax.set_title("Rotation scan (skipped: init given)")
    ax.grid(True, alpha=0.3)

    sigmas = np.array([h["sigma"] for h in history], dtype=float)
    metrics = [
        ("rotation_deg", "Rotation (deg)"),
        ("transported_mass", "Transported mass"),
        ("median_residual", "Median residual"),
        ("mean_cosine_sim", "Coupled cosine similarity"),
    ]
    for ax, (key, title) in zip(axes[1:], metrics):
        ax.plot(sigmas, [h[key] for h in history], marker="o")
        ax.invert_xaxis()  # coarse -> fine
        ax.set_title(title)
        ax.set_xlabel("OT length scale sigma")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Joint Rigid Alignment", fontsize=14)
    plt.show()
