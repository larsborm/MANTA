"""Placement of the end sections of the stack against a prediction extrapolated from their inner neighbours.

An end section is a grazing cut through the surface of the specimen (mostly the outer cell layer). Matched against
its inner neighbour by expression alone, its best match is the neighbour's rim, where the neighbour also cuts the
outer layer, so the section slides off to the side. Here every end section is matched against a PREDICTED section:
the expression + occupancy images of its inner neighbours, extrapolated one step outward along z. Where the tissue
ends the prediction falls towards zero, so a placement that hangs over the neighbour's rim is penalised; a stack that
grows outward is extrapolated just as well, no shape is assumed.

See docs/development/3D_reconstruction.md for the full description.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._features import raster, scan_features
from ._joint import refine_rigid
from ._stack import SectionStack, Similarity, rotation


def end_zone(n_cells, zone_frac: float = 0.4):
    """Stack positions of the end zones: from each end inward, the contiguous run of sections with fewer than
    zone_frac x the robust maximum cell count (median of the 3 largest sections). The first section at or above the
    threshold, and everything inside it, is not part of the zone."""
    n = np.asarray(n_cells, float)
    K = len(n)
    thr = zone_frac * np.median(np.sort(n)[-3:])
    zone = []
    for rng in (range(K), range(K - 1, -1, -1)):
        for k in rng:
            if n[k] >= thr:
                break
            zone.append(k)
    return sorted(set(zone))


def dominant_angle(angles, weights) -> float:
    """Weighted circular mean (deg) of the angles, recomputed on the angles within 90 deg of the first estimate."""
    th, w = np.radians(np.asarray(angles, float)), np.asarray(weights, float)
    m = np.arctan2((w * np.sin(th)).sum(), (w * np.cos(th)).sum())
    keep = np.cos(th - m) > 0
    if keep.any():
        m = np.arctan2((w * np.sin(th))[keep].sum(), (w * np.cos(th))[keep].sum())
    return float(np.degrees(m) % 360)


def _wrap(deg):
    return float((deg + 180) % 360 - 180)


def _image(P, F, centre, n, px, sigma_px, occ):
    """Feature channels + one occupancy channel (value occ per cell) of points P on a grid centred on `centre`."""
    G = np.c_[F, np.full(len(F), occ, np.float32)] if occ > 0 else F
    return raster(P - centre, G, -np.full(2, n * px / 2), n, px, sigma_px)


def _rms_radius(P):
    return float(np.sqrt(((P - P.mean(0)) ** 2).sum(1).mean()))


def _ncc(S, T):
    return float((S * T).sum() / (np.sqrt((S ** 2).sum()) * np.sqrt((T ** 2).sum()) + 1e-12))


def predict_and_place(stack: SectionStack, k: int, F, inner, z, weight: float = 0.5, angles=np.arange(0, 360, 3),
                      px: float = 4.0, sigma: float = 10.0, occ: float = 1.0, max_f: float = 2.0,
                      max_shift: float = 1.0):
    """Best pose of section k against the prediction from its inner neighbours `inner` (nearest first).

    Prediction T = R1 + weight * (trend(z_k) - R1): R1 = image of the nearest inner section, trend = least-squares
    line through the images of the inner sections along z (two sections: R1 + f (R1 - R2)), evaluated at z_k with
    the extrapolation distance capped at max_f section spacings. Every rotation in `angles` and every shift is scored
    by normalised cross-correlation (FFT); the section centroid may move at most max_shift x the rms radius of the
    nearest inner section away from that section's centroid.
    Returns (pose, J_new, J_current, J0_new, J0_current); J0 = the same scores against the plain image R1."""
    s = stack.ids[k]
    src = stack.raw[stack.rows[s]]
    fs = F[stack.rows[s]]
    cs = src.mean(0)
    k1 = inner[0]
    P1 = stack.xy(k1)
    centre = P1.mean(0)
    ext = max(np.linalg.norm(src - cs, axis=1).max(), np.linalg.norm(P1 - centre, axis=1).max()) + 3 * sigma
    n = int(2 ** np.ceil(np.log2(2.5 * ext / px)))
    sp = sigma / px
    R1 = _image(P1, F[stack.rows[stack.ids[k1]]], centre, n, px, sp, occ)
    T = R1
    if len(inner) > 1 and weight > 0:
        Rs = [R1] + [_image(stack.xy(l), F[stack.rows[stack.ids[l]]], centre, n, px, sp, occ) for l in inner[1:]]
        zi = np.array([z[l] for l in inner])
        zbar = zi.mean()
        wz = (zi - zbar) / ((zi - zbar) ** 2).sum()
        step = abs(z[k1] - z[inner[1]])
        dz = np.clip(z[k] - z[k1], -max_f * step, max_f * step)
        slope = sum(w_ * R_ for w_, R_ in zip(wz, Rs))
        pred = sum(Rs) / len(Rs) + slope * (z[k1] + dz - zbar)
        T = R1 + weight * (pred - R1)
    Tn = np.sqrt((T ** 2).sum()) + 1e-12
    TF = np.fft.rfft2(T)
    sh = ((np.indices((n, n)).reshape(2, -1).T + n // 2) % n - n // 2) * px
    pen = np.where(np.linalg.norm(sh, axis=1) > max_shift * _rms_radius(P1), np.inf, 0.0)
    S_cur = _image(stack.xy(k), fs, centre, n, px, sp, occ)
    J_cur = _ncc(S_cur, T)
    best = (-np.inf, 0.0, np.zeros(2))
    for ang in angles:
        Q = (src - cs) @ rotation(ang).T + centre
        S = _image(Q, fs, centre, n, px, sp, occ)
        Sn = np.sqrt((S ** 2).sum()) + 1e-12
        cc = np.fft.irfft2((np.conj(np.fft.rfft2(S)) * TF).sum(0), s=(n, n)).ravel() / (Sn * Tn) - pen
        j = int(np.argmax(cc))
        if cc[j] > best[0]:
            best = (float(cc[j]), ang, sh[j])
    J, ang, shift = best
    Rm = rotation(ang)
    pose = Similarity(Rm, centre + shift - Rm @ cs)
    S_new = _image(pose(src), fs, centre, n, px, sp, occ)
    return pose, J, J_cur, _ncc(S_new, R1), _ncc(S_cur, R1)


def place_end_sections(stack: SectionStack, emb, pairs, zone_frac: float = 0.4, weight: float = 0.5,
                       n_predict: int = 2, orient_margin: float | None = 0.1, tol: float = 0.02,
                       min_move_um: float = 5.0, max_data_loss: float = 0.05, polish=(10, 6), min_cells: int = 10,
                       verbose: bool = False, **scan_kw) -> SectionStack:
    """Re-place the sections of the two end zones (`end_zone`), from the inside outward, each against the
    prediction from its `n_predict` inner neighbours (`predict_and_place`), then polish the whole stack with a short
    joint rigid refinement.

    Orientation (orient_margin; None = off): sections are imaged in roughly the same orientation, so the rotation of
    the large sections (outside the zones) relative to their slide coordinates gives a common angle theta0. Each end
    section is scanned in the half of the rotations within 90 deg of theta0 ('upright') and in the opposite half
    ('flipped'). The flipped half is used only if its best score beats the upright one by more than orient_margin;
    small end sections, whose outline looks alike after a half turn, therefore default to the common orientation, and
    one that sits flipped is turned back.

    A section moves to the new pose when the prediction score improves by more than tol and the pose changes by more
    than min_move_um or 5 deg, and the plain match with its nearest inner section drops by at most max_data_loss.
    Returns a new SectionStack with .end_log (one row per examined section), .zone and .theta0."""
    st = stack.copy()
    F = scan_features(emb)
    z = st.z_sections
    K = st.K
    ncell = st.n_cells
    zone = set(end_zone(ncell, zone_frac))
    ka = st.ids.index(st.anchor)
    theta0, half = None, np.arange(-90, 90, 3)
    if orient_margin is not None:
        rest = [k for k in range(K) if k not in zone]
        theta0 = dominant_angle([st.T[st.ids[k]].angle_deg for k in rest], ncell[rest])
    order = ([(k, -1) for k in range(ka + 1, K) if k in zone] +
             [(k, +1) for k in range(ka - 1, -1, -1) if k in zone])
    rows, moved = [], []
    for k, d in order:
        inner = [k + d * i for i in range(1, n_predict + 1) if 0 <= k + d * i < K]
        if not inner or ncell[k] < min_cells:
            continue
        s = st.ids[k]
        row = {"k": k, "section": s, "n_cells": int(ncell[k])}
        if theta0 is None:
            pose, J, J_cur, J0_new, J0_cur = predict_and_place(st, k, F, inner, z, weight=weight, **scan_kw)
            correction = False
        else:
            up = predict_and_place(st, k, F, inner, z, weight=weight, angles=(theta0 + half) % 360, **scan_kw)
            fl = predict_and_place(st, k, F, inner, z, weight=weight, angles=(theta0 + 180 + half) % 360, **scan_kw)
            use_flipped = fl[1] > up[1] + orient_margin
            pose, J, J_cur, J0_new, J0_cur = fl if use_flipped else up
            cur_flipped = abs(_wrap(st.T[s].angle_deg - theta0)) > 90
            correction = cur_flipped != use_flipped
            row.update(score_upright=up[1], score_flipped=fl[1], orientation="flipped" if use_flipped else "upright")
        raw = st.raw[st.rows[s]]
        dshift = float(np.linalg.norm(pose(raw).mean(0) - st.xy(k).mean(0)))
        dang = _wrap(pose.angle_deg - st.T[s].angle_deg)
        if correction:
            move = J >= J_cur - orient_margin
        else:
            move = J > J_cur + tol and (dshift > min_move_um or abs(dang) > 5)
        if max_data_loss is not None and J0_cur - J0_new > max_data_loss:
            move = False
        if move:
            st.T[s] = pose
            moved.append(k)
        row.update(score_current=J_cur, score_new=J, plain_current=J0_cur, plain_new=J0_new, shift_um=dshift,
                   rotation_deg=dang, orientation_correction=bool(correction), moved=bool(move))
        rows.append(row)
        if verbose:
            print(f"  end section {k:2d} ({ncell[k]} cells): score {J_cur:.3f} -> {J:.3f}, shift {dshift:5.1f} um, "
                  f"rotation {dang:+6.1f} deg{'  moved' if move else ''}", flush=True)
    if moved and polish:
        st = refine_rigid(st, emb, pairs, sigmas=tuple(polish))
    st.end_log = pd.DataFrame(rows)
    st.zone = sorted(zone)
    st.theta0 = theta0
    return st
