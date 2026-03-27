# -*- coding: utf-8 -*-
"""
General_Lattice_Generator.py
------------------------------
Fills an arbitrary watertight STL with a TPMS gyroid lattice at a specified
pore size and solid fraction.  The original STL is never modified; output is
saved alongside it with a descriptive suffix.

Supported lattice types : gyroid   (Schwartz-P, Diamond, etc. to follow)
Gradient modes          : Uniform  (gradient along axis or from surface planned)

Usage (edit the CONFIG section below, then run):
    python General_Lattice_Generator.py

Output filename convention:
    <OriginalName>_<LatticeType>_<PoreUm>um_<SolidPct>pct_<Mode>.stl
"""

import os
import sys
import numpy as np
import trimesh
from scipy.ndimage import distance_transform_edt

# ---------------------------------------------------------------------------
# CONFIG  – edit these values before running
# ---------------------------------------------------------------------------

INPUT_STL          = r"SkullCutout_OriginToZero.stl"  # main part

LATTICE_TYPE       = "gyroid"             # "gyroid" | "gyroid_inv" | "schwartz_p"
GYROID_MODE        = "sheet"           # "sheet" | "network"  (sheet = both channels open)
TARGET_PORE_UM     =2000                 # target pore diameter (um)
TARGET_SOLID_FRAC  = 0.25                # solid fraction in the main region

# ── Gradient modifier ──────────────────────────────────────────────────────
# Set MODIFIER_STL to None for a uniform lattice.
# The modifier is a second watertight STL that defines a sub-region of the
# part where a different solid fraction is desired.  The unit cell period
# (spatial frequency) stays the same everywhere; only the isovalue threshold
# changes, so the pore size naturally adjusts with the solid fraction.
MODIFIER_STL           = r"NU_OriginToSkullCutout.stl"
MODIFIER_SOLID_FRAC    = 0.1     # solid fraction inside the modifier region (NU)
MODIFIER_TRANSITION_MM = 0.0      # blend width in mm; 0 = sharp boundary
MODIFIER_FULL_SOLID    = False    # True = NU region is 100% solid (union with solid NU)
MODIFIER_LATTICE_MODE  = "network"  # "sheet" | "network" | None (None = same as GYROID_MODE)
# Modifier unit cell: scale factor for X and Y only (Z unchanged). 1.0 = same as main.
MODIFIER_L_SCALE_XY   = 2.5        # e.g. 2.0 = modifier has 2x larger cells in XY

# Rotation (degrees) around X, Y, Z to align pores. (0,0,0) = default.
# Try (0, 0, 45) for 45 deg around Z; (45, 0, 0) around X; (0, 45, 0) around Y.
# For vertical (Z-aligned) pores: gyroid channels are along <111>; ~54.7 deg from Z.
ROTATION_DEG           = (15, 15, 0)

# Resolution: voxels per gyroid unit-cell period.
# Higher = smoother surface but more memory / time.
#   8  → fast, suitable for large parts (>50 mm)
#  16  → production quality
VOXELS_PER_PERIOD  = 24

# ---------------------------------------------------------------------------
# Rotation helpers
# ---------------------------------------------------------------------------

def rotation_matrix_deg(rx, ry, rz):
    """Build rotation matrix R = Rz(rz) @ Ry(ry) @ Rx(rx) in degrees."""
    def rad(d):
        return np.deg2rad(d)
    cx, sx = np.cos(rad(rx)), np.sin(rad(rx))
    cy, sy = np.cos(rad(ry)), np.sin(rad(ry))
    cz, sz = np.cos(rad(rz)), np.sin(rad(rz))
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def apply_rotation(X, Y, Z, R):
    """Rotate (X,Y,Z) by matrix R. Returns (Xr, Yr, Zr) in rotated frame."""
    pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    pts_rot = (R @ pts.T).T
    sh = X.shape
    return (pts_rot[:, 0].reshape(sh), pts_rot[:, 1].reshape(sh),
            pts_rot[:, 2].reshape(sh))


def make_rotated_gradient(base_gradient_fn, R):
    """Return gradient fn that rotates coords in, computes gradient, rotates out."""
    def rotated_gradient(X, Y, Z, L_mm):
        Xr, Yr, Zr = apply_rotation(X, Y, Z, R)
        GXr, GYr, GZr = base_gradient_fn(Xr, Yr, Zr, L_mm)
        # Gradient in world frame: grad_world = R^T @ grad_rot
        G = np.stack([GXr.ravel(), GYr.ravel(), GZr.ravel()], axis=1)
        G_world = (R.T @ G.T).T
        sh = X.shape
        return (G_world[:, 0].reshape(sh), G_world[:, 1].reshape(sh),
                G_world[:, 2].reshape(sh))
    return rotated_gradient


# ---------------------------------------------------------------------------
# Gyroid helpers
# ---------------------------------------------------------------------------

def gyroid_unit_cell_period(pore_mm, solid_fraction):
    """
    Unit cell period L from target pore diameter and solid fraction.

    Formula (Ref: Al-Ketan & Abu Al-Rub, 2019; community empirical fit):
        L = pore_mm / (1 - 1.15 * solid_fraction)

    At sf=0   : L = pore_mm          (no material → pore ≈ full cell)
    At sf=0.5 : L = pore_mm / 0.425  (half-half → cell much larger than pore)
    Valid for solid fractions 0.10 – 0.60.
    """
    if not (0 < solid_fraction < 0.8):
        print("  WARNING: solid fraction outside reliable range (0–0.8).")
    L = pore_mm / (1.0 - 1.15 * solid_fraction)
    return L


def gyroid_isovalue(solid_fraction):
    """
    Isovalue t for a given solid fraction using the analytical linear
    approximation for the gyroid level set:
        t = 1.5 * (2 * solid_fraction - 1)

    Convention: solid = F < t  (marching cubes encloses the lower-field phase).
    At sf=0.5: t=0  (equal volumes).
    At sf=0.25: t=-0.75  (25% solid in the minority/trough phase).
    """
    t = 1.5 * (2.0 * solid_fraction - 1.0)
    print(f"  Gyroid isovalue  t = {t:.5f}  "
          f"(target solid fraction {solid_fraction*100:.1f}%)")
    return t


def gyroid_field(X, Y, Z, L_mm):
    """
    Evaluate F = sin(kx)cos(ky) + sin(ky)cos(kz) + sin(kz)cos(kx)
    for arrays X, Y, Z (mm) with unit-cell period L_mm.
    Solid region: F < isovalue  (network) or |F| < δ  (sheet).
    """
    k = 2.0 * np.pi / L_mm
    return (np.sin(k * X) * np.cos(k * Y) +
            np.sin(k * Y) * np.cos(k * Z) +
            np.sin(k * Z) * np.cos(k * X))


def gyroid_field_aniso(X, Y, Z, Lx, Ly, Lz):
    """
    Anisotropic gyroid: F = sin(kx*X)cos(ky*Y) + sin(ky*Y)cos(kz*Z) + sin(kz*Z)cos(kx*X).
    Lx, Ly, Lz can be scalars or arrays (same shape as X,Y,Z).
    """
    kx = 2.0 * np.pi / np.asarray(Lx, dtype=np.float64)
    ky = 2.0 * np.pi / np.asarray(Ly, dtype=np.float64)
    kz = 2.0 * np.pi / np.asarray(Lz, dtype=np.float64)
    return (np.sin(kx * X) * np.cos(ky * Y) +
            np.sin(ky * Y) * np.cos(kz * Z) +
            np.sin(kz * Z) * np.cos(kx * X))


def gyroid_gradient_field_aniso(X, Y, Z, Lx, Ly, Lz):
    """Analytical gradient of anisotropic gyroid."""
    kx = 2.0 * np.pi / np.asarray(Lx, dtype=np.float64)
    ky = 2.0 * np.pi / np.asarray(Ly, dtype=np.float64)
    kz = 2.0 * np.pi / np.asarray(Lz, dtype=np.float64)
    GX = kx * (np.cos(kx*X) * np.cos(ky*Y) - np.sin(kx*X) * np.sin(kz*Z))
    GY = ky * (np.cos(ky*Y) * np.cos(kz*Z) - np.sin(ky*Y) * np.sin(kx*X))
    GZ = kz * (np.cos(kz*Z) * np.cos(kx*X) - np.sin(kz*Z) * np.sin(ky*Y))
    return GX, GY, GZ


def gyroid_sheet_delta(solid_fraction):
    """
    Shell thickness δ for sheet gyroid such that vol(|F| < δ) = solid_fraction.

    The sheet is centered on the minimal surface (F = 0).  Numerically
    calibrated on a 64³ unit-cell sample.
    """
    n = 64
    x = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    Xc, Yc, Zc = np.meshgrid(x, x, x, indexing='ij')
    F = (np.sin(Xc) * np.cos(Yc) + np.sin(Yc) * np.cos(Zc) +
         np.sin(Zc) * np.cos(Xc))
    F = F.ravel()
    nv = len(F)

    # Binary search for δ
    lo, hi = 0.0, 1.6
    for _ in range(30):
        mid = (lo + hi) / 2
        frac = (np.abs(F) < mid).sum() / nv
        if frac < solid_fraction:
            lo = mid
        else:
            hi = mid
    delta = (lo + hi) / 2
    print(f"  Gyroid sheet delta = {delta:.5f}  "
          f"(target solid fraction {solid_fraction*100:.1f}%)")
    return delta


def compute_L_field(shape, inside_modifier, L_main, modifier_l_scale_xy):
    """
    Build (Lx, Ly, Lz) fields. Main uses L_main for all. Modifier uses
    L_main * modifier_l_scale_xy for X and Y only; Lz unchanged.
    """
    Lx = np.full(shape, L_main, dtype=np.float64)
    Ly = np.full(shape, L_main, dtype=np.float64)
    Lz = np.full(shape, L_main, dtype=np.float64)
    if inside_modifier is not None and modifier_l_scale_xy != 1.0:
        L_mod_xy = L_main * modifier_l_scale_xy
        Lx[inside_modifier] = L_mod_xy
        Ly[inside_modifier] = L_mod_xy
        # Lz unchanged
    return Lx, Ly, Lz


def compute_delta_field(shape, inside_modifier, delta_main, delta_modifier,
                       modifier_transition_mm, res):
    """
    Build spatially-varying shell thickness δ(x,y,z) for sheet mode.
    Same logic as compute_t_field but for the half-thickness δ.
    """
    delta_field = np.full(shape, delta_main, dtype=np.float64)
    if inside_modifier is None or not inside_modifier.any():
        return delta_field
    if modifier_transition_mm <= 0:
        delta_field[inside_modifier] = delta_modifier
    else:
        dist_out = distance_transform_edt(~inside_modifier) * res
        dist_in  = distance_transform_edt( inside_modifier) * res
        signed   = dist_out - dist_in
        alpha    = np.clip(signed / modifier_transition_mm, 0.0, 1.0)
        delta_field = delta_modifier + alpha * (delta_main - delta_modifier)
    return delta_field


def compute_t_field(shape, inside_modifier, t_main, t_modifier,
                    modifier_transition_mm, res):
    """
    Build a spatially-varying isovalue array t(x,y,z).

    Every voxel starts at t_main (the default threshold for TARGET_SOLID_FRAC).
    Voxels inside the modifier region are set to t_modifier.  If
    modifier_transition_mm > 0 a smooth blend is computed using the Euclidean
    distance transform so that the solid fraction ramps continuously between
    the two values over that distance.

    G = F - t_field is then extracted at G = 0, which is equivalent to
    extracting F = t(x,y,z) — the heterogeneous isovalue surface.
    """
    t_field = np.full(shape, t_main, dtype=np.float64)
    if inside_modifier is None or not inside_modifier.any():
        return t_field

    if modifier_transition_mm <= 0:
        # Sharp boundary: inside modifier → t_modifier, everywhere else → t_main
        t_field[inside_modifier] = t_modifier
    else:
        # Smooth transition using signed distance from the modifier surface.
        # distance_transform_edt gives distance to the nearest False voxel.
        dist_out  = distance_transform_edt(~inside_modifier) * res   # +outside
        dist_in   = distance_transform_edt( inside_modifier) * res   # +inside
        signed    = dist_out - dist_in       # positive = outside, negative = inside
        alpha     = np.clip(signed / modifier_transition_mm, 0.0, 1.0)
        t_field   = t_modifier + alpha * (t_main - t_modifier)

    return t_field


def gyroid_gradient_field(X, Y, Z, L_mm):
    """Analytical gradient (GX, GY, GZ) of the gyroid field at every point."""
    k = 2.0 * np.pi / L_mm
    GX = k * (np.cos(k*X) * np.cos(k*Y) - np.sin(k*X) * np.sin(k*Z))
    GY = k * (np.cos(k*Y) * np.cos(k*Z) - np.sin(k*Y) * np.sin(k*X))
    GZ = k * (np.cos(k*Z) * np.cos(k*X) - np.sin(k*Z) * np.sin(k*Y))
    return GX, GY, GZ


# ---------------------------------------------------------------------------
# Gyroid – inverted (complementary network)
# ---------------------------------------------------------------------------
# The standard gyroid has "solid = F < t", which selects the trough-side
# network.  Negating the field selects the peak-side network ("solid = -F < t"
# ↔ "solid = F > -t"), i.e. the other of the two interpenetrating gyroid
# networks.  Visually the two look similar at low solid fractions (both are
# rod-like networks), but they are geometrically distinct and complement each
# other perfectly to fill space.
# ---------------------------------------------------------------------------

def gyroid_inv_field(X, Y, Z, L_mm):
    """Negated gyroid field — selects the complementary network."""
    return -gyroid_field(X, Y, Z, L_mm)


def gyroid_inv_gradient_field(X, Y, Z, L_mm):
    """Gradient of the negated gyroid field."""
    GX, GY, GZ = gyroid_gradient_field(X, Y, Z, L_mm)
    return -GX, -GY, -GZ


# ---------------------------------------------------------------------------
# Schwartz-P helpers
# ---------------------------------------------------------------------------
# F_P = -(cos(kx) + cos(ky) + cos(kz))
#
# The negation keeps the convention "solid = F < t" consistent with the gyroid:
#   F_P < 0  →  cos-sum > 0  →  the face-connected strut network (solid) ✓
#   F_P > 0  →  cos-sum < 0  →  the central spherical pore channels (void) ✓
#
# The isovalue is numerically calibrated rather than an analytical fit,
# so it is exact regardless of solid fraction.
# ---------------------------------------------------------------------------

def schwartz_p_field(X, Y, Z, L_mm):
    """
    Evaluate F_P = -(cos(kx) + cos(ky) + cos(kz)).
    Negated so that 'solid = F_P < t' matches the same convention as gyroid.
    Range: [-3, 3].  Solid struts near the cube faces are the low-F region.
    """
    k = 2.0 * np.pi / L_mm
    return -(np.cos(k * X) + np.cos(k * Y) + np.cos(k * Z))


def schwartz_p_gradient_field(X, Y, Z, L_mm):
    """Analytical gradient of F_P = -(cos(kx)+cos(ky)+cos(kz))."""
    k = 2.0 * np.pi / L_mm
    GX = k * np.sin(k * X)
    GY = k * np.sin(k * Y)
    GZ = k * np.sin(k * Z)
    return GX, GY, GZ


def schwartz_p_isovalue(solid_fraction):
    """
    Numerically calibrated isovalue t such that
        vol(F_P < t) / vol(unit_cell) == solid_fraction.

    Uses np.quantile on a 64³ sample of the unit cell — exact for any sf,
    no analytical approximation needed.
    """
    n = 64
    x = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    Xc, Yc, Zc = np.meshgrid(x, x, x, indexing='ij')
    F_unit = -(np.cos(Xc) + np.cos(Yc) + np.cos(Zc))
    t = float(np.quantile(F_unit, solid_fraction))
    print(f"  Schwartz-P isovalue  t = {t:.5f}  "
          f"(target solid fraction {solid_fraction*100:.1f}%)")
    return t


def schwartz_p_unit_cell_period(pore_mm, solid_fraction):
    """
    Unit cell period for Schwartz-P.  Uses the same empirical formula as the
    gyroid (Al-Ketan & Abu Al-Rub, 2019) for a comparable spatial scale.
    True pore size calibration for Schwartz-P is topology-dependent and can
    be refined in a future version.
    """
    return gyroid_unit_cell_period(pore_mm, solid_fraction)


# ---------------------------------------------------------------------------
# TPMS registry – add new surface types here
# ---------------------------------------------------------------------------

TPMS_REGISTRY = {
    "gyroid": {
        "field":    gyroid_field,
        "gradient": gyroid_gradient_field,
        "isovalue": gyroid_isovalue,
        "period":   gyroid_unit_cell_period,
    },
    "gyroid_inv": {
        "field":    gyroid_inv_field,
        "gradient": gyroid_inv_gradient_field,
        "isovalue": gyroid_isovalue,          # same calibration, field is negated
        "period":   gyroid_unit_cell_period,
    },
    "schwartz_p": {
        "field":    schwartz_p_field,
        "gradient": schwartz_p_gradient_field,
        "isovalue": schwartz_p_isovalue,
        "period":   schwartz_p_unit_cell_period,
    },
}


# ---------------------------------------------------------------------------
# Dual contouring helpers
# ---------------------------------------------------------------------------

def _accumulate_edges(axis, F, GX, GY, GZ, xi, yi, zi, res, iso,
                      ATA, ATb, mass_sum, cnt):
    """
    Find all isovalue crossings along `axis` and scatter their QEF
    contributions (outer-product of normal, rhs, mass point) to the
    four adjacent cells.
    """
    ncx, ncy, ncz = ATA.shape[:3]

    if axis == 0:
        F0, F1 = F[:-1, :, :], F[1:, :, :]
        G0 = np.stack([GX[:-1,:,:], GY[:-1,:,:], GZ[:-1,:,:]], axis=-1)
        G1 = np.stack([GX[1:, :,:], GY[1:, :,:], GZ[1:, :,:]], axis=-1)
    elif axis == 1:
        F0, F1 = F[:, :-1, :], F[:, 1:, :]
        G0 = np.stack([GX[:,:-1,:], GY[:,:-1,:], GZ[:,:-1,:]], axis=-1)
        G1 = np.stack([GX[:, 1:,:], GY[:, 1:,:], GZ[:, 1:,:]], axis=-1)
    else:
        F0, F1 = F[:, :, :-1], F[:, :, 1:]
        G0 = np.stack([GX[:,:,:-1], GY[:,:,:-1], GZ[:,:,:-1]], axis=-1)
        G1 = np.stack([GX[:,:, 1:], GY[:,:, 1:], GZ[:,:, 1:]], axis=-1)

    ei, ej, ek = np.where((F0 < iso) != (F1 < iso))
    if len(ei) == 0:
        return

    f0, f1 = F0[ei, ej, ek], F1[ei, ej, ek]
    t = np.clip((iso - f0) / (f1 - f0 + 1e-30), 0.0, 1.0)

    if axis == 0:
        p = np.stack([xi[ei] + t * res, yi[ej], zi[ek]], axis=1)
    elif axis == 1:
        p = np.stack([xi[ei], yi[ej] + t * res, zi[ek]], axis=1)
    else:
        p = np.stack([xi[ei], yi[ej], zi[ek] + t * res], axis=1)

    n = G0[ei, ej, ek] + t[:, None] * (G1[ei, ej, ek] - G0[ei, ej, ek])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-10)

    dp    = (n * p).sum(axis=1)
    outer = n[:, :, None] * n[:, None, :]   # (N, 3, 3)

    # Each crossing is shared by 4 cells in the two perpendicular directions.
    if axis == 0:
        cell_pairs = [(ej - 1, ek - 1), (ej, ek - 1), (ej, ek), (ej - 1, ek)]
        for (cj, ck) in cell_pairs:
            v = (cj >= 0) & (cj < ncy) & (ck >= 0) & (ck < ncz)
            ci = ei[v]; cjv = cj[v]; ckv = ck[v]
            np.add.at(ATA,      (ci, cjv, ckv), outer[v])
            np.add.at(ATb,      (ci, cjv, ckv), (n * dp[:, None])[v])
            np.add.at(mass_sum, (ci, cjv, ckv), p[v])
            np.add.at(cnt,      (ci, cjv, ckv), 1)
    elif axis == 1:
        cell_pairs = [(ei - 1, ek - 1), (ei - 1, ek), (ei, ek), (ei, ek - 1)]
        for (ci, ck) in cell_pairs:
            v = (ci >= 0) & (ci < ncx) & (ck >= 0) & (ck < ncz)
            civ = ci[v]; cjv = ej[v]; ckv = ck[v]
            np.add.at(ATA,      (civ, cjv, ckv), outer[v])
            np.add.at(ATb,      (civ, cjv, ckv), (n * dp[:, None])[v])
            np.add.at(mass_sum, (civ, cjv, ckv), p[v])
            np.add.at(cnt,      (civ, cjv, ckv), 1)
    else:
        cell_pairs = [(ei - 1, ej - 1), (ei, ej - 1), (ei, ej), (ei - 1, ej)]
        for (ci, cj) in cell_pairs:
            v = (ci >= 0) & (ci < ncx) & (cj >= 0) & (cj < ncy)
            civ = ci[v]; cjv = cj[v]; ckv = ek[v]
            np.add.at(ATA,      (civ, cjv, ckv), outer[v])
            np.add.at(ATb,      (civ, cjv, ckv), (n * dp[:, None])[v])
            np.add.at(mass_sum, (civ, cjv, ckv), p[v])
            np.add.at(cnt,      (civ, cjv, ckv), 1)


def _gen_quads(axis, F, iso, cell_idx):
    """
    For every crossing edge along `axis` emit two triangles whose winding
    gives a normal pointing away from the solid (F < iso) region.

    Verified CCW orderings (right-hand cross-product gives +axis normal):
      axis=0 (X): V0=(j-1,k-1), V1=(j,k-1), V2=(j,k), V3=(j-1,k)
      axis=1 (Y): V0=(i-1,k-1), V1=(i-1,k),  V2=(i,k), V3=(i,k-1)
      axis=2 (Z): V0=(i-1,j-1), V1=(i,j-1),  V2=(i,j), V3=(i-1,j)
    """
    ncx, ncy, ncz = cell_idx.shape

    if axis == 0:
        F0, F1 = F[:-1, :, :], F[1:, :, :]
    elif axis == 1:
        F0, F1 = F[:, :-1, :], F[:, 1:, :]
    else:
        F0, F1 = F[:, :, :-1], F[:, :, 1:]

    ei, ej, ek = np.where((F0 < iso) != (F1 < iso))
    if len(ei) == 0:
        return np.empty((0, 3), dtype=np.int64)

    f0v = F0[ei, ej, ek]

    if axis == 0:
        # valid: all 4 cells (ei, ej±, ek±) must be in bounds
        v = (ej >= 1) & (ej < ncy) & (ek >= 1) & (ek < ncz)
        ei, ej, ek, f0v = ei[v], ej[v], ek[v], f0v[v]
        id0 = cell_idx[ei, ej - 1, ek - 1]
        id1 = cell_idx[ei, ej,     ek - 1]
        id2 = cell_idx[ei, ej,     ek    ]
        id3 = cell_idx[ei, ej - 1, ek    ]
    elif axis == 1:
        v = (ei >= 1) & (ei < ncx) & (ek >= 1) & (ek < ncz)
        ei, ej, ek, f0v = ei[v], ej[v], ek[v], f0v[v]
        id0 = cell_idx[ei - 1, ej, ek - 1]
        id1 = cell_idx[ei - 1, ej, ek    ]
        id2 = cell_idx[ei,     ej, ek    ]
        id3 = cell_idx[ei,     ej, ek - 1]
    else:
        v = (ei >= 1) & (ei < ncx) & (ej >= 1) & (ej < ncy)
        ei, ej, ek, f0v = ei[v], ej[v], ek[v], f0v[v]
        id0 = cell_idx[ei - 1, ej - 1, ek]
        id1 = cell_idx[ei,     ej - 1, ek]
        id2 = cell_idx[ei,     ej,     ek]
        id3 = cell_idx[ei - 1, ej,     ek]

    # Only emit where all 4 surrounding cells have a vertex
    ok = (id0 >= 0) & (id1 >= 0) & (id2 >= 0) & (id3 >= 0)
    id0, id1, id2, id3 = id0[ok], id1[ok], id2[ok], id3[ok]
    solid0 = f0v[ok] < iso

    tris = []
    if solid0.any():
        # F0 < iso → normal in +axis → CCW: (V0,V1,V2) and (V0,V2,V3)
        a, b, c, d = id0[solid0], id1[solid0], id2[solid0], id3[solid0]
        tris += [np.stack([a, b, c], axis=1), np.stack([a, c, d], axis=1)]
    s1 = ~solid0
    if s1.any():
        # F0 > iso → normal in -axis → CW: (V0,V2,V1) and (V0,V3,V2)
        a, b, c, d = id0[s1], id1[s1], id2[s1], id3[s1]
        tris += [np.stack([a, c, b], axis=1), np.stack([a, d, c], axis=1)]

    return np.vstack(tris) if tris else np.empty((0, 3), dtype=np.int64)


def dual_contour(F, iso, res, origin, L_mm, gradient_fn=None):
    """
    Extract the isosurface at `iso` from a TPMS scalar field `F` using
    dual contouring with analytical surface normals.

    For each grid cell containing a surface crossing, a vertex is placed at
    the position that best satisfies all local normal constraints (QEF
    minimisation).  Adjacent cells sharing a crossing edge are then connected
    as quads (two triangles each).  This produces fewer, better-shaped
    triangles than marching cubes and naturally preserves sharp features.

    Returns (vertices, faces) numpy arrays in world-space mm coordinates.
    """
    nx, ny, nz = F.shape
    ncx, ncy, ncz = nx - 1, ny - 1, nz - 1

    xi = origin[0] + np.arange(nx, dtype=np.float64) * res
    yi = origin[1] + np.arange(ny, dtype=np.float64) * res
    zi = origin[2] + np.arange(nz, dtype=np.float64) * res
    X, Y, Z = np.meshgrid(xi, yi, zi, indexing='ij')

    print("  Computing analytical gradient ...")
    grad_fn = gradient_fn if gradient_fn is not None else gyroid_gradient_field
    GX, GY, GZ = grad_fn(X, Y, Z, L_mm)
    del X, Y, Z

    # Per-cell QEF accumulators
    ATA      = np.zeros((ncx, ncy, ncz, 3, 3), dtype=np.float64)
    ATb      = np.zeros((ncx, ncy, ncz, 3),    dtype=np.float64)
    mass_sum = np.zeros((ncx, ncy, ncz, 3),    dtype=np.float64)
    cnt      = np.zeros((ncx, ncy, ncz),        dtype=np.int32)

    print("  Accumulating QEF matrices ...")
    for axis in range(3):
        _accumulate_edges(axis, F, GX, GY, GZ, xi, yi, zi, res, iso,
                          ATA, ATb, mass_sum, cnt)
    del GX, GY, GZ

    # Solve QEF for every cell that has at least one edge crossing
    active = cnt > 0
    n_active = int(active.sum())
    print(f"  Active cells : {n_active:,}")

    ai, aj, ak = np.where(active)
    mass_pts = mass_sum[ai, aj, ak] / cnt[ai, aj, ak, None]
    ATA_a    = ATA[ai, aj, ak]      # (n_active, 3, 3)
    ATb_a    = ATb[ai, aj, ak]      # (n_active, 3)
    del ATA, ATb, mass_sum, cnt

    # Tikhonov regularisation: pull underdetermined cells toward the centroid
    # of their edge crossings.  eps=0.1 is small relative to typical QEF
    # eigenvalues (~2-8) so it only activates for nearly flat configurations.
    eps = 0.1
    ATA_a += eps * np.eye(3)[None]
    ATb_a += eps * mass_pts

    print("  Solving QEF per cell ...")
    # np.linalg.solve batch mode needs b as (..., m, k); add trailing dim then squeeze
    verts = np.linalg.solve(ATA_a, ATb_a[..., None])[..., 0]  # (n_active, 3)
    del ATA_a, ATb_a

    # Clamp each vertex to its cell's bounding box to prevent floaters
    cell_min = np.stack([origin[0] + ai * res,
                         origin[1] + aj * res,
                         origin[2] + ak * res], axis=1)
    verts = np.clip(verts, cell_min, cell_min + res)

    # Build sparse cell → vertex index lookup
    cell_idx          = np.full((ncx, ncy, ncz), -1, dtype=np.int64)
    cell_idx[ai, aj, ak] = np.arange(n_active)

    # Connect crossing edges into quads (2 triangles each)
    print("  Generating quads ...")
    tri_list = [_gen_quads(ax, F, iso, cell_idx) for ax in range(3)]
    faces    = np.vstack([t for t in tri_list if len(t) > 0])

    print(f"  Dual contour : {len(verts):,} vertices, {len(faces):,} faces")
    return verts, faces


# ---------------------------------------------------------------------------
# Pre-flight size estimator
# ---------------------------------------------------------------------------

def preflight_check(mesh, pore_um, solid_fraction, voxels_per_period, L_mm=None):
    """
    Estimate grid size, RAM, face count and output file size BEFORE running.
    Prints a report and returns True if safe to proceed, False if the user
    declines after a warning.  Pass L_mm to override the default gyroid formula.
    """
    if L_mm is None:
        L_mm = gyroid_unit_cell_period(pore_um / 1000.0, solid_fraction)
    res  = L_mm / voxels_per_period

    extents = mesh.extents
    pad     = res
    nx = int(np.ceil((extents[0] + 2*pad) / res)) + 1
    ny = int(np.ceil((extents[1] + 2*pad) / res)) + 1
    nz = int(np.ceil((extents[2] + 2*pad) / res)) + 1
    n_vox = nx * ny * nz

    # Part volume fraction inside bounding box (rough: part vol / bbox vol)
    part_fill = mesh.volume / np.prod(extents) if mesh.is_volume else 0.5
    n_part_vox = n_vox * part_fill

    # RAM estimate (bytes):
    #   inside mask (bool)  +  lattice result (bool)  +  gyroid chunk (float32)
    CHUNK_Z   = 40
    ram_bool  = 2 * n_vox                          # two bool arrays
    ram_chunk = nx * ny * CHUNK_Z * 4              # one float32 Z-slab
    ram_total_mb = (ram_bool + ram_chunk) / 1024**2

    # Face estimate: empirical factor from gyroid marching-cubes runs
    #   faces ≈ n_part_vox * 5.6 * sf * (1-sf)
    sf = solid_fraction
    n_faces_est = int(n_part_vox * 5.6 * sf * (1.0 - sf))

    # STL binary file size: 84-byte header + 50 bytes per triangle
    stl_mb = (84 + n_faces_est * 50) / 1024**2

    # Trim-mesh voxelization RAM: trimesh stores the fill result as bool
    vox_ram_mb = n_vox / 1024**2   # 1 byte per voxel

    print("\n" + "="*60)
    print("  PRE-FLIGHT SIZE ESTIMATE")
    print("="*60)
    print(f"  Unit cell period L : {L_mm*1000:.1f} um")
    print(f"  Voxel resolution   : {res*1000:.1f} um")
    print(f"  Grid dimensions    : {nx} x {ny} x {nz}")
    print(f"  Total voxels       : {n_vox/1e6:.1f} M")
    print(f"  Peak RAM (approx)  : {ram_total_mb:.0f} MB")
    print(f"  Est. output faces  : {n_faces_est/1e6:.1f} M")
    print(f"  Est. STL file size : {stl_mb:.0f} MB")
    print("="*60)

    warnings = []
    if n_vox > 200e6:
        warnings.append(f"Very large grid ({n_vox/1e6:.0f} M voxels). "
                        f"Consider reducing VOXELS_PER_PERIOD or increasing "
                        f"TARGET_PORE_UM.")
    if ram_total_mb > 4000:
        warnings.append(f"Estimated peak RAM {ram_total_mb:.0f} MB may exceed "
                        f"available memory.")
    if n_faces_est > 20e6:
        warnings.append(f"Estimated {n_faces_est/1e6:.0f} M output faces. "
                        f"The STL will be very large and slow to open. "
                        f"Consider decimating afterwards.")
    if stl_mb > 500:
        warnings.append(f"Estimated output file {stl_mb:.0f} MB. "
                        f"Ensure sufficient disk space.")

    if warnings:
        print("\n  WARNINGS:")
        for w in warnings:
            print(f"    ! {w}")
        print()
        ans = input("  Continue anyway? [y/N]: ").strip().lower()
        if ans != 'y':
            print("  Aborted.")
            return False

    print()
    return True


# ---------------------------------------------------------------------------
# Core generator
# ---------------------------------------------------------------------------

def generate(input_stl, lattice_type, pore_um, solid_fraction,
             voxels_per_period=VOXELS_PER_PERIOD,
             modifier_stl=None, modifier_solid_frac=None,
             modifier_transition_mm=0.0,
             modifier_full_solid=False,
             modifier_lattice_mode=None,
             modifier_l_scale_xy=1.0,
             rotation_deg=(0, 0, 0),
             gyroid_mode="sheet"):

    # ── 1. Load the input mesh ─────────────────────────────────────────────
    stl_path = os.path.abspath(input_stl)
    if not os.path.isfile(stl_path):
        # try relative to script location, then one level up
        for base in (os.path.dirname(__file__),
                     os.path.dirname(os.path.dirname(__file__))):
            candidate = os.path.join(base, input_stl)
            if os.path.isfile(candidate):
                stl_path = candidate
                break
    print(f"\nLoading '{stl_path}' ...")
    mesh = trimesh.load(stl_path)
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = mesh.dump(concatenate=True)
    bb_min = mesh.bounds[0]
    bb_max = mesh.bounds[1]
    print(f"  Extents   : {mesh.extents[0]:.2f} x {mesh.extents[1]:.2f} x "
          f"{mesh.extents[2]:.2f} mm")
    print(f"  Watertight: {mesh.is_watertight}")
    if not mesh.is_watertight:
        print("  WARNING: mesh is not watertight; interior detection may be "
              "inaccurate.")

    # ── 1b. Look up TPMS type ──────────────────────────────────────────────
    lt = lattice_type.lower()
    if lt not in TPMS_REGISTRY:
        raise ValueError(f"Unknown lattice type '{lattice_type}'. "
                         f"Supported: {list(TPMS_REGISTRY)}")
    tpms        = TPMS_REGISTRY[lt]
    field_fn    = tpms["field"]
    gradient_fn = tpms["gradient"]
    isovalue_fn = tpms["isovalue"]
    period_fn   = tpms["period"]

    # ── 1c. Pre-flight check ───────────────────────────────────────────────
    sf_for_preflight = (max(solid_fraction, modifier_solid_frac)
                        if modifier_solid_frac is not None else solid_fraction)
    L_pre = period_fn(pore_um / 1000.0, sf_for_preflight)
    ok = preflight_check(mesh, pore_um, sf_for_preflight, voxels_per_period,
                         L_mm=L_pre)
    if not ok:
        return None

    # ── 2. Derive lattice parameters ───────────────────────────────────────
    # When a modifier is present the unit cell period is based on the LARGER
    # of the two solid fractions.  This keeps both the forward and inverse
    # configurations on the same grid (same L, same voxel resolution), and
    # avoids an unnecessarily fine grid when the main region is thin.
    sf_for_L = solid_fraction
    if modifier_solid_frac is not None:
        sf_for_L = max(solid_fraction, modifier_solid_frac)
    L_mm = period_fn(pore_um / 1000.0, sf_for_L)
    res  = L_mm / voxels_per_period          # voxel pitch (mm)
    print(f"\n{lt.replace('_',' ').title()} parameters:")
    print(f"  Target pore diam    : {pore_um:.0f} um")
    print(f"  Target solid frac   : {solid_fraction*100:.1f}%")
    if sf_for_L != solid_fraction:
        print(f"  L derived from SF   : {sf_for_L*100:.1f}%  "
              f"(larger of main / modifier)")
    print(f"  Unit cell period L  : {L_mm*1000:.1f} um")
    print(f"  Voxel resolution    : {res*1000:.1f} um")

    # ── 3. Voxelise the input STL to get interior mask ─────────────────────
    # trimesh.voxelized uses ray-casting fill and is reliable for watertight
    # meshes.  The voxel grid origin / shape is returned in the VoxelGrid obj.
    print(f"\nVoxelising input mesh at {res*1000:.1f} um pitch ...")
    vox   = mesh.voxelized(pitch=res).fill()
    shape = vox.matrix.shape                 # (nx, ny, nz)
    origin = vox.translation                 # (3,) world-coord of voxel [0,0,0]
    inside = vox.matrix                      # True = inside the part

    actual_fill = inside.mean() * 100
    print(f"  Grid : {shape[0]} x {shape[1]} x {shape[2]}  "
          f"(fill {actual_fill:.1f}%)")

    # ── 3b. Load modifier mesh and compute inside mask ─────────────────────
    inside_modifier = None
    mod_mesh = None
    if modifier_stl is not None and (modifier_solid_frac is not None or modifier_full_solid
                                      or modifier_lattice_mode is not None
                                      or modifier_l_scale_xy != 1.0):
        mod_path = os.path.abspath(modifier_stl)
        if not os.path.isfile(mod_path):
            for base in (os.path.dirname(__file__),
                         os.path.dirname(os.path.dirname(__file__))):
                c = os.path.join(base, modifier_stl)
                if os.path.isfile(c):
                    mod_path = c
                    break
        print(f"\nLoading modifier '{mod_path}' ...")
        mod_mesh = trimesh.load(mod_path)
        if not isinstance(mod_mesh, trimesh.Trimesh):
            mod_mesh = mod_mesh.dump(concatenate=True)
        print(f"  Modifier extents : "
              f"{mod_mesh.extents[0]:.2f} x {mod_mesh.extents[1]:.2f} x "
              f"{mod_mesh.extents[2]:.2f} mm")

        # Voxelize the modifier at the same pitch as the main part, then map
        # its voxels onto the main grid by computing the integer index offset
        # between the two grid origins.  This is O(modifier voxels) — orders
        # of magnitude faster than a per-point ray-cast.
        print("  Voxelising modifier ...")
        mod_vox = mod_mesh.voxelized(pitch=res).fill()
        offset  = np.round((mod_vox.translation - origin) / res).astype(int)

        mi, mj, mk = np.where(mod_vox.matrix)
        gi = mi + offset[0]
        gj = mj + offset[1]
        gk = mk + offset[2]
        valid = ((gi >= 0) & (gi < shape[0]) &
                 (gj >= 0) & (gj < shape[1]) &
                 (gk >= 0) & (gk < shape[2]))
        inside_modifier = np.zeros(shape, dtype=bool)
        inside_modifier[gi[valid], gj[valid], gk[valid]] = True
        n_mod = int(inside_modifier.sum())
        print(f"  Modifier voxels  : {n_mod:,}  "
              f"({n_mod / max(inside.sum(), 1) * 100:.1f}% of part)")

    # ── 3c. Rotation setup ────────────────────────────────────────────────
    rx, ry, rz = rotation_deg if len(rotation_deg) >= 3 else (0, 0, 0)
    R = rotation_matrix_deg(rx, ry, rz) if (rx or ry or rz) else None
    if R is not None:
        print(f"\n  Rotation : ({rx:.0f}, {ry:.0f}, {rz:.0f}) deg around X, Y, Z")

    # ── 4. Compute TPMS field ──────────────────────────────────────────────
    use_aniso_L = (modifier_l_scale_xy != 1.0 and inside_modifier is not None
                   and lt in ("gyroid", "gyroid_inv"))
    if use_aniso_L:
        print(f"\n  Modifier L scale (XY only): {modifier_l_scale_xy:.2f}x")

    print(f"\nComputing {lt.replace('_',' ').title()} field ...")
    xi = origin[0] + np.arange(shape[0]) * res
    yi = origin[1] + np.arange(shape[1]) * res
    zi = origin[2] + np.arange(shape[2]) * res
    Xg, Yg, Zg = np.meshgrid(xi, yi, zi, indexing='ij')
    if R is not None:
        Xg, Yg, Zg = apply_rotation(Xg, Yg, Zg, R)

    if use_aniso_L:
        Lx_f, Ly_f, Lz_f = compute_L_field(shape, inside_modifier, L_mm,
                                            modifier_l_scale_xy)
        F = gyroid_field_aniso(Xg, Yg, Zg, Lx_f, Ly_f, Lz_f).astype(np.float64)
        if lt == "gyroid_inv":
            F = -F

        def _aniso_grad(X, Y, Z, L_mm):
            GX, GY, GZ = gyroid_gradient_field_aniso(X, Y, Z, Lx_f, Ly_f, Lz_f)
            return (-GX, -GY, -GZ) if lt == "gyroid_inv" else (GX, GY, GZ)

        base_gradient_fn = _aniso_grad
    else:
        F = field_fn(Xg, Yg, Zg, L_mm).astype(np.float64)
        base_gradient_fn = gradient_fn
    del Xg, Yg, Zg

    use_sheet = (gyroid_mode == "sheet" and lt in ("gyroid", "gyroid_inv"))
    mod_mode = (modifier_lattice_mode or gyroid_mode).lower()
    use_hybrid = (modifier_stl and modifier_solid_frac is not None
                  and inside_modifier is not None and lt in ("gyroid", "gyroid_inv")
                  and mod_mode != gyroid_mode.lower())

    if use_hybrid:
        # ── Hybrid: main = one mode (network/sheet), modifier = other mode ──
        # Generate both meshes, clip to part, then: main_region = lattice - modifier,
        # mod_region = lattice ∩ modifier, final = main_region ∪ mod_region
        print(f"\nHybrid mode: main={gyroid_mode} ({solid_fraction*100:.1f}% SF), "
              f"modifier={mod_mode} ({modifier_solid_frac*100:.1f}% SF)")
        grad_fn = (make_rotated_gradient(base_gradient_fn, R)
                  if R is not None else base_gradient_fn)

        def _clip_to_part(m):
            try:
                import manifold3d as mf
                def _to_mf(tm):
                    v = np.ascontiguousarray(tm.vertices, dtype=np.float32)
                    f = np.ascontiguousarray(tm.faces, dtype=np.uint32)
                    return mf.Manifold(mf.Mesh(vert_properties=v, tri_verts=f))
                raw = (_to_mf(m) ^ _to_mf(mesh)).to_mesh()
                return trimesh.Trimesh(
                    np.array(raw.vert_properties, dtype=np.float64),
                    np.array(raw.tri_verts, dtype=np.int64).reshape(-1, 3), process=True)
            except Exception:
                import pyvista as pv
                def _pv(tm):
                    return pv.PolyData(tm.vertices,
                        np.c_[np.full(len(tm.faces), 3, dtype=int), tm.faces])
                c = _pv(m).clip_surface(_pv(mesh), invert=True).triangulate().clean()
                return trimesh.Trimesh(
                    np.array(c.points), np.array(c.faces).reshape(-1, 4)[:, 1:], process=True)

        def _bool_op(a, b, op):
            try:
                import manifold3d as mf
                def _to_mf(tm):
                    v = np.ascontiguousarray(tm.vertices, dtype=np.float32)
                    f = np.ascontiguousarray(tm.faces, dtype=np.uint32)
                    return mf.Manifold(mf.Mesh(vert_properties=v, tri_verts=f))
                ma, mb = _to_mf(a), _to_mf(b)
                if op == "diff":   r = ma - mb
                elif op == "inter": r = ma ^ mb
                else:               r = ma + mb
                raw = r.to_mesh()
                return trimesh.Trimesh(
                    np.array(raw.vert_properties, dtype=np.float64),
                    np.array(raw.tri_verts, dtype=np.int64).reshape(-1, 3), process=True)
            except Exception:
                import pyvista as pv
                def _pv(tm):
                    return pv.PolyData(tm.vertices,
                        np.c_[np.full(len(tm.faces), 3, dtype=int), tm.faces])
                if op == "diff":   c = _pv(a).boolean_difference(_pv(b))
                elif op == "inter": c = _pv(a).boolean_intersection(_pv(b))
                else:               c = _pv(a).boolean_union(_pv(b))
                return trimesh.Trimesh(
                    np.array(c.points), np.array(c.faces).reshape(-1, 4)[:, 1:], process=True)

        # 1. Network mesh (main region)
        t_net = gyroid_isovalue(solid_fraction)
        G_net = F - t_net
        G_net[0,:,:] = 1.0; G_net[-1,:,:] = 1.0
        G_net[:,0,:] = 1.0; G_net[:,-1,:] = 1.0
        G_net[:,:,0] = 1.0; G_net[:,:,-1] = 1.0
        print("\nExtracting network lattice (main region) ...")
        v_net, f_net = dual_contour(G_net, 0.0, res, origin, L_mm, gradient_fn=grad_fn)
        mesh_net = trimesh.Trimesh(vertices=v_net, faces=f_net, process=True)
        mesh_net = _clip_to_part(mesh_net)
        network_main = _bool_op(mesh_net, mod_mesh, "diff") if len(mesh_net.faces) > 0 else None

        # 2. Sheet mesh (modifier region)
        delta_sheet = gyroid_sheet_delta(modifier_solid_frac)
        G_outer = F - delta_sheet
        G_outer[0,:,:] = 1.0; G_outer[-1,:,:] = 1.0
        G_outer[:,0,:] = 1.0; G_outer[:,-1,:] = 1.0
        G_outer[:,:,0] = 1.0; G_outer[:,:,-1] = 1.0
        G_inner = F + delta_sheet
        G_inner[0,:,:] = 1.0; G_inner[-1,:,:] = 1.0
        G_inner[:,0,:] = 1.0; G_inner[:,-1,:] = 1.0
        G_inner[:,:,0] = 1.0; G_inner[:,:,-1] = 1.0
        print("Extracting sheet lattice (modifier region) ...")
        v_o, f_o = dual_contour(G_outer, 0.0, res, origin, L_mm, gradient_fn=grad_fn)
        v_i, f_i = dual_contour(G_inner, 0.0, res, origin, L_mm, gradient_fn=grad_fn)
        mesh_sheet = trimesh.Trimesh(
            vertices=np.vstack([v_o, v_i]),
            faces=np.vstack([f_o, f_i[:, [0, 2, 1]] + len(v_o)]), process=True)
        mesh_sheet = _clip_to_part(mesh_sheet)
        sheet_mod = _bool_op(mesh_sheet, mod_mesh, "inter") if len(mesh_sheet.faces) > 0 else None

        # 3. Union
        if network_main is not None and len(network_main.faces) > 0:
            if sheet_mod is not None and len(sheet_mod.faces) > 0:
                out_mesh = _bool_op(network_main, sheet_mod, "union")
            else:
                out_mesh = network_main
        elif sheet_mod is not None and len(sheet_mod.faces) > 0:
            out_mesh = sheet_mod
        else:
            out_mesh = mesh_net if len(mesh_net.faces) > 0 else mesh_sheet

        del F, G_net, G_outer, G_inner
        skip_part_clip = True
        mode_tag = "Hybrid"

    else:
        skip_part_clip = False
        if use_sheet:
            # ── Sheet mode: solid = |F| < δ  (both channel networks stay open)
            delta_main = gyroid_sheet_delta(solid_fraction)
            if inside_modifier is not None:
                sf_mod = (modifier_solid_frac if modifier_solid_frac is not None
                          else solid_fraction)
                if modifier_full_solid:
                    sf_mod = solid_fraction  # lattice in NU will be replaced by solid
                delta_mod = gyroid_sheet_delta(sf_mod)
                print(f"\nGradient mode (sheet):")
                print(f"  Main delta     : {delta_main:.4f}  ({solid_fraction*100:.1f}% SF)")
                mod_label = "NU full solid" if modifier_full_solid else f"{sf_mod*100:.1f}% SF"
                print(f"  Modifier delta : {delta_mod:.4f}  ({mod_label})")
                delta_field = compute_delta_field(shape, inside_modifier,
                                                 delta_main, delta_mod,
                                                 modifier_transition_mm, res)
                mode_tag = "Gradient"
            else:
                delta_field = np.full(shape, delta_main, dtype=np.float64)
                mode_tag = "Uniform"
            del inside_modifier

            n_inside = int(inside.sum())
            n_solid  = int(((np.abs(F) < delta_field) & inside).sum())
            eff_sf   = n_solid / n_inside if n_inside > 0 else 0.0
            print(f"\n  Effective solid fraction inside part : {eff_sf*100:.1f}%")
            del inside

            G_outer = F - delta_field
            G_outer[0,:,:] = 1.0; G_outer[-1,:,:] = 1.0
            G_outer[:,0,:] = 1.0; G_outer[:,-1,:] = 1.0
            G_outer[:,:,0] = 1.0; G_outer[:,:,-1] = 1.0
            grad_fn = (make_rotated_gradient(base_gradient_fn, R)
                      if R is not None else base_gradient_fn)
            print("\nRunning dual contouring (outer sheet) ...")
            v_outer, f_outer = dual_contour(G_outer, 0.0, res, origin, L_mm,
                                            gradient_fn=grad_fn)

            G_inner = F + delta_field
            G_inner[0,:,:] = 1.0; G_inner[-1,:,:] = 1.0
            G_inner[:,0,:] = 1.0; G_inner[:,-1,:] = 1.0
            G_inner[:,:,0] = 1.0; G_inner[:,:,-1] = 1.0
            print("Running dual contouring (inner sheet) ...")
            v_inner, f_inner = dual_contour(G_inner, 0.0, res, origin, L_mm,
                                            gradient_fn=grad_fn)

            n_outer = len(v_outer)
            verts_dc = np.vstack([v_outer, v_inner])
            faces_dc = np.vstack([f_outer, f_inner[:, [0, 2, 1]] + n_outer])
            del F, delta_field, G_outer, G_inner

        else:
            # ── Network mode: solid = F < t_field
            t_main = isovalue_fn(solid_fraction)
            if inside_modifier is not None:
                t_mod = isovalue_fn(modifier_solid_frac)
                print(f"\nGradient mode:")
                print(f"  Main solid frac     : {solid_fraction*100:.1f}%  (t = {t_main:.4f})")
                print(f"  Modifier solid frac : {modifier_solid_frac*100:.1f}%  "
                      f"(t = {t_mod:.4f})")
                print(f"  Transition blend    : {modifier_transition_mm:.1f} mm")
                t_field = compute_t_field(shape, inside_modifier, t_main, t_mod,
                                          modifier_transition_mm, res)
                mode_tag = "Gradient"
            else:
                t_field = np.full(shape, t_main, dtype=np.float64)
                mode_tag = "Uniform"
            del inside_modifier

            n_inside = int(inside.sum())
            n_solid  = int(((F < t_field) & inside).sum())
            eff_sf   = n_solid / n_inside if n_inside > 0 else 0.0
            print(f"\n  Effective solid fraction inside part : {eff_sf*100:.1f}%")
            del inside

            G = F - t_field
            del F, t_field
            G[0,:,:] = 1.0; G[-1,:,:] = 1.0
            G[:,0,:] = 1.0; G[:,-1,:] = 1.0
            G[:,:,0] = 1.0; G[:,:,-1] = 1.0
            grad_fn = (make_rotated_gradient(base_gradient_fn, R)
                      if R is not None else base_gradient_fn)
            print("\nRunning dual contouring ...")
            verts_dc, faces_dc = dual_contour(G, 0.0, res, origin, L_mm,
                                              gradient_fn=grad_fn)
            del G

        out_mesh = trimesh.Trimesh(vertices=verts_dc, faces=faces_dc, process=True)

    print(f"  Vertices : {len(out_mesh.vertices):,}")
    print(f"  Faces    : {len(out_mesh.faces):,}")
    print(f"  Watertight output: {out_mesh.is_watertight}")

    # ── 5b. Boolean intersection – clip to exact 3-D part surface ────────
    # (Skipped for hybrid mode; already clipped in _clip_to_part.)
    clipped = None
    if not skip_part_clip:
        print("\nBoolean intersection with original part surface ...")
        try:
            import manifold3d as mf

            def _to_manifold(tm):
                vp = np.ascontiguousarray(tm.vertices, dtype=np.float32)
                tv = np.ascontiguousarray(tm.faces,    dtype=np.uint32)
                return mf.Manifold(mf.Mesh(vert_properties=vp, tri_verts=tv))

            m_gyroid = _to_manifold(out_mesh)
            m_part   = _to_manifold(mesh)

            if m_gyroid.is_empty():
                raise ValueError("Gyroid mesh is not manifold — grid padding may not "
                                 "have closed all boundary edges.")

            m_result = m_gyroid ^ m_part          # intersection (^)
            raw      = m_result.to_mesh()
            vb = np.array(raw.vert_properties, dtype=np.float64)
            fb = np.array(raw.tri_verts, dtype=np.int64).reshape(-1, 3)
            clipped = trimesh.Trimesh(vertices=vb, faces=fb, process=True)
            if len(clipped.faces) == 0:
                clipped = None
                raise ValueError("manifold3d intersection returned empty mesh.")

            print(f"  Faces after boolean  : {len(clipped.faces):,}")
            print(f"  Watertight           : {clipped.is_watertight}")

        except Exception as e:
            print(f"  manifold3d failed ({e})")

        if clipped is None:
            print("  Falling back to PyVista clip_surface (result will be open) ...")
            try:
                import pyvista as pv

                def _to_pv(tm):
                    return pv.PolyData(
                        tm.vertices,
                        np.c_[np.full(len(tm.faces), 3, dtype=int), tm.faces])

                clipped_pv = (_to_pv(out_mesh)
                              .clip_surface(_to_pv(mesh), invert=True)
                              .triangulate().clean())
                pts = np.array(clipped_pv.points)
                fc  = np.array(clipped_pv.faces).reshape(-1, 4)[:, 1:]
                clipped = trimesh.Trimesh(vertices=pts, faces=fc, process=True)
                if len(clipped.faces) == 0:
                    clipped = None
                else:
                    print(f"  Faces after clip : {len(clipped.faces):,}")
                    print(f"  Watertight       : {clipped.is_watertight}")
            except Exception as e2:
                print(f"  PyVista clip also failed ({e2})")

        if clipped is not None and len(clipped.faces) > 0:
            out_mesh = clipped
        else:
            print("  Boolean unavailable; saving unclipped mesh.")

    # ── 6. Union with solid NU when modifier_full_solid ─────────────────────
    # (Skipped for hybrid mode; modifier uses sheet lattice instead.)
    if modifier_full_solid and mod_mesh is not None and not use_hybrid:
        print("\nUnion with solid NU (modifier full solid) ...")
        try:
            import manifold3d as mf
            def _to_mf(tm):
                v = np.ascontiguousarray(tm.vertices, dtype=np.float32)
                f = np.ascontiguousarray(tm.faces, dtype=np.uint32)
                return mf.Manifold(mf.Mesh(vert_properties=v, tri_verts=f))
            m_lat = _to_mf(out_mesh)
            m_nu  = _to_mf(mod_mesh)
            m_result = m_lat + m_nu   # union
            raw = m_result.to_mesh()
            vb = np.array(raw.vert_properties, dtype=np.float64)
            fb = np.array(raw.tri_verts, dtype=np.int64).reshape(-1, 3)
            out_mesh = trimesh.Trimesh(vertices=vb, faces=fb, process=True)
            print(f"  Faces after union : {len(out_mesh.faces):,}")
        except Exception as e:
            print(f"  manifold3d union failed ({e}), trying PyVista ...")
            try:
                import pyvista as pv
                def _pv(tm):
                    return pv.PolyData(tm.vertices,
                        np.c_[np.full(len(tm.faces), 3, dtype=int), tm.faces])
                combined = _pv(out_mesh).boolean_union(_pv(mod_mesh))
                pts = np.array(combined.points)
                fc = np.array(combined.faces).reshape(-1, 4)[:, 1:]
                out_mesh = trimesh.Trimesh(vertices=pts, faces=fc, process=True)
                print(f"  Faces after union : {len(out_mesh.faces):,}")
            except Exception as e2:
                print(f"  Union failed ({e2}); saving lattice without solid NU.")

    # ── 7. Save ────────────────────────────────────────────────────────────
    stem     = os.path.splitext(os.path.basename(stl_path))[0]
    def _pct(v):
        s = f"{v*100:.4g}"          # e.g. "50", "7.5", "12.5"
        return s + "pct"

    sf_tag   = (_pct(solid_fraction)
                if modifier_stl is None
                else f"{_pct(solid_fraction)}-{_pct(modifier_solid_frac or 0)}")
    if modifier_full_solid and modifier_stl:
        sf_tag = f"{sf_tag}_NUfull"
    lt_tag   = lattice_type.replace("_", "").title()
    sheet_suffix = "_Sheet" if use_sheet else ""
    if use_hybrid:
        sheet_suffix = f"_Net{_pct(solid_fraction)}_Sheet{_pct(modifier_solid_frac)}"
    rot_suffix = f"_Rot{rx:.0f}_{ry:.0f}_{rz:.0f}" if R is not None else ""
    lscale_suffix = f"_LscaleXY{modifier_l_scale_xy:.2g}" if (modifier_l_scale_xy != 1.0 and modifier_stl) else ""
    if use_hybrid:
        out_name = f"{stem}_{lt_tag}{sheet_suffix}{rot_suffix}{lscale_suffix}_{pore_um:.0f}um_{mode_tag}.stl"
    else:
        out_name = (f"{stem}_{lt_tag}{sheet_suffix}{rot_suffix}{lscale_suffix}"
                    f"_{pore_um:.0f}um"
                    f"_{sf_tag}"
                    f"_{mode_tag}.stl")
    out_path = os.path.join(os.path.dirname(stl_path), out_name)
    out_mesh.export(out_path)
    print(f"\nSaved: '{out_path}'")
    return out_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    generate(
        input_stl              = INPUT_STL,
        lattice_type           = LATTICE_TYPE,
        pore_um                = TARGET_PORE_UM,
        solid_fraction         = TARGET_SOLID_FRAC,
        voxels_per_period      = VOXELS_PER_PERIOD,
        modifier_stl           = MODIFIER_STL,
        modifier_solid_frac    = MODIFIER_SOLID_FRAC,
        modifier_transition_mm = MODIFIER_TRANSITION_MM,
        modifier_full_solid    = MODIFIER_FULL_SOLID,
        modifier_lattice_mode  = MODIFIER_LATTICE_MODE,
        modifier_l_scale_xy    = MODIFIER_L_SCALE_XY,
        rotation_deg           = ROTATION_DEG,
        gyroid_mode            = GYROID_MODE,
    )
