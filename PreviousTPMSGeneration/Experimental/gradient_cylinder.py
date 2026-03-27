# -*- coding: utf-8 -*-
"""
Gradient Cylinder - Chirped TPMS (Flat-top and Angled-top versions)
--------------------------------------------------------------------
Pore size transition: geometric (exponential) interpolation in log(k) space.
  P(t) = P_bottom^(1-W) * P_top^W  ->  midpoint = geometric mean = ~400um
  W(t) = smoothstep = 3t^2 - 2t^3
    - Zero slope at both faces: no abrupt start/stop
    - Transition evenly spread across full height

FLAT-TOP  (generate_flat):
  t = z / H  -- purely Z-based, radially symmetric.
  Every horizontal cross-section has identical pore size.
  Outputs: gradient_cylinder_full.stl, gradient_cylinder_half.stl

ANGLED-TOP  (generate_angled):
  t = z / z_top(x)  -- column-local normalized height.
  z_top(x) = H_MAX + (x - R) * tan(ANGLE_DEG)
  Each vertical column [0 -> z_top(x)] maps to its own [0 -> 1] range,
  so the gradient rate adapts to the local column height. The result:
    - Flat bottom (z=0): t=0 everywhere -> 800um
    - Angled top face (z=z_top): t=1 everywhere -> 200um
  The pore size on each face is uniform regardless of the cut angle.
  Outputs: gradient_angled_full.stl, gradient_angled_half.stl
"""

import numpy as np
from skimage import measure
from stl import mesh
import time

# --- Parameters ---
R         = 1.0    # Cylinder radius (mm) -> 2mm diameter
H         = 5.0    # Height (mm) for flat-top; tallest point for angled-top
ANGLE_DEG = 30.0   # Angled-top: cut angle from horizontal (degrees)
RES_XY    = 260    # Voxels across diameter  (~8 um/voxel)
RES_Z     = 430    # Voxels along height     (~12 um/voxel)
TARGET_VF = 0.33

P_BOTTOM  = 0.8    # Pore size at bottom face (800 um)
P_TOP     = 0.2    # Pore size at top face    (200 um)

# -----------------------------------------------------------------------

def smoothstep(t):
    """W = 3t^2 - 2t^3. Zero slope at t=0 and t=1, transition spread across full height."""
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)

def split_p_field(X, Y, Z, k):
    """Split-P TPMS. k may be scalar or same-shaped array."""
    T1 = 1.1 * (np.sin(2*k*X)*np.sin(k*Z)*np.cos(k*Y) +
                np.sin(2*k*Y)*np.sin(k*X)*np.cos(k*Z) +
                np.sin(2*k*Z)*np.sin(k*Y)*np.cos(k*X))
    T2 = -0.2 * (np.cos(2*k*X)*np.cos(2*k*Y) +
                 np.cos(2*k*Y)*np.cos(2*k*Z) +
                 np.cos(2*k*Z)*np.cos(2*k*X))
    T3 = -0.4 * (np.cos(2*k*X) + np.cos(2*k*Y) + np.cos(2*k*Z))
    return T1 + T2 + T3

def calibrate_tau(F_interior, target_vf):
    tau_range = np.linspace(0, np.max(F_interior), 500)
    best_tau, min_err = 0.0, 1.0
    for t in tau_range:
        err = abs(np.sum(F_interior < t) / F_interior.size - target_vf)
        if err < min_err:
            min_err = err
            best_tau = t
    return best_tau, min_err

def export_stl(F_abs, tau, sp_xy, sp_z, filename):
    verts, faces, _, _ = measure.marching_cubes(
        F_abs, level=tau, spacing=(sp_xy, sp_xy, sp_z)
    )
    # Grid ran from -R to +R; marching_cubes starts at 0, so shift back
    verts[:, 0] -= R
    verts[:, 1] -= R
    out = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            out.vectors[i][j] = verts[f[j], :]
    out.save(filename)
    print(f"  Saved '{filename}'")

# -----------------------------------------------------------------------

def _build_grid():
    """Shared grid construction."""
    v_xy = np.linspace(-R, R, RES_XY)
    v_z  = np.linspace(0, H, RES_Z)
    X, Y, Z = np.meshgrid(v_xy, v_xy, v_z, indexing='ij')
    sp_xy = 2 * R / (RES_XY - 1)
    sp_z  = H     / (RES_Z  - 1)
    return X, Y, Z, sp_xy, sp_z

def _chirped_field(X, Y, Z, t_norm):
    """Evaluate chirped Split-P field given a [0,1] gradient parameter array."""
    k_b = (2 * np.pi) / (P_BOTTOM * 2.5)
    k_t = (2 * np.pi) / (P_TOP    * 2.5)
    W   = smoothstep(t_norm)
    # Geometric (log-space) interpolation: pore size P = P_b^(1-W) * P_t^W
    k_field = np.exp((1 - W) * np.log(k_b) + W * np.log(k_t))
    return np.abs(split_p_field(X, Y, Z, k_field))

def _mesh_and_export(F_abs, tau, sp_xy, sp_z, mask_outer, prefix):
    """Apply mask, cap bottom, export full and half STLs."""
    # Full
    F_full = F_abs.copy()
    F_full[mask_outer]  = 10
    F_full[:, :,  0]    = 10   # cap bottom
    F_full[:, :, -1]    = 10   # cap top boundary
    export_stl(F_full, tau, sp_xy, sp_z, f"{prefix}_full.stl")

    # Half (y >= 0 only — reveals cross-section)
    F_half = F_abs.copy()
    F_half[mask_outer | (Y_GLOBAL < 0)] = 10
    F_half[:, :,  0]    = 10
    F_half[:, :, -1]    = 10
    export_stl(F_half, tau, sp_xy, sp_z, f"{prefix}_half.stl")

# Module-level Y reference used by _mesh_and_export
Y_GLOBAL = None

# -----------------------------------------------------------------------

def generate_flat():
    """
    Flat-top cylinder. t = z/H (radially symmetric).
    Every horizontal slice has uniform pore size.
    """
    global Y_GLOBAL
    t0 = time.time()
    print("\n=== FLAT-TOP Gradient Cylinder ===")
    print(f"  {2*R}mm dia x {H}mm  |  {P_BOTTOM*1000:.0f}um (z=0) -> {P_TOP*1000:.0f}um (z={H}mm)")

    X, Y, Z, sp_xy, sp_z = _build_grid()
    Y_GLOBAL = Y

    mask_cyl = np.sqrt(X**2 + Y**2) > R

    print("  Evaluating field...")
    t_norm = Z / H                         # purely Z-based
    F_abs  = _chirped_field(X, Y, Z, t_norm)

    tau, err = calibrate_tau(F_abs[~mask_cyl], TARGET_VF)
    print(f"  tau = {tau:.4f}  (VF error: {err*100:.2f}%)")

    _mesh_and_export(F_abs, tau, sp_xy, sp_z, mask_cyl, "gradient_cylinder")
    print(f"  Done in {time.time()-t0:.1f}s")


def generate_angled():
    """
    Angled-top cylinder. t = z / z_top(x) (column-local height).
    z_top(x) = H + (x - R) * tan(ANGLE_DEG)

    Each vertical column is independently mapped to [0,1], so:
      - Bottom face (z=0): t=0 -> P_BOTTOM everywhere
      - Angled top face (z=z_top): t=1 -> P_TOP everywhere
    The gradient rate is faster for shorter columns (x=-R side) and
    slower for taller columns (x=+R side) to keep both faces uniform.
    """
    global Y_GLOBAL
    t0 = time.time()
    angle_rad = np.radians(ANGLE_DEG)
    h_min = H - 2 * R * np.tan(angle_rad)
    print("\n=== ANGLED-TOP Gradient Cylinder ===")
    print(f"  {2*R}mm dia  |  Height: {h_min:.2f}mm (short) to {H:.2f}mm (tall)  |  {ANGLE_DEG}deg cut")
    print(f"  {P_BOTTOM*1000:.0f}um (bottom) -> {P_TOP*1000:.0f}um (angled top face)")

    X, Y, Z, sp_xy, sp_z = _build_grid()
    Y_GLOBAL = Y

    # Angled top surface: tallest at x=+R, drops toward x=-R
    Z_top = H + (X - R) * np.tan(angle_rad)   # shape (RES_XY, RES_XY, RES_Z)

    mask_cyl  = np.sqrt(X**2 + Y**2) > R
    mask_cut  = Z > Z_top
    mask_outer = mask_cyl | mask_cut

    # Column-local t: each (x,y) column maps its full height to [0,1]
    # Short columns (x=-R) compress the gradient; tall columns (x=+R) stretch it.
    t_norm = np.clip(Z / Z_top, 0.0, 1.0)

    print("  Evaluating field...")
    F_abs = _chirped_field(X, Y, Z, t_norm)

    tau, err = calibrate_tau(F_abs[~mask_outer], TARGET_VF)
    print(f"  tau = {tau:.4f}  (VF error: {err*100:.2f}%)")

    _mesh_and_export(F_abs, tau, sp_xy, sp_z, mask_outer, "gradient_angled")
    print(f"  Done in {time.time()-t0:.1f}s")


def generate_angled_twostage():
    """
    Angled-top cylinder with a two-zone pore gradient:

    Zone 1  (top, 1mm perpendicular thickness):
      200um at angled face  ->  400um at boundary plane
      Parameterized by perpendicular distance from the angled face, so the
      transition rate is uniform across the entire zone regardless of x/y.

    Zone 2  (bottom, from boundary plane down to flat base):
      400um at boundary plane  ->  800um at flat bottom
      Uses column-local height t = z / z_mid(x), the same X,Y,Z-adaptive
      scaling as generate_angled().

    The boundary between zones is the plane parallel to the angled top face,
    offset 1mm downward along the face normal:
      z_mid(x) = z_top(x) - ZONE1_MM / cos(ANGLE_DEG)
    """
    global Y_GLOBAL
    t0 = time.time()
    angle_rad  = np.radians(ANGLE_DEG)
    zone1_perp = 1.0   # mm, perpendicular thickness of top zone
    h_min = H - 2 * R * np.tan(angle_rad)

    print("\n=== ANGLED-TOP Two-Stage Gradient Cylinder ===")
    print(f"  {2*R}mm dia  |  {h_min:.2f}-{H:.2f}mm tall  |  {ANGLE_DEG}deg cut")
    print(f"  Zone 1 (top {zone1_perp}mm perp): 200um -> 400um  (parallel to angled face)")
    print(f"  Zone 2 (remainder):          400um -> 800um  (column-local scaling)")

    X, Y, Z, sp_xy, sp_z = _build_grid()
    Y_GLOBAL = Y

    # Angled top surface and 400um boundary (1mm perp below it)
    z_top = H + (X - R) * np.tan(angle_rad)
    z_mid = z_top - zone1_perp / np.cos(angle_rad)   # parallel plane, 1mm lower

    mask_cyl   = np.sqrt(X**2 + Y**2) > R
    mask_cut   = Z > z_top
    mask_outer = mask_cyl | mask_cut

    # ------------------------------------------------------------------
    # Zone 1 parameter: perpendicular distance from the angled top face.
    #   dist_perp = (z_top - z) * cos(angle)
    #   t1 = 0 at the top face (200um), t1 = 1 at the 400um boundary
    # ------------------------------------------------------------------
    dist_perp = (z_top - Z) * np.cos(angle_rad)
    t1 = np.clip(dist_perp / zone1_perp, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Zone 2 parameter: column-local height from z=0 to z=z_mid(x).
    #   t2 = 0 at flat bottom (800um), t2 = 1 at 400um boundary
    # ------------------------------------------------------------------
    t2 = np.where(z_mid > 0, np.clip(Z / z_mid, 0.0, 1.0), 0.0)

    # k values for the three pore sizes
    k_200 = (2 * np.pi) / (0.2 * 2.5)
    k_400 = (2 * np.pi) / (0.4 * 2.5)
    k_800 = (2 * np.pi) / (0.8 * 2.5)

    # Zone 1: 200um (t=0) -> 400um (t=1)
    W1 = smoothstep(t1)
    k_zone1 = np.exp((1 - W1) * np.log(k_200) + W1 * np.log(k_400))

    # Zone 2: 800um (t=0) -> 400um (t=1)
    W2 = smoothstep(t2)
    k_zone2 = np.exp((1 - W2) * np.log(k_800) + W2 * np.log(k_400))

    # Select zone by position
    in_zone1 = Z >= z_mid
    k_field = np.where(in_zone1, k_zone1, k_zone2)

    print("  Evaluating field...")
    F_abs = np.abs(split_p_field(X, Y, Z, k_field))

    tau, err = calibrate_tau(F_abs[~mask_outer], TARGET_VF)
    print(f"  tau = {tau:.4f}  (VF error: {err*100:.2f}%)")

    _mesh_and_export(F_abs, tau, sp_xy, sp_z, mask_outer, "gradient_angled_twostage")
    print(f"  Done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    generate_flat()
    generate_angled()
    generate_angled_twostage()
    print("\nAll done. Six STLs saved:")
    print("  gradient_cylinder_full/half.stl       -- flat top, single gradient")
    print("  gradient_angled_full/half.stl         -- angled top, single gradient")
    print("  gradient_angled_twostage_full/half.stl -- angled top, two-zone gradient")
