# -*- coding: utf-8 -*-
"""
Angled-Cut Cylinder Chirped TPMS Generator
-------------------------------------------
Geometry:
  - 2mm diameter cylinder, 5mm tall at its tallest point.
  - Top face is a plane angled at 30 degrees from horizontal.
  - The cut runs along the X-axis: tallest at x = +R, shortest at x = -R.
    z_top(x) = H_MAX + (x - R) * tan(30 deg)
    Tallest:  z = 5.000 mm at x = +1mm
    Shortest: z = 5 - 2*tan(30) ~ 3.845 mm at x = -1mm

Gradient (Z-axis symmetric):
  - Pore size depends only on absolute z (not on x or y).
  - t = z / H_REF  where H_REF = average height of the angled top face.
  - This makes the gradient radially symmetric: every point at the same height
    has the same pore size, so the blooming distortion is uniform around the axis.
  - The angled cut is purely a geometry mask; it does not warp the gradient.

Transition curve:
  - Smoothstep: W(t) = 3t^2 - 2t^3
  - Zero slope at t=0 and t=1 -> no abrupt start/stop at the faces.
  - Linear slope at t=0.5 -> transition is spread evenly across the full height.
  - Compare to sigmoid(steepness=10) which compresses 90% of the change into the
    central 60% of height.

Method: Chirped TPMS (single Split-P field, spatially-varying k).
  k(t) is log-interpolated so the 4x pore size change is perceptually even.
"""

import numpy as np
from skimage import measure
from stl import mesh
import time

# --- Parameters ---
R         = 1.0    # Cylinder radius (mm) -> 2mm diameter
H_MAX     = 5.0    # Height at tallest point (mm)
ANGLE_DEG = 30.0   # Angle of top cut from horizontal (degrees)
RES_XY    = 180    # Voxels across diameter  (~11 um/voxel)
RES_Z     = 300    # Voxels along height     (~17 um/voxel)
TARGET_VF = 0.33   # 33% solid fraction

P_BOTTOM  = 0.8    # Pore size at flat bottom (800 um)
P_TOP     = 0.2    # Pore size at angled top face (200 um)

# -----------------------------------------------------------------------

def top_height(X):
    """
    Z-coordinate of the angled top face at grid position X.
    The cut plane tilts along the X-axis: tallest at x=+R, drops toward x=-R.
    """
    angle_rad = np.radians(ANGLE_DEG)
    return H_MAX + (X - R) * np.tan(angle_rad)

def smoothstep(t):
    """
    Cubic smoothstep: W = 3t^2 - 2t^3
    - W(0) = 0, W(1) = 1  (hits endpoints exactly)
    - W'(0) = W'(1) = 0   (zero slope at faces -> no abrupt start/stop)
    - W'(0.5) = 1.5       (max rate of change at midpoint, but spread across full height)
    Compare to sigmoid(steepness=10): the sigmoid concentrates 90% of the
    transition into the central 60% of height; smoothstep uses 100%.
    """
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)

def split_p_field(X, Y, Z, k):
    """
    Split-P TPMS. k may be scalar or array (same shape as X/Y/Z).
    T3 uses cos(2k*r) per the reference formula.
    """
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

def save_stl(verts, faces, filename):
    out_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            out_mesh.vectors[i][j] = verts[f[j], :]
    out_mesh.save(filename)

# -----------------------------------------------------------------------

def generate_angled_cylinder():
    t0 = time.time()
    angle_rad = np.radians(ANGLE_DEG)
    h_min = H_MAX - 2 * R * np.tan(angle_rad)
    print("=== Angled-Cut Cylinder: Chirped TPMS ===")
    print(f"  Diameter: {2*R:.1f}mm  |  Height: {h_min:.2f}mm (short side) to {H_MAX:.2f}mm (tall side)")
    print(f"  Cut angle: {ANGLE_DEG} deg  |  Gradient: {P_BOTTOM*1000:.0f}um (bottom) -> {P_TOP*1000:.0f}um (angled top)")

    # --- 1. Grid (x: -R..+R, y: -R..+R, z: 0..H_MAX) ---
    v_xy = np.linspace(-R, R, RES_XY)
    v_z  = np.linspace(0, H_MAX, RES_Z)
    X, Y, Z = np.meshgrid(v_xy, v_xy, v_z, indexing='ij')
    sp_xy = 2 * R  / (RES_XY - 1)
    sp_z  = H_MAX  / (RES_Z  - 1)

    # --- 2. Geometry masks ---
    Z_top = top_height(X)                      # Angled surface height at each (x,y,z)
    r_dist = np.sqrt(X**2 + Y**2)
    mask_outside = (r_dist > R) | (Z > Z_top)  # Outside cylinder or above cut plane

    # --- 3. Normalized height t in [0, 1] (Z-axis symmetric) ---
    # t depends only on absolute z, NOT on x or y.
    # H_REF = average height of the angled top face, which equals the
    # centroid z * 2. Using this as the reference centers the gradient
    # midpoint (t=0.5) at the center of mass of the scaffold in z.
    angle_rad = np.radians(ANGLE_DEG)
    h_avg = H_MAX - R * np.tan(angle_rad)   # avg z_top over the circle (x_bar=0)
    H_REF = h_avg                           # t=1 at the average top height
    t_norm = np.clip(Z / H_REF, 0.0, 1.0)
    print(f"  H_REF (avg top height) = {H_REF:.3f}mm  -> gradient midpoint at z = {H_REF/2:.3f}mm")

    # --- 4. Spatially-varying k via log-space interpolation ---
    k_bottom = (2 * np.pi) / (P_BOTTOM * 2.5)
    k_top    = (2 * np.pi) / (P_TOP    * 2.5)
    W = smoothstep(t_norm)
    k_field = np.exp((1 - W) * np.log(k_bottom) + W * np.log(k_top))

    # --- 5. Evaluate field ---
    print("  Evaluating chirped Split-P field...")
    F_abs = np.abs(split_p_field(X, Y, Z, k_field))

    # --- 6. Calibrate tau on interior voxels only ---
    tau, err = calibrate_tau(F_abs[~mask_outside], TARGET_VF)
    print(f"  tau = {tau:.4f}  (VF error: {err*100:.2f}%)")

    # --- 7. Mask outside and bottom face -> void ---
    F_abs[mask_outside] = 10
    F_abs[:, :, 0] = 10   # Cap flat bottom for closed mesh

    # --- 8. Mesh ---
    print("  Meshing...")
    verts, faces, _, _ = measure.marching_cubes(
        F_abs, level=tau, spacing=(sp_xy, sp_xy, sp_z)
    )

    # Shift vertices so x and y are centered at 0 (grid ran from -R to +R)
    # marching_cubes returns indices * spacing starting from 0, so shift back
    verts[:, 0] -= R
    verts[:, 1] -= R

    # --- 9. Export ---
    fname = "angled_cylinder_chirped.stl"
    save_stl(verts, faces, fname)
    print(f"  Saved '{fname}'  ({time.time()-t0:.1f}s)")

if __name__ == "__main__":
    generate_angled_cylinder()
