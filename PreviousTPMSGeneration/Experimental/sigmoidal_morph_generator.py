# -*- coding: utf-8 -*-
"""
TPMS Gradient Scaffold Generator - Two Methods for Comparison
--------------------------------------------------------------
Method A | Chirped TPMS:
  A single Split-P field is evaluated with a spatially-varying k(z).
  k is interpolated in log-space (appropriate for the 4x pore size range).
  There is no second field, so there are no inter-field beating artifacts.
  Trade-off: the lattice cells distort slightly in the z-direction as k changes.

Method B | Sigmoid Field Morph:
  Two separate Split-P fields (one per pore size) are each evaluated over
  the full grid, then blended: F = (1-W)*F_A + W*F_B, where W(z) is a
  sigmoid spanning the full 5mm height.
  Trade-off: the blended region can produce mild interference fringes where
  the two fields are out of phase.

Both output an STL. Open both in Blender to compare.
"""

import numpy as np
from skimage import measure
from stl import mesh
import time

# --- Shared Parameters ---
S_XY     = 2.0    # Width / depth (mm)
S_Z      = 5.0    # Height (mm)
RES_XY   = 150    # Voxels in X, Y
RES_Z    = 250    # Voxels in Z
TARGET_VF = 0.33  # 33% solid fraction

# Gradient pore sizes
P_BOTTOM = 0.2    # 200um at z = 0
P_TOP    = 0.8    # 800um at z = S_Z

# -----------------------------------------------------------------------
# Shared utility functions
# -----------------------------------------------------------------------

def build_grid():
    """Returns X, Y, Z meshgrids and their spacing values."""
    v_xy = np.linspace(0, S_XY, RES_XY)
    v_z  = np.linspace(0, S_Z,  RES_Z)
    X, Y, Z = np.meshgrid(v_xy, v_xy, v_z, indexing='ij')
    sp_xy = S_XY / (RES_XY - 1)
    sp_z  = S_Z  / (RES_Z  - 1)
    return X, Y, Z, sp_xy, sp_z

def sigmoid_weight(Z, z_start, z_end):
    """Smooth S-curve: ~0 at z_start, ~1 at z_end."""
    mid = (z_start + z_end) / 2.0
    steepness = 10.0 / (z_end - z_start)
    return 1.0 / (1.0 + np.exp(-steepness * (Z - mid)))

def split_p_field(X, Y, Z, k):
    """
    Split-P TPMS field. k may be a scalar or an array the same shape as X/Y/Z.
    Formula (T3 uses cos(2k*r)):
      T1 =  1.1*(sin(2kx)*sin(kz)*cos(ky) + sin(2ky)*sin(kx)*cos(kz) + sin(2kz)*sin(ky)*cos(kx))
      T2 = -0.2*(cos(2kx)*cos(2ky) + cos(2ky)*cos(2kz) + cos(2kz)*cos(2kx))
      T3 = -0.4*(cos(2kx) + cos(2ky) + cos(2kz))
    """
    T1 = 1.1 * (np.sin(2*k*X)*np.sin(k*Z)*np.cos(k*Y) +
                np.sin(2*k*Y)*np.sin(k*X)*np.cos(k*Z) +
                np.sin(2*k*Z)*np.sin(k*Y)*np.cos(k*X))
    T2 = -0.2 * (np.cos(2*k*X)*np.cos(2*k*Y) +
                 np.cos(2*k*Y)*np.cos(2*k*Z) +
                 np.cos(2*k*Z)*np.cos(2*k*X))
    T3 = -0.4 * (np.cos(2*k*X) + np.cos(2*k*Y) + np.cos(2*k*Z))
    return T1 + T2 + T3

def calibrate_tau(F_abs, target_vf):
    """Find tau so that the fraction of voxels with F_abs < tau == target_vf."""
    tau_range = np.linspace(0, np.max(F_abs), 500)
    best_tau, min_err = 0.0, 1.0
    for t in tau_range:
        err = abs(np.sum(F_abs < t) / F_abs.size - target_vf)
        if err < min_err:
            min_err = err
            best_tau = t
    return best_tau, min_err

def cap_boundaries(F_abs):
    """Force all six faces to void so marching cubes produces a closed mesh."""
    F_abs[0,:,:] = F_abs[-1,:,:] = 10
    F_abs[:,0,:] = F_abs[:,-1,:] = 10
    F_abs[:,:,0] = F_abs[:,:,-1] = 10

def save_stl(verts, faces, filename):
    out_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            out_mesh.vectors[i][j] = verts[f[j], :]
    out_mesh.save(filename)

# -----------------------------------------------------------------------
# Method A: Chirped TPMS
# -----------------------------------------------------------------------

def generate_chirped():
    """
    Single-field gradient via spatially-varying k(z).
    k is log-interpolated from k_bottom (200um) to k_top (800um).
    No second field -> no inter-field beating.
    """
    t0 = time.time()
    print("\n=== Method A: Chirped TPMS ===")

    X, Y, Z, sp_xy, sp_z = build_grid()

    k_bottom = (2 * np.pi) / (P_BOTTOM * 2.5)
    k_top    = (2 * np.pi) / (P_TOP    * 2.5)
    W = sigmoid_weight(Z, 0.0, S_Z)
    # Log-space: equal multiplicative steps suit the 4x pore size range
    k_field = np.exp((1 - W) * np.log(k_bottom) + W * np.log(k_top))
    print(f"  k: {k_bottom:.3f} (200um, bottom) -> {k_top:.3f} (800um, top)")

    print("  Evaluating field...")
    F_abs = np.abs(split_p_field(X, Y, Z, k_field))

    tau, err = calibrate_tau(F_abs, TARGET_VF)
    print(f"  tau = {tau:.4f}  (VF error: {err*100:.2f}%)")

    cap_boundaries(F_abs)

    verts, faces, _, _ = measure.marching_cubes(F_abs, level=tau, spacing=(sp_xy, sp_xy, sp_z))
    save_stl(verts, faces, "gradient_chirped.stl")
    print(f"  Saved 'gradient_chirped.stl'  ({time.time()-t0:.1f}s)")

# -----------------------------------------------------------------------
# Method B: Sigmoid Field Morph (two-field blend)
# -----------------------------------------------------------------------

def generate_sigmoid_morph():
    """
    Two-field blend: F = (1-W)*F_A + W*F_B, where W(z) is a sigmoid
    spanning the full 5mm height. Each field uses a fixed k for its pore size.
    May produce mild interference fringes in the transition zone.
    """
    t0 = time.time()
    print("\n=== Method B: Sigmoid Field Morph ===")

    X, Y, Z, sp_xy, sp_z = build_grid()

    k_bottom = (2 * np.pi) / (P_BOTTOM * 2.5)
    k_top    = (2 * np.pi) / (P_TOP    * 2.5)

    print("  Evaluating F_A (200um, fixed k)...")
    F_A = split_p_field(X, Y, Z, k_bottom)

    print("  Evaluating F_B (800um, fixed k)...")
    F_B = split_p_field(X, Y, Z, k_top)

    print("  Blending via sigmoid over 5mm...")
    W = sigmoid_weight(Z, 0.0, S_Z)
    # Blend the absolute (sheet) fields, not the signed fields.
    # Blending signed fields causes cancellation near the transition (W~0.5),
    # which drives F toward zero and produces phantom walls / elevated VF.
    F_abs = (1 - W) * np.abs(F_A) + W * np.abs(F_B)

    tau, err = calibrate_tau(F_abs, TARGET_VF)
    print(f"  tau = {tau:.4f}  (VF error: {err*100:.2f}%)")

    cap_boundaries(F_abs)

    verts, faces, _, _ = measure.marching_cubes(F_abs, level=tau, spacing=(sp_xy, sp_xy, sp_z))
    save_stl(verts, faces, "gradient_sigmoid_morph.stl")
    print(f"  Saved 'gradient_sigmoid_morph.stl'  ({time.time()-t0:.1f}s)")

# -----------------------------------------------------------------------

if __name__ == "__main__":
    generate_chirped()
    generate_sigmoid_morph()
    print("\nDone. Open both STLs in Blender to compare.")
