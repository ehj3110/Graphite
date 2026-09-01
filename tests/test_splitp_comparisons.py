"""
Three-way Split-P comparison: basic chirp, Jacobian W-phase, MATLAB reverse-lookup (4 mm base).
Sheet rendering: |F| < 0.2 → white walls, else black void. No griddata.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import cumulative_trapezoid

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def evaluate_split_p(U, V, W):
    T1 = 1.1 * (
        np.sin(2 * U) * np.sin(W) * np.cos(V)
        + np.sin(2 * V) * np.sin(U) * np.cos(W)
        + np.sin(2 * W) * np.sin(V) * np.cos(U)
    )
    T2 = -0.2 * (
        np.cos(2 * U) * np.cos(2 * V)
        + np.cos(2 * V) * np.cos(2 * W)
        + np.cos(2 * W) * np.cos(2 * U)
    )
    T3 = -0.4 * (np.cos(2 * U) + np.cos(2 * V) + np.cos(2 * W))
    return T1 + T2 + T3


def get_L_final(z):
    """Final-space pore scale L(z) in mm."""
    z = np.asarray(z, dtype=float)
    return np.interp(z, [0.0, 16.5, 17.0, 20.0], [16.0, 8.0, 8.0, 4.0])


# Jacobian W along Z in [0, 20] (Methods 1 & 2)
z_1d = np.linspace(0.0, 20.0, 2000)
omega_1d = 2.0 * np.pi / get_L_final(z_1d)
W_int_1d = cumulative_trapezoid(omega_1d, z_1d, initial=0)

# Method 3: pre-flip stretch knots → scale S(Z_orig); Z_warped = Z_orig * S
Z_target_pre = [0.0, 3.0, 3.5, 20.0]
S_target_pre = [1.0, 2.0, 2.0, 4.0]
z_orig_dense = np.linspace(0.0, 30.0, 5000)
z_warped_dense = z_orig_dense * np.interp(z_orig_dense, Z_target_pre, S_target_pre)
omega_0 = 2.0 * np.pi / 4.0  # 4 mm base unit cell


def eval_method3(X, Y, Z):
    """Reverse map from final (X,Y,Z) to pre-scale base lattice coordinates."""
    Z_cropped = 20.0 - Z
    Z_orig = np.interp(Z_cropped, z_warped_dense, z_orig_dense)
    S_applied = np.interp(Z_orig, Z_target_pre, S_target_pre)
    X_orig = X / S_applied
    Y_orig = Y / S_applied
    return evaluate_split_p(X_orig * omega_0, Y_orig * omega_0, Z_orig * omega_0)


def main() -> None:
    out_dir = _REPO_ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    dpi = 300

    target_zs = [20.0, 17.0, 16.5, 8.0, 0.0]
    x_grid = np.linspace(-12.0, 12.0, 600)
    y_grid = np.linspace(-12.0, 12.0, 600)
    X, Y = np.meshgrid(x_grid, y_grid)
    mask_xy = (X**2 + Y**2) <= 10.0**2

    fig1, axes1 = plt.subplots(
        len(target_zs), 3, figsize=(12, 18), facecolor="black"
    )
    if axes1.ndim == 1:
        axes1 = axes1.reshape(1, -1)

    for ax in axes1.ravel():
        ax.set_facecolor("black")

    for i, z_eval in enumerate(target_zs):
        L_val = float(get_L_final(z_eval))
        omega_val = 2.0 * np.pi / L_val
        W_chirp = np.full_like(X, z_eval * omega_val, dtype=float)
        W_int = np.full_like(X, np.interp(z_eval, z_1d, W_int_1d), dtype=float)

        F1 = evaluate_split_p(X * omega_val, Y * omega_val, W_chirp)
        F2 = evaluate_split_p(X * omega_val, Y * omega_val, W_int)
        F3 = eval_method3(X, Y, np.full_like(X, z_eval, dtype=float))

        for j, F in enumerate([F1, F2, F3]):
            ax = axes1[i, j]
            img = np.where((np.abs(F) < 0.2) & mask_xy, 1.0, 0.0)
            ax.imshow(
                img,
                cmap="gray",
                vmin=0.0,
                vmax=1.0,
                origin="lower",
                extent=[-12, 12, -12, 12],
            )
            ax.axis("off")
            if i == 0:
                titles = ["Basic Chirp", "Jacobian Integrated", "MATLAB Legacy"]
                ax.set_title(titles[j], color="white", pad=10)
            if j == 0:
                ax.text(
                    -13.0,
                    0.0,
                    f"Z = {z_eval} mm\nL = {L_val:.1f} mm",
                    color="white",
                    va="center",
                    ha="right",
                    rotation=90,
                    fontsize=11,
                )

    fig1.tight_layout()
    p1 = out_dir / "SplitP_Z_Slices.png"
    fig1.savefig(p1, dpi=dpi, facecolor="black")
    plt.close(fig1)

    x_vert = np.linspace(-12.0, 12.0, 800)
    z_vert = np.linspace(0.0, 20.0, 1000)
    X_v, Z_v = np.meshgrid(x_vert, z_vert)
    mask_v = np.abs(X_v) <= 10.0

    omega_2d = 2.0 * np.pi / get_L_final(Z_v)
    W_chirp_2d = Z_v * omega_2d
    W_int_2d = np.interp(Z_v, z_1d, W_int_1d)

    F1_v = evaluate_split_p(X_v * omega_2d, np.zeros_like(X_v), W_chirp_2d)
    F2_v = evaluate_split_p(X_v * omega_2d, np.zeros_like(X_v), W_int_2d)
    F3_v = eval_method3(X_v, np.zeros_like(X_v), Z_v)

    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 8), facecolor="black")
    titles_v = ["Basic Chirp", "Jacobian Integrated", "MATLAB Legacy"]
    for j, F in enumerate([F1_v, F2_v, F3_v]):
        ax = axes2[j]
        ax.set_facecolor("black")
        img = np.where((np.abs(F) < 0.2) & mask_v, 1.0, 0.0)
        ax.imshow(
            img,
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
            origin="lower",
            extent=[-12, 12, 0, 20],
        )
        ax.set_title(titles_v[j], color="white")
        ax.set_xlabel("X (mm)", color="white")
        ax.set_ylabel("Z (mm)", color="white")
        ax.tick_params(colors="white")

    fig2.tight_layout()
    p2 = out_dir / "SplitP_Vertical_Slice.png"
    fig2.savefig(p2, dpi=dpi, facecolor="black")
    plt.close(fig2)

    print(f"Wrote {p1}")
    print(f"Wrote {p2}")


if __name__ == "__main__":
    main()
