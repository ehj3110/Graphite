from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def evaluate_woodpile_field(X, Y, Z, pore_size: float, true_woodpile: bool):
    pitch = 2.0 * pore_size
    layer_idx = np.floor(Z / pore_size).astype(int)

    X_eff = X.copy()
    Y_eff = Y.copy()
    if true_woodpile:
        shift_x_mask = (layer_idx % 4) == 3
        shift_y_mask = (layer_idx % 4) == 2
        X_eff = X_eff + np.where(shift_x_mask, pore_size, 0.0)
        Y_eff = Y_eff + np.where(shift_y_mask, pore_size, 0.0)

    wave_X = np.abs(np.mod(X_eff + pore_size, pitch) - pore_size) - (pore_size / 2.0)
    wave_Y = np.abs(np.mod(Y_eff + pore_size, pitch) - pore_size) - (pore_size / 2.0)
    field = np.where(layer_idx % 2 == 0, wave_Y, wave_X)
    return field


def run_woodpile_diagnostic(output_dir: str = "outputs/diagnostics/") -> bool:
    try:
        pore_size = 1.0
        x = np.linspace(-4.0, 4.0, 600)
        y = np.linspace(-4.0, 4.0, 600)
        X, Y = np.meshgrid(x, y, indexing="ij")
        z_slices = [0.0, 0.5, 1.0, 1.5]

        fig, axes = plt.subplots(2, 4, figsize=(14, 7), facecolor="black")
        for ax in axes.ravel():
            ax.set_facecolor("black")

        for c, z_val in enumerate(z_slices):
            Z = np.full_like(X, z_val)

            field_simple = evaluate_woodpile_field(X, Y, Z, pore_size, true_woodpile=False)
            img_simple = np.where(field_simple <= 0.0, 1.0, 0.0)
            ax0 = axes[0, c]
            ax0.imshow(
                img_simple,
                cmap="gray",
                vmin=0.0,
                vmax=1.0,
                origin="lower",
                extent=[x.min(), x.max(), y.min(), y.max()],
            )
            ax0.set_title(f"Simple | Z={z_val:.2f}", color="white")
            ax0.axis("off")

            field_true = evaluate_woodpile_field(X, Y, Z, pore_size, true_woodpile=True)
            img_true = np.where(field_true <= 0.0, 1.0, 0.0)
            ax1 = axes[1, c]
            ax1.imshow(
                img_true,
                cmap="gray",
                vmin=0.0,
                vmax=1.0,
                origin="lower",
                extent=[x.min(), x.max(), y.min(), y.max()],
            )
            ax1.set_title(f"True Woodpile | Z={z_val:.2f}", color="white")
            ax1.axis("off")

        fig.tight_layout()
        out_dir_path = Path(output_dir).resolve()
        out_dir_path.mkdir(parents=True, exist_ok=True)
        out_path = out_dir_path / "Woodpile_Sandbox_Slices.png"
        fig.savefig(out_path, dpi=300, facecolor="black")
        plt.close(fig)
        return True
    except Exception as e:
        print(f"Woodpile Generation Test failed: {e}")
        return False


if __name__ == "__main__":
    out_dir_default = Path(__file__).resolve().parents[1] / "outputs" / "diagnostics"
    success = run_woodpile_diagnostic(str(out_dir_default))
    print(f"Test Success: {success}")
