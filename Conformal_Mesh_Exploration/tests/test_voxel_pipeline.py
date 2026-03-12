"""
Voxel Pipeline & Visualizer — Conformal Hex Inside-Out.

Runs generate_conformal_hexes on 20mm cube and rounded cube, then
visualizes surface hexes and full grid (internal + surface) in matplotlib.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

_root = Path(__file__).resolve().parents[2]  # Graphite project root
TEST_PARTS = _root / "test_parts"
CUBE_STL = TEST_PARTS / "20mm_cube.stl"
ROUNDED_STL = TEST_PARTS / "rounded-cube.stl"

# Standard hex face indices (6 quads, 4 vertices each)
HEX_FACES = [
    [0, 1, 2, 3],  # bottom
    [4, 5, 6, 7],  # top
    [0, 1, 5, 4],  # front
    [1, 2, 6, 5],  # right
    [2, 3, 7, 6],  # back
    [3, 0, 4, 7],  # left
]


def hexes_to_faces(hexes: np.ndarray) -> list[np.ndarray]:
    """Convert (N, 8, 3) hex array to list of (4, 3) face vertex arrays."""
    faces = []
    for h in hexes:
        for fi in HEX_FACES:
            faces.append(h[fi])
    return faces


def add_hexes_to_ax(
    ax,
    hexes: np.ndarray,
    facecolor: str | tuple,
    edgecolor: str = "black",
    alpha: float = 0.6,
) -> None:
    """Add hexahedra to a 3D axes as a Poly3DCollection."""
    if hexes.size == 0:
        return
    faces = hexes_to_faces(hexes)
    coll = Poly3DCollection(
        faces,
        facecolors=facecolor,
        edgecolors=edgecolor,
        alpha=alpha,
        linewidths=0.5,
    )
    ax.add_collection3d(coll)


def main() -> None:
    import sys

    sys.path.insert(0, str(_root))
    from Conformal_Mesh_Exploration.core.voxelizer import generate_conformal_hexes

    target_size = 5.0

    print("Processing 20mm cube...")
    cube_hexes, _ = generate_conformal_hexes(CUBE_STL, target_size)

    print("Processing rounded cube...")
    rounded_hexes, _ = generate_conformal_hexes(ROUNDED_STL, target_size)

    # PLOT 1: All hexes (1x2) — light blue
    fig1, (ax1a, ax1b) = plt.subplots(1, 2, subplot_kw={"projection": "3d"})
    fig1.suptitle("Conformal Hex Mesh (Centroid Rule)", fontsize=12)

    add_hexes_to_ax(ax1a, cube_hexes, facecolor="lightblue", edgecolor="black", alpha=0.6)
    ax1a.set_title("20mm Cube")
    ax1a.set_xlabel("X")
    ax1a.set_ylabel("Y")
    ax1a.set_zlabel("Z")
    _set_equal_aspect(ax1a, cube_hexes)

    add_hexes_to_ax(ax1b, rounded_hexes, facecolor="lightblue", edgecolor="black", alpha=0.6)
    ax1b.set_title("Rounded Cube")
    ax1b.set_xlabel("X")
    ax1b.set_ylabel("Y")
    ax1b.set_zlabel("Z")
    _set_equal_aspect(ax1b, rounded_hexes)

    fig1.tight_layout()

    # PLOT 2: Full grid (all hexes)
    fig2, (ax2a, ax2b) = plt.subplots(1, 2, subplot_kw={"projection": "3d"})
    fig2.suptitle("Full Grid", fontsize=12)

    add_hexes_to_ax(ax2a, cube_hexes, facecolor="green", edgecolor="black", alpha=0.6)
    ax2a.set_title("20mm Cube")
    ax2a.set_xlabel("X")
    ax2a.set_ylabel("Y")
    ax2a.set_zlabel("Z")
    _set_equal_aspect(ax2a, cube_hexes)

    add_hexes_to_ax(ax2b, rounded_hexes, facecolor="green", edgecolor="black", alpha=0.6)
    ax2b.set_title("Rounded Cube")
    ax2b.set_xlabel("X")
    ax2b.set_ylabel("Y")
    ax2b.set_zlabel("Z")
    _set_equal_aspect(ax2b, rounded_hexes)

    fig2.tight_layout()

    plt.show()


def _set_equal_aspect(ax, hexes: np.ndarray) -> None:
    """Set axis limits from hex bounds for equal aspect."""
    if hexes.size == 0:
        return
    pts = hexes.reshape(-1, 3)
    mn, mx = pts.min(axis=0), pts.max(axis=0)
    c = (mn + mx) / 2
    r = max(mx - mn) / 2
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)


if __name__ == "__main__":
    main()
