"""
Surface Mesh Visualizer — Inspect algorithmic irregularity of tetrahedral meshes.

Loads the surface STL files from the tet compatibility test (5.0mm and 2.5mm)
and displays them side-by-side with light grey faces and black edges so the
surface tessellation is starkly visible. Use the interactive plot to rotate
and inspect the irregularity.

Prerequisite: Run test_tet_compatibility first to generate:
    output/Raw_Tet_Mesh_5.0mm_Surface.stl
    output/Raw_Tet_Mesh_2.5mm_Surface.stl

Run from Graphite root:
    python -m Universal_Lattice_Engine.tests.visualize_surface_mesh
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
SURFACE_FILES = [
    ("5.0 mm", OUTPUT_DIR / "Raw_Tet_Mesh_5.0mm_Surface.stl"),
    ("2.5 mm", OUTPUT_DIR / "Raw_Tet_Mesh_2.5mm_Surface.stl"),
]


def load_stl(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load STL and return (vertices, faces)."""
    try:
        import trimesh
    except ImportError:
        raise ImportError(
            "trimesh is required for loading STL. Install with: pip install trimesh"
        ) from None

    mesh = trimesh.load(str(path))
    if isinstance(mesh, trimesh.Scene):
        meshes = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        mesh = meshes[0] if meshes else None
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Could not load mesh from {path}")
    return np.asarray(mesh.vertices), np.asarray(mesh.faces)


def plot_surface(ax, vertices: np.ndarray, faces: np.ndarray, title: str) -> None:
    """Plot surface mesh with grey faces and black edges."""
    verts = vertices[faces]   # (N, 3, 3)
    poly = Poly3DCollection(
        verts,
        facecolor=(0.85, 0.85, 0.85),
        alpha=0.8,
        edgecolor="black",
        linewidths=0.5,
    )
    ax.add_collection3d(poly)
    ax.set_xlim(vertices[:, 0].min(), vertices[:, 0].max())
    ax.set_ylim(vertices[:, 1].min(), vertices[:, 1].max())
    ax.set_zlim(vertices[:, 2].min(), vertices[:, 2].max())
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)
    ax.set_box_aspect([1, 1, 1])


def main() -> None:
    print("=" * 60)
    print("Surface Mesh Visualizer — Tet Mesh Irregularity")
    print("=" * 60)

    missing = [label for label, p in SURFACE_FILES if not p.exists()]
    if missing:
        print(
            f"Missing surface files for: {missing}. "
            "Run: python -m Universal_Lattice_Engine.tests.test_tet_compatibility"
        )
        return

    fig = plt.figure(figsize=(12, 6))
    for i, (label, path) in enumerate(SURFACE_FILES):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        vertices, faces = load_stl(path)
        print(f"  {label}: {len(vertices)} vertices, {len(faces)} triangles")
        plot_surface(ax, vertices, faces, f"Tet Surface — {label}")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
