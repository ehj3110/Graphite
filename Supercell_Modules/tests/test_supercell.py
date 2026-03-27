"""
Supercell Testing Harness — 3-step export pipeline for Cartesian lattices.

Run from Graphite root: python -m Supercell_Modules.tests.test_supercell
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import trimesh

# Add Graphite root for geometry_module
_graphite_root = Path(__file__).resolve().parents[2]
if str(_graphite_root) not in sys.path:
    sys.path.insert(0, str(_graphite_root))

# Add Supercell_Modules for core imports
_supercell_dir = Path(__file__).resolve().parent.parent
if str(_supercell_dir) not in sys.path:
    sys.path.insert(0, str(_supercell_dir))

from graphite.explicit.geometry_module import generate_geometry, manifold_to_trimesh

from core.grid_generator import generate_cartesian_nodes, trim_to_stl
from core.topologies import get_bcc_supercell, get_cartesian_kagome, get_voronoi_foam

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
STRUT_RADIUS = 0.3  # mm


def _get_dummy_cube_stl(size_mm: float = 20.0) -> trimesh.Trimesh:
    """Create a 20mm cube centered at origin (watertight for contains)."""
    box = trimesh.creation.box(extents=[size_mm, size_mm, size_mm])
    return box


def _classify_struts(
    nodes: np.ndarray,
    struts: np.ndarray,
    stl_mesh: trimesh.Trimesh,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split struts into: (boundary_intersecting, strictly_internal).
    Boundary-intersecting: at least one endpoint outside or on boundary.
    Strictly internal: both endpoints inside.
    """
    inside = np.asarray(stl_mesh.contains(nodes), dtype=bool)
    a_in = inside[struts[:, 0]]
    b_in = inside[struts[:, 1]]
    strictly_internal = a_in & b_in
    boundary_intersecting = ~strictly_internal
    return struts[boundary_intersecting], struts[strictly_internal]


def export_surface_plot(
    nodes: np.ndarray,
    struts: np.ndarray,
    stl_mesh: trimesh.Trimesh,
    output_path: str | Path | None = None,
    prefix: str = "",
) -> None:
    """
    Generate an interactive 3D plot of ONLY struts that intersect the STL boundary.
    """
    boundary_struts, _ = _classify_struts(nodes, struts, stl_mesh)
    if boundary_struts.shape[0] == 0:
        print("[Surface Plot] No boundary-intersecting struts.")
        return

    try:
        import pyvista as pv
    except ImportError:
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            _plot_matplotlib(nodes, boundary_struts, stl_mesh, output_path, prefix)
            return
        except ImportError:
            print("[Surface Plot] Install pyvista or matplotlib for 3D plotting.")
            return

    # PyVista: build polyline mesh
    lines = []
    for a_idx, b_idx in boundary_struts:
        lines.append(2)
        lines.append(int(a_idx))
        lines.append(int(b_idx))
    lines = np.array(lines, dtype=np.int64)
    cloud = pv.PolyData(nodes)
    cloud.lines = lines

    plotter = pv.Plotter(off_screen=(output_path is not None or bool(prefix)))
    plotter.add_mesh(cloud, color="red", line_width=2, render_lines_as_tubes=True)
    plotter.add_mesh(
        pv.wrap(stl_mesh),
        color="lightgray",
        opacity=0.3,
        style="wireframe",
    )
    plotter.add_axes()

    if output_path or prefix:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        name = f"{prefix}Surface_Struts.png" if prefix else (output_path or "Surface_Struts.png")
        path = OUTPUT_DIR / name
        plotter.screenshot(str(path))
        print(f"[Surface Plot] Saved: {path}")
    else:
        plotter.show()
    plotter.close()


def _plot_matplotlib(
    nodes: np.ndarray,
    boundary_struts: np.ndarray,
    stl_mesh: trimesh.Trimesh,
    output_path: str | Path | None,
    prefix: str = "",
) -> None:
    """Fallback: matplotlib 3D plot."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Struts
    segments = np.array([[nodes[i], nodes[j]] for i, j in boundary_struts])
    ax.add_collection3d(Line3DCollection(segments, colors="red", linewidths=1.5))

    # STL wireframe
    verts = stl_mesh.vertices
    faces = stl_mesh.faces
    for f in faces:
        tri = verts[f]
        ax.plot(tri[:, 0], tri[:, 1], tri[:, 2], "gray", alpha=0.5, linewidth=0.5)

    ax.set_xlim(nodes[:, 0].min() - 1, nodes[:, 0].max() + 1)
    ax.set_ylim(nodes[:, 1].min() - 1, nodes[:, 1].max() + 1)
    ax.set_zlim(nodes[:, 2].min() - 1, nodes[:, 2].max() + 1)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    if output_path or prefix:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        name = f"{prefix}Surface_Struts.png" if prefix else (str(output_path) if output_path else "Surface_Struts.png")
        path = OUTPUT_DIR / name
        plt.savefig(str(path), dpi=150)
        print(f"[Surface Plot] Saved: {path}")
    plt.close()


def export_core_stl(
    nodes: np.ndarray,
    struts: np.ndarray,
    stl_mesh: trimesh.Trimesh,
    prefix: str = "",
) -> Path:
    """
    Build cylinders for ONLY strictly internal struts. Save to output/Core_Only.stl.
    """
    _, internal_struts = _classify_struts(nodes, struts, stl_mesh)
    if internal_struts.shape[0] == 0:
        print("[Core STL] No strictly internal struts.")
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        path = OUTPUT_DIR / f"{prefix}Core_Only.stl"
        # Empty mesh
        empty = trimesh.Trimesh(vertices=np.zeros((0, 3)), faces=np.zeros((0, 3), dtype=np.int32))
        empty.export(str(path))
        return path

    lattice = generate_geometry(
        nodes=nodes,
        struts=internal_struts,
        strut_radius=STRUT_RADIUS,
        boundary_mesh=None,
        crop_to_boundary=False,
    )
    if isinstance(lattice, tuple):
        lattice = lattice[0]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{prefix}Core_Only.stl"
    lattice.export(str(path))
    print(f"[Core STL] Saved: {path} ({internal_struts.shape[0]} struts)")
    return path


def export_final_stl(
    nodes: np.ndarray,
    struts: np.ndarray,
    stl_mesh: trimesh.Trimesh,
    prefix: str = "",
) -> Path:
    """
    Build cylinders for all struts, clip against STL boundary. Save to output/Final_Composite.stl.
    """
    lattice = generate_geometry(
        nodes=nodes,
        struts=struts,
        strut_radius=STRUT_RADIUS,
        boundary_mesh=stl_mesh,
        crop_to_boundary=True,
    )
    if isinstance(lattice, tuple):
        lattice = lattice[0]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{prefix}Final_Composite.stl"
    lattice.export(str(path))
    print(f"[Final STL] Saved: {path} ({struts.shape[0]} struts, clipped)")
    return path


def _run_pipeline(
    name: str,
    nodes: np.ndarray,
    struts: np.ndarray,
    stl_mesh: trimesh.Trimesh,
    prefix: str,
) -> None:
    """Run 3-step export for a given topology."""
    print(f"\n{'=' * 60}")
    print(f"{name}")
    print("=" * 60)
    print(f"Raw: {nodes.shape[0]} nodes, {struts.shape[0]} struts")

    nodes, struts = trim_to_stl(nodes, struts, stl_mesh)
    print(f"Trimmed: {nodes.shape[0]} nodes, {struts.shape[0]} struts")

    print("\n--- Step 1: Surface Plot ---")
    export_surface_plot(nodes, struts, stl_mesh, prefix=prefix)

    print("\n--- Step 2: Core Only ---")
    export_core_stl(nodes, struts, stl_mesh, prefix=prefix)

    print("\n--- Step 3: Final Composite ---")
    export_final_stl(nodes, struts, stl_mesh, prefix=prefix)


def main() -> None:
    """Run the 3-step export for Cartesian Kagome and BCC Voronoi Foam."""
    print("=" * 60)
    print("Supercell Pipeline — Kagome + Voronoi in 20mm Cube")
    print("=" * 60)

    stl_mesh = _get_dummy_cube_stl(20.0)
    print(f"STL: 20mm cube, watertight={stl_mesh.is_watertight}")

    bbox = ((-10, -10, -10), (10, 10, 10))
    cell_size = 4.0

    # Cartesian Kagome
    nodes_kagome, struts_kagome = get_cartesian_kagome(bbox, cell_size)
    _run_pipeline("Cartesian Kagome", nodes_kagome, struts_kagome, stl_mesh, "Kagome_")

    # BCC Voronoi Foam
    nodes_voronoi, struts_voronoi = get_voronoi_foam(bbox, cell_size, seed_type="BCC", jitter=0.0)
    _run_pipeline("BCC Voronoi Foam", nodes_voronoi, struts_voronoi, stl_mesh, "Voronoi_")

    print("\nDone.")


if __name__ == "__main__":
    main()
