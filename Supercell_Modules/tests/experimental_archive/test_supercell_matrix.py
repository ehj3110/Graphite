"""
Supercell discovery matrix:
  - One rule (Kagome) across 4 topologies
  - One topology (Bitruncated) across 4 rules

Exports 8 STLs to:
  Supercell_Modules/output/Supercell_Discovery/

Run:
  python -m Supercell_Modules.tests.test_supercell_matrix
"""

from __future__ import annotations

import sys
from pathlib import Path

import manifold3d
import numpy as np

# Add Graphite root for geometry_module import
_graphite_root = Path(__file__).resolve().parents[2]
if str(_graphite_root) not in sys.path:
    sys.path.insert(0, str(_graphite_root))

# Add Supercell_Modules for core imports
_supercell_dir = Path(__file__).resolve().parent.parent
if str(_supercell_dir) not in sys.path:
    sys.path.insert(0, str(_supercell_dir))

from geometry_module import manifold_to_trimesh
from core.factory import LatticeFactory


OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output" / "Supercell_Discovery"
CELL_SIZE = 10.0
NODE_RADIUS = 0.25
STRUT_RADIUS = 0.1


def _frame_from_segment(p0: np.ndarray, p1: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    vec = p1 - p0
    length = float(np.linalg.norm(vec))
    if length <= 1e-9:
        raise ValueError("Zero-length segment.")

    z_axis = vec / length
    ref = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(np.dot(z_axis, ref)) > 0.98:
        ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    x_axis = np.cross(ref, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    R = np.column_stack((x_axis, y_axis, z_axis))
    return R, length, 0.5 * (p0 + p1)


def build_mesh(nodes: np.ndarray, struts: np.ndarray) -> manifold3d.Manifold:
    """Render nodes/struts as manifold mesh."""
    parts: list[manifold3d.Manifold] = []
    for i, j in struts:
        p0 = nodes[int(i)]
        p1 = nodes[int(j)]
        if np.linalg.norm(p1 - p0) <= 1e-9:
            continue
        R, seg_len, mid = _frame_from_segment(p0, p1)
        cyl = manifold3d.Manifold.cylinder(
            height=seg_len,
            radius_low=STRUT_RADIUS,
            radius_high=STRUT_RADIUS,
            circular_segments=16,
            center=True,
        )
        affine = [
            [float(R[0, 0]), float(R[0, 1]), float(R[0, 2]), float(mid[0])],
            [float(R[1, 0]), float(R[1, 1]), float(R[1, 2]), float(mid[1])],
            [float(R[2, 0]), float(R[2, 1]), float(R[2, 2]), float(mid[2])],
        ]
        parts.append(cyl.transform(affine))

    for xyz in nodes:
        parts.append(manifold3d.Manifold.sphere(NODE_RADIUS).translate(tuple(float(v) for v in xyz)))

    if not parts:
        raise RuntimeError("No primitives generated for discovery mesh.")
    return manifold3d.Manifold.compose(parts)


def export_case(factory: LatticeFactory, topology: str, rule: str, filename: str) -> None:
    nodes, struts = factory.generate_lattice(topology, rule, cell_size=CELL_SIZE, nx=1, ny=1, nz=1)
    mesh = build_mesh(nodes, struts)
    path = OUTPUT_DIR / filename
    manifold_to_trimesh(mesh).export(str(path))
    print(f"{filename}: nodes={nodes.shape[0]}, struts={struts.shape[0]}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    factory = LatticeFactory()

    print("=" * 72)
    print("SUPERCELL DISCOVERY MATRIX")
    print("=" * 72)

    # Row sweep: one rule across all topologies
    kagome_topologies = [
        "bitruncated",
        "truncated_oct_tet",
        "rhombicuboct",
        "a15",
    ]
    for topo in kagome_topologies:
        export_case(
            factory,
            topology=topo,
            rule="kagome",
            filename=f"{topo}_Kagome.stl",
        )

    # Column sweep: one topology across all rules
    rules = ["kagome", "voronoi", "icosahedral", "rhombic"]
    for rule in rules:
        export_case(
            factory,
            topology="bitruncated",
            rule=rule,
            filename=f"Bitruncated_{rule}.stl",
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
