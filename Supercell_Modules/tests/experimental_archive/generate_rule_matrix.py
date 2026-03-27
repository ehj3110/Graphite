"""
Generate rule-matrix STLs from Freudenthal 6-tet cube structures.

Exports to:
  Supercell_Modules/output/Rule_Matrix/
    - Kagome_Single.stl
    - Voronoi_Single.stl
    - Icosahedral_Single.stl
    - Rhombic_Single.stl
    - Kagome_2x2x2.stl
    - Voronoi_2x2x2.stl
    - Icosahedral_2x2x2.stl
    - Rhombic_2x2x2.stl

Run from Graphite root:
    python -m Supercell_Modules.tests.generate_rule_matrix
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

from graphite.explicit.geometry_module import manifold_to_trimesh
from core.lattice_rules import (
    apply_icosahedral_rule,
    apply_kagome_rule,
    apply_rhombic_rule,
    apply_voronoi_rule,
)


OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output" / "Rule_Matrix"
CELL_SIZE = 10.0

# Freudenthal-like 6-tet decomposition along cube diagonal (0 -> 7).
FREUDENTHAL_TETS = (
    (0, 1, 3, 7),
    (0, 1, 5, 7),
    (0, 4, 5, 7),
    (0, 2, 3, 7),
    (0, 2, 6, 7),
    (0, 4, 6, 7),
)


def cube_vertices(origin: np.ndarray, cell_size: float) -> np.ndarray:
    """Return the 8 cube corners with fixed local index ordering."""
    ox, oy, oz = origin
    L = float(cell_size)
    return np.array(
        [
            [ox, oy, oz],  # 0
            [ox + L, oy, oz],  # 1
            [ox, oy + L, oz],  # 2
            [ox + L, oy + L, oz],  # 3
            [ox, oy, oz + L],  # 4
            [ox + L, oy, oz + L],  # 5
            [ox, oy + L, oz + L],  # 6
            [ox + L, oy + L, oz + L],  # 7
        ],
        dtype=np.float64,
    )


def generate_6tet_supercell(nx: int, ny: int, nz: int, cell_size: float) -> list[np.ndarray]:
    """Create list of tetrahedra coords (4,3) for a nx*ny*nz cube stack."""
    tets: list[np.ndarray] = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                origin = np.array([i, j, k], dtype=np.float64) * float(cell_size)
                verts = cube_vertices(origin, cell_size)
                for a, b, c, d in FREUDENTHAL_TETS:
                    tets.append(verts[[a, b, c, d]])
    return tets


def _key(xyz: np.ndarray, ndigits: int = 8) -> tuple[float, float, float]:
    return (round(float(xyz[0]), ndigits), round(float(xyz[1]), ndigits), round(float(xyz[2]), ndigits))


def build_global_graph(
    tets: list[np.ndarray],
    rule_fn,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply rule on each tet with global node/strut merging.
    """
    node_map: dict[tuple[float, float, float], int] = {}
    nodes_list: list[np.ndarray] = []
    strut_set: set[tuple[int, int]] = set()

    def add_node(pt: np.ndarray) -> int:
        k = _key(pt)
        idx = node_map.get(k)
        if idx is None:
            idx = len(nodes_list)
            nodes_list.append(np.asarray(pt, dtype=np.float64))
            node_map[k] = idx
        return idx

    for tet in tets:
        local_nodes, local_struts = rule_fn(tet)
        global_ids = [add_node(local_nodes[i]) for i in range(local_nodes.shape[0])]
        for u, v in local_struts:
            a = global_ids[int(u)]
            b = global_ids[int(v)]
            if a == b:
                continue
            strut_set.add((min(a, b), max(a, b)))

    nodes = np.vstack(nodes_list) if nodes_list else np.empty((0, 3), dtype=np.float64)
    struts = np.array(sorted(strut_set), dtype=np.int64) if strut_set else np.empty((0, 2), dtype=np.int64)
    return nodes, struts


def _frame_from_segment(p0: np.ndarray, p1: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    """Return orthonormal frame, segment length, midpoint."""
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
    midpoint = 0.5 * (p0 + p1)
    return R, length, midpoint


def build_manifold(
    nodes: np.ndarray,
    struts: np.ndarray,
    node_radius: float,
    strut_radius: float,
) -> manifold3d.Manifold:
    """Render struts and nodes as manifold primitives."""
    parts: list[manifold3d.Manifold] = []

    for i, j in struts:
        p0 = nodes[int(i)]
        p1 = nodes[int(j)]
        length = float(np.linalg.norm(p1 - p0))
        if length <= 1e-9:
            continue
        R, seg_len, mid = _frame_from_segment(p0, p1)
        cyl = manifold3d.Manifold.cylinder(
            height=seg_len,
            radius_low=float(strut_radius),
            radius_high=float(strut_radius),
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
        parts.append(manifold3d.Manifold.sphere(float(node_radius)).translate(tuple(float(v) for v in xyz)))

    if not parts:
        raise RuntimeError("No manifold parts were generated.")
    return manifold3d.Manifold.compose(parts)


def export_rule_variant(
    name: str,
    rule_fn,
    tets: list[np.ndarray],
    node_radius: float,
    strut_radius: float,
    output_filename: str,
) -> None:
    nodes, struts = build_global_graph(tets, rule_fn)
    manifold_mesh = build_manifold(nodes, struts, node_radius=node_radius, strut_radius=strut_radius)
    out_path = OUTPUT_DIR / output_filename
    manifold_to_trimesh(manifold_mesh).export(str(out_path))
    print(f"{name}: nodes={nodes.shape[0]}, struts={struts.shape[0]} -> {out_path}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 72)
    print("RULE MATRIX EXPORT")
    print("=" * 72)

    tets_single = generate_6tet_supercell(1, 1, 1, CELL_SIZE)
    tets_super = generate_6tet_supercell(3, 3, 2, CELL_SIZE)

    # Category A: Single unit cell
    export_rule_variant(
        "Kagome_Single",
        apply_kagome_rule,
        tets_single,
        node_radius=1.0,
        strut_radius=0.6,
        output_filename="Kagome_Single.stl",
    )
    export_rule_variant(
        "Voronoi_Single",
        apply_voronoi_rule,
        tets_single,
        node_radius=1.0,
        strut_radius=0.6,
        output_filename="Voronoi_Single.stl",
    )
    export_rule_variant(
        "Icosahedral_Single",
        apply_icosahedral_rule,
        tets_single,
        node_radius=1.0,
        strut_radius=0.6,
        output_filename="Icosahedral_Single.stl",
    )
    export_rule_variant(
        "Rhombic_Single",
        apply_rhombic_rule,
        tets_single,
        node_radius=1.0,
        strut_radius=0.6,
        output_filename="Rhombic_Single.stl",
    )

    # Category B: 2x2x2 supercell
    export_rule_variant(
        "Kagome_2x2x2",
        apply_kagome_rule,
        tets_super,
        node_radius=0.5,
        strut_radius=0.3,
        output_filename="Kagome_3x3x2.stl",
    )
    export_rule_variant(
        "Voronoi_2x2x2",
        apply_voronoi_rule,
        tets_super,
        node_radius=0.5,
        strut_radius=0.3,
        output_filename="Voronoi_3x3x2.stl",
    )
    export_rule_variant(
        "Icosahedral_2x2x2",
        apply_icosahedral_rule,
        tets_super,
        node_radius=0.5,
        strut_radius=0.3,
        output_filename="Icosahedral_3x3x2.stl",
    )
    export_rule_variant(
        "Rhombic_2x2x2",
        apply_rhombic_rule,
        tets_super,
        node_radius=0.5,
        strut_radius=0.3,
        output_filename="Rhombic_3x3x2.stl",
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
