"""
20mm Single Hexahedral Element — Rule Validation & STL Export.

Defines one 20mm cube hex, applies all 3 hex rules, renders with Manifold3D,
and exports:
    output/20mm_Hex_Voronoi.stl
    output/20mm_Hex_Rhombic.stl
    output/20mm_Hex_Kagome.stl

Run from Graphite root:
    python -m Hex_Sandbox.tests.test_20mm_hex
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import manifold3d

# ---------------------------------------------------------------------------
# Path setup — allow imports from Hex_Sandbox and Graphite root
# ---------------------------------------------------------------------------
_root = Path(__file__).resolve().parents[2]
_hex_sandbox = Path(__file__).resolve().parents[1]
for _p in (_root, _hex_sandbox):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from core.hex_rules import apply_hex_voronoi, apply_hex_rhombic, apply_hex_kagome

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output"
STRUT_RADIUS = 0.8   # mm
NODE_RADIUS = 1.2    # mm


# ---------------------------------------------------------------------------
# Rendering helpers (self-contained — no external geometry_module dependency)
# ---------------------------------------------------------------------------

def _frame_from_segment(
    p0: np.ndarray, p1: np.ndarray
) -> tuple[np.ndarray, float, np.ndarray]:
    """
    Return (R 3x3, length, midpoint) for a cylinder aligned with p0 -> p1.

    manifold3d cylinders are Z-aligned and centered at origin when center=True.
    We build an orthonormal frame whose Z-axis points along the segment and
    translate to the midpoint via Manifold.transform().
    """
    vec = p1 - p0
    length = float(np.linalg.norm(vec))
    if length < 1e-9:
        raise ValueError("Degenerate segment (zero length).")

    z_axis = vec / length
    ref = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(np.dot(z_axis, ref)) > 0.98:
        ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    x_axis = np.cross(ref, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)

    R = np.column_stack((x_axis, y_axis, z_axis))   # (3, 3)
    midpoint = 0.5 * (p0 + p1)
    return R, length, midpoint


def build_manifold(
    nodes: np.ndarray,
    struts: np.ndarray,
    strut_radius: float = STRUT_RADIUS,
    node_radius: float = NODE_RADIUS,
) -> manifold3d.Manifold:
    """Render nodes as spheres and struts as cylinders, return composed Manifold."""
    parts: list[manifold3d.Manifold] = []

    # Nodes
    for pt in nodes:
        sphere = manifold3d.Manifold.sphere(radius=node_radius, circular_segments=16)
        sphere = sphere.translate(tuple(float(v) for v in pt))
        parts.append(sphere)

    # Struts — centered cylinder transformed to midpoint via 3x4 affine
    for a_idx, b_idx in struts:
        p0 = nodes[int(a_idx)]
        p1 = nodes[int(b_idx)]
        try:
            R, length, mid = _frame_from_segment(p0, p1)
        except ValueError:
            continue
        cyl = manifold3d.Manifold.cylinder(
            height=length,
            radius_low=strut_radius,
            radius_high=strut_radius,
            circular_segments=12,
            center=True,
        )
        affine = [
            [float(R[0, 0]), float(R[0, 1]), float(R[0, 2]), float(mid[0])],
            [float(R[1, 0]), float(R[1, 1]), float(R[1, 2]), float(mid[1])],
            [float(R[2, 0]), float(R[2, 1]), float(R[2, 2]), float(mid[2])],
        ]
        parts.append(cyl.transform(affine))

    if not parts:
        raise RuntimeError("No geometry produced — check nodes/struts arrays.")

    return manifold3d.Manifold.compose(parts)


def export_stl(manifold: manifold3d.Manifold, path: Path) -> None:
    mesh = manifold.to_mesh()
    verts = np.array(mesh.vert_properties, dtype=np.float32)
    tris = np.array(mesh.tri_verts, dtype=np.int32)
    import struct
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"\x00" * 80)
        f.write(struct.pack("<I", len(tris)))
        for tri in tris:
            v0, v1, v2 = verts[tri[0]], verts[tri[1]], verts[tri[2]]
            normal = np.cross(v1 - v0, v2 - v0)
            n_len = np.linalg.norm(normal)
            if n_len > 0:
                normal /= n_len
            f.write(struct.pack("<fff", *normal))
            f.write(struct.pack("<fff", *v0))
            f.write(struct.pack("<fff", *v1))
            f.write(struct.pack("<fff", *v2))
            f.write(b"\x00\x00")
    print(f"  Saved: {path}  ({len(tris):,} triangles)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

#: Standard 20mm cube hex — nodes in the order expected by hex_rules.py
HEX_20MM = np.array([
    [ 0,  0,  0],   # 0 bottom
    [20,  0,  0],   # 1
    [20, 20,  0],   # 2
    [ 0, 20,  0],   # 3
    [ 0,  0, 20],   # 4 top
    [20,  0, 20],   # 5
    [20, 20, 20],   # 6
    [ 0, 20, 20],   # 7
], dtype=np.float64)


def run_rule(name: str, rule_fn, coords: np.ndarray) -> None:
    print(f"\n{'=' * 60}")
    print(f"Rule: {name}")
    print("=" * 60)

    nodes, struts = rule_fn(coords)
    print(f"  Nodes  : {nodes.shape[0]}")
    print(f"  Struts : {struts.shape[0]}")
    for i, (a, b) in enumerate(struts):
        length = np.linalg.norm(nodes[b] - nodes[a])
        print(f"    Strut {i:>2d}: node {a} -> node {b}  |  length = {length:.3f} mm")

    geo = build_manifold(nodes, struts)
    out = OUTPUT_DIR / f"20mm_Hex_{name}.stl"
    export_stl(geo, out)


def main() -> None:
    print("=" * 60)
    print("20mm Hexahedral Element — Rule Export")
    print("=" * 60)
    print(f"Hex corners:\n{HEX_20MM}\n")

    run_rule("Voronoi",  apply_hex_voronoi,  HEX_20MM)
    run_rule("Rhombic",  apply_hex_rhombic,  HEX_20MM)
    run_rule("Kagome",   apply_hex_kagome,   HEX_20MM)

    print("\nDone.")


if __name__ == "__main__":
    main()
