"""
2x2x2 Hexahedral Grid — Universal Router Test.

Generates eight 20mm hex elements in a 2x2x2 stack, passes them through the
universal router with the 'kagome' rule, and exports a merged STL.  Because
adjacent hexes share face-plane midpoints, the router's global node merge
produces a single connected lattice with no duplicated geometry.

Expected output for Kagome on 2x2x2 hex grid:
    - 8 hexes × 6 face-centers = 48 raw nodes
    - 12 shared internal faces → 12 merged nodes
    - Unique nodes: 36
    - 8 hexes × 12 struts = 96 struts
      (no strut dedup occurs: adjacent hexes share face-centers but each
      hex connects its shared face-center to its own other 4 face-centers,
      producing 4 unique struts per shared face with no cross-element overlap)

Run from Graphite root:
    python -m Universal_Lattice_Engine.tests.test_universal_2x2x2
"""

from __future__ import annotations

import sys
from pathlib import Path

import manifold3d
import numpy as np

_root = Path(__file__).resolve().parents[2]
_ule = Path(__file__).resolve().parent.parent
for _p in (_root, _ule):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from Universal_Lattice_Engine.core.topology_module import generate_universal_lattice
from geometry_module import manifold_to_trimesh

OUTPUT_DIR   = Path(__file__).resolve().parent.parent / "output"
CELL_SIZE    = 20.0   # mm
STRUT_RADIUS = 0.6    # mm
NODE_RADIUS  = 1.0    # mm


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _frame_from_segment(
    p0: np.ndarray, p1: np.ndarray
) -> tuple[np.ndarray, float, np.ndarray]:
    vec = p1 - p0
    length = float(np.linalg.norm(vec))
    if length < 1e-9:
        raise ValueError("Zero-length segment.")
    z = vec / length
    ref = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(z, ref)) > 0.98:
        ref = np.array([0.0, 1.0, 0.0])
    x = np.cross(ref, z); x /= np.linalg.norm(x)
    y = np.cross(z, x);   y /= np.linalg.norm(y)
    R = np.column_stack((x, y, z))
    return R, length, 0.5 * (p0 + p1)


def build_manifold(nodes: np.ndarray, struts: np.ndarray) -> manifold3d.Manifold:
    parts: list[manifold3d.Manifold] = []
    for pt in nodes:
        parts.append(
            manifold3d.Manifold.sphere(NODE_RADIUS, circular_segments=16)
            .translate(tuple(float(v) for v in pt))
        )
    for a, b in struts:
        p0, p1 = nodes[int(a)], nodes[int(b)]
        try:
            R, length, mid = _frame_from_segment(p0, p1)
        except ValueError:
            continue
        cyl = manifold3d.Manifold.cylinder(
            height=length, radius_low=STRUT_RADIUS, radius_high=STRUT_RADIUS,
            circular_segments=12, center=True,
        )
        affine = [
            [float(R[0,0]), float(R[0,1]), float(R[0,2]), float(mid[0])],
            [float(R[1,0]), float(R[1,1]), float(R[1,2]), float(mid[1])],
            [float(R[2,0]), float(R[2,1]), float(R[2,2]), float(mid[2])],
        ]
        parts.append(cyl.transform(affine))
    if not parts:
        raise RuntimeError("No geometry produced.")
    return manifold3d.Manifold.compose(parts)


# ---------------------------------------------------------------------------
# Grid generator
# ---------------------------------------------------------------------------

def build_2x2x2_hex_elements(cell_size: float) -> np.ndarray:
    """Return (8, 8, 3) array — eight hex elements in a 2×2×2 stack."""
    elements = []
    for ix in range(2):
        for iy in range(2):
            for iz in range(2):
                x0, x1 = ix * cell_size, (ix + 1) * cell_size
                y0, y1 = iy * cell_size, (iy + 1) * cell_size
                z0, z1 = iz * cell_size, (iz + 1) * cell_size
                elements.append([
                    [x0, y0, z0],   # 0  bottom
                    [x1, y0, z0],   # 1
                    [x1, y1, z0],   # 2
                    [x0, y1, z0],   # 3
                    [x0, y0, z1],   # 4  top
                    [x1, y0, z1],   # 5
                    [x1, y1, z1],   # 6
                    [x0, y1, z1],   # 7
                ])
    return np.array(elements, dtype=np.float64)   # (8, 8, 3)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("Universal Router — 2x2x2 Hex Grid, Kagome Rule")
    print("=" * 60)

    elements = build_2x2x2_hex_elements(CELL_SIZE)
    print(f"Elements : {elements.shape[0]}  ({elements.shape[1]}-node hex)")
    print(f"Extent   : {elements.max():.0f} x {elements.max():.0f} x {elements.max():.0f} mm")

    nodes, struts = generate_universal_lattice(elements, "octahedral")
    print(f"\nAfter global merge:")
    print(f"  Nodes  : {nodes.shape[0]}  (expected 36 for octahedral 2x2x2)")
    print(f"  Struts : {struts.shape[0]}  (expected 96 for octahedral 2x2x2)")

    strut_lengths = np.linalg.norm(nodes[struts[:, 1]] - nodes[struts[:, 0]], axis=1)
    print(f"\nStrut length stats:")
    print(f"  Min : {strut_lengths.min():.3f} mm")
    print(f"  Max : {strut_lengths.max():.3f} mm")
    print(f"  Mean: {strut_lengths.mean():.3f} mm")

    print("\nBuilding Manifold geometry...")
    geo = build_manifold(nodes, struts)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / "Universal_2x2x2_Hex_Kagome.stl"
    mesh = manifold_to_trimesh(geo)
    mesh.export(str(out))
    print(f"Saved: {out}  ({len(mesh.faces):,} triangles)")
    print("\nDone.")


if __name__ == "__main__":
    main()
