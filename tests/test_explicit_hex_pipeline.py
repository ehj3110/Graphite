from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from graphite.explicit.geometry_module import generate_geometry
from graphite.explicit.hex_scaffold_module import (
    generate_conformed_hex_scaffold,
    generate_cropped_hex_scaffold,
)
from graphite.explicit.hex_topology_module import _HEX_RULES, generate_hex_topology


def _build_structured_hex_grid(nx: int, ny: int, nz: int, cell_size: float = 1.0) -> np.ndarray:
    xs = np.arange(nx + 1, dtype=np.float64) * float(cell_size)
    ys = np.arange(ny + 1, dtype=np.float64) * float(cell_size)
    zs = np.arange(nz + 1, dtype=np.float64) * float(cell_size)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    points = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
    nny, nnz = len(ys), len(zs)

    def idx(i: int, j: int, k: int) -> int:
        return i * (nny * nnz) + j * nnz + k

    corners = [
        (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
        (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1),
    ]
    out: list[np.ndarray] = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                ids = [idx(i + di, j + dj, k + dk) for di, dj, dk in corners]
                out.append(points[ids])
    return np.asarray(out, dtype=np.float64)


def test_route2_cropped_hex_boolean_crop_baseline() -> None:
    box = trimesh.creation.box(extents=[10.0, 10.0, 10.0])
    hexes = generate_cropped_hex_scaffold(box, target_element_size=2.5)
    assert len(hexes) > 0

    nodes, struts = generate_hex_topology(hexes, rule_name="grid")
    out = generate_geometry(
        nodes,
        struts,
        strut_radius=0.1,
        boundary_mesh=box,
        crop_to_boundary=True,
    )
    mesh = out[0] if isinstance(out, tuple) else out
    assert mesh.is_watertight
    assert np.allclose(mesh.bounds[:, 2], np.array([-5.0, 5.0]), atol=1e-6)


def test_route3_conformed_hex_report_and_watertight() -> None:
    sphere = trimesh.creation.icosphere(subdivisions=2, radius=5.0)
    hexes, report = generate_conformed_hex_scaffold(sphere, target_element_size=2.0)
    assert len(hexes) > 0
    assert "shrink_hexes" in report
    assert "grow_hexes" in report
    assert "rejected_inversion" in report
    assert report.get("neighbor_stretch_nodes", 0) > 0
    assert report.get("dropped_hexes", 0) > 0

    nodes, struts = generate_hex_topology(hexes, rule_name="grid")
    out = generate_geometry(
        nodes,
        struts,
        strut_radius=0.1,
        boundary_mesh=sphere,
        crop_to_boundary=True,
    )
    mesh = out[0] if isinstance(out, tuple) else out
    assert mesh.is_watertight


def test_hex_topology_adjacency_for_all_rules_and_hex_dual_counts() -> None:
    hexes = _build_structured_hex_grid(2, 2, 2, cell_size=1.0)  # 8 cells
    assert len(hexes) == 8

    for rule_name in sorted(_HEX_RULES.keys()):
        nodes, struts = generate_hex_topology(hexes, rule_name=rule_name)
        assert nodes.ndim == 2 and nodes.shape[1] == 3
        assert struts.ndim == 2 and struts.shape[1] == 2
        if rule_name == "hex_dual":
            assert len(nodes) == 32
            assert len(struts) == 36
        if rule_name == "hex_face_dual":
            assert len(nodes) == 36
            assert len(struts) == 96

