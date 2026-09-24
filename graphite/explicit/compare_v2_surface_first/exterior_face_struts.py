"""Exterior faces only — complete quads fully outside the CAD (no morph, no stubs).

How this differs from the failed attempt
--------------------------------------
Failed approach:
  1. Keep surface-straddling hexes (a thick shell).
  2. Take *all* exposed faces of that shell — including faces that look into the
     hollow where fully-inside hexes were removed (those faces sit *inside* the
     part).
  3. Emit every edge, then delete edges that intersect CAD → **hanging stubs**.

Correct approach:
  1. Same surface-straddling hexes.
  2. Keep a face only if **all four corners are outside** the CAD (SDF > 0).
     That drops cavity-facing faces and any face that cuts the solid.
  3. Emit the **whole** face perimeter (4 edges) or nothing — never orphan an edge.
  4. No morph / projection.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh

from graphite.explicit.compare_v2_surface_first.surface_shell_dual import (
    cull_surface_intersecting_hexes,
)
from graphite.explicit.conformal_core import (
    hex_node_ids_from_elements,
    safe_signed_distance,
)
from graphite.explicit.hex_rules import _HEX_FACES


@dataclass
class ExteriorFaceOnlyReport:
    n_background_hex: int = 0
    n_surface_hex: int = 0
    n_faces_considered: int = 0
    n_faces_all_outside: int = 0
    n_faces_rejected_inside_or_mixed: int = 0
    n_nodes: int = 0
    n_struts: int = 0
    min_corner_sdf: float = 0.0


def build_fully_outside_exterior_faces(
    cad_mesh: trimesh.Trimesh,
    cell_size: float | tuple[float, float, float] | np.ndarray,
    *,
    outside_eps: float = 1e-3,
    round_decimals: int = 6,
) -> tuple[np.ndarray, np.ndarray, ExteriorFaceOnlyReport]:
    """
    Surface-straddling hexes → faces with all corners outside → closed perimeters.

    Returns unique nodes and undirected struts (exterior face edges only).
    """
    report = ExteriorFaceOnlyReport()
    hex_shell, info = cull_surface_intersecting_hexes(cad_mesh, cell_size)
    report.n_background_hex = int(info["n_background_hex"])
    report.n_surface_hex = int(info["n_surface_hex"])

    scaffold, hex_ids = hex_node_ids_from_elements(
        hex_shell, round_decimals=max(round_decimals, 6)
    )
    # Every local face of every surface hex (not "exposed of shell" — that
    # incorrectly includes cavity walls). A face is kept iff all 4 corners
    # are outside; shared exterior faces appear once via the edge set.
    face_list: list[np.ndarray] = []
    for elem in hex_ids:
        for fv in _HEX_FACES:
            face_list.append(np.asarray(elem)[list(fv)].astype(np.int64))
    faces = np.asarray(face_list, dtype=np.int64)
    report.n_faces_considered = int(len(faces))

    sd = safe_signed_distance(cad_mesh, scaffold)
    keep_mask = np.ones(len(faces), dtype=bool)
    for i, face in enumerate(faces):
        s = sd[face]
        if not bool(np.all(s > float(outside_eps))):
            keep_mask[i] = False
    kept = faces[keep_mask]
    report.n_faces_all_outside = int(len(kept))
    report.n_faces_rejected_inside_or_mixed = int(np.count_nonzero(~keep_mask))

    if len(kept) == 0:
        raise ValueError(
            "No fully-outside faces on surface-straddling hexes; "
            "try a finer cell size"
        )

    edge_set: set[tuple[int, int]] = set()
    used_nodes: set[int] = set()
    for face in kept:
        corners = [int(c) for c in face]
        used_nodes.update(corners)
        for a, b in zip(corners, corners[1:] + corners[:1]):
            if a != b:
                edge_set.add((min(a, b), max(a, b)))

    used = sorted(used_nodes)
    old_to_new = {old: new for new, old in enumerate(used)}
    nodes = scaffold[np.asarray(used, dtype=np.int64)].copy()
    struts = np.asarray(
        [(old_to_new[a], old_to_new[b]) for a, b in sorted(edge_set)],
        dtype=np.int64,
    )
    report.n_nodes = int(len(nodes))
    report.n_struts = int(len(struts))
    report.min_corner_sdf = float(sd[np.asarray(used, dtype=np.int64)].min())

    print(
        f"  Fully-outside exterior faces: surface_hex={report.n_surface_hex}, "
        f"faces kept={report.n_faces_all_outside}/"
        f"{report.n_faces_considered} "
        f"(rejected inside/mixed={report.n_faces_rejected_inside_or_mixed}), "
        f"closed edges={report.n_struts}, nodes={report.n_nodes}, "
        f"min_corner_sdf={report.min_corner_sdf:.3f}"
    )
    return nodes, struts, report
