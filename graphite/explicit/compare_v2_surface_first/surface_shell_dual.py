"""Surface-intersecting hex shell → exterior-only dual outside the CAD.

Cull keeps only hexes that *straddle* the surface (some face centroids in,
some out). From that shell, keep exposed faces whose outward normal points
into free space. Dual nodes are placed on the CAD along that normal, then
pushed outward by ``clearance`` so the dual sits slightly outside the part
and does not intersect it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh

from graphite.explicit.conformal_core import (
    classify_boundary_from_hex_elems,
    safe_signed_distance,
)
from graphite.explicit.hex_rules import _HEX_FACES
from graphite.explicit.hex_surface_dual import (
    boundary_quad_centroids,
    get_boundary_quad_adjacency,
)
from graphite.explicit.proven_topologies import generate_background_grid

from .dual import _project_along_rays, _quad_outward_normals


@dataclass
class SurfaceShellReport:
    n_background_hex: int = 0
    n_surface_hex: int = 0
    n_exposed_faces: int = 0
    n_exterior_faces: int = 0
    n_dual_nodes: int = 0
    n_dual_struts: int = 0
    clearance_mm: float = 0.0
    min_dual_sdf: float = 0.0
    n_dual_inside: int = 0
    placement: str = "exterior_face_corners_on_cad_plus_clearance"


def cull_surface_intersecting_hexes(
    cad_mesh: trimesh.Trimesh,
    cell_size: float | tuple[float, float, float] | np.ndarray,
) -> tuple[np.ndarray, dict]:
    """
    Keep only SC hexes whose face-centroids straddle the CAD surface.

    Fully inside (all 6 face centroids in) and fully outside (all out) are
    dropped. Surviving cells are the surface shell.
    """
    grid_nodes, cells = generate_background_grid("SC", cad_mesh.bounds, cell_size)
    all_centroids: list[np.ndarray] = []
    cell_to_centroid_idx: list[list[int]] = []
    for cell in cells:
        idxs: list[int] = []
        corners = grid_nodes[cell]
        for fv in _HEX_FACES:
            idxs.append(len(all_centroids))
            all_centroids.append(corners[list(fv)].mean(axis=0))
        cell_to_centroid_idx.append(idxs)

    cents = np.asarray(all_centroids, dtype=np.float64)
    s_dists = safe_signed_distance(cad_mesh, cents)

    kept: list[np.ndarray] = []
    for i, cell in enumerate(cells):
        c_dists = s_dists[cell_to_centroid_idx[i]]
        n_inside = int(np.sum(c_dists <= 1e-5))
        n_faces = len(c_dists)
        # Straddle: at least one in, at least one out.
        if 0 < n_inside < n_faces:
            kept.append(grid_nodes[cell])

    info = {
        "n_background_hex": int(len(cells)),
        "n_surface_hex": int(len(kept)),
    }
    if not kept:
        raise ValueError("No surface-intersecting hex cells found")
    return np.asarray(kept, dtype=np.float64), info


def _filter_exterior_facing_quads(
    scaffold: np.ndarray,
    quads: np.ndarray,
    normals: np.ndarray,
    cad_mesh: trimesh.Trimesh,
    *,
    probe: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Keep exposed faces whose outward normal points into free space.

    Probe a short step from the face centroid along the outward normal; keep
    the face if that probe is outside the CAD (or more outside than the
    centroid). Drops faces that look into the hollow interior of the shell.
    """
    if len(quads) == 0:
        return quads
    cents = boundary_quad_centroids(scaffold, quads)
    sd0 = safe_signed_distance(cad_mesh, cents)
    probes = cents + float(probe) * normals
    sd1 = safe_signed_distance(cad_mesh, probes)
    # Exterior-facing: probe is outside, or clearly moved outward.
    keep = (sd1 > 1e-3) | ((sd1 - sd0) > 0.1)
    return quads[keep], normals[keep]


def build_exterior_surface_shell_dual(
    cad_mesh: trimesh.Trimesh,
    cell_size: float | tuple[float, float, float] | np.ndarray,
    *,
    clearance: float = 0.6,
    dual_topology: str = "corners",
    ray_length: float = 80.0,
    round_decimals: int = 6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, SurfaceShellReport]:
    """
    Surface-shell hexes → exterior faces → dual outside the CAD.

    Parameters
    ----------
    clearance :
        Extra offset along the outward CAD/face normal after surface hit so
        dual nodes sit outside the solid (mm).
    dual_topology :
        ``"corners"`` — unique corners of exterior faces (grid skin).
        ``"centroids"`` — one node per exterior face (face-centroid dual).

    Returns
    -------
    hex_shell, dual_nodes, dual_struts, dual_normals, report
    """
    report = SurfaceShellReport(clearance_mm=float(clearance))
    hex_shell, info = cull_surface_intersecting_hexes(cad_mesh, cell_size)
    report.n_background_hex = int(info["n_background_hex"])
    report.n_surface_hex = int(info["n_surface_hex"])
    print(
        f"  Surface-shell hexes: {report.n_surface_hex} / {report.n_background_hex} "
        f"(straddling surface only)"
    )

    scaffold, hex_ids, boundary_quads, _ = classify_boundary_from_hex_elems(
        hex_shell, round_decimals=max(round_decimals, 6)
    )
    report.n_exposed_faces = int(len(boundary_quads))
    if boundary_quads.size == 0:
        raise ValueError("Surface shell has no exposed faces")

    hex_cents = hex_shell.mean(axis=1)
    interior_ref = hex_cents.mean(axis=0)
    face_normals = _quad_outward_normals(scaffold, boundary_quads, interior_ref)
    ext_quads, ext_normals = _filter_exterior_facing_quads(
        scaffold, boundary_quads, face_normals, cad_mesh
    )
    report.n_exterior_faces = int(len(ext_quads))
    print(
        f"  Exterior faces: {report.n_exterior_faces} / {report.n_exposed_faces} "
        f"exposed on surface shell"
    )
    if len(ext_quads) == 0:
        raise ValueError("No exterior-facing faces on surface shell")

    topo = str(dual_topology).strip().lower()
    if topo not in ("corners", "centroids"):
        raise ValueError("dual_topology must be 'corners' or 'centroids'")

    if topo == "centroids":
        origins = boundary_quad_centroids(scaffold, ext_quads)
        normals = ext_normals
        # Face–face adjacency among exterior faces only (shared edge).
        pairs, _ = get_boundary_quad_adjacency(ext_quads)
        strut_set = {
            (int(min(i, j)), int(max(i, j))) for i, j in pairs if int(i) != int(j)
        }
        report.placement = "exterior_face_centroids_on_cad_plus_clearance"
    else:
        corner_nsum: dict[int, np.ndarray] = {}
        for face, n in zip(ext_quads, ext_normals):
            for cid in face:
                cid = int(cid)
                corner_nsum.setdefault(cid, np.zeros(3, dtype=np.float64))
                corner_nsum[cid] += n
        corner_ids = sorted(corner_nsum.keys())
        id_to_local = {cid: i for i, cid in enumerate(corner_ids)}
        origins = scaffold[np.asarray(corner_ids, dtype=np.int64)].copy()
        normals = np.zeros((len(corner_ids), 3), dtype=np.float64)
        for cid in corner_ids:
            s = corner_nsum[cid]
            nn = float(np.linalg.norm(s))
            normals[id_to_local[cid]] = (
                s / nn if nn > 1e-12 else np.array([0.0, 0.0, 1.0])
            )
        strut_set = set()
        for face in ext_quads:
            corners = [int(c) for c in face]
            for a, b in zip(corners, corners[1:] + corners[:1]):
                if a == b:
                    continue
                la, lb = id_to_local[a], id_to_local[b]
                strut_set.add((min(la, lb), max(la, lb)))
        report.placement = "exterior_face_corners_on_cad_plus_clearance"

    # Raycast to CAD along ± outward normal; prefer outward hit.
    on_surface, hit_mask = _project_along_rays(
        origins, normals, cad_mesh, ray_length=ray_length
    )
    # Push outside along the dual's placement normal (face/avg), falling back
    # to CAD closest-point normal if needed.
    clearance = float(clearance)
    dual_nodes = on_surface + clearance * normals
    # If a node is still inside (concave / bad normal), force closest-point +
    # CAD face normal offset.
    sd = safe_signed_distance(cad_mesh, dual_nodes)
    bad = sd <= float(clearance) * 0.25
    if np.any(bad):
        pq = trimesh.proximity.ProximityQuery(cad_mesh)
        closest, _, tri = pq.on_surface(dual_nodes[bad])
        cad_n = np.asarray(cad_mesh.face_normals, dtype=np.float64)[
            np.asarray(tri, dtype=np.int64)
        ]
        nn = np.linalg.norm(cad_n, axis=1, keepdims=True)
        cad_n = cad_n / np.maximum(nn, 1e-12)
        # Orient CAD normal outward (positive SDF direction ≈ outside).
        # trimesh normals may point either way; flip so offset increases SDF.
        trial = closest + clearance * cad_n
        sd_trial = safe_signed_distance(cad_mesh, trial)
        flip = sd_trial < safe_signed_distance(cad_mesh, closest)
        cad_n[flip] *= -1.0
        dual_nodes[bad] = closest + clearance * cad_n
        normals[bad] = cad_n

    sd_final = safe_signed_distance(cad_mesh, dual_nodes)
    report.min_dual_sdf = float(sd_final.min()) if len(sd_final) else 0.0
    report.n_dual_inside = int(np.count_nonzero(sd_final <= 1e-3))
    report.n_dual_nodes = int(len(dual_nodes))
    dual_struts = (
        np.asarray(sorted(strut_set), dtype=np.int64)
        if strut_set
        else np.empty((0, 2), dtype=np.int64)
    )
    report.n_dual_struts = int(len(dual_struts))
    print(
        f"  Exterior dual: nodes={report.n_dual_nodes}, struts={report.n_dual_struts}, "
        f"clearance={clearance:g} mm, min_sdf={report.min_dual_sdf:.3f}, "
        f"inside_nodes={report.n_dual_inside}"
    )
    return (
        hex_shell,
        dual_nodes.astype(np.float64, copy=False),
        dual_struts,
        normals.astype(np.float64, copy=False),
        report,
    )
