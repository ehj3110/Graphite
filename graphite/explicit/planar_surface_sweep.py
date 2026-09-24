"""Planar Slicing Contour Sweep — Conformal Surface Dual for SC Lattices.

Extracts organic CAD surface contours along planar slicing cross-sections between
surface dual nodes, lofts continuous swept solids with in-plane normal alignment,
and applies boundary volume strut promotion and empty valley pruning to produce
watertight, clean conforming surface dual scaffolds.

See docs/PLANAR_SLICING_SURFACE_DUAL_RETROSPECTIVE.md for full engineering details.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy.spatial import cKDTree
import trimesh

try:
    import manifold3d
except ImportError:
    manifold3d = None  # type: ignore[assignment]

from graphite.explicit.conformal_core import safe_signed_distance
from graphite.explicit.geometry_module import _union_manifolds_for_joint


@dataclass
class PlanarSweepConfig:
    """Configuration parameters for the Planar Slicing Contour Sweep surface dual."""

    dual_width: float = 1.6
    """Full ribbon width (lateral dimension parallel to CAD surface) in mm."""

    dual_thickness: float = 0.8
    """Target finished strut thickness (depth dimension normal to CAD surface) in mm."""

    inward_depth_factor: float = 1.5
    """Multiplier on dual_thickness determining initial inward sweep depth before CAD boolean.
    Default 1.5x (e.g. 1.2 mm for 0.8 mm thickness) safely bridges chord sagitta up to 0.4 mm
    without protruding through acute re-entrant corners."""

    outer_margin: float = 0.4
    """Outward extension past CAD surface in mm. Ensures the outer CAD boolean leaves a flush,
    clean surface with zero gap."""

    n_sweep_pts: int = 8
    """Number of sample stations along each chord arc. 8 stations gives chord-sagitta error < 0.05 mm."""

    extend_factor: float = 0.4
    """Endpoint elongation multiplier (x dual_width) along the tangent to ensure full volume
    overlap at multi-strut vertex junctions before boolean union."""

    valley_prune_sdf: float = 0.5
    """Midpoint signed distance (mm) above which non-volume dual struts bridging exterior empty
    valleys / concave gaps are pruned."""

    boundary_promote_eps: float = 0.1
    """Distance threshold (mm) from CAD surface for promoting boundary volume struts into the
    surface dual edge set, sealing rim and corner gaps."""


def generate_planar_sweep_strut_manifold(
    p0: np.ndarray,
    p1: np.ndarray,
    cad: trimesh.Trimesh,
    *,
    width: float = 1.6,
    thick_inward: float = 1.2,
    outer_margin: float = 0.4,
    n_sweep_pts: int = 8,
    extend_factor: float = 0.4,
) -> Any:
    """Generate a single continuous swept solid manifold along the planar surface contour.

    Args:
        p0: Starting point on CAD surface (3,).
        p1: Ending point on CAD surface (3,).
        cad: Reference CAD trimesh.
        width: Lateral strut ribbon width in mm.
        thick_inward: Inward depth extension in mm.
        outer_margin: Outward extension in mm.
        n_sweep_pts: Number of profile stations.
        extend_factor: Fraction of width to extend at endpoints.

    Returns:
        manifold3d.Manifold if successful, or None if degenerate.
    """
    if manifold3d is None:
        raise ImportError("manifold3d is required for planar slicing surface dual generation.")

    p0 = np.asarray(p0, dtype=np.float64).reshape(3)
    p1 = np.asarray(p1, dtype=np.float64).reshape(3)

    t_chord = p1 - p0
    length = float(np.linalg.norm(t_chord))
    if length < 1e-6:
        return None
    t_chord = t_chord / length

    mid = 0.5 * (p0 + p1)
    _, _, tri_id = trimesh.proximity.closest_point(cad, [mid])
    norm_mid = cad.face_normals[tri_id[0]]

    # Compute slicing plane normal w = normalize(t_chord x norm_mid)
    w = np.cross(t_chord, norm_mid)
    w_len = np.linalg.norm(w)
    if w_len < 1e-4:
        w = np.cross(t_chord, np.array([0.0, 0.0, 1.0]))
        w_len = np.linalg.norm(w)
        if w_len < 1e-4:
            w = np.cross(t_chord, np.array([1.0, 0.0, 0.0]))
            w_len = np.linalg.norm(w)
    if np.linalg.norm(w) < 1e-6:
        return None
    w = w / np.linalg.norm(w)

    # Sample chord stations, project onto CAD surface, then project into slicing plane Pi
    ts = np.linspace(0.0, 1.0, n_sweep_pts)
    interp = (1.0 - ts[:, None]) * p0 + ts[:, None] * p1
    closest, _, _ = trimesh.proximity.closest_point(cad, interp)
    curve = closest - np.sum((closest - p0) * w, axis=1, keepdims=True) * w

    half_w = width / 2.0
    extend = extend_factor * width

    rings = []
    for k in range(n_sweep_pts):
        pt = curve[k]
        if k < n_sweep_pts - 1:
            t_k = curve[k + 1] - pt
        else:
            t_k = pt - curve[k - 1]
        t_len = np.linalg.norm(t_k)
        t_k = t_k / (t_len if t_len > 1e-6 else 1.0)

        d_k = np.cross(t_k, w)
        d_len = np.linalg.norm(d_k)
        d_k = d_k / (d_len if d_len > 1e-6 else 1.0)
        if np.dot(d_k, norm_mid) < 0:
            d_k = -d_k

        if k == 0:
            pt = pt - extend * t_k
        elif k == n_sweep_pts - 1:
            pt = pt + extend * t_k

        v0 = pt - half_w * w + outer_margin * d_k
        v1 = pt + half_w * w + outer_margin * d_k
        v2 = pt + half_w * w - thick_inward * d_k
        v3 = pt - half_w * w - thick_inward * d_k
        rings.append([v0, v1, v2, v3])

    verts = np.array(rings).reshape(-1, 3)
    faces = []
    # Start cap
    faces.append([0, 2, 1])
    faces.append([0, 3, 2])
    # Body quads split into triangles
    for k in range(n_sweep_pts - 1):
        r0 = k * 4
        r1 = (k + 1) * 4
        faces.append([r0 + 0, r0 + 1, r1 + 1])
        faces.append([r0 + 0, r1 + 1, r1 + 0])
        faces.append([r0 + 1, r0 + 2, r1 + 2])
        faces.append([r0 + 1, r1 + 2, r1 + 1])
        faces.append([r0 + 2, r0 + 3, r1 + 3])
        faces.append([r0 + 2, r1 + 3, r1 + 2])
        faces.append([r0 + 3, r0 + 0, r1 + 0])
        faces.append([r0 + 3, r1 + 0, r1 + 3])
    # End cap
    end_r = (n_sweep_pts - 1) * 4
    faces.append([end_r + 0, end_r + 1, end_r + 2])
    faces.append([end_r + 0, end_r + 2, end_r + 3])

    mesh = manifold3d.Mesh(
        vert_properties=np.asarray(verts, dtype=np.float32),
        tri_verts=np.asarray(faces, dtype=np.uint32),
    )
    m_strut = manifold3d.Manifold(mesh)
    if m_strut.status() == manifold3d.Error.NoError:
        return m_strut
    return None


def build_planar_slicing_surface_dual(
    cad: trimesh.Trimesh,
    dual_nodes_projected: np.ndarray,
    dual_struts: np.ndarray,
    volume_nodes: np.ndarray | None = None,
    volume_struts: np.ndarray | None = None,
    volume_nodes_deformed: np.ndarray | None = None,
    dual_nodes_cartesian: np.ndarray | None = None,
    *,
    config: PlanarSweepConfig | None = None,
) -> tuple[Any, dict]:
    """Build the raw un-booleaned surface dual manifold via Planar Slicing Contour Sweep.

    Incorporates boundary volume strut promotion (to seal rims/corners) and empty
    valley pruning (to remove floating struts across concave gaps).

    Args:
        cad: Reference CAD trimesh.
        dual_nodes_projected: Surface-projected dual nodes (N, 3).
        dual_struts: Strut connectivity index pairs into dual_nodes_projected (E, 2).
        volume_nodes: Undeformed/Cartesian volume lattice nodes (M, 3), optional.
        volume_struts: Volume lattice connectivity (K, 2), optional.
        volume_nodes_deformed: Boundary-conformed volume nodes (M, 3), optional.
        dual_nodes_cartesian: Undeformed dual nodes (N, 3) matching volume coordinate space,
            used for accurate mapping to volume nodes. Defaults to dual_nodes_projected if omitted.
        config: PlanarSweepConfig instance.

    Returns:
        (dual_raw_manifold, report)
        The returned manifold is the raw union of all swept struts. The caller should apply
        the double boolean: ``(dual_raw ^ cad_outer) - cad_inner`` where ``cad_inner`` is shrunk
        by ``config.dual_thickness``.
    """
    if manifold3d is None:
        raise ImportError("manifold3d is required for planar slicing surface dual generation.")

    if config is None:
        config = PlanarSweepConfig()

    t0 = time.time()
    dual_pts = np.asarray(dual_nodes_projected, dtype=np.float64)
    d_struts = np.asarray(dual_struts, dtype=np.int64).reshape(-1, 2) if len(dual_struts) else np.empty((0, 2), dtype=np.int64)

    has_vol = volume_nodes is not None and volume_struts is not None and len(volume_nodes) > 0 and len(volume_struts) > 0
    vol_pts = np.asarray(volume_nodes, dtype=np.float64) if has_vol else np.empty((0, 3), dtype=np.float64)
    v_struts = np.asarray(volume_struts, dtype=np.int64).reshape(-1, 2) if has_vol else np.empty((0, 2), dtype=np.int64)

    if has_vol:
        if volume_nodes_deformed is not None:
            vol_pts_def = np.asarray(volume_nodes_deformed, dtype=np.float64)
        else:
            from graphite.explicit.nodal_conformation import deform_outside_nodes
            vol_pts_def, _ = deform_outside_nodes(vol_pts, cad)
    else:
        vol_pts_def = vol_pts

    # Map dual nodes to volume nodes for provenance tracking
    dual_vol_edges = set()
    vol_edges = set((min(int(u), int(v)), max(int(u), int(v))) for u, v in v_struts) if has_vol else set()
    boundary_vol_struts = []
    if has_vol and len(d_struts) > 0:
        query_nodes = dual_nodes_cartesian if dual_nodes_cartesian is not None else dual_pts
        tree_vol = cKDTree(vol_pts)
        _, dual_to_vol = tree_vol.query(query_nodes)

        for da, db in d_struts:
            va, vb = dual_to_vol[da], dual_to_vol[db]
            dual_vol_edges.add((min(va, vb), max(va, vb)))

        # Boundary volume strut promotion: find struts where both endpoints lie on boundary
        _, d_vol, _ = trimesh.proximity.closest_point(cad, vol_pts_def)
        for va, vb in v_struts:
            if d_vol[va] < config.boundary_promote_eps and d_vol[vb] < config.boundary_promote_eps:
                pair = (min(va, vb), max(va, vb))
                if pair not in dual_vol_edges:
                    boundary_vol_struts.append(pair)

    # Collect dual edge pairs while pruning spurious floating struts across empty valleys
    all_dual_edges_pts: list[tuple[np.ndarray, np.ndarray]] = []
    n_pruned_valleys = 0

    for da, db in d_struts:
        p0 = dual_pts[da]
        p1 = dual_pts[db]
        mid = 0.5 * (p0 + p1)
        sdf = safe_signed_distance(cad, np.array([mid]))[0]

        is_vol = False
        if has_vol and len(d_struts) > 0:
            va, vb = dual_to_vol[da], dual_to_vol[db]
            is_vol = (min(va, vb), max(va, vb)) in vol_edges

        if not is_vol and sdf > config.valley_prune_sdf:
            n_pruned_valleys += 1
            continue

        all_dual_edges_pts.append((p0, p1))

    # Promote boundary volume struts to dual edges to seal corner and rim gaps
    for va, vb in boundary_vol_struts:
        all_dual_edges_pts.append((vol_pts_def[va], vol_pts_def[vb]))

    thick_inward = config.inward_depth_factor * config.dual_thickness

    dual_parts = []
    for p0, p1 in all_dual_edges_pts:
        m = generate_planar_sweep_strut_manifold(
            p0,
            p1,
            cad,
            width=config.dual_width,
            thick_inward=thick_inward,
            outer_margin=config.outer_margin,
            n_sweep_pts=config.n_sweep_pts,
            extend_factor=config.extend_factor,
        )
        if m is not None:
            dual_parts.append(m)

    if dual_parts:
        dual_raw, _ = _union_manifolds_for_joint(dual_parts)
    else:
        dual_raw = None

    elapsed = time.time() - t0
    report = {
        "config": asdict(config),
        "n_initial_dual_struts": int(len(d_struts)),
        "n_promoted_boundary_struts": int(len(boundary_vol_struts)),
        "n_pruned_valleys": int(n_pruned_valleys),
        "n_total_sweep_paths": int(len(all_dual_edges_pts)),
        "n_valid_manifolds": int(len(dual_parts)),
        "build_time_s": float(elapsed),
        "status": "ok" if dual_raw is not None else "empty",
    }

    return dual_raw, report
