"""
Graphite Explicit Interlinked — Universal Topological Catenation & Link Checker Engine

Implements universal verification of topological catenation (interlinking)
for all polycatenated architected materials (PAMs) and chainmail fabrics.

Core Methodology:
1. Spanning Solid Generation:
   - Wireframe polyhedra (TET, OCT, CUBE, CO, ICO, TT) -> Solid convex hull or boundary mesh.
   - Wireframe rings (planar circles / tori) -> Spanning thin solid cylindrical disk.
2. Dual-Stage Linking Verification:
   - Stage 1 (Broad Phase): Manifold3D boolean intersection volume Vol(Solid_A ∩ Solid_B) > 0.
   - Stage 2 (Narrow Phase): Boundary face traversal / segment-triangle piercing
     or Gauss linking integral |Lk| >= 0.7 for closed 1D curves.
   - Stage 3 (DfAM Clearance): Physical surface-to-surface gap Delta = d_min - (r_A + r_B) > 0.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence, Any
import numpy as np
import trimesh
import manifold3d as m3d
from scipy.spatial import cKDTree
from scipy.sparse import csgraph, csr_matrix

from graphite.explicit.geometry_module import _rotation_align_local_z_to_unit
from graphite.explicit.interlinked.clearance import segment_segment_distance
from graphite.explicit.interlinked.epitaxy import (
    curve_curve_linking_number,
    extract_triangulated_faces,
)
from graphite.explicit.interlinked.particle import InterlinkedParticle


@dataclass(frozen=True)
class LinkingResult:
    """Diagnostic result of topological linking verification between two particles."""
    is_linked: bool
    solid_intersection_volume: float
    min_surface_clearance: float
    mechanism: str  # "face_piercing", "gauss_linking", "none"
    colliding: bool  # True if min_surface_clearance < 0


@dataclass
class LatticeCatenationReport:
    """Comprehensive catenation and clearance report for an entire interlinked lattice."""
    num_particles: int
    total_links: int
    num_linked_particles: int
    num_isolated_particles: int
    num_components: int
    min_clearance: float
    all_linked: bool
    has_collisions: bool
    adjacency_matrix: np.ndarray
    is_continuous: bool = False
    component_labels: np.ndarray | None = None
    degrees: np.ndarray | None = None
    dangling_particles: list[int] = field(default_factory=list)
    zone_components: dict[str, int] = field(default_factory=dict)
    inter_zone_links: list[tuple[int, int]] = field(default_factory=list)
    summary_dict: dict[str, Any] = field(default_factory=dict)


def segment_triangle_intersection(
    p0: np.ndarray,
    p1: np.ndarray,
    v0: np.ndarray,
    v1: np.ndarray,
    v2: np.ndarray,
    eps: float = 1e-5,
) -> bool:
    """Möller–Trumbore ray-triangle intersection clamped to segment parameter t in [eps, 1-eps]."""
    e1 = v1 - v0
    e2 = v2 - v0
    d = p1 - p0
    h = np.cross(d, e2)
    a = float(np.dot(e1, h))
    if abs(a) < 1e-9:
        return False
    f = 1.0 / a
    s = p0 - v0
    u = f * float(np.dot(s, h))
    if u < eps or u > 1.0 - eps:
        return False
    q = np.cross(s, e1)
    v = f * float(np.dot(d, q))
    if v < eps or u + v > 1.0 - eps:
        return False
    t = f * float(np.dot(e2, q))
    return eps <= t <= 1.0 - eps


def to_spanning_solid(
    particle: InterlinkedParticle,
    disk_thickness: float = 0.40,
    disk_segments: int = 32,
) -> trimesh.Trimesh:
    """
    Convert an InterlinkedParticle into its spanning watertight solid trimesh.

    - Rings: converted to a spanning circular disk (thin cylinder) centered on the ring.
    - Wireframe polyhedra: converted to their convex hull solid.
    """
    cache_key = f"_spanning_solid_{disk_thickness:.3f}_{disk_segments}"
    if cache_key in particle.metadata:
        return particle.metadata[cache_key]

    g = particle.geometry
    g_type = str(g.geometry_type).lower().strip()
    nodes_w = particle.global_nodes()

    # 1. Ring / circular loop
    if g_type == "ring" or "major_radius" in g.metadata or "radius" in particle.metadata:
        R = float(g.metadata.get("major_radius", particle.metadata.get("radius", g.bounding_radius)))
        cyl = trimesh.creation.cylinder(radius=R, height=disk_thickness, sections=disk_segments)
        
        # Align cylinder axis (Z) to the ring's local normal
        normal_local = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        if len(g.nodes) >= 3:
            v1 = g.nodes[1] - g.nodes[0]
            v2 = g.nodes[2] - g.nodes[0]
            cross = np.cross(v1, v2)
            norm_val = np.linalg.norm(cross)
            if norm_val > 1e-8:
                normal_local = cross / norm_val

        # Transform to world space
        R_align = _rotation_align_local_z_to_unit(normal_local)
        T_local = np.eye(4, dtype=np.float64)
        T_local[:3, :3] = R_align
        cyl.apply_transform(T_local)
        cyl.apply_transform(particle.transform)
        particle.metadata[cache_key] = cyl
        return cyl

    # 2. General Polyhedra: convex hull
    try:
        hull = trimesh.convex.convex_hull(nodes_w)
        if not hull.is_watertight:
            hull.fix_normals()
        particle.metadata[cache_key] = hull
        return hull
    except Exception:
        # Fallback bounding box
        box = trimesh.creation.box(bounds=particle.bounds)
        particle.metadata[cache_key] = box
        return box


def _to_manifold(mesh: trimesh.Trimesh) -> m3d.Manifold:
    """Convert trimesh.Trimesh to manifold3d.Manifold."""
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.uint32)
    return m3d.Manifold(m3d.Mesh(vert_properties=verts, tri_verts=faces))


def verify_particle_linking(
    pA: InterlinkedParticle,
    pB: InterlinkedParticle,
    strut_radius: float = 0.35,
    min_clearance_threshold: float = 0.05,
    disk_thickness: float = 0.40,
) -> LinkingResult:
    """
    Verify whether two discrete particles are topologically catenated (interlinked).

    Three-stage evaluation:
    1. Distance broad phase: if d > rA + rB + 2r, cannot intersect.
    2. Solid-solid Boolean intersection volume Vol(Solid_A ∩ Solid_B) > 0.
    3. Topological boundary traversal (piercing of spanning faces or Gauss linking number).
    4. Physical clearance check: Delta = d_min - 2*r.
    """
    # Early distance rejection
    dist_c = float(np.linalg.norm(pA.center - pB.center))
    rA = float(pA.bounding_radius)
    rB = float(pB.bounding_radius)
    if dist_c > rA + rB + 2.0 * strut_radius:
        return LinkingResult(
            is_linked=False,
            solid_intersection_volume=0.0,
            min_surface_clearance=dist_c - (rA + rB + 2.0 * strut_radius),
            mechanism="none",
            colliding=False,
        )

    nA, sA = pA.global_nodes(), pA.struts
    nB, sB = pB.global_nodes(), pB.struts

    # Evaluate physical surface clearance
    min_d = 1e9
    for s0 in sA:
        for s1 in sB:
            d = segment_segment_distance(nA[s0[0]], nA[s0[1]], nB[s1[0]], nB[s1[1]])
            if d < min_d:
                min_d = d
    surface_clearance = min_d - (2.0 * strut_radius)
    is_colliding = surface_clearance < min_clearance_threshold

    # Stage 1: Broad-Phase Solid Intersection
    solidA = to_spanning_solid(pA, disk_thickness=disk_thickness)
    solidB = to_spanning_solid(pB, disk_thickness=disk_thickness)

    try:
        mA = _to_manifold(solidA)
        mB = _to_manifold(solidB)
        inter_vol = float((mA ^ mB).volume())
    except Exception:
        inter_vol = 0.0

    if inter_vol <= 1e-5:
        # Solids do not even overlap in space; cannot be linked
        return LinkingResult(
            is_linked=False,
            solid_intersection_volume=inter_vol,
            min_surface_clearance=surface_clearance,
            mechanism="none",
            colliding=is_colliding,
        )

    # Stage 2: Narrow-Phase Topological Verification
    # Case A: Closed 1D curves (rings or single-loop polygons)
    if len(sA) == len(nA) and len(sB) == len(nB):
        p0A, p1A = nA, np.roll(nA, -1, axis=0)
        p0B, p1B = nB, np.roll(nB, -1, axis=0)
        lk = curve_curve_linking_number(p0A, p1A, p0B, p1B)
        if abs(lk) >= 0.7:
            return LinkingResult(
                is_linked=True,
                solid_intersection_volume=inter_vol,
                min_surface_clearance=surface_clearance,
                mechanism="gauss_linking",
                colliding=is_colliding,
            )

    # Case B: Polyhedral wireframe cages or hybrid ring-cage
    fA_wire = extract_triangulated_faces(nA, sA)
    fB_wire = extract_triangulated_faces(nB, sB)

    vertsA = nA if len(fA_wire) > 0 else solidA.vertices
    facesA = fA_wire if len(fA_wire) > 0 else solidA.faces

    vertsB = nB if len(fB_wire) > 0 else solidB.vertices
    facesB = fB_wire if len(fB_wire) > 0 else solidB.faces

    # Test if struts of B pierce faces of A
    pierces_A = False
    for s in sB:
        p0, p1 = nB[s[0]], nB[s[1]]
        for f in facesA:
            if segment_triangle_intersection(p0, p1, vertsA[f[0]], vertsA[f[1]], vertsA[f[2]]):
                pierces_A = True
                break
        if pierces_A:
            break

    # Test if struts of A pierce faces of B
    pierces_B = False
    for s in sA:
        p0, p1 = nA[s[0]], nA[s[1]]
        for f in facesB:
            if segment_triangle_intersection(p0, p1, vertsB[f[0]], vertsB[f[1]], vertsB[f[2]]):
                pierces_B = True
                break
        if pierces_B:
            break

    # Mutual piercing requirement
    mutual_piercing = pierces_A and pierces_B

    # Cycle-cycle Gauss linking
    max_cycle_lk = 0.0
    if not mutual_piercing:
        from .epitaxy import extract_chordless_cycles
        cycles_A = extract_chordless_cycles(nA, sA)
        cycles_B = extract_chordless_cycles(nB, sB)
        for cA in cycles_A:
            sp0 = nA[cA]
            sp1 = nA[np.roll(cA, -1)]
            for cB in cycles_B:
                sq0 = nB[cB]
                sq1 = nB[np.roll(cB, -1)]
                lk = abs(curve_curve_linking_number(sp0, sp1, sq0, sq1))
                if lk > max_cycle_lk:
                    max_cycle_lk = lk
                if max_cycle_lk >= 0.75:
                    break
            if max_cycle_lk >= 0.75:
                break

    is_linked = mutual_piercing or (max_cycle_lk >= 0.75)
    if mutual_piercing:
        mechanism = "mutual_piercing"
    elif max_cycle_lk >= 0.75:
        mechanism = "cycle_gauss_linking"
    else:
        mechanism = "none"

    return LinkingResult(
        is_linked=is_linked,
        solid_intersection_volume=inter_vol,
        min_surface_clearance=surface_clearance,
        mechanism=mechanism,
        colliding=is_colliding,
    )


def audit_lattice_catenation(
    particles: Sequence[InterlinkedParticle],
    wire_radius: float = 0.35,
    neighbor_search_radius: float | None = None,
    min_clearance_threshold: float = 0.05,
    check_closest_k_csg: int = 10,
) -> LatticeCatenationReport:
    """
    Perform a complete topological and clearance audit of an entire PAM lattice assembly.
    Uses localized sub-section cKDTree pair queries and exact cycle linking.
    """
    N = len(particles)
    if N == 0:
        return LatticeCatenationReport(
            num_particles=0,
            total_links=0,
            num_linked_particles=0,
            num_isolated_particles=0,
            num_components=0,
            min_clearance=float("inf"),
            all_linked=True,
            has_collisions=False,
            adjacency_matrix=np.zeros((0, 0), dtype=int),
            is_continuous=True,
        )

    centers = np.array([p.center for p in particles], dtype=np.float64)
    r_bound_max = max(p.bounding_radius for p in particles)

    if neighbor_search_radius is None:
        search_r = float(2.5 * r_bound_max + 2.0 * wire_radius)
    else:
        search_r = float(neighbor_search_radius)

    tree = cKDTree(centers)
    pairs = sorted(list(tree.query_pairs(r=search_r)))

    adj = np.zeros((N, N), dtype=int)
    min_clr_global = float("inf")
    any_collision = False

    pair_clearances = []

    for i, j in pairs:
        res = verify_particle_linking(
            particles[i],
            particles[j],
            strut_radius=wire_radius,
            min_clearance_threshold=min_clearance_threshold,
        )
        pair_clearances.append((res.min_surface_clearance, i, j, res))
        if res.min_surface_clearance < min_clr_global:
            min_clr_global = res.min_surface_clearance
        if res.colliding:
            any_collision = True
        if res.is_linked:
            adj[i, j] = 1
            adj[j, i] = 1

    # Optional: exact Manifold3D CSG boolean checks on the K closest candidate pairs
    if check_closest_k_csg > 0 and len(pair_clearances) > 0 and m3d is not None:
        from ..geometry_module import build_clean_miter_truss
        pair_clearances.sort(key=lambda x: x[0])
        for clr_val, i, j, _ in pair_clearances[:check_closest_k_csg]:
            if clr_val < min_clearance_threshold * 2.0:
                pA, pB = particles[i], particles[j]
                rA = pA.effective_wire_radius(fallback=wire_radius)
                rB = pB.effective_wire_radius(fallback=wire_radius)
                mA = _to_manifold(build_clean_miter_truss(pA.global_nodes(), pA.struts, rA))
                mB = _to_manifold(build_clean_miter_truss(pB.global_nodes(), pB.struts, rB))
                inter_vol = float((mA ^ mB).volume())
                if inter_vol > 1e-4:
                    any_collision = True

    deg = adj.sum(axis=1)
    num_linked = int(np.count_nonzero(deg > 0))
    num_isolated = N - num_linked
    total_links = int(adj.sum() // 2)

    n_comp, labels = csgraph.connected_components(csr_matrix(adj), directed=False)
    is_continuous = (n_comp == 1) and (num_isolated == 0) and (total_links > 0)
    dangling = [int(i) for i, d in enumerate(deg) if d < 1]

    # Zone analysis
    zone_comps: dict[str, int] = {}
    inter_zone_links: list[tuple[int, int]] = []
    zones_present = set(p.metadata.get("zone") for p in particles if p.metadata.get("zone"))
    if len(zones_present) > 1:
        for z in zones_present:
            z_indices = [idx for idx, p in enumerate(particles) if p.metadata.get("zone") == z]
            if len(z_indices) > 0:
                sub_adj = adj[np.ix_(z_indices, z_indices)]
                n_z, _ = csgraph.connected_components(csr_matrix(sub_adj), directed=False)
                zone_comps[str(z)] = int(n_z)
        for i, j in pairs:
            if adj[i, j] and particles[i].metadata.get("zone") != particles[j].metadata.get("zone"):
                inter_zone_links.append((i, j))

    all_linked = (num_isolated == 0) and (total_links > 0)

    summary = {
        "num_particles": N,
        "total_links": total_links,
        "num_linked_particles": num_linked,
        "num_isolated_particles": num_isolated,
        "num_components": int(n_comp),
        "is_continuous": bool(is_continuous),
        "min_clearance": float(min_clr_global),
        "all_linked": bool(all_linked),
        "has_collisions": bool(any_collision),
        "zone_components": zone_comps,
        "inter_zone_links_count": len(inter_zone_links),
    }

    return LatticeCatenationReport(
        num_particles=N,
        total_links=total_links,
        num_linked_particles=num_linked,
        num_isolated_particles=num_isolated,
        num_components=int(n_comp),
        min_clearance=float(min_clr_global),
        all_linked=bool(all_linked),
        has_collisions=bool(any_collision),
        adjacency_matrix=adj,
        is_continuous=bool(is_continuous),
        component_labels=labels,
        degrees=deg,
        dangling_particles=dangling,
        zone_components=zone_comps,
        inter_zone_links=inter_zone_links,
        summary_dict=summary,
    )


def audit_intra_lattice_unit_cells(
    cell: Any,
    repeats: tuple[int, int, int] = (2, 2, 2),
    pitch: float = 12.0,
    wire_radius: float = 0.35,
    min_clearance: float = 0.10,
) -> LatticeCatenationReport:
    """
    Sub-section pre-flight check: validates that a small (2x2x2) lattice of the same cell
    type forms a single connected component with zero collisions and degree regularity.
    """
    from .generator import InterlinkedConfig, generate_interlinked_lattice
    cfg = InterlinkedConfig(
        cell=cell,
        grid_size=repeats,
        pitch=pitch,
        wire_radius=wire_radius,
        min_clearance=min_clearance,
    )
    res = generate_interlinked_lattice(cfg, check_clearance=False)
    return audit_lattice_catenation(
        res.particles,
        wire_radius=wire_radius,
        min_clearance_threshold=min_clearance,
    )
