"""
graphite/explicit/interlinked/validator.py

Fast, exact 2x2 and 2x2x2 interface collision and topological catenation validator
for unbonded metamaterial heteroepitaxy and PAM transitions.

Tests instantiated 3D solid meshes using manifold3d CSG boolean intersection
and evaluates exact discrete Gauss linking numbers across boundary layers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence
import numpy as np
import trimesh

try:
    import manifold3d
except ImportError:
    manifold3d = None

from .particle import InterlinkedParticle, ParticleGeometry
from .cell import InterlinkedCell, InterlinkedRegistry
from .clearance import (
    segment_segment_distance,
    particle_linking_number,
    gauss_linking_number,
)
from .epitaxy import (
    EpitaxialAlignment,
    get_habit_plane_rotation_matrix,
    apply_strut_taper,
    apply_cage_shrinkage,
    extract_chordless_cycles,
    curve_curve_linking_number,
)
from ..geometry_module import (
    build_clean_miter_truss,
    _trimesh_to_manifold,
    _affine_rows_from_R_t,
)


@dataclass
class InterfaceIntersectionReport:
    """
    Diagnostic report returned by the 2x2 / 2x2x2 interface validator.

    Attributes:
        is_valid: True if zero solid collisions and global min clearance >= min_clearance.
        has_piercings: True if at least one inter-zone loop pair has |Lk| >= 1.
        num_particles_a: Total particles instantiated in Zone A patch.
        num_particles_b: Total particles instantiated in Zone B patch.
        collision_pairs: List of (particle_a_id, particle_b_id) with solid intersection volume > 1e-5 mm^3.
        collision_volumes_mm3: Measured solid intersection volumes in mm^3 for collision pairs.
        min_clearance_mm: Minimum surface-to-surface Euclidean clearance across all candidate pairs.
        violations: Pairs whose clearance is below min_clearance threshold.
        linking_pairs: List of inter-zone pairs exhibiting topological catenation (|Lk| >= 1).
        diagnostics: Additional geometric metrics and timings.
    """
    is_valid: bool
    has_piercings: bool
    num_particles_a: int
    num_particles_b: int
    collision_pairs: list[tuple[int, int]] = field(default_factory=list)
    collision_volumes_mm3: list[float] = field(default_factory=list)
    min_clearance_mm: float = 0.0
    violations: list[dict[str, Any]] = field(default_factory=list)
    linking_pairs: list[dict[str, Any]] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)


def instantiate_interface_patch(
    cell: InterlinkedCell | str,
    patch_size: tuple[int, int] = (2, 2),
    pitch: float = 10.0,
    x_offset: float = 0.0,
    transverse_offset: tuple[float, float] = (0.0, 0.0),
    rotation_matrix: np.ndarray | None = None,
    id_start: int = 0,
    zone_tag: str = "Zone",
    habit_plane: str = "100",
) -> list[InterlinkedParticle]:
    """
    Instantiate a 2D boundary layer (Ny x Nz) of a unit cell across the transverse interface.
    Handles Cartesian, Diamond, and Rotated bases correctly.
    """
    if isinstance(cell, str):
        cell_cls = InterlinkedRegistry.get(cell)
        cell = cell_cls() if callable(cell_cls) else cell_cls

    R_rot = np.eye(3, dtype=np.float64) if rotation_matrix is None else np.asarray(rotation_matrix, dtype=np.float64)
    R_plane = get_habit_plane_rotation_matrix(habit_plane)
    R_total = R_rot @ R_plane

    ny, nz = patch_size
    particles: list[InterlinkedParticle] = []
    pid = id_start
    dy, dz = transverse_offset

    net = getattr(cell, "parent_network", "pcu").lower().strip()

    if net == "dia":
        from .pams import diamond_network_sites
        sites = diamond_network_sites((1, ny, nz), pitch)
        for pos, sub in sites:
            site_parts = cell.instantiate_site(
                grid_index=(0, int(np.floor(pos[1] / pitch)), int(np.floor(pos[2] / pitch))),
                site_origin=np.zeros(3, dtype=np.float64),
                cell_pitch=pitch,
                id_start=pid,
                context={"sublattice": sub, "site_frame": R_total},
            )
            for p in site_parts:
                R_world = R_total @ p.rotation
                t_world = (R_total @ pos) + np.array([x_offset, dy, dz]) + R_total @ p.center
                T = np.eye(4, dtype=np.float64)
                T[:3, :3] = R_world
                T[:3, 3] = t_world
                p.transform = T
                p.particle_id = pid
                p.sublattice_id = sub
                p.metadata["zone"] = zone_tag
                p.metadata["layer_x"] = 0
                particles.append(p)
                pid += 1
    else:
        # Simple Cubic / Cartesian basis
        for iy in range(ny):
            for iz in range(nz):
                origin = np.array([x_offset, iy * pitch + dy, iz * pitch + dz], dtype=np.float64)
                parts = cell.instantiate_site(
                    (0, iy, iz),
                    origin,
                    pitch,
                    id_start=pid,
                    context={"site_frame": R_total},
                )
                for p in parts:
                    if R_total is not None and not np.allclose(R_total, np.eye(3)):
                        R_world = R_total @ p.rotation
                        t_world = (R_total @ (p.center - origin)) + origin
                        T = np.eye(4, dtype=np.float64)
                        T[:3, :3] = R_world
                        T[:3, 3] = t_world
                        p.transform = T
                    p.metadata["zone"] = zone_tag
                    p.metadata["layer_x"] = 0
                    particles.append(p)
                    pid += 1

    return particles


def verify_interface_patch(
    cell_a: InterlinkedCell | str,
    cell_b: InterlinkedCell | str,
    alignment: EpitaxialAlignment | None = None,
    patch_size: tuple[int, int] = (2, 2),
    pitch_a: float = 10.0,
    pitch_b: float | None = None,
    wire_radius: float = 0.35,
    min_clearance: float = 0.100,
    habit_plane: str = "100",
    check_mesh_intersection: bool = True,
) -> InterfaceIntersectionReport:
    """
    Verify physical non-intersection and topological catenation across a 2x2 boundary patch.

    Args:
        cell_a: Zone A host cell or name.
        cell_b: Zone B entering cell or name.
        alignment: Solved EpitaxialAlignment containing phase_shift, rotation_matrix, taper_gamma, shrink_eta.
        patch_size: (Ny, Nz) transverse repeats across the boundary (default 2x2).
        pitch_a: Unit cell pitch of Cell A in mm.
        pitch_b: Unit cell pitch of Cell B in mm (auto-resolved from network if None).
        wire_radius: Strut wire radius in mm.
        min_clearance: Minimum required free-play surface gap in mm.
        habit_plane: Interface plane ('100', '110', '111').
        check_mesh_intersection: If True, evaluates manifold3d solid boolean intersection.

    Returns:
        InterfaceIntersectionReport detailing collisions, min clearance, and piercings.
    """
    if isinstance(cell_a, str):
        cell_a = InterlinkedRegistry.get(cell_a)()
    if isinstance(cell_b, str):
        cell_b = InterlinkedRegistry.get(cell_b)()

    p_a = float(pitch_a)
    net_a = getattr(cell_a, "parent_network", "pcu").lower().strip()
    net_b = getattr(cell_b, "parent_network", "pcu").lower().strip()

    if pitch_b is None:
        if net_a in ("pcu", "bcc") and net_b == "dia":
            p_b = float(np.sqrt(2.0) * p_a)
        elif net_a == "dia" and net_b in ("pcu", "bcc"):
            p_b = float(p_a / np.sqrt(2.0))
        else:
            p_b = p_a
    else:
        p_b = float(pitch_b)

    if alignment is None:
        from .epitaxy import solve_epitaxial_transition
        alignment = solve_epitaxial_transition(
            cell_a=cell_a,
            cell_b=cell_b,
            pitch_a=p_a,
            pitch_b=p_b,
            wire_radius=wire_radius,
            min_clearance=min_clearance,
            habit_plane=habit_plane,
        )

    dx, dy, dz = alignment.phase_shift
    R_rot = alignment.rotation_matrix

    # 1. Instantiate Zone A (at X = 0) and Zone B (at X = dx)
    parts_a = instantiate_interface_patch(
        cell=cell_a,
        patch_size=patch_size,
        pitch=p_a,
        x_offset=0.0,
        transverse_offset=(0.0, 0.0),
        rotation_matrix=np.eye(3),
        id_start=0,
        zone_tag="Zone_A",
        habit_plane=habit_plane,
    )

    parts_b = instantiate_interface_patch(
        cell=cell_b,
        patch_size=patch_size,
        pitch=p_b,
        x_offset=dx,
        transverse_offset=(dy, dz),
        rotation_matrix=R_rot,
        id_start=len(parts_a),
        zone_tag="Zone_B",
        habit_plane=habit_plane,
    )

    # 2. Apply Boundary Tapering and Shrinkage if solved
    all_parts = parts_a + parts_b
    if getattr(alignment, "taper_gamma", 1.0) < 1.0:
        apply_strut_taper(all_parts, gamma=alignment.taper_gamma, boundary_layers=(0,))
    if getattr(alignment, "shrink_eta", 1.0) < 1.0:
        apply_cage_shrinkage(parts_b, eta=alignment.shrink_eta, boundary_layers=(0,))

    # 3. Two-Tier Collision & Clearance Evaluation
    collision_pairs: list[tuple[int, int]] = []
    collision_volumes: list[float] = []
    violations: list[dict[str, Any]] = []
    linking_pairs: list[dict[str, Any]] = []
    global_min_clr = float("inf")

    # Pre-build solid manifolds lazily if manifold3d is available and clr < 0.0
    manifolds_a: dict[int, Any] = {}
    manifolds_b: dict[int, Any] = {}

    for pa in parts_a:
        nodes_a = pa.global_nodes()
        struts_a = pa.struts
        r_a = pa.effective_wire_radius(fallback=wire_radius)
        p0A = nodes_a[struts_a[:, 0]]
        p1A = nodes_a[struts_a[:, 1]]
        cA = pa.center
        r_bound_a = pa.bounding_radius + r_a

        for pb in parts_b:
            nodes_b = pb.global_nodes()
            struts_b = pb.struts
            r_b = pb.effective_wire_radius(fallback=wire_radius)
            cB = pb.center
            r_bound_b = pb.bounding_radius + r_b

            # Broadphase bounding sphere check
            dist_c = float(np.linalg.norm(cA - cB))
            if dist_c > (r_bound_a + r_bound_b + min_clearance):
                continue

            p0B = nodes_b[struts_b[:, 0]]
            p1B = nodes_b[struts_b[:, 1]]

            # Segment-segment surface clearance
            d_centerline = segment_segment_distance(p0A, p1A, p0B, p1B)
            clr = float(d_centerline - (r_a + r_b))
            if clr < global_min_clr:
                global_min_clr = clr

            if clr < float(min_clearance):
                violations.append({
                    "particle_a": pa.particle_id,
                    "particle_b": pb.particle_id,
                    "clearance_mm": clr,
                })

            # Exact Solid Boolean Intersection Check (lazy evaluation only if surfaces penetrate)
            if check_mesh_intersection and manifold3d is not None and clr < 0.0:
                if pa.particle_id not in manifolds_a:
                    t_mesh = build_clean_miter_truss(pa.global_nodes(), pa.geometry.struts, r_a)
                    manifolds_a[pa.particle_id] = _trimesh_to_manifold(t_mesh)
                if pb.particle_id not in manifolds_b:
                    t_mesh = build_clean_miter_truss(pb.global_nodes(), pb.geometry.struts, r_b)
                    manifolds_b[pb.particle_id] = _trimesh_to_manifold(t_mesh)
                ma = manifolds_a.get(pa.particle_id)
                mb = manifolds_b.get(pb.particle_id)
                if ma is not None and mb is not None:
                    inter = ma ^ mb
                    vol = float(inter.volume())
                    if vol > 1e-5:
                        collision_pairs.append((pa.particle_id, pb.particle_id))
                        collision_volumes.append(vol)

            # Topological Linking & Piercing Check via Universal Catenation Engine
            from .catenation import verify_particle_linking
            link_res = verify_particle_linking(
                pa,
                pb,
                strut_radius=wire_radius,
                min_clearance_threshold=float(min_clearance),
            )
            if link_res.is_linked:
                linking_pairs.append({
                    "particle_a": pa.particle_id,
                    "particle_b": pb.particle_id,
                    "mechanism": link_res.mechanism,
                    "min_clearance": link_res.min_surface_clearance,
                })

    if global_min_clr == float("inf"):
        min_d = float("inf")
        for pa in parts_a:
            nodes_a = pa.global_nodes()
            p0A = nodes_a[pa.struts[:, 0]]
            p1A = nodes_a[pa.struts[:, 1]]
            r_a = pa.effective_wire_radius(fallback=wire_radius)
            for pb in parts_b:
                nodes_b = pb.global_nodes()
                p0B = nodes_b[pb.struts[:, 0]]
                p1B = nodes_b[pb.struts[:, 1]]
                r_b = pb.effective_wire_radius(fallback=wire_radius)
                d = float(segment_segment_distance(p0A, p1A, p0B, p1B) - (r_a + r_b))
                if d < min_d:
                    min_d = d
        global_min_clr = float(min_d if min_d != float("inf") else 0.0)

    is_valid = bool(len(collision_pairs) == 0 and global_min_clr >= float(min_clearance))
    has_piercings = bool(len(linking_pairs) > 0)

    return InterfaceIntersectionReport(
        is_valid=is_valid,
        has_piercings=has_piercings,
        num_particles_a=len(parts_a),
        num_particles_b=len(parts_b),
        collision_pairs=collision_pairs,
        collision_volumes_mm3=collision_volumes,
        min_clearance_mm=global_min_clr,
        violations=violations,
        linking_pairs=linking_pairs,
        diagnostics={
            "patch_size": patch_size,
            "pitch_a": p_a,
            "pitch_b": p_b,
            "phase_shift": list(alignment.phase_shift),
            "habit_plane": habit_plane,
        },
    )


# =============================================================================
# Two-Step Physical & Topological Verification Architecture
# Step 1: Literal Mesh Intersecting Faces & Non-Manifold Edges Check
# Step 2: Solid Polyhedral Envelope Penetration & Dangling Links Check
# =============================================================================

@dataclass
class FaceIntersectionReport:
    """Report for Step 1: Physical Collisions & Surface Mesh Quality."""
    passed: bool
    has_intersecting_faces: bool
    num_intersecting_pairs: int
    intersecting_pairs: list[tuple[int, int, float]]
    non_manifold_edges: int
    is_watertight: bool
    num_bodies: int
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class EnvelopePenetrationReport:
    """Report for Step 2: Envelope Penetration, Catenation & Dangling Links."""
    passed: bool
    all_interlinked: bool
    num_particles: int
    num_interlinked: int
    num_hanging: int
    hanging_particle_ids: list[int]
    degrees: np.ndarray
    min_degree: int
    max_degree: int
    avg_degree: float
    total_interlinks: int
    interlink_pairs: list[tuple[int, int, float]]
    num_linked: int = 0
    num_entangled: int = 0
    linked_pairs: list[tuple[int, int, int]] = field(default_factory=list)
    entangled_pairs: list[tuple[int, int, float]] = field(default_factory=list)
    inter_zone_links: int = 0
    inter_zone_catenation_passed: bool = True
    inter_zone_linked_pairs: list[tuple[int, int, int]] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class TwoStepQualityReport:
    """Consolidated report for the 2-step verification protocol."""
    passed: bool
    step1_report: FaceIntersectionReport
    step2_report: EnvelopePenetrationReport | None
    failure_reason: str | None = None


def count_polyhedron_face_punctures(
    struts: np.ndarray,
    strut_nodes: np.ndarray,
    hull_faces: np.ndarray,
    hull_nodes: np.ndarray,
) -> set[int]:
    """
    Find distinct face indices of a polyhedral hull that are punctured by the given struts.
    Uses exact ray-triangle plane intersection and in-triangle point tests.
    """
    hit_faces: set[int] = set()
    for s in struts:
        p0 = strut_nodes[s[0]]
        p1 = strut_nodes[s[1]]
        for f_idx, f in enumerate(hull_faces):
            tri = hull_nodes[f]
            v0, v1, v2 = tri[0], tri[1], tri[2]
            norm = np.cross(v1 - v0, v2 - v0)
            nl = np.linalg.norm(norm)
            if nl < 1e-9:
                continue
            norm = norm / nl
            d0 = float(np.dot(p0 - v0, norm))
            d1 = float(np.dot(p1 - v0, norm))
            if d0 * d1 < -1e-8:
                t = d0 / (d0 - d1)
                pt = p0 + t * (p1 - p0)
                c0 = np.cross(v1 - v0, pt - v0)
                c1 = np.cross(v2 - v1, pt - v1)
                c2 = np.cross(v0 - v2, pt - v2)
                if (
                    np.dot(c0, norm) >= -1e-7
                    and np.dot(c1, norm) >= -1e-7
                    and np.dot(c2, norm) >= -1e-7
                ):
                    hit_faces.add(f_idx)
    return hit_faces


def evaluate_parent_polyhedron_face_catenation(
    pa: InterlinkedParticle,
    pb: InterlinkedParticle,
) -> tuple[int, int, int]:
    """
    Evaluate topological catenation between two particles using the Solid Parent Polyhedron
    Multi-Face Penetration Check.

    Solidifies both particles into their parent polyhedral shapes (convex hulls).
    Counts how many distinct faces of Parent B are punctured by struts of A (n_b),
    and how many distinct faces of Parent A are punctured by struts of B (n_a).

    Returns:
        tuple[int, int, int]: (linking_number, n_a_hit, n_b_hit)
        - If n_a >= 2 and n_b >= 2: True Link (threads through >= 2 faces of each cage). Returns (1, n_a, n_b).
        - If n_a == 1 or n_b == 1: Pierce / Poke (tip enters one face without exiting). Returns (0, n_a, n_b).
        - If n_a == 0 and n_b == 0: Disjoint. Returns (0, 0, 0).
    """
    nodes_a = pa.global_nodes()
    struts_a = pa.geometry.struts
    nodes_b = pb.global_nodes()
    struts_b = pb.geometry.struts

    hull_a = trimesh.convex.convex_hull(nodes_a)
    hull_b = trimesh.convex.convex_hull(nodes_b)

    faces_b_hit = count_polyhedron_face_punctures(struts_a, nodes_a, hull_b.faces, nodes_b)
    faces_a_hit = count_polyhedron_face_punctures(struts_b, nodes_b, hull_a.faces, nodes_a)

    n_a = len(faces_a_hit)
    n_b = len(faces_b_hit)

    if n_a >= 2 and n_b >= 2:
        return 1, n_a, n_b
    return 0, n_a, n_b


def face_cycle_linking(
    pa: InterlinkedParticle,
    pb: InterlinkedParticle,
    wire_radius: float = 0.35,
) -> int:
    """
    Evaluate exact topological catenation between two particles.

    Primary Method: Solid Parent Polyhedron Multi-Face Penetration Check.
    - If n_a >= 2 and n_b >= 2: The struts pass all the way through the host cages,
      guaranteeing true mechanical interlocking (|Lk| = 1).
    - If n_a <= 1 or n_b <= 1: The cages are only piercing an open aperture without
      threading through, which is strictly rejected as unlinked (|Lk| = 0).

    For 1D ring/torus geometries, evaluates exact discrete Stokes polygon-strut signed puncture crossing.

    Returns:
        int: Integer linking number (>= 1 if catenated, 0 if unlinked/pierced only).
    """
    nodes_a = pa.global_nodes()
    nodes_b = pb.global_nodes()

    # Broadphase AABB pre-filter
    min_a, max_a = nodes_a.min(axis=0), nodes_a.max(axis=0)
    min_b, max_b = nodes_b.min(axis=0), nodes_b.max(axis=0)
    r_margin = float(wire_radius) * 2.0
    if not (np.all(min_a <= max_b + r_margin) and np.all(min_b <= max_a + r_margin)):
        return 0

    g_type_a = getattr(pa.geometry, "geometry_type", "").lower()
    g_type_b = getattr(pb.geometry, "geometry_type", "").lower()
    is_ring_a = (g_type_a in ("ring", "torus") or "major_radius" in pa.geometry.metadata)
    is_ring_b = (g_type_b in ("ring", "torus") or "major_radius" in pb.geometry.metadata)

    if not is_ring_a and not is_ring_b:
        # Standard polyhedral wireframe cages (D4-CUBE, D4-TET, C-6-TT, B-8-CO, etc.)
        lk, _, _ = evaluate_parent_polyhedron_face_catenation(pa, pb)
        return lk

    # For ring/torus particles, evaluate exact discrete Stokes polygon puncture
    cycles_a = [list(range(len(nodes_a)))] if is_ring_a else extract_chordless_cycles(nodes_a, pa.geometry.struts)
    cycles_b = [list(range(len(nodes_b)))] if is_ring_b else extract_chordless_cycles(nodes_b, pb.geometry.struts)

    if not cycles_a or not cycles_b:
        return 0

    max_lk = 0
    for ca in cycles_a:
        pts_a = nodes_a[ca]
        sp0 = pts_a
        sp1 = nodes_a[np.roll(ca, -1)]

        for cb in cycles_b:
            pts_b = nodes_b[cb]
            sq0 = pts_b
            sq1 = nodes_b[np.roll(cb, -1)]

            lk = int(abs(round(curve_curve_linking_number(sp0, sp1, sq0, sq1))))
            if lk > max_lk:
                max_lk = lk
            if max_lk >= 1:
                return max_lk

    return max_lk


def check_mesh_face_intersections(
    mesh_or_path: str | trimesh.Trimesh,
) -> FaceIntersectionReport:
    """
    Step 1: Literal Physical Collision Check on Exported Surface Mesh.
    Checks:
    1. Zero non-manifold edges across the assembly.
    2. Watertightness of the multi-body solid.
    3. Zero intersecting faces between distinct bodies in 3D space.
    """
    if isinstance(mesh_or_path, (str, bytes)) or hasattr(mesh_or_path, "__fspath__"):
        mesh = trimesh.load(str(mesh_or_path))
    else:
        mesh = mesh_or_path

    if isinstance(mesh, trimesh.Scene):
        bodies = []
        for name, data in mesh.graph.to_flattened().items():
            geom_name = data.get("geometry")
            if geom_name and geom_name in mesh.geometry:
                m_inst = mesh.geometry[geom_name].copy()
                m_inst.apply_transform(data["transform"])
                bodies.append(m_inst)
        non_manifold = sum(
            int(np.sum(np.unique(b.edges_sorted, axis=0, return_counts=True)[1] != 2))
            for b in bodies
        )
        is_watertight = all(b.is_watertight for b in bodies) if bodies else False
    else:
        edges = mesh.edges_sorted
        unique, counts = np.unique(edges, axis=0, return_counts=True)
        non_manifold = int(np.sum(counts != 2))
        # Filter out zero-volume / degenerate sliver fragments if multibody concatenation had slivers
        split_candidates = mesh.split(only_watertight=True)
        if len(split_candidates) > 1:
            bodies = split_candidates
        else:
            bodies = [b for b in mesh.split(only_watertight=False) if len(b.faces) >= 4 and b.volume > 1e-4]
        is_watertight = bool(mesh.is_watertight)

    num_bodies = len(bodies)

    colliding_pairs: list[tuple[int, int, float]] = []
    if num_bodies > 1 and manifold3d is not None:
        bounds = [b.bounds for b in bodies]
        manifolds = [_trimesh_to_manifold(b) for b in bodies]
        for i in range(num_bodies):
            for j in range(i + 1, num_bodies):
                # AABB pre-filter
                if np.all(bounds[i][0] <= bounds[j][1]) and np.all(bounds[j][0] <= bounds[i][1]):
                    inter = manifolds[i] ^ manifolds[j]
                    vol = float(inter.volume())
                    if vol > 1e-4:
                        colliding_pairs.append((i, j, vol))

    has_intersecting_faces = (len(colliding_pairs) > 0)
    passed = bool(not has_intersecting_faces and non_manifold == 0 and is_watertight)

    return FaceIntersectionReport(
        passed=passed,
        has_intersecting_faces=has_intersecting_faces,
        num_intersecting_pairs=len(colliding_pairs),
        intersecting_pairs=colliding_pairs,
        non_manifold_edges=non_manifold,
        is_watertight=bool(is_watertight),
        num_bodies=num_bodies,
        diagnostics={
            "num_faces": sum(len(b.faces) for b in bodies),
            "num_vertices": sum(len(b.vertices) for b in bodies),
        },
    )


def check_envelope_penetration_and_catenation(
    particles: Sequence[InterlinkedParticle],
    wire_radius: float = 0.35,
) -> EnvelopePenetrationReport:
    """
    Step 2: Topological Penetration Check via Solid Polyhedral Envelopes & Exact Linking Numbers.
    Converts each closed particle into its canonical solid interior envelope
    (tetrahedron for D4-TET, cube for D4-CUBE, planar cylinder/disk for rings).
    Tests whether envelopes intersect in 3D space:
    - If envelopes intersect AND face cycles have |Lk| >= 1: mechanically interlinked.
    - If envelopes intersect BUT face cycles have |Lk| == 0: entangled (poked tip or colliding).
    - If a particle's envelope intersects 0 other envelopes: hanging/dangling.
    """
    if manifold3d is None:
        raise RuntimeError("manifold3d is required for solid envelope penetration testing.")

    N = len(particles)
    if N == 0:
        return EnvelopePenetrationReport(
            passed=True,
            all_interlinked=True,
            num_particles=0,
            num_interlinked=0,
            num_hanging=0,
            hanging_particle_ids=[],
            degrees=np.zeros(0, dtype=int),
            min_degree=0,
            max_degree=0,
            avg_degree=0.0,
            total_interlinks=0,
            interlink_pairs=[],
            num_linked=0,
            num_entangled=0,
            linked_pairs=[],
            entangled_pairs=[],
        )

    from scipy.spatial import cKDTree

    # Construct canonical solid envelope manifolds
    mani_envelopes: list[manifold3d.Manifold] = []
    env_mins: list[np.ndarray] = []
    env_maxs: list[np.ndarray] = []
    centers: list[np.ndarray] = []
    max_radii: list[float] = []

    for p in particles:
        nodes = p.global_nodes()
        centers.append(p.center)
        b_rad = float(p.bounding_radius or 1.0)
        max_radii.append(b_rad)
        g_type = getattr(p.geometry, "geometry_type", "").lower()
        if g_type in ("ring", "torus") or "major_radius" in p.geometry.metadata:
            R_major = float(p.geometry.metadata.get("major_radius", p.geometry.bounding_radius))
            # Planar disk/low-height cylinder
            cyl = manifold3d.Manifold.cylinder(
                height=1.0,
                radius_low=R_major,
                radius_high=R_major,
                circular_segments=32,
                center=True,
            )
            mani_envelopes.append(cyl.transform(_affine_rows_from_R_t(p.rotation, p.center)))
            env_mins.append(p.center - R_major)
            env_maxs.append(p.center + R_major)
        else:
            # Polyhedral cage envelope (convex hull of centerline nodes)
            hull = trimesh.convex.convex_hull(nodes)
            m_hull = _trimesh_to_manifold(hull)
            mani_envelopes.append(m_hull)
            env_mins.append(nodes.min(axis=0))
            env_maxs.append(nodes.max(axis=0))

    # KD-Tree Broadphase
    centers_arr = np.array(centers, dtype=np.float64)
    tree = cKDTree(centers_arr)
    max_search_r = 2.5 * max(max_radii)
    candidate_pairs = tree.query_pairs(r=max_search_r)

    linked_per_particle: dict[int, list[int]] = {i: [] for i in range(N)}
    entangled_per_particle: dict[int, list[int]] = {i: [] for i in range(N)}
    linked_pairs: list[tuple[int, int, int]] = []
    entangled_pairs: list[tuple[int, int, float]] = []
    interlink_pairs_legacy: list[tuple[int, int, float]] = []

    for i, j in candidate_pairs:
        # AABB overlap pre-check
        if np.all(env_mins[i] <= env_maxs[j]) and np.all(env_mins[j] <= env_maxs[i]):
            # Check envelope intersection volume
            inter = mani_envelopes[i] ^ mani_envelopes[j]
            vol = float(inter.volume())
            if vol > 1e-4:
                interlink_pairs_legacy.append((i, j, vol))
                # Evaluate exact face cycle linking
                lk = face_cycle_linking(particles[i], particles[j], wire_radius=wire_radius)
                if lk >= 1:
                    linked_per_particle[i].append(j)
                    linked_per_particle[j].append(i)
                    linked_pairs.append((i, j, lk))
                else:
                    entangled_per_particle[i].append(j)
                    entangled_per_particle[j].append(i)
                    entangled_pairs.append((i, j, vol))

    degrees_linked = np.array([len(linked_per_particle[i]) for i in range(N)], dtype=int)
    degrees_total = np.array([len(linked_per_particle[i]) + len(entangled_per_particle[i]) for i in range(N)], dtype=int)

    # Inter-zone boundary link verification
    zone_indices = [p.metadata.get("zone_index", 0) for p in particles]
    unique_zones = set(zone_indices)
    inter_zone_linked_pairs = [(i, j, lk) for (i, j, lk) in linked_pairs if zone_indices[i] != zone_indices[j]]
    inter_zone_links = len(inter_zone_linked_pairs)
    inter_zone_passed = (inter_zone_links > 0) if len(unique_zones) > 1 else True

    hanging_ids = [int(i) for i in range(N) if degrees_total[i] == 0]
    num_hanging = len(hanging_ids)
    num_linked = int(np.count_nonzero(degrees_linked > 0))
    num_entangled = int(np.count_nonzero((degrees_linked == 0) & (degrees_total > 0)))
    num_interlinked = num_linked

    # Passed if all particles are linked with zero hanging, zero entangled, and inter-zone catenated if multi-zone
    all_interlinked = (num_hanging == 0 and num_entangled == 0 and N > 0 and inter_zone_passed)
    passed = (num_hanging == 0 and N > 0 and inter_zone_passed)

    return EnvelopePenetrationReport(
        passed=passed,
        all_interlinked=all_interlinked,
        num_particles=N,
        num_interlinked=num_interlinked,
        num_hanging=num_hanging,
        hanging_particle_ids=hanging_ids,
        degrees=degrees_linked,
        min_degree=int(np.min(degrees_linked)) if N > 0 else 0,
        max_degree=int(np.max(degrees_linked)) if N > 0 else 0,
        avg_degree=float(np.mean(degrees_linked)) if N > 0 else 0.0,
        total_interlinks=len(linked_pairs) if len(linked_pairs) > 0 else len(interlink_pairs_legacy),
        interlink_pairs=interlink_pairs_legacy,
        num_linked=num_linked,
        num_entangled=num_entangled,
        linked_pairs=linked_pairs,
        entangled_pairs=entangled_pairs,
        inter_zone_links=inter_zone_links,
        inter_zone_catenation_passed=inter_zone_passed,
        inter_zone_linked_pairs=inter_zone_linked_pairs,
    )


@dataclass
class InterZoneBoundaryReport:
    """
    Dedicated diagnostic report for boundary interfaces between adjacent metamaterial zones.

    Attributes:
        is_valid: True if at least one inter-zone pair is catenated (|Lk| >= 1),
            zero solid collisions across zones, and min clearance >= min_clearance.
        num_zones: Total number of discrete zones detected.
        boundary_linked_count: Total cross-boundary pairs with |Lk| >= 1.
        boundary_min_clearance_mm: Minimum surface-to-surface Euclidean clearance between zones.
        boundary_linked_pairs: List of (id_i, id_j, lk) for cross-boundary links.
        boundary_colliding_pairs: List of (id_i, id_j, vol) for cross-boundary collisions.
        boundary_entangled_pairs: List of (id_i, id_j, vol) for cross-boundary entanglements.
        boundary_violations: Cross-boundary pairs below min_clearance threshold.
        diagnostics: Additional geometric metrics.
    """
    is_valid: bool
    num_zones: int
    boundary_linked_count: int
    boundary_min_clearance_mm: float
    boundary_linked_pairs: list[tuple[int, int, int]] = field(default_factory=list)
    boundary_colliding_pairs: list[tuple[int, int, float]] = field(default_factory=list)
    boundary_entangled_pairs: list[tuple[int, int, float]] = field(default_factory=list)
    boundary_violations: list[dict[str, Any]] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)


def evaluate_inter_zone_catenation(
    particles: Sequence[InterlinkedParticle],
    wire_radius: float = 0.35,
    min_clearance: float = 0.15,
) -> InterZoneBoundaryReport:
    """
    Evaluate topological catenation and physical clearance strictly across zone boundaries.
    Isolates pairs (p_i, p_j) where p_i and p_j belong to different zones.
    Guarantees that multi-zone assemblies cannot pass if components are disconnected islands.
    """
    N = len(particles)
    if N == 0:
        return InterZoneBoundaryReport(
            is_valid=True,
            num_zones=0,
            boundary_linked_count=0,
            boundary_min_clearance_mm=float("inf"),
        )

    zone_indices = [int(p.metadata.get("zone_index", 0)) for p in particles]
    unique_zones = sorted(list(set(zone_indices)))
    num_zones = len(unique_zones)

    if num_zones <= 1:
        return InterZoneBoundaryReport(
            is_valid=True,
            num_zones=num_zones,
            boundary_linked_count=0,
            boundary_min_clearance_mm=float("inf"),
            diagnostics={"note": "Single zone assembly, no inter-zone boundaries."},
        )

    b_linked_pairs: list[tuple[int, int, int]] = []
    b_colliding_pairs: list[tuple[int, int, float]] = []
    b_entangled_pairs: list[tuple[int, int, float]] = []
    b_violations: list[dict[str, Any]] = []
    min_clr_found = float("inf")

    # Evaluate all candidate cross-zone pairs
    for i in range(N):
        zi = zone_indices[i]
        pa = particles[i]
        nodes_a = pa.global_nodes()
        struts_a = pa.struts
        r_a = pa.effective_wire_radius(fallback=wire_radius)
        p0A = nodes_a[struts_a[:, 0]]
        p1A = nodes_a[struts_a[:, 1]]
        box_a_min, box_a_max = nodes_a.min(axis=0), nodes_a.max(axis=0)

        for j in range(i + 1, N):
            zj = zone_indices[j]
            if zi == zj:
                continue

            pb = particles[j]
            nodes_b = pb.global_nodes()
            struts_b = pb.struts
            r_b = pb.effective_wire_radius(fallback=wire_radius)
            box_b_min, box_b_max = nodes_b.min(axis=0), nodes_b.max(axis=0)

            # Broad-phase distance filter
            search_margin = 2.0 * max(r_a, r_b) + 5.0
            if not (np.all(box_a_min <= box_b_max + search_margin) and np.all(box_b_min <= box_a_max + search_margin)):
                continue

            # Exact segment-segment clearance
            p0B = nodes_b[struts_b[:, 0]]
            p1B = nodes_b[struts_b[:, 1]]
            d_center = segment_segment_distance(p0A, p1A, p0B, p1B)
            clr = float(d_center - (r_a + r_b))

            if clr < min_clr_found:
                min_clr_found = clr

            if clr < float(min_clearance):
                b_violations.append({
                    "particle_i": i,
                    "particle_j": j,
                    "zone_i": zi,
                    "zone_j": zj,
                    "clearance_mm": clr,
                })
                if clr < 0.0:
                    b_colliding_pairs.append((i, j, abs(clr)))

            # Face cycle catenation
            lk = face_cycle_linking(pa, pb, wire_radius=wire_radius)
            if lk >= 1:
                b_linked_pairs.append((i, j, lk))
            elif clr < 0.0:
                b_entangled_pairs.append((i, j, abs(clr)))

    boundary_min_clr = float(min_clr_found if min_clr_found != float("inf") else 0.0)
    is_valid = bool(len(b_linked_pairs) > 0 and len(b_colliding_pairs) == 0 and boundary_min_clr >= float(min_clearance))

    return InterZoneBoundaryReport(
        is_valid=is_valid,
        num_zones=num_zones,
        boundary_linked_count=len(b_linked_pairs),
        boundary_min_clearance_mm=boundary_min_clr,
        boundary_linked_pairs=b_linked_pairs,
        boundary_colliding_pairs=b_colliding_pairs,
        boundary_entangled_pairs=b_entangled_pairs,
        boundary_violations=b_violations,
        diagnostics={
            "num_particles": N,
            "unique_zones": unique_zones,
            "min_clearance_threshold": float(min_clearance),
        },
    )


def verify_two_step_interlinked_quality(
    mesh_or_path: str | trimesh.Trimesh,
    particles: Sequence[InterlinkedParticle],
) -> TwoStepQualityReport:
    """
    Unified 2-Step Quality Gate Execution:
    Step 1: Literal physical collision and manifold integrity check.
            If Step 1 fails, halts immediately with exact collision diagnostics.
    Step 2: Solid envelope penetration, dangling links, and inter-zone catenation check.
            If Step 2 fails, flags unlinked / dangling links or disconnected zones.
    """
    step1 = check_mesh_face_intersections(mesh_or_path)
    if not step1.passed:
        reason = (
            f"Step 1 Physical Collision Failure: {step1.num_intersecting_pairs} colliding body pairs, "
            f"{step1.non_manifold_edges} non-manifold edges (watertight={step1.is_watertight})."
        )
        return TwoStepQualityReport(
            passed=False,
            step1_report=step1,
            step2_report=None,
            failure_reason=reason,
        )

    step2 = check_envelope_penetration_and_catenation(particles)
    if not step2.passed:
        if not step2.inter_zone_catenation_passed:
            reason = (
                f"Step 2 Multi-Zone Catenation Failure: 0 catenated links crossing inter-zone boundaries."
            )
        else:
            reason = (
                f"Step 2 Hanging Links Failure: {step2.num_hanging} particles are unlinked/floating "
                f"(IDs: {step2.hanging_particle_ids})."
            )
        return TwoStepQualityReport(
            passed=False,
            step1_report=step1,
            step2_report=step2,
            failure_reason=reason,
        )

    return TwoStepQualityReport(
        passed=True,
        step1_report=step1,
        step2_report=step2,
        failure_reason=None,
    )


@dataclass
class InsufficientLinkingReport:
    """
    Diagnostic report evaluating intra-zone vs inter-zone connectivity of particles.
    Identifies orphaned boundary particles: particles that link across a boundary
    to an adjacent zone (inter_degree >= 1), but have zero connections into their
    own zone's bulk network (intra_degree == 0).

    Attributes:
        is_valid: True if zero orphaned boundary particles exist across all zones.
        num_orphaned_boundary_particles: Number of boundary particles with inter >= 1 and intra == 0.
        orphaned_particle_ids: List of global particle IDs flagged as orphaned boundary links.
        orphaned_details: Detailed breakdown of each orphaned particle (zone, coordinates, inter links).
        intra_degrees: Mapping from particle ID to intra-zone link count.
        inter_degrees: Mapping from particle ID to inter-zone link count.
        diagnostics: Additional connectivity metrics per zone.
    """
    is_valid: bool
    num_orphaned_boundary_particles: int
    orphaned_particle_ids: list[int] = field(default_factory=list)
    orphaned_details: list[dict[str, Any]] = field(default_factory=list)
    intra_degrees: dict[int, int] = field(default_factory=dict)
    inter_degrees: dict[int, int] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)


def evaluate_insufficient_linking(
    particles: Sequence[InterlinkedParticle],
    wire_radius: float = 0.35,
    linked_pairs: list[tuple[int, int, int]] | None = None,
) -> InsufficientLinkingReport:
    """
    Evaluate structural anchoring of boundary particles across multi-zone metamaterial assemblies.

    Checks the dual-degree anchoring condition:
    For every boundary particle exhibiting inter-zone catenation (inter_degree >= 1),
    verifies whether it is structurally anchored into its own zone's bulk network (intra_degree >= 1).
    Particles with inter_degree >= 1 but intra_degree == 0 are flagged as orphaned boundary particles.
    """
    N = len(particles)
    if N == 0:
        return InsufficientLinkingReport(
            is_valid=True,
            num_orphaned_boundary_particles=0,
            orphaned_particle_ids=[],
            orphaned_details=[],
            intra_degrees={},
            inter_degrees={},
            diagnostics={"num_particles": 0},
        )

    if linked_pairs is None:
        cat_rep = check_envelope_penetration_and_catenation(particles, wire_radius=wire_radius)
        linked_pairs = getattr(cat_rep, "linked_pairs", [])

    zone_indices = [int(p.metadata.get("zone_index", 0)) for p in particles]
    zone_counts: dict[int, int] = {}
    for z in zone_indices:
        zone_counts[z] = zone_counts.get(z, 0) + 1

    intra_deg = {i: 0 for i in range(N)}
    inter_deg = {i: 0 for i in range(N)}
    inter_link_targets: dict[int, list[int]] = {i: [] for i in range(N)}

    for pair in linked_pairs:
        i, j = int(pair[0]), int(pair[1])
        if 0 <= i < N and 0 <= j < N:
            if zone_indices[i] == zone_indices[j]:
                intra_deg[i] += 1
                intra_deg[j] += 1
            else:
                inter_deg[i] += 1
                inter_deg[j] += 1
                inter_link_targets[i].append(j)
                inter_link_targets[j].append(i)

    orphaned_ids: list[int] = []
    orphaned_details: list[dict[str, Any]] = []

    for i in range(N):
        zi = zone_indices[i]
        # Only evaluate anchoring condition if the home zone has > 1 particle
        if zone_counts.get(zi, 0) > 1:
            if inter_deg[i] >= 1 and intra_deg[i] == 0:
                orphaned_ids.append(i)
                p = particles[i]
                orphaned_details.append({
                    "particle_id": i,
                    "zone_index": zi,
                    "zone_name": p.metadata.get("zone_name", f"Zone_{zi}"),
                    "cell_name": p.metadata.get("cell_name", ""),
                    "center": p.center.tolist(),
                    "inter_degree": inter_deg[i],
                    "intra_degree": intra_deg[i],
                    "linked_targets": inter_link_targets[i],
                })

    is_valid = (len(orphaned_ids) == 0)

    return InsufficientLinkingReport(
        is_valid=is_valid,
        num_orphaned_boundary_particles=len(orphaned_ids),
        orphaned_particle_ids=orphaned_ids,
        orphaned_details=orphaned_details,
        intra_degrees=intra_deg,
        inter_degrees=inter_deg,
        diagnostics={
            "num_particles": N,
            "zone_counts": zone_counts,
            "total_linked_pairs": len(linked_pairs),
            "wire_radius": float(wire_radius),
        },
    )


@dataclass
class ConnectivityReport:
    """
    Diagnostic report evaluating full-assembly catenation graph connected components.

    Attributes:
        num_components: Total number of connected components in the catenation graph.
        component_sizes: List of sizes of each component, sorted descending.
        largest_fraction: Fraction of all particles belonging to the largest component (0.0 to 1.0).
        is_single_component: True if num_components <= 1 (the entire assembly is topologically united).
        component_of: Mapping from particle index to component ID (0-indexed, where 0 is the largest component).
        diagnostics: Additional connectivity metrics.
    """
    num_components: int
    component_sizes: list[int] = field(default_factory=list)
    largest_fraction: float = 0.0
    is_single_component: bool = True
    component_of: dict[int, int] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)


def evaluate_assembly_connectivity(
    particles: Sequence[InterlinkedParticle],
    wire_radius: float = 0.35,
    linked_pairs: list[tuple[int, int, int]] | None = None,
) -> ConnectivityReport:
    """
    Evaluate connected components of the whole interlinked assembly catenation graph.
    """
    N = len(particles)
    if N == 0:
        return ConnectivityReport(
            num_components=0,
            component_sizes=[],
            largest_fraction=1.0,
            is_single_component=True,
            component_of={},
            diagnostics={"num_particles": 0},
        )
    if N == 1:
        return ConnectivityReport(
            num_components=1,
            component_sizes=[1],
            largest_fraction=1.0,
            is_single_component=True,
            component_of={0: 0},
            diagnostics={"num_particles": 1},
        )

    if linked_pairs is None:
        cat_rep = check_envelope_penetration_and_catenation(particles, wire_radius=wire_radius)
        linked_pairs = getattr(cat_rep, "linked_pairs", [])

    from scipy.sparse import csgraph, lil_matrix

    mat = lil_matrix((N, N), dtype=bool)
    for pair in linked_pairs:
        i, j = int(pair[0]), int(pair[1])
        if 0 <= i < N and 0 <= j < N:
            mat[i, j] = True
            mat[j, i] = True

    n_comps, raw_labels = csgraph.connected_components(mat, directed=False)
    raw_counts = np.bincount(raw_labels, minlength=n_comps)
    # Order components by descending size
    order = np.argsort(-raw_counts)
    comp_map = {int(old_lbl): int(new_lbl) for new_lbl, old_lbl in enumerate(order)}
    component_sizes = [int(raw_counts[old_lbl]) for old_lbl in order]
    largest_fraction = float(component_sizes[0]) / float(N) if N > 0 else 1.0
    component_of = {i: comp_map[int(raw_labels[i])] for i in range(N)}
    is_single = (n_comps <= 1)

    return ConnectivityReport(
        num_components=int(n_comps),
        component_sizes=component_sizes,
        largest_fraction=largest_fraction,
        is_single_component=is_single,
        component_of=component_of,
        diagnostics={
            "num_particles": N,
            "total_linked_pairs": len(linked_pairs),
            "wire_radius": float(wire_radius),
        },
    )

