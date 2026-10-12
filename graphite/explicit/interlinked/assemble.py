"""
Graphite Explicit Interlinked — Multi-Zone Permissive Assembly Engine (Phase 3)

Implements the general multi-zone assembly protocol:
- ZoneSpec: Cell specification, parent network, pitch, habit plane/frame, region predicate, and site props.
- InterfaceSpec: Arbitrary interface plane normal, boundary gap, in-plane shift, and relative twist.
- assemble(zones, interfaces): Deterministic spatial assembly with zero numerical solver loops.
  "Always Generate, Always Diagnose" standard: never raises on physical incompatibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence, Any
from pathlib import Path
import numpy as np
import trimesh

from .particle import ParticleGeometry, InterlinkedParticle
from .cell import InterlinkedCell, InterlinkedRegistry
from .cells import CustomWireframeCell
from .generator import (
    SiteProps,
    InterlinkedLatticeResult,
    _particles_to_combined_mesh,
)
from .seeding import (
    orthogonal_supercell,
    seed_orthogonal_lattice,
    seed_cartesian_lattice,
    seed_diamond_lattice,
    seed_bcc_lattice,
    instantiate_lattice_on_sites,
    resolve_orientation_matrix,
)
from .validator import (
    check_envelope_penetration_and_catenation,
    evaluate_inter_zone_catenation,
    InterZoneBoundaryReport,
    evaluate_insufficient_linking,
    InsufficientLinkingReport,
    evaluate_assembly_connectivity,
    ConnectivityReport,
)
from .clearance import (
    check_particle_clearance,
    compute_pairwise_particle_clearances,
)


@dataclass
class ZoneSpec:
    """
    Specification for a discrete structural zone within an interlinked metamaterial assembly.

    Attributes:
        cell: Registered cell name, InterlinkedCell instance, (nodes, struts) tuple, or recipe dict.
        network: Parent crystallographic network ('dia', 'pcu', 'bcc', 'square_2d', etc.).
        pitch: Characteristic unit cell spacing in mm.
        grid_size: (nx, ny, nz) cell repeat counts.
        frame: 3x3 orientation matrix, Miller plane name (e.g. '111', '100'), or Euler angles.
        origin: Cartesian origin (x0, y0, z0) in mm.
        region: Optional spatial predicate function (pt: np.ndarray) -> bool for boundary clipping.
        site_props: Optional callable (pt: np.ndarray) -> SiteProps | dict for per-site properties.
        name: Human-readable zone identifier.
        wire_radius: Nominal wire/strut radius in mm.
        cull_dangling: If True, culls intra-zone hanging particles (degree 0 within this zone's lattice).
    """
    cell: InterlinkedCell | str | tuple | dict
    network: str = "dia"
    pitch: float = 12.0
    grid_size: tuple[int, int, int] = (2, 2, 2)
    frame: np.ndarray | str | Sequence[float] | None = None
    origin: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    region: Callable[[np.ndarray], bool] | None = None
    site_props: Callable[[np.ndarray], Any] | None = None
    name: str = "Zone"
    wire_radius: float = 0.40
    cull_dangling: bool = False
    is_twin: bool = False


@dataclass
class InterfaceSpec:
    """
    Geometric boundary relationship bridging two adjacent structural zones.

    Attributes:
        normal: Habit plane unit normal vector orienting the interface boundary (e.g. [0,0,1], '111', '100').
        point: Anchor point lying on the interface plane in mm.
        gap: Physical spacing along normal between the boundary layers of Zone A and Zone B.
        shift: 3D translational offset applied to Zone B in the transverse habit plane.
        twist_deg: Relative azimuthal twist rotation of Zone B about the interface normal in degrees.
        reflection_matrix: Optional 3x3 reflection matrix for coherent twin boundaries (Sigma 3).
    """
    normal: np.ndarray | str | Sequence[float] = field(default_factory=lambda: np.array([0.0, 0.0, 1.0], dtype=np.float64))
    point: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    gap: float = 0.0
    shift: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    twist_deg: float = 0.0
    reflection_matrix: np.ndarray | None = None


def _resolve_normal(normal: Any, habit_frame: Any = None) -> np.ndarray:
    """
    Resolve normal parameter into a normalized 3D unit vector.

    When habit_frame is an orthogonal supercell habit plane ('111', '110', '100', '001', '011'),
    the lattice has already been rotated so that the habit plane normal lies along +Z = [0, 0, 1].
    In that local frame, if normal is specified as the habit plane name or '001', the normal is [0, 0, 1].
    """
    if isinstance(habit_frame, str):
        hf = habit_frame.strip().replace("{", "").replace("}", "")
        if hf in ("111", "110", "100", "001", "011"):
            if isinstance(normal, str):
                s = normal.strip().replace("{", "").replace("}", "")
                if s == hf or s in ("001", "00-1"):
                    return np.array([0.0, 0.0, 1.0], dtype=np.float64)
            elif isinstance(normal, (list, tuple, np.ndarray)):
                n_arr = np.asarray(normal, dtype=np.float64).reshape(-1)
                if len(n_arr) == 3 and np.allclose(n_arr / (np.linalg.norm(n_arr) + 1e-12), [0, 0, 1]):
                    return np.array([0.0, 0.0, 1.0], dtype=np.float64)

    if isinstance(normal, str):
        s = normal.strip().replace("{", "").replace("}", "")
        if s == "111":
            n = np.array([1.0, 1.0, 1.0], dtype=np.float64)
        elif s == "110":
            n = np.array([1.0, 1.0, 0.0], dtype=np.float64)
        elif s in ("100", "001", "010"):
            n = np.array([1.0, 0.0, 0.0]) if s == "100" else (np.array([0.0, 1.0, 0.0]) if s == "010" else np.array([0.0, 0.0, 1.0]))
        else:
            try:
                n = np.array([float(c) for c in s], dtype=np.float64)
            except Exception:
                n = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        n = np.asarray(normal, dtype=np.float64).reshape(3)
    norm = np.linalg.norm(n)
    return n / norm if norm > 1e-12 else np.array([0.0, 0.0, 1.0], dtype=np.float64)


def _apply_props_to_particle(p: InterlinkedParticle, props: Any) -> None:
    """Helper to apply per-site properties (wire_radius, size_ratio, rotation) to a particle."""
    if props is None:
        return
    if isinstance(props, dict):
        w_r = props.get("wire_radius")
        s_r = props.get("size_ratio")
        tw_deg = props.get("twist_deg")
        rot_mat = props.get("rotation")
    else:
        w_r = getattr(props, "wire_radius", None)
        s_r = getattr(props, "size_ratio", None)
        tw_deg = getattr(props, "twist_deg", None)
        rot_mat = getattr(props, "rotation", None)

    if w_r is not None:
        p.wire_radius = float(w_r)
    if s_r is not None and abs(float(s_r) - 1.0) > 1e-4:
        scale_factor = float(s_r)
        new_nodes = p.geometry.nodes * scale_factor
        p.geometry = ParticleGeometry(
            nodes=new_nodes,
            struts=p.geometry.struts,
            bounding_radius=p.geometry.bounding_radius * scale_factor,
            geometry_type=p.geometry.geometry_type,
            metadata=dict(p.geometry.metadata),
        )
    if rot_mat is not None:
        R_add = np.asarray(rot_mat, dtype=np.float64).reshape(3, 3)
        p.rotation = R_add @ p.rotation
    elif tw_deg is not None and abs(float(tw_deg)) > 1e-4:
        R_twist_site = _rotation_about_axis(np.array([0.0, 0.0, 1.0]), float(tw_deg))
        p.rotation = R_twist_site @ p.rotation


def _resolve_cell_instance(cell_param: Any) -> InterlinkedCell:
    """Resolve cell parameter into a valid InterlinkedCell instance."""
    if isinstance(cell_param, str):
        cell_cls = InterlinkedRegistry.get(cell_param)
        return cell_cls() if callable(cell_cls) else cell_cls
    elif isinstance(cell_param, InterlinkedCell):
        return cell_param
    elif isinstance(cell_param, (tuple, list)) and len(cell_param) == 2:
        return CustomWireframeCell(nodes=cell_param[0], struts=cell_param[1])
    elif isinstance(cell_param, dict):
        return CustomWireframeCell(recipe=cell_param)
    elif callable(cell_param):
        return cell_param()
    raise TypeError(f"Unsupported cell parameter type: {type(cell_param)}")


def _rotation_about_axis(axis: np.ndarray, angle_deg: float) -> np.ndarray:
    """Compute 3x3 rotation matrix for rotation by angle_deg around unit axis."""
    rad = np.radians(float(angle_deg))
    u = np.asarray(axis, dtype=np.float64)
    u_norm = np.linalg.norm(u)
    if u_norm < 1e-12 or abs(angle_deg) < 1e-6:
        return np.eye(3, dtype=np.float64)
    u = u / u_norm
    cos_a = np.cos(rad)
    sin_a = np.sin(rad)
    K = np.array([
        [0.0, -u[2], u[1]],
        [u[2], 0.0, -u[0]],
        [-u[1], u[0], 0.0],
    ], dtype=np.float64)
    return cos_a * np.eye(3, dtype=np.float64) + sin_a * K + (1.0 - cos_a) * np.outer(u, u)


def _cull_to_largest_component(
    parts: list[InterlinkedParticle],
    wire_radius: float,
) -> list[InterlinkedParticle]:
    """
    Cull dangling particles by retaining strictly the largest connected component
    of the intra-zone catenation graph. Removes isolated particles and disconnected islands.
    """
    if len(parts) <= 1:
        return parts

    cat_rep = check_envelope_penetration_and_catenation(parts, wire_radius=wire_radius)
    linked_pairs = getattr(cat_rep, "linked_pairs", [])
    if not linked_pairs:
        return parts

    from scipy.sparse import csgraph, lil_matrix
    N = len(parts)
    mat = lil_matrix((N, N), dtype=bool)
    for pair in linked_pairs:
        i, j = int(pair[0]), int(pair[1])
        if 0 <= i < N and 0 <= j < N:
            mat[i, j] = True
            mat[j, i] = True

    n_comps, labels = csgraph.connected_components(mat, directed=False)
    if n_comps <= 1:
        return parts

    comp_sizes = np.bincount(labels, minlength=n_comps)
    largest_comp = int(np.argmax(comp_sizes))
    surviving = [p for idx, p in enumerate(parts) if labels[idx] == largest_comp]
    return surviving


def assemble(
    zones: Sequence[ZoneSpec],
    interfaces: Sequence[InterfaceSpec] | None = None,
    wire_radius: float = 0.40,
    min_clearance: float = 0.20,
    check_clearance: bool = True,
    circular_segments: int = 24,
    num_thickness_buckets: int = 16,
    strict: bool = False,
    mode: str = "discrete",
    cull_dangling: bool = False,
) -> InterlinkedLatticeResult:
    """
    Assemble multi-zone interlinked metamaterial structures deterministically.

    "Always Generate, Always Diagnose" standard:
    Places all particles according to ZoneSpec and InterfaceSpec definitions,
    builds the solid multi-body meshes via clean mitered joints, measures
    physical clearance and topological catenation, and returns a fully diagnosed
    result without raising exceptions on geometric mismatch (unless strict=True).

    Supports:
        - Continuous Mode (mode='continuous' or 'auto'): Seeds a continuous crystallographic
          lattice across multiple zones sharing the same parent network and habit frame,
          guaranteeing exact crystallographic registry and layer interleaving.
        - Discrete Mode (mode='discrete'): Positions adjacent zones along the habit plane normal
          using center-to-center layer spacing (iface.gap) and azimuthal twist about the habit axis.

    Args:
        zones: Sequence of ZoneSpec definitions.
        interfaces: Optional sequence of InterfaceSpec transition relationships.
        wire_radius: Fallback strut wire radius in mm.
        min_clearance: Target minimum surface clearance threshold in mm.
        check_clearance: If True, evaluates pairwise physical distances.
        circular_segments: Cross-sectional cylinder discretization resolution.
        num_thickness_buckets: Cylinder mesh bucketing resolution.
        strict: If True, raises ValueError on clearance or catenation failure.
        mode: Assembly mode ('auto', 'continuous', or 'discrete').
        cull_dangling: If True, culls intra-zone hanging particles across all zones.

    Returns:
        InterlinkedLatticeResult payload containing the combined solid mesh,
        surviving particles with zone metadata, and physical quality report.
    """
    if not zones:
        raise ValueError("assemble() requires at least one ZoneSpec")

    all_particles: list[InterlinkedParticle] = []
    zone_particle_lists: list[list[InterlinkedParticle]] = []

    # Determine whether continuous lattice mode applies
    use_continuous = bool(mode == "continuous" and len(zones) == 2)

    if use_continuous:
        z0 = zones[0]
        z1 = zones[1]
        cell_a = _resolve_cell_instance(z0.cell)
        cell_b = _resolve_cell_instance(z1.cell)
        pitch = float(z0.pitch)
        net = str(z0.network).lower().strip()
        frame_str = str(z0.frame).strip().replace("{", "").replace("}", "") if isinstance(z0.frame, str) else None
        z_origin = np.asarray(z0.origin, dtype=np.float64).reshape(3)
        g0 = tuple(z0.grid_size)
        g1 = tuple(z1.grid_size)
        combined_repeats = (g0[0], g0[1], g0[2] + g1[2])

        if frame_str and frame_str in ("100", "110", "111", "001", "011"):
            cull_active = bool(cull_dangling or any(getattr(z, "cull_dangling", False) for z in zones))
            centers, frames, indices, sublattice_ids = seed_orthogonal_lattice(
                repeats=combined_repeats,
                pitch=pitch,
                network=net,
                plane=frame_str,
                origin=z_origin,
                prune_unbonded=cull_active,
            )
            _R_sup, periods = orthogonal_supercell(network=net, plane=frame_str)
            period_z = pitch * periods[2]
            z_split = z_origin[2] + period_z * g0[2]
        elif net in ("dia", "diamond"):
            centers, frames, indices, sublattice_ids = seed_diamond_lattice(
                repeats=combined_repeats,
                conventional_cell_size=pitch,
                origin=z_origin,
            )
            z_split = z_origin[2] + pitch * g0[2]
        elif net in ("bcc", "body_centered"):
            centers, frames, indices, sublattice_ids = seed_bcc_lattice(
                repeats=combined_repeats,
                cell_size=pitch,
                origin=z_origin,
            )
            z_split = z_origin[2] + pitch * g0[2]
        else:
            centers, frames, indices = seed_cartesian_lattice(
                repeats=combined_repeats,
                pitch=pitch,
                origin=z_origin,
            )
            sublattice_ids = None
            z_split = z_origin[2] + pitch * g0[2]

        mask_a = np.round(centers[:, 2], 4) <= round(z_split, 4)
        mask_b = ~mask_a

        subs_a = [sublattice_ids[i] for i in range(len(sublattice_ids)) if mask_a[i]] if sublattice_ids is not None else None
        subs_b = [sublattice_ids[i] for i in range(len(sublattice_ids)) if mask_b[i]] if sublattice_ids is not None else None

        parts_a = instantiate_lattice_on_sites(
            cell=cell_a,
            centers=centers[mask_a],
            frames=frames[mask_a],
            grid_indices=indices[mask_a],
            cell_pitch=pitch,
            id_start=0,
            sublattice_ids=subs_a,
        )
        parts_b = instantiate_lattice_on_sites(
            cell=cell_b,
            centers=centers[mask_b],
            frames=frames[mask_b],
            grid_indices=indices[mask_b],
            cell_pitch=pitch,
            id_start=len(parts_a),
            sublattice_ids=subs_b,
        )

        # Apply interface transformation to Zone B
        if interfaces and len(interfaces) > 0:
            iface = interfaces[0]
            if getattr(iface, "reflection_matrix", None) is not None:
                R_refl = np.asarray(iface.reflection_matrix, dtype=np.float64)
                for p in parts_b:
                    p.rotation = R_refl @ p.rotation
            if abs(float(iface.twist_deg)) > 1e-4:
                R_twist = _rotation_about_axis(np.array([0.0, 0.0, 1.0]), iface.twist_deg)
                for p in parts_b:
                    p.rotation = R_twist @ p.rotation
            if np.linalg.norm(iface.shift) > 1e-4:
                shift_vec = np.asarray(iface.shift, dtype=np.float64).reshape(3)
                for p in parts_b:
                    p.center = p.center + shift_vec

        # Apply region predicate and site props
        w_r0 = float(z0.wire_radius) if z0.wire_radius is not None else float(wire_radius)
        w_r1 = float(z1.wire_radius) if z1.wire_radius is not None else float(wire_radius)

        if z0.region is not None:
            parts_a = [p for p in parts_a if z0.region(p.center)]
        if z1.region is not None:
            parts_b = [p for p in parts_b if z1.region(p.center)]

        if (getattr(z0, "cull_dangling", False) or cull_dangling) and len(parts_a) > 1:
            parts_a = _cull_to_largest_component(parts_a, wire_radius=w_r0)

        if (getattr(z1, "cull_dangling", False) or cull_dangling) and len(parts_b) > 1:
            parts_b = _cull_to_largest_component(parts_b, wire_radius=w_r1)

        for p in parts_a:
            p.wire_radius = w_r0
            p.metadata["zone_index"] = 0
            p.metadata["zone_name"] = z0.name
            p.metadata["cell_name"] = getattr(cell_a, "name", str(cell_a))
            if z0.site_props is not None:
                props = z0.site_props(p.center)
                _apply_props_to_particle(p, props)

        for p in parts_b:
            p.wire_radius = w_r1
            p.metadata["zone_index"] = 1
            p.metadata["zone_name"] = z1.name
            p.metadata["cell_name"] = getattr(cell_b, "name", str(cell_b))
            if z1.site_props is not None:
                props = z1.site_props(p.center)
                _apply_props_to_particle(p, props)

        zone_particle_lists = [parts_a, parts_b]
        all_particles = parts_a + parts_b

    else:
        # Standard discrete multi-zone placement loop
        pid = 0
        for z_idx, z_spec in enumerate(zones):
            cell = _resolve_cell_instance(z_spec.cell)
            pitch = float(z_spec.pitch)
            grid_size = tuple(z_spec.grid_size)
            net = str(z_spec.network).lower().strip()
            z_origin = np.asarray(z_spec.origin, dtype=np.float64).reshape(3)
            z_wire_r = float(z_spec.wire_radius) if z_spec.wire_radius is not None else float(wire_radius)

            sublattice_ids = None
            plane_str = None
            if isinstance(z_spec.frame, str):
                plane_str = z_spec.frame.strip().replace("{", "").replace("}", "")

            if plane_str and plane_str in ("100", "110", "111", "001", "011"):
                cull_active = bool(cull_dangling or getattr(z_spec, "cull_dangling", False))
                centers, frames, indices, sublattice_ids = seed_orthogonal_lattice(
                    repeats=grid_size,
                    pitch=pitch,
                    network=net,
                    plane=plane_str,
                    origin=z_origin,
                    prune_unbonded=cull_active,
                )
            elif net in ("dia", "diamond"):
                centers, frames, indices, sublattice_ids = seed_diamond_lattice(
                    repeats=grid_size,
                    conventional_cell_size=pitch,
                    origin=z_origin,
                )
            elif net in ("bcc", "body_centered"):
                centers, frames, indices, sublattice_ids = seed_bcc_lattice(
                    repeats=grid_size,
                    cell_size=pitch,
                    origin=z_origin,
                )
            else:
                centers, frames, indices = seed_cartesian_lattice(
                    repeats=grid_size,
                    pitch=pitch,
                    origin=z_origin,
                )

            if z_spec.frame is not None and not (isinstance(z_spec.frame, str) and plane_str in ("100", "110", "111")):
                R_rot = resolve_orientation_matrix(z_spec.frame)
                centers = (R_rot @ (centers - z_origin).T).T + z_origin
                frames = np.array([R_rot @ f for f in frames], dtype=np.float64)

            z_particles = instantiate_lattice_on_sites(
                cell=cell,
                centers=centers,
                frames=frames,
                grid_indices=indices,
                cell_pitch=pitch,
                id_start=pid,
                sublattice_ids=sublattice_ids,
            )

            if (getattr(z_spec, "cull_dangling", False) or cull_dangling) and len(z_particles) > 1:
                z_particles = _cull_to_largest_component(z_particles, wire_radius=z_wire_r)

            if z_idx > 0 and interfaces and len(interfaces) >= z_idx:
                iface = interfaces[z_idx - 1]
                n_unit = _resolve_normal(iface.normal, habit_frame=plane_str)

                prev_particles = zone_particle_lists[z_idx - 1]
                if prev_particles:
                    prev_centers = np.array([p.center for p in prev_particles], dtype=np.float64)
                    z_prev_max = float(np.max(prev_centers @ n_unit))
                else:
                    z_prev_max = 0.0

                cur_centers = np.array([p.center for p in z_particles], dtype=np.float64) if z_particles else np.zeros((0, 3))
                z_cur_min = float(np.min(cur_centers @ n_unit)) if len(cur_centers) > 0 else 0.0

                delta_n = (z_prev_max - z_cur_min + float(iface.gap)) * n_unit
                delta_shift = np.asarray(iface.shift, dtype=np.float64).reshape(3)
                total_trans = delta_n + delta_shift
                R_twist = _rotation_about_axis(n_unit, iface.twist_deg)

                pivot = np.asarray(iface.point, dtype=np.float64).reshape(3)
                R_refl = getattr(iface, "reflection_matrix", None)
                for p in z_particles:
                    c_rel = p.center - pivot
                    c_rot = (R_twist @ c_rel) + pivot + total_trans
                    R_new = R_twist @ p.rotation
                    if R_refl is not None:
                        R_new = np.asarray(R_refl, dtype=np.float64) @ R_new
                    T_new = np.eye(4, dtype=np.float64)
                    T_new[:3, :3] = R_new
                    T_new[:3, 3] = c_rot
                    p.transform = T_new

            if z_spec.region is not None:
                z_particles = [p for p in z_particles if z_spec.region(p.center)]

            for p in z_particles:
                p.wire_radius = z_wire_r
                p.metadata["zone_index"] = z_idx
                p.metadata["zone_name"] = z_spec.name
                p.metadata["cell_name"] = getattr(cell, "name", str(cell))
                if z_spec.site_props is not None:
                    props = z_spec.site_props(p.center)
                    _apply_props_to_particle(p, props)

            zone_particle_lists.append(z_particles)
            all_particles.extend(z_particles)
            pid += len(z_particles)

    # Re-index all particle IDs consecutively
    for i, p in enumerate(all_particles):
        p.particle_id = i

    # Physical Clearance & Clash Verification
    clearance_valid = True
    min_clr_found = float("inf")
    clr_report: list[dict[str, Any]] = []
    strut_r_input = [p.effective_wire_radius(fallback=wire_radius) for p in all_particles]

    if check_clearance and len(all_particles) > 1:
        clearance_valid, min_clr_found, clr_report = check_particle_clearance(
            all_particles,
            min_clearance=min_clearance,
            strut_radius=strut_r_input,
        )

    # Envelope Penetration & Catenation Check
    catenation_report = None
    all_linked_pairs: list[tuple[int, int, int]] = []
    if len(all_particles) > 1:
        catenation_report = check_envelope_penetration_and_catenation(all_particles)
        all_linked_pairs = getattr(catenation_report, "linked_pairs", [])

    # Dedicated Inter-Zone Boundary Metrics
    boundary_report = None
    boundary_linked_count = 0
    boundary_min_clr = float("inf")
    inter_zone_linked = True
    if len(zones) > 1 and len(all_particles) > 1:
        boundary_report = evaluate_inter_zone_catenation(
            all_particles,
            wire_radius=wire_radius,
            min_clearance=min_clearance,
        )
        boundary_linked_count = boundary_report.boundary_linked_count
        boundary_min_clr = boundary_report.boundary_min_clearance_mm
        inter_zone_linked = bool(boundary_linked_count > 0)
        # Tag boundary-linked particles
        for i_p, j_p, _lk in boundary_report.boundary_linked_pairs:
            all_particles[i_p].metadata["boundary_linked"] = True
            all_particles[j_p].metadata["boundary_linked"] = True

    # Dedicated Insufficient Linking / Boundary Anchoring Evaluation
    insufficient_linking_report = evaluate_insufficient_linking(
        all_particles,
        wire_radius=wire_radius,
        linked_pairs=all_linked_pairs,
    )
    orphaned_indices = set(insufficient_linking_report.orphaned_particle_ids)

    # Full Assembly Connected Components Evaluation
    connectivity_report = evaluate_assembly_connectivity(
        all_particles,
        wire_radius=wire_radius,
        linked_pairs=all_linked_pairs,
    )

    # Tag particle statuses for reporting and color-coded 3MF export
    colliding_indices = set()
    for rec in clr_report:
        clr_val = float(rec.get("clearance", 0.0))
        if clr_val < min_clearance or not rec.get("valid", True):
            i_idx = rec.get("particle_i", rec.get("id_i", None))
            j_idx = rec.get("particle_j", rec.get("id_j", None))
            if i_idx is not None:
                colliding_indices.add(int(i_idx))
            if j_idx is not None:
                colliding_indices.add(int(j_idx))

    hanging_indices = set()
    entangled_indices = set()
    if catenation_report is not None:
        hanging_indices = set(getattr(catenation_report, "hanging_particle_ids", []))
        degrees_linked = getattr(catenation_report, "degrees", np.zeros(len(all_particles), dtype=int))
        for i_p, j_p, _vol in getattr(catenation_report, "entangled_pairs", []):
            if i_p < len(degrees_linked) and degrees_linked[i_p] == 0:
                entangled_indices.add(i_p)
            if j_p < len(degrees_linked) and degrees_linked[j_p] == 0:
                entangled_indices.add(j_p)

    detached_indices = set()
    if connectivity_report.num_components > 1 and connectivity_report.component_sizes and connectivity_report.component_sizes[0] > 1:
        detached_indices = {
            pid for pid, comp_id in connectivity_report.component_of.items()
            if comp_id != 0
        }

    status_counts: dict[str, int] = {}
    for p in all_particles:
        pid_i = p.particle_id
        if pid_i in colliding_indices:
            status = "colliding"
        elif pid_i in orphaned_indices:
            status = f"Zone_{p.metadata.get('zone_index', 0)}_orphaned"
        elif pid_i in detached_indices:
            status = f"Zone_{p.metadata.get('zone_index', 0)}_detached"
        elif pid_i in hanging_indices:
            status = "hanging"
        elif pid_i in entangled_indices:
            status = "entangled"
        elif p.metadata.get("boundary_linked", False):
            status = f"Zone_{p.metadata.get('zone_index', 0)}_boundary_linked"
        else:
            status = f"Zone_{p.metadata.get('zone_index', 0)}"

        p.metadata["status"] = status
        status_counts[status] = status_counts.get(status, 0) + 1

    # Strict Mode Quality Gate
    if strict:
        if not clearance_valid:
            raise ValueError(
                f"Multi-zone clearance violation under strict=True: "
                f"min clearance {min_clr_found:.4f} mm < target {min_clearance:.4f} mm "
                f"({len(colliding_indices)} particles involved)"
            )
        if len(zones) > 1 and not inter_zone_linked:
            raise ValueError(
                f"Multi-zone inter-zone catenation violation under strict=True: "
                f"{boundary_linked_count} boundary links found across zones "
                f"(min clearance: {boundary_min_clr:.4f} mm)"
            )
        if (cull_dangling or any(getattr(z, "cull_dangling", False) for z in zones)):
            if len(zones) > 1 and not insufficient_linking_report.is_valid:
                raise ValueError(
                    f"Multi-zone insufficient linking violation under strict=True: "
                    f"{insufficient_linking_report.num_orphaned_boundary_particles} orphaned boundary particles "
                    f"(IDs: {insufficient_linking_report.orphaned_particle_ids})"
                )
            if connectivity_report.num_components > 1:
                raise ValueError(
                    f"Multi-zone assembly connectivity violation under strict=True with cull_dangling: "
                    f"assembly has {connectivity_report.num_components} disconnected components "
                    f"(sizes: {connectivity_report.component_sizes})"
                )
        if catenation_report is not None:
            if catenation_report.num_entangled > 0:
                raise ValueError(
                    f"Multi-zone catenation violation under strict=True: "
                    f"{catenation_report.num_entangled} entangled pairs"
                )
            if len(zones) == 1 and not catenation_report.passed:
                raise ValueError(
                    f"Multi-zone catenation violation under strict=True: "
                    f"{catenation_report.num_hanging} hanging particles"
                )

    # Build combined multi-body watertight mesh
    mesh = _particles_to_combined_mesh(
        all_particles,
        wire_radius=wire_radius,
        circular_segments=circular_segments,
        num_thickness_buckets=num_thickness_buckets,
    )

    volume = float(mesh.volume) if hasattr(mesh, "volume") else 0.0
    bounds = np.asarray(mesh.bounds, dtype=np.float64) if len(mesh.vertices) > 0 else np.zeros((2, 3))
    total_nodes = sum(len(p.geometry.nodes) for p in all_particles)
    total_struts = sum(len(p.geometry.struts) for p in all_particles)

    metadata = {
        "num_zones": len(zones),
        "zone_names": [z.name for z in zones],
        "particles_per_zone": [len(zl) for zl in zone_particle_lists],
        "total_particles": len(all_particles),
        "min_clearance": min_clr_found,
        "clearance_valid": clearance_valid,
        "catenation_report": catenation_report,
        "clearance_report": clr_report,
        "boundary_report": boundary_report,
        "boundary_linked_count": boundary_linked_count,
        "boundary_min_clearance": boundary_min_clr,
        "inter_zone_linked": inter_zone_linked,
        "insufficient_linking_report": insufficient_linking_report,
        "orphaned_boundary_count": insufficient_linking_report.num_orphaned_boundary_particles,
        "is_sufficiently_linked": insufficient_linking_report.is_valid,
        "connectivity_report": connectivity_report,
        "num_components": connectivity_report.num_components,
        "is_single_component": connectivity_report.is_single_component,
        "status_counts": status_counts,
        "is_watertight": bool(mesh.is_watertight),
        "num_vertices": len(mesh.vertices),
        "num_faces": len(mesh.faces),
    }

    return InterlinkedLatticeResult(
        mesh=mesh,
        rings=[],
        num_rings=len(all_particles),
        num_nodes=total_nodes,
        num_struts=total_struts,
        min_clearance=min_clr_found,
        clearance_valid=clearance_valid,
        volume=volume,
        bounds=bounds,
        metadata=metadata,
        particles=all_particles,
        wire_radius=wire_radius,
    )
