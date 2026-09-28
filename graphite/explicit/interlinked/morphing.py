"""
Graphite Explicit Interlinked — Epitaxially-Constrained Continuous Basis Morphing (Idea 3)

Implements continuous vertex/face truncation morphing across an epitaxial habit plane
transition zone, where particle geometries, orientations, and lattice bases smoothly
interpolate from Phase A (e.g. C-6-TT) to Phase B (e.g. D-4-TET).

Governing Principles:
1. Epitaxially-Constrained Transverse Pinning:
   Locks transverse habit plane pitch to a_trans = a_sc (0% transverse strain)
   across all intermediate slices, strictly preventing vertical/lateral layer tearing.
2. Parametric Truncated Tetrahedron Morpher:
   Continuous vertex truncation tau(t) in [0, 1/3]:
     tau = 1/3: Archimedean Truncated Tetrahedron (C-6-TT, 12 vertices, 18 struts)
     0 < tau < 1/3: Hybrid truncated tetrahedron (12 vertices, 18 struts)
     tau = 0: Platonic Regular Tetrahedron (D-4-TET, 4 vertices, 6 struts)
3. Geodesic SO(3) Habit Plane Orientation:
   Smoothly rotates local particle orientation from identity R = I towards the rotated
   habit plane R_habit (e.g. up to 45-deg around X-axis for Diamond coincidence).
4. Scale & Longitudinal Spacing Interpolation:
   Smoothly interpolates particle scale s(t) = (1-t)*s_a + t*s_b and longitudinal
   layer spacing dx(t) along the habit plane normal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence
import numpy as np

from .particle import ParticleGeometry, InterlinkedParticle
from .cell import InterlinkedCell, InterlinkedRegistry, C6TTCell
from .clearance import check_particle_clearance, compute_pairwise_particle_clearances
from .gradient import ThicknessGradient, resolve_thickness_gradient, LinearThicknessGradient
from .epitaxy import solve_epitaxial_transition, EpitaxialAlignment


# =============================================================================
# Parametric Truncated Tetrahedron Geometry Builder
# =============================================================================

def build_morphing_tetrahedron_geometry(
    size: float = 8.0,
    tau: float = 1.0 / 3.0,
    min_strut_length: float = 1e-4,
) -> ParticleGeometry:
    """
    Parametric polyhedron with vertex truncation tau in [0, 1/3].

    - tau = 1/3: Archimedean Truncated Tetrahedron (C-6-TT, 12 vertices, 18 struts).
    - 0 < tau < 1/3: Intermediate Truncated Tetrahedron (12 vertices, 18 struts).
    - tau = 0.0: Platonic Regular Tetrahedron (D-4-TET, 4 vertices, 6 struts).

    Args:
        size: Bounding scale / nominal cell size in mm.
        tau: Truncation ratio in [0, 1/3]. 1/3 gives Archimedean TT, 0 gives regular TET.
        min_strut_length: Threshold below which collapsed corner struts are pruned.

    Returns:
        ParticleGeometry instance.
    """
    s = float(size)
    tau_clipped = float(np.clip(tau, 0.0, 1.0 / 3.0))

    # Base regular tetrahedron vertices (edge length = 2 * sqrt(2) * (s / (2*sqrt(2))) = s)
    V = np.array(
        [
            [1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
        ],
        dtype=np.float64,
    ) * (s / np.sqrt(2.0))

    if tau_clipped < 1e-4:
        # Fully collapsed to regular tetrahedron
        nodes = V.copy()
        struts = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]], dtype=np.int64)
        bounding_radius = float(np.max(np.linalg.norm(nodes, axis=1)))
        return ParticleGeometry(
            nodes=nodes,
            struts=struts,
            bounding_radius=bounding_radius,
            geometry_type="TET",
            metadata={"size": s, "tau": 0.0, "num_vertices": 4, "num_struts": 6},
        )

    edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    nodes_list: list[np.ndarray] = []
    corner_verts: dict[int, list[int]] = {0: [], 1: [], 2: [], 3: []}
    edge_struts: list[tuple[int, int]] = []

    idx = 0
    for i, j in edges:
        p_ij = (1.0 - tau_clipped) * V[i] + tau_clipped * V[j]
        p_ji = tau_clipped * V[i] + (1.0 - tau_clipped) * V[j]

        nodes_list.append(p_ij)
        corner_verts[i].append(idx)
        idx += 1

        nodes_list.append(p_ji)
        corner_verts[j].append(idx)
        idx += 1

        edge_struts.append((idx - 2, idx - 1))

    corner_struts: list[tuple[int, int]] = []
    for c_idx in range(4):
        cv = corner_verts[c_idx]
        corner_struts.append((cv[0], cv[1]))
        corner_struts.append((cv[1], cv[2]))
        corner_struts.append((cv[2], cv[0]))

    all_struts = edge_struts + corner_struts
    nodes_arr = np.array(nodes_list, dtype=np.float64)

    # Filter out any degenerate / zero-length struts if tau is very small
    valid_struts: list[tuple[int, int]] = []
    for u, v in all_struts:
        if np.linalg.norm(nodes_arr[u] - nodes_arr[v]) >= min_strut_length:
            valid_struts.append((u, v))

    struts_arr = np.array(valid_struts, dtype=np.int64)
    bounding_radius = float(np.max(np.linalg.norm(nodes_arr, axis=1)))

    return ParticleGeometry(
        nodes=nodes_arr,
        struts=struts_arr,
        bounding_radius=bounding_radius,
        geometry_type="TT_MORPH",
        metadata={
            "size": s,
            "tau": tau_clipped,
            "num_vertices": len(nodes_arr),
            "num_struts": len(struts_arr),
        },
    )


# =============================================================================
# Epitaxial Basis Morpher Class
# =============================================================================

@dataclass
class EpitaxialMorphConfig:
    """
    Configuration for an epitaxially-constrained continuous basis morphing PAM assembly.
    """
    cell_a: InterlinkedCell | str = "c6tt"
    cell_b: InterlinkedCell | str = "d4tet"
    grid_size_a: tuple[int, int, int] = (2, 2, 2)  # (nx_A, ny, nz)
    grid_size_b: tuple[int, int, int] = (2, 2, 2)  # (nx_B, ny, nz)
    num_morph_layers: int = 2                       # Number of intermediate morphed layers
    pitch: float = 10.0                            # Transverse pitch a_trans = a_sc (mm)
    wire_radius: float = 0.35                       # Strut wire radius (mm)
    min_clearance: float = 0.150                    # Target minimum clearance (mm)
    habit_plane: str = "100"                        # Habit plane
    thickness_gradient: ThicknessGradient | None = None
    gradient_axis: str | None = None
    gradient_radius_range: tuple[float, float] | None = None
    gradient_bounds: tuple[float, float] | None = None
    num_thickness_buckets: int = 16


class EpitaxialBasisMorpher:
    """
    Evaluates continuous morphing states along a heteroepitaxial transition path.
    """
    def __init__(
        self,
        pitch_a: float = 10.0,
        pitch_b: float | None = None,
        habit_angle_rad: float = np.pi / 4.0,
        size_ratio_a: float = 0.80,
        size_ratio_b: float = 0.603,
    ):
        self.pitch_a = float(pitch_a)
        self.pitch_b = float(pitch_b if pitch_b is not None else np.sqrt(2.0) * pitch_a)
        self.habit_angle_rad = float(habit_angle_rad)
        self.size_ratio_a = float(size_ratio_a)
        self.size_ratio_b = float(size_ratio_b)

        self.size_a = self.pitch_a * self.size_ratio_a
        self.size_b = self.pitch_b * self.size_ratio_b

    def evaluate_state(self, t: float) -> dict[str, Any]:
        """
        Evaluate morphing state at normalized transition coordinate t in [0, 1].

        - t = 0.0: Pure Phase A (C-6-TT)
        - t = 1.0: Pure Phase B (D-4-TET orientation)
        """
        t_clamped = float(np.clip(t, 0.0, 1.0))
        tau = (1.0 / 3.0) * (1.0 - t_clamped)
        size = (1.0 - t_clamped) * self.size_a + t_clamped * self.size_b
        theta = t_clamped * self.habit_angle_rad

        R = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, np.cos(theta), -np.sin(theta)],
                [0.0, np.sin(theta), np.cos(theta)],
            ],
            dtype=np.float64,
        )

        geom = build_morphing_tetrahedron_geometry(size=size, tau=tau)
        return {
            "t": t_clamped,
            "tau": tau,
            "size": size,
            "theta_rad": theta,
            "theta_deg": float(np.degrees(theta)),
            "rotation_matrix": R,
            "geometry": geom,
        }


# =============================================================================
# Epitaxial Morphing Assembly Generator
# =============================================================================

def generate_epitaxial_morph_lattice(
    config: EpitaxialMorphConfig,
    check_clearance: bool = True,
) -> Any:
    """
    Generate a full multi-zone PAM lattice featuring:
      - Zone 1: Pure Phase A (e.g. C-6-TT)
      - Zone 2: Epitaxially-Constrained Continuous Basis Morphing Band (tau: 1/3 -> 0)
      - Zone 3: Pure Phase B (e.g. Rotated D-4-TET)

    All intermediate slices have transverse pitch locked to a_sc (0% strain),
    strictly preventing vertical layer detachment or tearing.
    """
    from .generator import (
        InterlinkedLatticeResult,
        _particles_to_combined_mesh,
    )
    from .pams import _tet_ab_templates, _TET_STRUTS

    # 1. Resolve Cells
    cell_a = config.cell_a
    if isinstance(cell_a, str):
        cell_a = InterlinkedRegistry.get(cell_a)()
    cell_b = config.cell_b
    if isinstance(cell_b, str):
        cell_b = InterlinkedRegistry.get(cell_b)()

    pitch_a = float(config.pitch)
    pitch_b = float(np.sqrt(2.0) * pitch_a)
    wire_r = float(config.wire_radius)
    min_clr = float(config.min_clearance)

    # 2. Alignment & Optimization
    alignment = solve_epitaxial_transition(
        cell_a=cell_a,
        cell_b=cell_b,
        pitch_a=pitch_a,
        pitch_b=pitch_b,
        wire_radius=wire_r,
        min_clearance=min_clr,
    )
    dx_int = float(alignment.interface_offset)
    R_habit = alignment.rotation_matrix

    morpher = EpitaxialBasisMorpher(pitch_a=pitch_a, pitch_b=pitch_b)

    particles: list[InterlinkedParticle] = []
    pid = 0
    grid_a = config.grid_size_a
    grid_b = config.grid_size_b
    num_morph = max(0, int(config.num_morph_layers))

    # Longitudinal layer spacing for morph layers:
    # Morph layers are spaced by pitch_a along -X between the interface and Zone A
    # Total morph offset along X:
    morph_layer_dx = pitch_a

    # Zone 1: Pure Cell A (nx_A layers along -X)
    x_zone_a_offset = dx_int + (num_morph * morph_layer_dx)
    for ix in range(-grid_a[0], 0):
        for iy in range(grid_a[1]):
            for iz in range(grid_a[2]):
                origin = np.array([(ix + 1) * pitch_a - x_zone_a_offset, iy * pitch_a, iz * pitch_a], dtype=np.float64)
                parts = cell_a.instantiate_site((ix, iy, iz), origin, pitch_a, id_start=pid)
                for p in parts:
                    p.metadata["zone"] = "Zone_A_Pure"
                    p.metadata["tau"] = 1.0 / 3.0
                    p.metadata["layer_x"] = ix
                    particles.append(p)
                    pid += 1

    # Zone 2: Continuous Epitaxial Morphing Band
    for m in range(num_morph):
        # Progress from t ~ 0 near Zone A to t ~ 1 near Zone B
        t_m = float(m + 1) / float(num_morph + 1)
        state = morpher.evaluate_state(t_m)
        x_m = -dx_int - ((num_morph - 1 - m) * morph_layer_dx)

        for iy in range(grid_a[1]):
            for iz in range(grid_a[2]):
                T = np.eye(4, dtype=np.float64)
                T[:3, :3] = state["rotation_matrix"]
                T[:3, 3] = [x_m, iy * pitch_a, iz * pitch_a]

                p = InterlinkedParticle(
                    particle_id=pid,
                    geometry=state["geometry"],
                    transform=T,
                    sublattice_id="A",
                    cell_index=(m, iy, iz),
                    metadata={
                        "zone": "Zone_Morph",
                        "t": t_m,
                        "tau": state["tau"],
                        "theta_deg": state["theta_deg"],
                        "layer_x": m,
                    },
                )
                particles.append(p)
                pid += 1

    # Zone 3: Pure Cell B (Rotated Diamond Cubic, extending along +X)
    L_tet = 0.603 * pitch_b
    proto_a = ParticleGeometry(nodes=_tet_ab_templates(L_tet)[0], struts=_TET_STRUTS.copy(), bounding_radius=L_tet, geometry_type="TET")
    proto_b = ParticleGeometry(nodes=_tet_ab_templates(L_tet)[1], struts=_TET_STRUTS.copy(), bounding_radius=L_tet, geometry_type="TET")

    ny_a, nz_a = grid_a[1], grid_a[2]
    w_y = (ny_a - 1) * pitch_a
    w_z = (nz_a - 1) * pitch_a
    margin = 1.0
    y_min, y_max = -margin, w_y + margin
    z_min, z_max = -margin, w_z + margin

    y_dia_min = (y_min + z_min) / np.sqrt(2.0)
    y_dia_max = (y_max + z_max) / np.sqrt(2.0)
    z_dia_min = (z_min - y_max) / np.sqrt(2.0)
    z_dia_max = (z_max - y_min) / np.sqrt(2.0)

    j_min = int(np.floor(y_dia_min / pitch_b)) - 1
    j_max = int(np.ceil(y_dia_max / pitch_b)) + 1
    k_min = int(np.floor(z_dia_min / pitch_b)) - 1
    k_max = int(np.ceil(z_dia_max / pitch_b)) + 1

    fcc_offsets = [(0.0, 0.0, 0.0), (0.5, 0.5, 0.0), (0.5, 0.0, 0.5), (0.0, 0.5, 0.5)]
    basis_b_vec = np.array([0.25, 0.25, 0.25], dtype=np.float64) * pitch_b
    seen_sites: set[tuple[float, float, float]] = set()

    for i in range(grid_b[0]):
        for j in range(j_min, j_max + 1):
            for k in range(k_min, k_max + 1):
                cell_orig = np.array([i, j, k], dtype=np.float64) * pitch_b
                for fcc in fcc_offsets:
                    pA = cell_orig + np.asarray(fcc, dtype=np.float64) * pitch_b
                    pB = pA + basis_b_vec
                    for p_raw, sub in [(pA, "A"), (pB, "B")]:
                        key = (round(float(p_raw[0]), 3), round(float(p_raw[1]), 3), round(float(p_raw[2]), 3))
                        if key in seen_sites:
                            continue
                        seen_sites.add(key)
                        pos_rot = R_habit @ p_raw
                        if (y_min - 1e-4 <= pos_rot[1] <= y_max + 1e-4 and
                            z_min - 1e-4 <= pos_rot[2] <= z_max + 1e-4 and
                            0.0 <= pos_rot[0] <= pitch_b * (grid_b[0] - 0.5)):
                            geom = proto_a if sub == "A" else proto_b
                            T = np.eye(4, dtype=np.float64)
                            T[:3, :3] = R_habit
                            T[:3, 3] = pos_rot
                            p = InterlinkedParticle(
                                particle_id=pid,
                                geometry=geom,
                                transform=T,
                                sublattice_id=sub,
                                metadata={"zone": "Zone_B_Pure", "layer_x": int(pos_rot[0] / (pitch_b / 4.0))},
                            )
                            particles.append(p)
                            pid += 1

    # 4. Thickness Grading (if configured)
    centers_all = np.array([p.center for p in particles], dtype=np.float64) if particles else np.zeros((0, 3))
    pts_envelope = None
    grad_axis = config.gradient_axis or (config.thickness_gradient.axis if isinstance(config.thickness_gradient, LinearThicknessGradient) else "x")
    ax_idx = {"x": 0, "y": 1, "z": 2}.get(grad_axis.lower(), 0)
    if len(centers_all) > 0:
        pts_envelope = (float(np.min(centers_all[:, ax_idx])), float(np.max(centers_all[:, ax_idx])))

    gradient = resolve_thickness_gradient(
        gradient=config.thickness_gradient,
        gradient_axis=config.gradient_axis,
        gradient_radius_range=config.gradient_radius_range,
        gradient_bounds=config.gradient_bounds,
        points_envelope=pts_envelope,
    )

    if gradient is not None:
        for p in particles:
            p.wire_radius = gradient.evaluate(p.center)
    else:
        for p in particles:
            if p.wire_radius is None:
                p.wire_radius = wire_r

    # 5. Clearance Verification
    clr_valid = True
    min_clr_found = float("inf")
    clr_report: list[dict[str, Any]] = []

    if check_clearance and len(particles) > 1:
        radii_list = [p.effective_wire_radius(fallback=wire_r) for p in particles]
        clr_valid, min_clr_found, clr_report = check_particle_clearance(
            particles,
            strut_radius=radii_list,
            min_clearance=min_clr,
        )

    # 6. Solid Mesh Generation
    combined_mesh = _particles_to_combined_mesh(
        particles,
        wire_radius=wire_r,
        num_thickness_buckets=config.num_thickness_buckets,
    )

    vol = float(combined_mesh.volume) if combined_mesh is not None and not combined_mesh.is_empty else 0.0
    bounds = combined_mesh.bounds if combined_mesh is not None and not combined_mesh.is_empty else np.zeros((2, 3))

    total_nodes = sum(len(p.geometry.nodes) for p in particles)
    total_struts = sum(len(p.geometry.struts) for p in particles)

    return InterlinkedLatticeResult(
        mesh=combined_mesh,
        rings=[],
        num_rings=len(particles),
        num_nodes=total_nodes,
        num_struts=total_struts,
        min_clearance=float(min_clr_found),
        clearance_valid=bool(clr_valid),
        volume=vol,
        bounds=bounds,
        metadata={
            "pipeline": "epitaxial_continuous_morph",
            "habit_plane": config.habit_plane,
            "num_particles": len(particles),
            "num_morph_layers": num_morph,
            "min_clearance_mm": float(min_clr_found),
            "clearance_valid": bool(clr_valid),
            "alignment": alignment,
        },
        particles=particles,
    )
