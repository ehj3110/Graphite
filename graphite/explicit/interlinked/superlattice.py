"""
Graphite Explicit Interlinked — Epitaxial Buffer Layers & Superlattices (Idea 2)

Implements multi-zone sequential heteroepitaxy and superlattice generation across
arbitrary chains of interlinkable unit cells, including transitional buffer layers
and alternating periodic superlattice stacks.

Governing Principles:
1. Sequential Heteroepitaxy Chain:
   Evaluates and solves the Three Laws sequentially across each consecutive
   coincidence interface: Zone_0 -> Zone_1 -> ... -> Zone_{N-1}.
2. Cumulative Habit Plane Transformations:
   Tracks cumulative longitudinal offsets dx_cumulative and SO(3) rotations
   across multi-stage habit planes to maintain global crystallographic coherence.
3. Transverse Commensurability Gate:
   Verifies that transverse supercell dimensions match across all interfaces,
   ensuring uniform cross-sections without dangling or disconnected struts.
4. Universal Graded Superlattice Support:
   Fully supports continuous thickness gradients across the entire multi-zone stack.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence
import numpy as np
import trimesh

import networkx as nx

from .particle import ParticleGeometry, InterlinkedParticle
from .cell import InterlinkedCell, InterlinkedRegistry, C6TTCell
from .clearance import check_particle_clearance, compute_pairwise_particle_clearances
from .gradient import ThicknessGradient, resolve_thickness_gradient, LinearThicknessGradient
from .epitaxy import (
    solve_epitaxial_transition,
    analyze_epitaxial_transition,
    EpitaxialAlignment,
    EpitaxialIncompatibilityError,
    apply_strut_taper,
    apply_cage_shrinkage,
)


@dataclass
class BufferPathResult:
    """
    Result of a Dijkstra pathfinding query through the PAM metamaterial transition graph.

    Attributes:
        source: Name or key of source unit cell (e.g. 'd4tet').
        target: Name or key of target unit cell (e.g. 'd4cube').
        path: Ordered list of cell names along the optimal path (e.g. ['d4tet', 'c6tt', 'd4cube']).
        total_cost: Summed Dijkstra transition cost along the path.
        is_direct: True if source transitions directly to target without intermediate buffer cells.
        intermediate_cells: Ordered list of intermediate buffer cells (excluding source and target).
        edge_diagnostics: Per-hop edge metadata including clearance, strain, and cost.
    """
    source: str
    target: str
    path: list[str]
    total_cost: float
    is_direct: bool
    intermediate_cells: list[str]
    edge_diagnostics: list[dict[str, Any]] = field(default_factory=list)


def find_buffer_path(
    cell_a: str | InterlinkedCell,
    cell_b: str | InterlinkedCell,
    habit_plane: str = "100",
    pitch_a: float = 10.0,
    wire_radius: float = 0.35,
    min_clearance: float = 0.100,
    max_hops: int = 4,
    w_strain: float = 10.0,
    w_clearance: float = 1.0,
    w_hop: float = 0.5,
    delta_target: float = 0.400,
) -> BufferPathResult:
    """
    Clearance & Strain Weighted Dijkstra Pathfinding (Strategy 3 Option B).

    Finds the optimal sequence of intermediate buffer cells to transition between
    two distinct PAM unit cell classes when direct planar transition is impeded
    by aperture frustration or lattice mismatch.

    Cost function per edge e = (u, v):
        Cost(e) = w_strain * epsilon + w_clearance * (delta_target / max(Delta, 1e-4)) + w_hop

    Args:
        cell_a: Source unit cell name or instance.
        cell_b: Target unit cell name or instance.
        habit_plane: Interface habit plane ('100', '110', '111').
        pitch_a: Unit cell pitch of source cell.
        wire_radius: Strut wire radius in mm.
        min_clearance: Required minimum physical clearance in mm.
        max_hops: Maximum intermediate path length.
        w_strain: Weight for lattice mismatch strain penalty.
        w_clearance: Weight for inverse clearance penalty.
        w_hop: Weight for path hop length penalty.
        delta_target: Target clearance baseline (default 0.400 mm = 400 um).

    Returns:
        BufferPathResult containing the optimal path and diagnostic metrics.
    """
    name_a = (
        getattr(cell_a, "name", str(cell_a))
        .lower()
        .replace("-", "")
        .replace("_", "")
    )
    name_b = (
        getattr(cell_b, "name", str(cell_b))
        .lower()
        .replace("-", "")
        .replace("_", "")
    )

    # 1. Quick Direct Transition Check
    try:
        direct_res = analyze_epitaxial_transition(
            cell_a,
            cell_b,
            habit_plane=habit_plane,
            pitch_a=pitch_a,
            wire_radius=wire_radius,
            min_clearance=min_clearance,
            enable_taper=True,
        )
        if direct_res.is_feasible and direct_res.min_clearance_mm >= min_clearance:
            clr = direct_res.min_clearance_mm
            strain = direct_res.alignment.strain if direct_res.alignment else 0.0
            cost = w_strain * strain + w_clearance * (delta_target / max(clr, 1e-4)) + w_hop
            return BufferPathResult(
                source=name_a,
                target=name_b,
                path=[name_a, name_b],
                total_cost=float(cost),
                is_direct=True,
                intermediate_cells=[],
                edge_diagnostics=[{
                    "from": name_a,
                    "to": name_b,
                    "clearance_mm": clr,
                    "strain": strain,
                    "cost": float(cost),
                }],
            )
    except Exception:
        pass

    # 2. Lazy Dijkstra Shortest Path Search using Priority Queue
    import heapq

    pam_nodes = [
        "c6tt", "d4tet", "d4cube", "d4ct", "d4ico",
        "b8co", "c6co", "j4oct", "j4ring", "t6ring", "t6hex",
    ]
    all_nodes = list(dict.fromkeys([name_a] + pam_nodes + [name_b]))

    def _eval_edge(u: str, v: str) -> dict[str, Any] | None:
        try:
            res = analyze_epitaxial_transition(
                u,
                v,
                habit_plane=habit_plane,
                pitch_a=pitch_a,
                wire_radius=wire_radius,
                min_clearance=min_clearance,
                enable_taper=True,
            )
            if res.is_feasible and res.min_clearance_mm >= min_clearance:
                clr = res.min_clearance_mm
                strain = res.alignment.strain if res.alignment else 0.0
                cost = float(w_strain * strain + w_clearance * (delta_target / max(clr, 1e-4)) + w_hop)
                return {
                    "weight": cost,
                    "clearance": clr,
                    "strain": strain,
                }
        except Exception:
            pass
        return None

    edge_cache: dict[tuple[str, str], dict[str, Any] | None] = {}
    pq: list[tuple[float, str, list[str]]] = [(0.0, name_a, [name_a])]
    visited: set[str] = set()

    while pq:
        cost, u, path = heapq.heappop(pq)
        if u == name_b and len(path) > 1:
            intermediates = path[1:-1]
            diagnostics = []
            for i in range(len(path) - 1):
                e_data = edge_cache.get((path[i], path[i + 1])) or {}
                diagnostics.append({
                    "from": path[i],
                    "to": path[i + 1],
                    "clearance_mm": e_data.get("clearance", 0.0),
                    "strain": e_data.get("strain", 0.0),
                    "cost": e_data.get("weight", 0.0),
                })
            return BufferPathResult(
                source=name_a,
                target=name_b,
                path=path,
                total_cost=float(cost),
                is_direct=(len(intermediates) == 0),
                intermediate_cells=intermediates,
                edge_diagnostics=diagnostics,
            )

        if u in visited or len(path) > max_hops + 1:
            continue
        visited.add(u)

        for v in all_nodes:
            if v == u or v in visited:
                continue
            pair = (u, v)
            if pair not in edge_cache:
                edge_cache[pair] = _eval_edge(u, v)
            e_info = edge_cache[pair]
            if e_info is not None:
                heapq.heappush(pq, (cost + e_info["weight"], v, path + [v]))

    return BufferPathResult(
        source=name_a,
        target=name_b,
        path=[],
        total_cost=float("inf"),
        is_direct=False,
        intermediate_cells=[],
    )


def build_auto_buffer_superlattice(
    cell_a: str | InterlinkedCell,
    cell_b: str | InterlinkedCell,
    grid_size: tuple[int, int, int] = (2, 2, 2),
    buffer_layers_per_zone: int = 2,
    habit_plane: str = "100",
    pitch_a: float | None = None,
    pitch_b: float | None = None,
    wire_radius: float = 0.35,
    min_clearance: float = 0.100,
    **kwargs: Any,
) -> SuperlatticeConfig:
    """
    Construct a multi-zone SuperlatticeConfig automatically inserting intermediate
    buffer layers (Strategy 3) between cell_a and cell_b when direct transition is incompatible.
    """
    p_a = float(pitch_a if pitch_a is not None else 10.0)
    buf_result = find_buffer_path(
        cell_a=cell_a,
        cell_b=cell_b,
        habit_plane=habit_plane,
        pitch_a=p_a,
        wire_radius=wire_radius,
        min_clearance=min_clearance,
    )

    if not buf_result.path:
        zones = [
            LatticeZoneConfig(cell=cell_a, num_layers=grid_size[0], pitch=pitch_a, habit_plane=habit_plane),
            LatticeZoneConfig(cell=cell_b, num_layers=grid_size[0], pitch=pitch_b, habit_plane=habit_plane),
        ]
    else:
        zones = []
        for i, cell_name in enumerate(buf_result.path):
            if i == 0:
                n_layers = grid_size[0]
                pitch = pitch_a
            elif i == len(buf_result.path) - 1:
                n_layers = grid_size[0]
                pitch = pitch_b
            else:
                n_layers = buffer_layers_per_zone
                pitch = None  # Resolved sequentially
            zones.append(
                LatticeZoneConfig(
                    cell=cell_name,
                    num_layers=n_layers,
                    pitch=pitch,
                    habit_plane=habit_plane,
                    metadata={"buffer_zone": (0 < i < len(buf_result.path) - 1)},
                )
            )

    return SuperlatticeConfig(
        zones=zones,
        grid_transverse=(grid_size[1], grid_size[2]),
        wire_radius=wire_radius,
        min_clearance=min_clearance,
        **kwargs,
    )


@dataclass
class LatticeZoneConfig:
    """
    Specification for a single crystallographic zone within a multi-zone lattice.

    Attributes:
        cell: InterlinkedCell instance or registered name (e.g. 'c6tt', 'd4tet').
        num_layers: Number of unit cell repeat layers along the longitudinal transition axis (X).
        pitch: Fundamental unit cell pitch (mm). If None, inferred or solved.
        habit_plane: Miller index habit plane normal (default '100').
        metadata: Custom zone annotations.
    """
    cell: InterlinkedCell | str
    num_layers: int = 2
    pitch: float | None = None
    habit_plane: str = "100"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SuperlatticeConfig:
    """
    Specification for an epitaxial multi-zone buffer or superlattice assembly.
    """
    zones: list[LatticeZoneConfig] = field(default_factory=list)
    repeat_count: int = 1                         # Repeat sequence count (e.g. [A, B, A, B])
    grid_transverse: tuple[int, int] = (2, 2)     # (ny, nz) transverse grid cells
    wire_radius: float = 0.35                     # Strut wire radius (mm)
    min_clearance: float = 0.150                  # Minimum physical clearance (mm)
    thickness_gradient: ThicknessGradient | None = None
    gradient_axis: str | None = None
    gradient_radius_range: tuple[float, float] | None = None
    gradient_bounds: tuple[float, float] | None = None
    num_thickness_buckets: int = 16


def generate_epitaxial_superlattice(
    config: SuperlatticeConfig,
    check_clearance: bool = True,
) -> Any:
    """
    Synthesize an epitaxial multi-zone buffer superlattice.

    Chains N zones sequentially along the transition axis (X), solving the Three
    Laws across each consecutive boundary, and assembling a unified watertight
    multi-body metamaterial.
    """
    from .generator import (
        InterlinkedLatticeResult,
        _particles_to_combined_mesh,
    )

    if not config.zones:
        raise ValueError("SuperlatticeConfig.zones must contain at least one LatticeZoneConfig.")

    # Expand repeating zones if repeat_count > 1
    expanded_zones: list[LatticeZoneConfig] = []
    for _ in range(max(1, config.repeat_count)):
        expanded_zones.extend(config.zones)

    wire_r = float(config.wire_radius)
    min_clr = float(config.min_clearance)
    ny, nz = config.grid_transverse

    # Resolve all cell instances
    resolved_zones: list[dict[str, Any]] = []
    for idx, z_cfg in enumerate(expanded_zones):
        cell_obj = z_cfg.cell
        if isinstance(cell_obj, str):
            cell_cls = InterlinkedRegistry.get(cell_obj)
            cell_inst = cell_cls() if callable(cell_cls) else cell_cls
        elif isinstance(cell_obj, InterlinkedCell):
            cell_inst = cell_obj
        elif callable(cell_obj):
            cell_inst = cell_obj()
        else:
            raise TypeError(f"Zone {idx}: Unsupported cell type {type(cell_obj)}")

        if z_cfg.pitch is not None:
            pitch = float(z_cfg.pitch)
        elif resolved_zones:
            prev_cell = resolved_zones[-1]["cell"]
            prev_pitch = resolved_zones[-1]["pitch"]
            prev_net = getattr(prev_cell, "parent_network", "pcu").lower().strip()
            curr_net = getattr(cell_inst, "parent_network", "pcu").lower().strip()
            if prev_net in ("pcu", "bcc") and curr_net == "dia":
                pitch = float(np.sqrt(2.0) * prev_pitch)
            elif prev_net == "dia" and curr_net in ("pcu", "bcc"):
                pitch = float(prev_pitch / np.sqrt(2.0))
            else:
                pitch = float(prev_pitch)
        else:
            pitch = 10.0

        resolved_zones.append({
            "config": z_cfg,
            "cell": cell_inst,
            "pitch": pitch,
            "num_layers": max(1, int(z_cfg.num_layers)),
            "habit_plane": z_cfg.habit_plane,
        })

    # Sequential Three-Law interface evaluation
    alignments: list[EpitaxialAlignment] = []
    for k in range(len(resolved_zones) - 1):
        zA = resolved_zones[k]
        zB = resolved_zones[k + 1]
        align = solve_epitaxial_transition(
            cell_a=zA["cell"],
            cell_b=zB["cell"],
            pitch_a=zA["pitch"],
            pitch_b=zB["pitch"],
            wire_radius=wire_r,
            min_clearance=min_clr,
            habit_plane=zA["habit_plane"],
        )
        alignments.append(align)

    # Instantiate Particles Zone-by-Zone along X
    particles: list[InterlinkedParticle] = []
    pid = 0
    current_x = 0.0
    cumulative_transverse = np.array([0.0, 0.0], dtype=np.float64)
    cumulative_R = np.eye(3, dtype=np.float64)

    for k, z_info in enumerate(resolved_zones):
        cell = z_info["cell"]
        pitch = z_info["pitch"]
        n_layers = z_info["num_layers"]
        zone_tag = f"Zone_{k}_{getattr(cell, 'name', 'Cell')}"

        if getattr(cell, "parent_network", "") == "dia":
            from .pams import diamond_network_sites
            sites = diamond_network_sites((n_layers, ny, nz), pitch)
            for pos, sub in sites:
                site_parts = cell.instantiate_site(
                    grid_index=(int(np.floor(pos[0] / pitch)), int(np.floor(pos[1] / pitch)), int(np.floor(pos[2] / pitch))),
                    site_origin=np.zeros(3, dtype=np.float64),
                    cell_pitch=pitch,
                    id_start=pid,
                    context={"sublattice": sub, "site_frame": cumulative_R},
                )
                for p in site_parts:
                    R_world = cumulative_R @ p.rotation
                    t_world = (cumulative_R @ pos) + np.array([current_x, cumulative_transverse[0], cumulative_transverse[1]]) + cumulative_R @ p.center
                    T = np.eye(4, dtype=np.float64)
                    T[:3, :3] = R_world
                    T[:3, 3] = t_world
                    p.transform = T
                    p.particle_id = pid
                    p.sublattice_id = sub
                    p.metadata["zone"] = zone_tag
                    p.metadata["zone_idx"] = k
                    p.metadata["layer_x"] = int(np.floor(pos[0] / pitch))
                    particles.append(p)
                    pid += 1

            last_layer_x = current_x + pitch * (n_layers - 1)

        else:
            # Simple Cubic / Cartesian basis
            for ix in range(n_layers):
                for iy in range(ny):
                    for iz in range(nz):
                        origin = np.array([current_x + ix * pitch, iy * pitch + cumulative_transverse[0], iz * pitch + cumulative_transverse[1]], dtype=np.float64)
                        parts = cell.instantiate_site(
                            (ix, iy, iz),
                            origin,
                            pitch,
                            id_start=pid,
                            context={"site_frame": cumulative_R},
                        )
                        for p in parts:
                            if cumulative_R is not None and not np.allclose(cumulative_R, np.eye(3)):
                                R_world = cumulative_R @ p.rotation
                                t_world = (cumulative_R @ (p.center - origin)) + origin
                                T = np.eye(4, dtype=np.float64)
                                T[:3, :3] = R_world
                                T[:3, 3] = t_world
                                p.transform = T
                            p.metadata["zone"] = zone_tag
                            p.metadata["zone_idx"] = k
                            p.metadata["layer_x"] = ix
                            particles.append(p)
                            pid += 1

            last_layer_x = current_x + pitch * (n_layers - 1)

        # Step interface offset to next zone if another zone follows
        if k < len(alignments):
            align = alignments[k]
            dx_int = float(align.phase_shift[0])
            dy_int = float(align.phase_shift[1])
            dz_int = float(align.phase_shift[2])
            current_x = last_layer_x + dx_int
            cumulative_transverse += np.array([dy_int, dz_int], dtype=np.float64)
            cumulative_R = cumulative_R @ align.rotation_matrix

    # Apply solved boundary tapering and shrinkage per interface
    for k in range(len(alignments)):
        align = alignments[k]
        if align.taper_gamma < 1.0:
            from .epitaxy import apply_strut_taper
            z_curr_parts = [p for p in particles if p.metadata.get("zone_idx") == k]
            z_next_parts = [p for p in particles if p.metadata.get("zone_idx") == k + 1]
            max_layer_curr = max((p.metadata.get("layer_x", 0) for p in z_curr_parts), default=0)
            apply_strut_taper(z_curr_parts, gamma=align.taper_gamma, boundary_layers=(max_layer_curr,))
            apply_strut_taper(z_next_parts, gamma=align.taper_gamma, boundary_layers=(0,))
        if align.shrink_eta < 1.0:
            from .epitaxy import apply_cage_shrinkage
            z_next_parts = [p for p in particles if p.metadata.get("zone_idx") == k + 1]
            apply_cage_shrinkage(z_next_parts, eta=align.shrink_eta, boundary_layers=(0,))

    # Thickness Grading (if configured)
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

    # Clearance Verification
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

    # Solid Mesh Generation
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
            "pipeline": "epitaxial_superlattice",
            "num_zones": len(resolved_zones),
            "num_particles": len(particles),
            "min_clearance_mm": float(min_clr_found),
            "clearance_valid": bool(clr_valid),
            "alignments": alignments,
            "cell_a": getattr(resolved_zones[0]["cell"], "name", str(resolved_zones[0]["cell"])),
            "cell_b": getattr(resolved_zones[-1]["cell"], "name", str(resolved_zones[-1]["cell"])),
        },
        particles=particles,
    )
