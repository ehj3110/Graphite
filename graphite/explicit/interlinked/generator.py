"""
Graphite Explicit Interlinked — Top-Level Generator Module

Orchestrates:
    - Modular unit-cell generation (C6TT, D4Tet, J4Oct, European 4-in-1, Japanese Kusari, Nasa Space Fabric).
    - Spatial lattice seeding (Cartesian, Staggered, Hexagonal, Diamond, Cylindrical Wrap, Spherical Shell).
    - Perimeter boundary management (Policy A: Inset SDF/Mesh culling; Policy C: Solid perimeter frame).
    - Two-tier vectorized clearance verification & auto-inversion pitch solving.
    - Multi-body instanced 3MF & watertight Manifold3D print-in-place mesh generation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence, Any
import numpy as np
import trimesh
import manifold3d as m3d

from graphite.explicit.geometry_module import (
    generate_geometry,
    _rotation_align_local_z_to_unit,
    _affine_rows_from_R_t,
    _manifold_to_trimesh,
)

from .patterns import (
    Ring,
    generate_european_4in1_rings,
    generate_kusari_rings,
    generate_cubic_8ring,
    generate_volumetric_kusari_rings,
)
from .clearance import (
    check_ring_clearance,
    check_particle_clearance,
    resolve_lattice_pitch,
)
from .conformal import (
    cull_rings_by_sdf,
    cull_rings_by_mesh,
)
from .particle import ParticleGeometry, InterlinkedParticle
from .cell import InterlinkedCell, InterlinkedRegistry
from .cells import (  # ensure cell implementations are loaded and registered
    D4TetCell,
    J4OctCell,
    European4in1Cell,
    JapaneseKusariCell,
    NasaSpaceFabricCell,
)
from .seeding import (
    seed_cartesian_lattice,
    seed_staggered_lattice,
    seed_hexagonal_lattice,
    seed_diamond_lattice,
    seed_cylindrical_wrap,
    seed_spherical_shell,
    instantiate_lattice_on_sites,
)
from .boundary import (
    cull_particles_by_sdf,
    cull_particles_by_mesh,
    build_perimeter_frame_solid,
    fuse_perimeter_frame,
)
from .writer_3mf import (
    export_interlinked_3mf,
    solidify_particle_prototype,
)
from .epitaxy import (
    EpitaxialAlignment,
    EpitaxialFeasibilityResult,
    InterlinkedTransitionConfig,
    EpitaxialIncompatibilityError,
    solve_epitaxial_transition,
    analyze_epitaxial_transition,
)

_LEGACY_PATTERNS = {
    "european_4in1",
    "european",
    "euro",
    "kusari",
    "japanese_kusari",
    "japanese",
    "cubic_8ring",
    "cube_2x2x2",
    "cube",
    "volumetric_kusari",
    "volumetric_cube",
    "kusari_3d",
}


@dataclass
class InterlinkedConfig:
    """
    Configuration specification for explicit interlinked lattice generation.

    Attributes:
        pattern: Name of pattern or registered cell.
        cell: Optional explicit InterlinkedCell class or instance (overrides pattern if given).
        seeding_type: Seeding topology ('cartesian', 'staggered', 'hexagonal', 'diamond', 'cylindrical', 'spherical').
        pitch: Grid cell spacing L in mm.
        radius_ratio: Ratio of ring major radius to pitch (R / L) for legacy ring patterns.
        wire_radius: Cross-sectional strut/wire radius r in mm.
        tilt_angle_deg: Tilt angle for alternating weave patterns.
        grid_size: (nx, ny, nz) lattice repetition counts.
        num_ring_segments: Discretization vertices per ring / circular segment resolution.
        min_clearance: Required minimum surface-to-surface gap in mm.
        auto_resolve_pitch: If True, solves unit cell pitch to satisfy min_clearance exactly.
        cull_margin: Inset buffer margin for SDF/mesh boundary culling in mm.
        add_spheres: If True, add fillet spheres at ring polygon nodes.
        flat_radius: Explicit major radius for Kusari flat rings in mm.
        arch_radius: Explicit major radius for Kusari arch links in mm.
        arch_z_radius: Explicit major radius for Volumetric Kusari Z-arch links in mm.
        origin: (x0, y0, z0) Cartesian origin in mm.
        cylinder_radius: Target cylinder radius for cylindrical wrap seeding in mm.
        cylinder_height: Target cylinder height for cylindrical wrap seeding in mm.
        sphere_radius: Target sphere radius for spherical shell seeding in mm.
        add_perimeter_frame: If True, synthesizes a solid boundary frame (Policy C).
        frame_wall_thickness: Wall thickness for boundary perimeter frame in mm.
        frame_margin: Gap clearance between particles and perimeter frame in mm.
        frame_shape: Frame cross-section/hull shape ('box' or 'cylinder').
        export_format: Default export format ('stl' or '3mf').
    """
    pattern: str = "european_4in1"
    cell: InterlinkedCell | str | None = None
    seeding_type: str = "cartesian"
    pitch: float = 10.0
    radius_ratio: float = 0.65
    wire_radius: float = 0.40
    tilt_angle_deg: float = 28.0
    grid_size: tuple[int, int, int] = (5, 5, 1)
    num_ring_segments: int = 24
    min_clearance: float = 0.30
    auto_resolve_pitch: bool = False
    cull_margin: float = 0.50
    add_spheres: bool = True
    flat_radius: float | None = None
    arch_radius: float | None = None
    arch_z_radius: float | None = None
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    cylinder_radius: float | None = None
    cylinder_height: float | None = None
    sphere_radius: float | None = None
    add_perimeter_frame: bool = False
    frame_wall_thickness: float = 1.0
    frame_margin: float = 0.50
    frame_shape: str = "box"
    export_format: str = "stl"
    support_recipe: str | None = None
    transition_to_cell: InterlinkedCell | str | None = None
    transition_config: Any | None = None
    transition_mode: str = "discrete"
    num_morph_layers: int = 2
    zones: Sequence[Any] | None = None
    thickness_gradient: Any | None = None
    gradient_axis: str | None = None
    gradient_radius_range: tuple[float, float] | None = None
    gradient_bounds: tuple[float, float] | None = None
    num_thickness_buckets: int = 16


@dataclass
class InterlinkedLatticeResult:
    """
    Output payload of the interlinked lattice generation pipeline.

    Attributes:
        mesh: Watertight multi-body trimesh.Trimesh containing all interlinked solid particles.
        rings: List of surviving Ring objects (for backwards compatibility).
        num_rings: Total number of surviving particles/rings in the multi-body mesh.
        num_nodes: Total number of explicit skeleton vertices across all particles.
        num_struts: Total number of explicit skeleton edges across all particles.
        min_clearance: Evaluated minimum surface clearance in mm.
        clearance_valid: True if min_clearance >= config.min_clearance.
        volume: Solid material volume in mm^3.
        bounds: (2, 3) min and max bounding box coordinates in mm.
        metadata: Diagnostic and summary dictionary.
        particles: List of surviving InterlinkedParticle instances.
        frame_mesh: Optional solid perimeter boundary frame mesh (Policy C).
        wire_radius: Strut/wire radius in mm.
    """
    mesh: trimesh.Trimesh
    rings: list[Ring]
    num_rings: int
    num_nodes: int
    num_struts: int
    min_clearance: float
    clearance_valid: bool
    volume: float
    bounds: np.ndarray
    metadata: dict = field(default_factory=dict)
    particles: list[InterlinkedParticle] = field(default_factory=list)
    frame_mesh: trimesh.Trimesh | None = None
    wire_radius: float = 0.40

    def export_stl(self, path: str | Path, include_frame: bool = True) -> Path:
        """
        Export multi-body lattice mesh as a monolithic STL file.

        Args:
            path: Destination file path.
            include_frame: If True and frame_mesh exists, combines lattice with frame.

        Returns:
            Path to exported STL.
        """
        out_p = Path(path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        if include_frame and self.frame_mesh is not None:
            combined = trimesh.util.concatenate([self.mesh, self.frame_mesh])
            combined.export(str(out_p))
        else:
            self.mesh.export(str(out_p))
        return out_p

    def export_3mf(
        self,
        path: str | Path,
        circular_segments: int = 24,
        include_frame: bool = True,
    ) -> Path:
        """
        Export assembly as an instanced 3MF package (O(1) prototype storage in <resources>).

        Args:
            path: Destination .3mf file path.
            circular_segments: Discretization resolution for prototype mesh generation.
            include_frame: If True and frame_mesh exists, includes frame in 3MF build.

        Returns:
            Path to exported 3MF.
        """
        out_p = Path(path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        frame = self.frame_mesh if include_frame else None

        if self.particles:
            return export_interlinked_3mf(
                particles=self.particles,
                strut_radius=self.wire_radius,
                output_path=out_p,
                solid_frames=frame,
                circular_segments=circular_segments,
            )
        else:
            # Wrap legacy rings into InterlinkedParticle instances
            parts: list[InterlinkedParticle] = []
            for idx, r in enumerate(self.rings):
                g = ParticleGeometry(
                    nodes=r.nodes - r.center,
                    struts=r.struts.copy(),
                    geometry_type="ring",
                    bounding_radius=r.radius + r.wire_radius,
                    metadata={"major_radius": r.radius},
                )
                R_mat = _rotation_align_local_z_to_unit(r.normal)
                T = np.eye(4, dtype=np.float64)
                T[:3, :3] = R_mat
                T[:3, 3] = r.center
                p = InterlinkedParticle(
                    particle_id=idx,
                    geometry=g,
                    transform=T,
                    sublattice_id=r.tag,
                    cell_index=tuple(r.cell_index),
                )
                parts.append(p)
            return export_interlinked_3mf(
                particles=parts,
                strut_radius=self.wire_radius,
                output_path=out_p,
                solid_frames=frame,
                circular_segments=circular_segments,
            )


def _particles_to_combined_mesh(
    particles: Sequence[InterlinkedParticle],
    wire_radius: float,
    circular_segments: int = 24,
    num_thickness_buckets: int = 16,
) -> trimesh.Trimesh:
    """
    Combine particles into a single multi-body trimesh using prototype instancing.

    Reuses prototype mesh geometry at the origin, transforming and indexing faces
    in O(N) array operations. Supports variable per-particle wire radii via bucketing.
    """
    if not particles:
        return trimesh.Trimesh()

    proto_cache: dict[str, trimesh.Trimesh] = {}

    # Inspect per-particle wire radii and establish bucket centers
    radii_list = [p.effective_wire_radius(fallback=wire_radius) for p in particles]
    min_r, max_r = min(radii_list), max(radii_list)

    if abs(max_r - min_r) > 1e-4 and num_thickness_buckets > 1:
        bucket_centers = np.linspace(min_r, max_r, num_thickness_buckets)
    else:
        bucket_centers = np.array([0.5 * (min_r + max_r)])

    def _quantized_radius(r_val: float) -> float:
        idx = int(np.argmin(np.abs(bucket_centers - r_val)))
        return float(bucket_centers[idx])

    def _key(p: InterlinkedParticle, r_q: float) -> str:
        g = p.geometry
        return f"{g.geometry_type}_{len(g.nodes)}_{len(g.struts)}_{p.sublattice_id}_r{r_q:.4f}"

    transformed_vertices: list[np.ndarray] = []
    transformed_faces: list[np.ndarray] = []
    vertex_offset = 0

    for p in particles:
        p_r = p.effective_wire_radius(fallback=wire_radius)
        r_bucket = _quantized_radius(p_r)
        k = _key(p, r_bucket)
        if k not in proto_cache:
            proto_cache[k] = solidify_particle_prototype(
                p.geometry,
                strut_radius=r_bucket,
                circular_segments=circular_segments,
            )
        base_mesh = proto_cache[k]

        # Rigid homogeneous transform
        V_homo = np.hstack([base_mesh.vertices, np.ones((len(base_mesh.vertices), 1), dtype=np.float64)])
        V_trans = (p.transform @ V_homo.T)[:3].T
        F_trans = base_mesh.faces + vertex_offset

        transformed_vertices.append(V_trans)
        transformed_faces.append(F_trans)
        vertex_offset += len(base_mesh.vertices)

    all_v = np.vstack(transformed_vertices)
    all_f = np.vstack(transformed_faces)
    return trimesh.Trimesh(vertices=all_v, faces=all_f, process=False)



def _generate_epitaxial_transition_lattice(
    config: InterlinkedConfig,
    check_clearance: bool = True,
) -> InterlinkedLatticeResult:
    """
    Synthesize a multi-zone epitaxial transition lattice bridging two distinct cell types.
    """
    # 1. Resolve Cell A
    if config.cell is not None:
        if isinstance(config.cell, str):
            cell_cls_a = InterlinkedRegistry.get(config.cell)
            cell_a = cell_cls_a() if callable(cell_cls_a) else cell_cls_a
        elif isinstance(config.cell, InterlinkedCell):
            cell_a = config.cell
        elif callable(config.cell):
            cell_a = config.cell()
        else:
            raise TypeError(f"Unsupported cell parameter type: {type(config.cell)}")
    else:
        cell_cls_a = InterlinkedRegistry.get(config.pattern)
        cell_a = cell_cls_a() if callable(cell_cls_a) else cell_cls_a

    # 2. Resolve Cell B
    cell_b_raw = config.transition_to_cell
    if isinstance(cell_b_raw, str):
        cell_cls_b = InterlinkedRegistry.get(cell_b_raw)
        cell_b = cell_cls_b() if callable(cell_cls_b) else cell_cls_b
    elif isinstance(cell_b_raw, InterlinkedCell):
        cell_b = cell_b_raw
    elif callable(cell_b_raw):
        cell_b = cell_b_raw()
    else:
        raise TypeError(f"Unsupported transition_to_cell type: {type(cell_b_raw)}")

    wire_r = float(config.wire_radius)
    min_clr = float(config.min_clearance)
    pitch_a = float(config.pitch) if config.pitch is not None else 9.10
    pitch_b = float(np.sqrt(2.0) * pitch_a)

    # 3. Solve Epitaxial Alignment via the Three Laws
    alignment = solve_epitaxial_transition(
        cell_a=cell_a,
        cell_b=cell_b,
        pitch_a=pitch_a,
        pitch_b=pitch_b,
        wire_radius=wire_r,
        min_clearance=min_clr,
    )

    # 4. Instantiate Multi-Zone Particles
    grid_a = config.grid_size  # (nx, ny, nz)
    grid_b = config.grid_size
    dx_int = float(alignment.interface_offset)
    R_habit = alignment.rotation_matrix

    particles: list[InterlinkedParticle] = []
    pid = 0

    # Zone 1: Cell A (Simple Cubic, nx layers along -X)
    for ix in range(-grid_a[0], 0):
        for iy in range(grid_a[1]):
            for iz in range(grid_a[2]):
                origin = np.array([(ix + 1) * pitch_a - dx_int, iy * pitch_a, iz * pitch_a], dtype=np.float64)
                parts_site = cell_a.instantiate_site((ix, iy, iz), origin, pitch_a, id_start=pid)
                for p in parts_site:
                    p.metadata["zone"] = "Zone_A"
                    p.metadata["layer_x"] = ix
                    particles.append(p)
                    pid += 1

    # Zone 3: Cell B (Diamond Cubic, extending along +X)
    from .pams import _tet_ab_templates, _TET_STRUTS

    L_tet = 0.603 * pitch_b
    proto_a = ParticleGeometry(nodes=_tet_ab_templates(L_tet)[0], struts=_TET_STRUTS.copy(), bounding_radius=L_tet, geometry_type="TET")
    proto_b = ParticleGeometry(nodes=_tet_ab_templates(L_tet)[1], struts=_TET_STRUTS.copy(), bounding_radius=L_tet, geometry_type="TET")

    ny_a, nz_a = grid_a[1], grid_a[2]
    w_y = (ny_a - 1) * pitch_a
    w_z = (nz_a - 1) * pitch_a
    margin = 1.0  # Buffer margin to capture boundary sites
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
                                metadata={"zone": "Zone_B", "layer_x": int(pos_rot[0] / (pitch_b / 4.0))},
                            )
                            particles.append(p)
                            pid += 1

    # 5. Spatial Strut Thickness Grading
    from .gradient import resolve_thickness_gradient, LinearThicknessGradient
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
            p.wire_radius = float(gradient.evaluate(p.center))

    strut_r_input: float | list[float] = (
        [p.effective_wire_radius(fallback=wire_r) for p in particles]
        if gradient is not None
        else wire_r
    )

    # 6. Clearance Verification
    clearance_valid = True
    min_clr_found = float("inf")
    clr_report = []
    if check_clearance and len(particles) > 1:
        clearance_valid, min_clr_found, clr_report = check_particle_clearance(
            particles,
            min_clearance=min_clr,
            strut_radius=strut_r_input,
        )

    # 7. Combined Mesh
    mesh = _particles_to_combined_mesh(
        particles,
        wire_radius=wire_r,
        circular_segments=config.num_ring_segments,
        num_thickness_buckets=config.num_thickness_buckets,
    )

    volume = float(mesh.volume) if hasattr(mesh, "volume") else 0.0
    bounds = np.asarray(mesh.bounds, dtype=np.float64) if len(mesh.vertices) > 0 else np.zeros((2, 3))

    metadata = {
        "transition_type": "epitaxial",
        "cell_a": getattr(cell_a, "name", str(cell_a)),
        "cell_b": getattr(cell_b, "name", str(cell_b)),
        "alignment": alignment,
        "pitch_a": pitch_a,
        "pitch_b": pitch_b,
        "wire_radius": wire_r,
        "min_clearance_target": min_clr,
        "min_clearance_found": min_clr_found,
        "clearance_valid": clearance_valid,
        "num_particles": len(particles),
    }

    return InterlinkedLatticeResult(
        mesh=mesh,
        rings=[],
        num_rings=len(particles),
        num_nodes=sum(len(p.geometry.nodes) for p in particles),
        num_struts=sum(len(p.geometry.struts) for p in particles),
        min_clearance=min_clr_found,
        clearance_valid=clearance_valid,
        volume=volume,
        bounds=bounds,
        metadata=metadata,
        particles=particles,
        wire_radius=wire_r,
    )


def generate_interlinked_lattice(
    config: InterlinkedConfig | None = None,
    boundary_mesh: trimesh.Trimesh | None = None,
    sdf_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    check_clearance: bool = True,
    **kwargs,
) -> InterlinkedLatticeResult:
    """
    Generate an explicit multi-body interlinked lattice mesh.
    """
    if config is None:
        config = InterlinkedConfig(**kwargs)
    elif kwargs:
        config_dict = {
            k: getattr(config, k)
            for k in config.__dataclass_fields__
        }
        config_dict.update(kwargs)
        config = InterlinkedConfig(**config_dict)

    # Check for multi-zone superlattice delegation
    if config.zones is not None:
        from .superlattice import SuperlatticeConfig, generate_epitaxial_superlattice
        sl_cfg = SuperlatticeConfig(
            zones=list(config.zones),
            grid_transverse=(config.grid_size[1], config.grid_size[2]),
            wire_radius=config.wire_radius,
            min_clearance=config.min_clearance,
            thickness_gradient=config.thickness_gradient,
            gradient_axis=config.gradient_axis,
            gradient_radius_range=config.gradient_radius_range,
            gradient_bounds=config.gradient_bounds,
            num_thickness_buckets=config.num_thickness_buckets,
        )
        return generate_epitaxial_superlattice(sl_cfg, check_clearance=check_clearance)

    # Check for multi-zone epitaxial transition delegation
    if config.transition_to_cell is not None or config.transition_config is not None:
        if getattr(config, "transition_mode", "discrete").lower() == "morph":
            from .morphing import EpitaxialMorphConfig, generate_epitaxial_morph_lattice
            m_cfg = EpitaxialMorphConfig(
                cell_a=config.cell or config.pattern,
                cell_b=config.transition_to_cell,
                grid_size_a=config.grid_size,
                grid_size_b=config.grid_size,
                num_morph_layers=config.num_morph_layers,
                pitch=config.pitch,
                wire_radius=config.wire_radius,
                min_clearance=config.min_clearance,
                thickness_gradient=config.thickness_gradient,
                gradient_axis=config.gradient_axis,
                gradient_radius_range=config.gradient_radius_range,
                gradient_bounds=config.gradient_bounds,
                num_thickness_buckets=config.num_thickness_buckets,
            )
            return generate_epitaxial_morph_lattice(m_cfg, check_clearance=check_clearance)
        return _generate_epitaxial_transition_lattice(config, check_clearance=check_clearance)

    pattern_raw = config.pattern.lower().strip()
    use_modular = (config.cell is not None) or (pattern_raw not in _LEGACY_PATTERNS)

    if use_modular:
        # =========================================================================
        # Modular Unit-Cell Pipeline (Phase 1–5 Architecture)
        # =========================================================================
        if config.cell is not None:
            if isinstance(config.cell, str):
                cell_cls = InterlinkedRegistry.get(config.cell)
                cell: InterlinkedCell = cell_cls() if callable(cell_cls) else cell_cls
            elif isinstance(config.cell, InterlinkedCell):
                cell = config.cell
            elif callable(config.cell):
                cell = config.cell()
            else:
                raise TypeError(f"Unsupported cell parameter type: {type(config.cell)}")
        else:
            cell_cls = InterlinkedRegistry.get(config.pattern)
            cell = cell_cls() if callable(cell_cls) else cell_cls

        # Pitch resolution
        if config.auto_resolve_pitch:
            pitch = cell.resolve_pitch(
                target_clearance=config.min_clearance,
                strut_diameter=2.0 * config.wire_radius,
            )
        else:
            pitch = float(config.pitch)

        wire_r = float(config.wire_radius)
        strut_d = 2.0 * wire_r

        # DfAM Pre-Flight Guardrail: analytical check before spatial allocation
        try:
            pred_clr = cell.forward_clearance(pitch, strut_d)
            if pred_clr < 0.0:
                min_p = cell.resolve_pitch(0.0, strut_d)
                raise ValueError(
                    f"DfAM Pre-Flight Rejection: Cell '{getattr(cell, 'cell_name', getattr(cell, 'name', 'unknown'))}' "
                    f"at pitch={pitch:.3f} mm and wire_radius={wire_r:.3f} mm has negative predicted clearance "
                    f"({pred_clr:.3f} mm). Components will physically collide and fuse. Minimum pitch required "
                    f"for zero collision: {min_p:.3f} mm."
                )
            if check_clearance and pred_clr < float(config.min_clearance) - 1e-6:
                import warnings
                warnings.warn(
                    f"DfAM Clearance Warning: Cell '{getattr(cell, 'cell_name', getattr(cell, 'name', 'unknown'))}' "
                    f"at pitch={pitch:.3f} mm has predicted clearance ({pred_clr:.3f} mm) below requested "
                    f"min_clearance ({config.min_clearance:.3f} mm).",
                    UserWarning,
                    stacklevel=2,
                )
        except (AttributeError, NotImplementedError):
            pass

        seeding_type = config.seeding_type.lower().strip()

        # Spatial Seeding
        sublattice_ids = None
        if seeding_type in ("cylindrical", "cylinder", "cylindrical_wrap"):
            R = config.cylinder_radius if config.cylinder_radius is not None else 20.0
            H = config.cylinder_height if config.cylinder_height is not None else 30.0
            cyl_dict = seed_cylindrical_wrap(
                radius=R,
                height=H,
                target_pitch=pitch,
                origin=config.origin,
            )
            centers = cyl_dict["centers"]
            frames = cyl_dict["frames"]
            indices = cyl_dict["indices"]
        elif seeding_type in ("spherical", "sphere", "spherical_shell"):
            R = config.sphere_radius if config.sphere_radius is not None else 20.0
            sph_dict = seed_spherical_shell(
                radius=R,
                target_pitch=pitch,
                origin=config.origin,
            )
            centers = sph_dict["centers"]
            frames = sph_dict["frames"]
            indices = sph_dict["indices"]
        elif seeding_type in ("diamond", "cubic_diamond"):
            centers, frames, indices, sublattice_ids = seed_diamond_lattice(
                repeats=config.grid_size,
                conventional_cell_size=pitch,
                origin=config.origin,
            )
        elif seeding_type in ("staggered", "brick", "bcc_staggered"):
            centers, frames, indices = seed_staggered_lattice(
                repeats=config.grid_size,
                pitch=pitch,
                origin=config.origin,
            )
        elif seeding_type in ("hexagonal", "hcp", "hex"):
            centers, frames, indices = seed_hexagonal_lattice(
                repeats=config.grid_size,
                pitch=pitch,
                origin=config.origin,
            )
        else:
            # Default: Cartesian grid
            centers, frames, indices = seed_cartesian_lattice(
                repeats=config.grid_size,
                pitch=pitch,
                origin=config.origin,
            )

        # Instantiate particles on sites
        particles = instantiate_lattice_on_sites(
            cell=cell,
            centers=centers,
            frames=frames,
            grid_indices=indices,
            cell_pitch=pitch,
            sublattice_ids=sublattice_ids,
        )

        # Boundary Culling (Policy A)
        culled_count = 0
        if boundary_mesh is not None:
            particles, culled = cull_particles_by_mesh(
                particles, boundary_mesh, margin=config.cull_margin
            )
            culled_count = len(culled)
        elif sdf_fn is not None:
            particles, culled = cull_particles_by_sdf(
                particles, sdf_fn, margin=config.cull_margin
            )
            culled_count = len(culled)

        if not particles:
            raise ValueError(
                "All candidate particles were culled by the domain boundary. "
                "Increase domain size or reduce cell pitch / cull_margin."
            )

        # Apply support recipe if requested
        if config.support_recipe and str(config.support_recipe).lower() not in ("none", "unprinted", "", "none (unprinted / free)"):
            from .support_recipes import apply_support_recipe
            for p in particles:
                p.geometry = apply_support_recipe(p.geometry, config.support_recipe, wire_radius=wire_r)

        # Boundary Perimeter Framing (Policy C)
        frame_mesh: trimesh.Trimesh | None = None
        if config.add_perimeter_frame:
            frame_mesh = build_perimeter_frame_solid(
                particles,
                wall_thickness=config.frame_wall_thickness,
                margin=config.frame_margin,
                frame_shape=config.frame_shape,
            )

        # Spatial Strut Thickness Grading
        from .gradient import resolve_thickness_gradient, LinearThicknessGradient
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
                p.wire_radius = float(gradient.evaluate(p.center))

        strut_r_input: float | list[float] = (
            [p.effective_wire_radius(fallback=wire_r) for p in particles]
            if gradient is not None
            else wire_r
        )

        # Clearance Verification
        min_clr = float("inf")
        clearance_valid = True
        clr_report: list[dict[str, Any]] = []
        if check_clearance and len(particles) > 1:
            clearance_valid, min_clr, clr_report = check_particle_clearance(
                particles,
                min_clearance=config.min_clearance,
                strut_radius=strut_r_input,
            )

        # Multi-body solid mesh generation
        mesh = _particles_to_combined_mesh(
            particles,
            wire_radius=wire_r,
            circular_segments=config.num_ring_segments,
            num_thickness_buckets=config.num_thickness_buckets,
        )

        total_nodes = sum(len(p.geometry.nodes) for p in particles)
        total_struts = sum(len(p.geometry.struts) for p in particles)

        # Backwards compatibility: populate rings if applicable
        rings: list[Ring] = []
        for p in particles:
            g = p.geometry
            if g.geometry_type.lower() == "ring" or "major_radius" in g.metadata:
                r_maj = float(g.metadata.get("major_radius", g.bounding_radius))
                r_obj = Ring(
                    nodes=p.global_nodes(),
                    struts=p.geometry.struts.copy(),
                    center=p.center,
                    normal=p.transform[:3, 2],
                    radius=r_maj,
                    wire_radius=wire_r,
                    tag=p.sublattice_id,
                    cell_index=tuple(p.metadata.get("grid_index", (0, 0, 0))),
                )
                rings.append(r_obj)

        volume = float(mesh.volume) if hasattr(mesh, "volume") else 0.0
        bounds = np.asarray(mesh.bounds, dtype=np.float64) if len(mesh.vertices) > 0 else np.zeros((2, 3))

        metadata = {
            "cell_name": getattr(cell, "name", str(config.cell or config.pattern)),
            "seeding_type": seeding_type,
            "pitch": pitch,
            "wire_radius": wire_r,
            "surviving_particles": len(particles),
            "culled_particles": culled_count,
            "clearance_valid": clearance_valid,
            "min_clearance": min_clr,
            "has_frame": frame_mesh is not None,
            "is_watertight": bool(mesh.is_watertight),
            "num_vertices": len(mesh.vertices),
            "num_faces": len(mesh.faces),
            "clearance_report": clr_report,
        }

        return InterlinkedLatticeResult(
            mesh=mesh,
            rings=rings,
            num_rings=len(particles),
            num_nodes=total_nodes,
            num_struts=total_struts,
            min_clearance=min_clr,
            clearance_valid=clearance_valid,
            volume=volume,
            bounds=bounds,
            metadata=metadata,
            particles=particles,
            frame_mesh=frame_mesh,
            wire_radius=wire_r,
        )

    # =========================================================================
    # Legacy Ring Generator Pipeline (Backwards Compatibility)
    # =========================================================================
    pattern = pattern_raw
    grid_size = config.grid_size
    pitch = float(config.pitch)
    wire_r = float(config.wire_radius)
    n_seg = int(config.num_ring_segments)
    orig = config.origin

    if pattern in ("european_4in1", "european", "euro"):
        rings = generate_european_4in1_rings(
            grid_size=grid_size,
            pitch=pitch,
            radius_ratio=config.radius_ratio,
            wire_radius=wire_r,
            tilt_angle_deg=config.tilt_angle_deg,
            num_segments=n_seg,
            origin=orig,
        )
    elif pattern in ("kusari", "japanese_kusari", "japanese"):
        rf = config.flat_radius if config.flat_radius is not None else 0.36 * pitch
        ra = config.arch_radius if config.arch_radius is not None else 0.41 * pitch
        rings = generate_kusari_rings(
            grid_size=grid_size,
            pitch=pitch,
            flat_radius=rf,
            arch_radius=ra,
            wire_radius=wire_r,
            num_segments=n_seg,
            origin=orig,
        )
    elif pattern in ("cubic_8ring", "cube_2x2x2", "cube"):
        r_cube = config.flat_radius if config.flat_radius is not None else 0.6375 * pitch
        rings = generate_cubic_8ring(
            pitch=pitch,
            radius=r_cube,
            wire_radius=wire_r,
            num_segments=n_seg,
            origin=orig,
        )
    elif pattern in ("volumetric_kusari", "volumetric_cube", "kusari_3d"):
        rf = config.flat_radius if config.flat_radius is not None else 0.36 * pitch
        ra = config.arch_radius if config.arch_radius is not None else 0.41 * pitch
        rz = config.arch_z_radius if config.arch_z_radius is not None else 0.54 * pitch
        rings = generate_volumetric_kusari_rings(
            grid_size=grid_size,
            pitch=pitch,
            flat_radius=rf,
            arch_radius=ra,
            arch_z_radius=rz,
            wire_radius=wire_r,
            num_segments=n_seg,
            origin=orig,
        )
    else:
        raise ValueError(
            f"Unknown interlinked pattern '{config.pattern}'. "
            f"Supported patterns: 'european_4in1', 'kusari', 'cubic_8ring', 'volumetric_kusari', "
            f"or specify a registered cell via cell='c6tt', 'd4tet', 'j4oct', etc."
        )

    # Inset Culling
    culled_count = 0
    if boundary_mesh is not None:
        rings, culled = cull_rings_by_mesh(rings, boundary_mesh, margin=config.cull_margin)
        culled_count = len(culled)
    elif sdf_fn is not None:
        rings, culled = cull_rings_by_sdf(rings, sdf_fn, margin=config.cull_margin)
        culled_count = len(culled)

    if not rings:
        raise ValueError(
            "All candidate rings were culled by the domain boundary. "
            "Increase domain size or reduce ring radius / cull_margin."
        )

    # Clearance Verification
    min_clr = float("inf")
    clearance_valid = True
    violations = []
    if check_clearance and len(rings) > 1:
        clearance_valid, min_clr, violations = check_ring_clearance(
            rings,
            min_clearance=config.min_clearance,
        )

    # Global Skeleton Assembly
    node_list = []
    strut_list = []
    radii_list = []
    offset = 0

    for ring in rings:
        node_list.append(ring.nodes)
        strut_list.append(ring.struts + offset)
        radii_list.append(np.full(len(ring.struts), ring.wire_radius, dtype=np.float64))
        offset += len(ring.nodes)

    all_nodes = np.vstack(node_list)
    all_struts = np.vstack(strut_list)
    all_radii = np.concatenate(radii_list)

    # Watertight Solid Geometry Generation
    cross_segs = max(16, int(config.num_ring_segments // 2))
    revolve_segs = max(32, int(config.num_ring_segments * 2))

    torus_solids: list[m3d.Manifold] = []
    for ring in rings:
        c_circle = m3d.CrossSection.circle(
            float(ring.wire_radius),
            circular_segments=cross_segs,
        ).translate([float(ring.radius), 0.0])
        torus = m3d.Manifold.revolve(c_circle, circular_segments=revolve_segs)
        R_mat = _rotation_align_local_z_to_unit(ring.normal)
        aff = _affine_rows_from_R_t(R_mat, ring.center)
        torus_solids.append(torus.transform(aff))

    united_manifold = m3d.Manifold.compose(torus_solids)
    mesh = _manifold_to_trimesh(united_manifold)

    # Build InterlinkedParticle representations for downstream 3MF/frame support
    parts: list[InterlinkedParticle] = []
    for idx, r in enumerate(rings):
        g = ParticleGeometry(
            nodes=r.nodes - r.center,
            struts=r.struts.copy(),
            geometry_type="ring",
            bounding_radius=r.radius + r.wire_radius,
            metadata={"major_radius": r.radius},
        )
        R_mat = _rotation_align_local_z_to_unit(r.normal)
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R_mat
        T[:3, 3] = r.center
        p = InterlinkedParticle(
            particle_id=idx,
            geometry=g,
            transform=T,
            sublattice_id=r.tag,
            cell_index=tuple(r.cell_index),
        )
        parts.append(p)

    frame_mesh: trimesh.Trimesh | None = None
    if config.add_perimeter_frame:
        frame_mesh = build_perimeter_frame_solid(
            parts,
            wall_thickness=config.frame_wall_thickness,
            margin=config.frame_margin,
            frame_shape=config.frame_shape,
        )

    volume = float(mesh.volume) if hasattr(mesh, "volume") else 0.0
    bounds = np.asarray(mesh.bounds, dtype=np.float64)

    metadata = {
        "pattern": config.pattern,
        "grid_size": list(config.grid_size),
        "pitch": config.pitch,
        "wire_radius": config.wire_radius,
        "surviving_rings": len(rings),
        "culled_rings": culled_count,
        "violations_count": len(violations),
        "is_watertight": bool(mesh.is_watertight),
        "num_faces": len(mesh.faces),
        "num_vertices": len(mesh.vertices),
    }

    return InterlinkedLatticeResult(
        mesh=mesh,
        rings=rings,
        num_rings=len(rings),
        num_nodes=len(all_nodes),
        num_struts=len(all_struts),
        min_clearance=min_clr,
        clearance_valid=clearance_valid,
        volume=volume,
        bounds=bounds,
        metadata=metadata,
        particles=parts,
        frame_mesh=frame_mesh,
        wire_radius=wire_r,
    )
