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
from .dimensioning import PAMDimensioner, DEFAULT_WARNING_CLEARANCE
from .cells import (  # ensure cell implementations are loaded and registered
    CustomWireframeCell,
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
    seed_orthogonal_lattice,
    orthogonal_supercell,
    seed_bcc_lattice,
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
class SiteProps:
    """Per-site particle properties applied during spatial assembly."""
    wire_radius: float | None = None
    size_ratio: float | None = None
    twist_deg: float | None = None
    rotation: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


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
        strict: If True, raises exceptions on negative clearance pre-flight or physical violations.
        site_props: Optional callable(pos: np.ndarray) -> dict | SiteProps evaluating per-site properties.
    """
    pattern: str = "european_4in1"
    cell: InterlinkedCell | str | tuple | dict | None = None
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
    habit_plane: str = "100"
    transition_to_cell: InterlinkedCell | str | None = None
    transition_config: Any | None = None
    transition_mode: str = "discrete"
    auto_buffer: bool = True
    buffer_layers_per_zone: int = 2
    enable_strut_taper: bool = True
    strut_taper_gamma: float = 0.70
    cage_shrink_eta: float = 0.90
    num_morph_layers: int = 2
    zones: Sequence[Any] | None = None
    thickness_gradient: Any | None = None
    gradient_axis: str | None = None
    gradient_radius_range: tuple[float, float] | None = None
    gradient_bounds: tuple[float, float] | None = None
    num_thickness_buckets: int = 16
    rotation_matrix: Any | None = None
    orientation: Any | None = None
    euler_angles_deg: Sequence[float] | None = None
    target_solid_fraction: float | None = None
    target_clearance: float | None = None
    warning_clearance: float = DEFAULT_WARNING_CLEARANCE
    strict: bool = False
    site_props: Callable[[np.ndarray], Any] | None = None


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
        color_by_status: bool = True,
        status_colors: dict[str, str] | None = None,
    ) -> Path:
        """
        Export assembly as an instanced 3MF package (O(1) prototype storage in <resources>).

        Args:
            path: Destination .3mf file path.
            circular_segments: Discretization resolution for prototype mesh generation.
            include_frame: If True and frame_mesh exists, includes frame in 3MF build.
            color_by_status: If True, assign color groups by particle status (Zone, Colliding, etc.).
            status_colors: Optional override dictionary mapping status names to RGBA hex strings.

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
                color_by_status=color_by_status,
                status_colors=status_colors,
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

    def export(self, path: str | Path, **kwargs: Any) -> Path:
        """
        Universal export method. Automatically selects instanced 3MF or monolithic STL
        based on the file extension (.3mf or .stl).
        """
        p = Path(path)
        if p.suffix.lower() == ".3mf":
            return self.export_3mf(p, **kwargs)
        return self.export_stl(p, **kwargs)

    def render(
        self,
        path: str | Path,
        cutaway: bool = False,
        theme: str = "light",
        title: str | None = None,
        subtitle: str | None = None,
        window_size: tuple[int, int] = (1920, 1080),
    ) -> Path:
        """
        Render a publication-grade matte scientific figure of the generated lattice.

        Args:
            path: Destination PNG file path.
            cutaway: If True, crops out half the domain along the X midplane to reveal interior links.
            theme: 'light', 'neutral', or 'dark'.
            title: Custom title header (defaults to cell/pattern name).
            subtitle: Custom subtitle (defaults to strut diameter & clearance metrics).
            window_size: (width, height) pixel resolution.

        Returns:
            Path to rendered PNG.
        """
        from .viz import render_matte_gallery_specimen
        out_p = Path(path)
        effective_title = title or f"{self.metadata.get('cell_name', 'Interlinked Lattice').upper()}"
        effective_sub = subtitle or (
            f"Strut: {2.0 * self.wire_radius * 1000.0:.0f} um | "
            f"Clearance: {self.min_clearance * 1000.0:.1f} um | "
            f"Links: {self.num_rings}"
        )
        parts = self.particles
        if not parts and self.rings:
            parts = []
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
        return render_matte_gallery_specimen(
            particles=parts,
            out_png=out_p,
            title=effective_title,
            subtitle=effective_sub,
            cutaway=cutaway,
            theme=theme,
            window_size=window_size,
        )

    def summary(self) -> str:
        """Return a human-readable diagnostic summary of the generated lattice."""
        bounds_span = self.bounds[1] - self.bounds[0] if len(self.bounds) == 2 else np.zeros(3)
        cell_name = self.metadata.get("cell_name", "interlinked")
        pitch = self.metadata.get("pitch", None)
        dim = self.metadata.get("dimensioning")

        lines = [
            f"=== Interlinked Lattice Summary ===",
            f"  Architecture:       {cell_name}",
        ]
        if pitch is not None:
            lines.append(f"  Unit Cell Pitch:    {float(pitch):.4f} mm")
        if dim is not None and hasattr(dim, "cage_diameter") and dim.cage_diameter > 0.0:
            lines.append(f"  Cage Outer Dia (D): {dim.cage_diameter:.4f} mm")
        if dim is not None and hasattr(dim, "solid_fraction_percent"):
            lines.append(f"  Solid Density (Vf): {dim.solid_fraction_percent:.2f}% (Jamming Limit: {dim.max_solid_fraction_percent:.2f}%)")

        lines.extend([
            f"  Particles / Rings:  {self.num_rings}",
            f"  Explicit Vertices:  {self.num_nodes}",
            f"  Explicit Struts:    {self.num_struts}",
            f"  Strut Wire Radius:  {self.wire_radius:.3f} mm (Diameter: {2.0 * self.wire_radius:.3f} mm)",
            f"  Minimum Clearance:  {self.min_clearance * 1000.0:.1f} um ({self.min_clearance:.4f} mm)",
            f"  Clearance Valid:    {'YES' if self.clearance_valid else 'NO'}",
            f"  Bounding Envelope:  {bounds_span[0]:.1f} x {bounds_span[1]:.1f} x {bounds_span[2]:.1f} mm",
            f"  Solid Mesh Faces:   {len(self.mesh.faces) if self.mesh is not None else 0}",
        ])
        if self.frame_mesh is not None:
            lines.append(f"  Handling Frame:     Attached (Solid CAD Perimeter)")
        return "\n".join(lines)


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
        if np.linalg.det(p.transform[:3, :3]) < 0:
            F_trans = base_mesh.faces[:, [0, 2, 1]] + vertex_offset
        else:
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

    net_a = getattr(cell_a, "parent_network", "pcu").lower().strip()
    net_b = getattr(cell_b, "parent_network", "pcu").lower().strip()
    if net_a in ("pcu", "bcc") and net_b == "dia":
        pitch_b = float(np.sqrt(2.0) * pitch_a)
    elif net_a == "dia" and net_b in ("pcu", "bcc"):
        pitch_b = float(pitch_a / np.sqrt(2.0))
    else:
        pitch_b = float(pitch_a)

    # 3. Resolve InterfaceSpec
    h_plane = getattr(config, "habit_plane", "100")
    tc = config.transition_config
    if tc is not None:
        from .assemble import InterfaceSpec
        if isinstance(tc, InterfaceSpec):
            iface = tc
        elif isinstance(tc, dict):
            iface = InterfaceSpec(**tc)
        else:
            nominal_gap = float(0.5 * (pitch_a + pitch_b))
            iface = InterfaceSpec(normal=h_plane, gap=nominal_gap)
    else:
        from .assemble import InterfaceSpec
        nominal_gap = float(0.5 * (pitch_a + pitch_b))
        iface = InterfaceSpec(
            normal=h_plane,
            gap=nominal_gap,
            shift=np.zeros(3, dtype=np.float64),
            twist_deg=0.0,
        )

    # 4. Construct ZoneSpecs
    from .assemble import ZoneSpec, assemble
    zone_a = ZoneSpec(
        cell=cell_a,
        network=net_a,
        pitch=pitch_a,
        grid_size=config.grid_size,
        frame=h_plane,
        name="Zone_A",
        wire_radius=wire_r,
        site_props=config.site_props,
    )
    zone_b = ZoneSpec(
        cell=cell_b,
        network=net_b,
        pitch=pitch_b,
        grid_size=config.grid_size,
        frame=h_plane,
        name="Zone_B",
        wire_radius=wire_r,
        site_props=config.site_props,
    )

    # 5. Permissive Multi-Zone Assembly (zero numerical solver loops)
    res = assemble(
        zones=[zone_a, zone_b],
        interfaces=[iface],
        wire_radius=wire_r,
        min_clearance=min_clr,
        check_clearance=check_clearance,
        circular_segments=config.num_ring_segments,
        num_thickness_buckets=config.num_thickness_buckets,
        strict=config.strict,
    )
    res.metadata["transition_type"] = "epitaxial"
    res.metadata["cell_a"] = cell_a.name
    res.metadata["cell_b"] = cell_b.name
    res.metadata["habit_plane"] = h_plane
    return res


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

    # Check for multi-zone assemble or superlattice delegation
    if config.zones is not None:
        if len(config.zones) > 0 and hasattr(config.zones[0], "cell") and hasattr(config.zones[0], "network"):
            from .assemble import assemble, InterfaceSpec
            ifaces = [config.transition_config] if isinstance(config.transition_config, InterfaceSpec) else None
            return assemble(
                zones=list(config.zones),
                interfaces=ifaces,
                wire_radius=config.wire_radius,
                min_clearance=config.min_clearance,
                check_clearance=check_clearance,
                circular_segments=config.num_ring_segments,
                num_thickness_buckets=config.num_thickness_buckets,
                strict=config.strict,
            )
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
            elif isinstance(config.cell, (tuple, list)) and len(config.cell) == 2:
                cell = CustomWireframeCell(nodes=config.cell[0], struts=config.cell[1])
            elif isinstance(config.cell, dict):
                cell = CustomWireframeCell(recipe=config.cell)
            elif callable(config.cell):
                cell = config.cell()
            else:
                raise TypeError(f"Unsupported cell parameter type: {type(config.cell)}")
        else:
            cell_cls = InterlinkedRegistry.get(config.pattern)
            cell = cell_cls() if callable(cell_cls) else cell_cls

        # Pitch and Wire Dimension resolution
        dimensioning_result = None
        if config.target_solid_fraction is not None:
            clr_target = config.target_clearance if config.target_clearance is not None else config.min_clearance
            dim_res = PAMDimensioner.solve_from_clearance_and_vf(
                cell=cell,
                clearance=clr_target,
                solid_fraction=config.target_solid_fraction,
                warning_clearance=config.warning_clearance,
                raise_on_jamming=True,
            )
            pitch = dim_res.pitch
            wire_r = dim_res.wire_radius
            strut_d = dim_res.wire_diameter
            dimensioning_result = dim_res
        elif config.auto_resolve_pitch:
            pitch = cell.resolve_pitch(
                target_clearance=config.min_clearance,
                strut_diameter=2.0 * config.wire_radius,
            )
            wire_r = float(config.wire_radius)
            strut_d = 2.0 * wire_r
        else:
            pitch = float(config.pitch)
            wire_r = float(config.wire_radius)
            strut_d = 2.0 * wire_r

        # DfAM Pre-Flight Guardrail: analytical check before spatial allocation
        preflight_warning = None
        try:
            pred_clr = cell.forward_clearance(pitch, strut_d)
            if pred_clr < 0.0:
                min_p = cell.resolve_pitch(0.0, strut_d)
                msg = (
                    f"DfAM Pre-Flight Rejection: Cell '{getattr(cell, 'cell_name', getattr(cell, 'name', 'unknown'))}' "
                    f"at pitch={pitch:.3f} mm and wire_radius={wire_r:.3f} mm has negative predicted clearance "
                    f"({pred_clr:.3f} mm). Components will physically collide and fuse. Minimum pitch required "
                    f"for zero collision: {min_p:.3f} mm."
                )
                if config.strict:
                    raise ValueError(msg)
                else:
                    import warnings
                    warnings.warn(msg, UserWarning, stacklevel=2)
                    preflight_warning = msg
            elif check_clearance and pred_clr < float(config.min_clearance) - 1e-6:
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

        # Auto-select seeding topology from cell.preferred_seeding_type when caller
        # has not overridden seeding_type (i.e., it remains at the default 'cartesian').
        # This prevents silent failure modes where diamond-network cells produce zero
        # links because the wrong seeding topology was used.
        if seeding_type == "cartesian":
            cell_preferred = getattr(cell, "preferred_seeding_type", None)
            if cell_preferred and cell_preferred.lower() != "cartesian":
                import warnings
                warnings.warn(
                    f"InterlinkedConfig: seeding_type='cartesian' (default) but cell "
                    f"'{getattr(cell, 'name', str(cell))}' has preferred_seeding_type="
                    f"'{cell_preferred}'. Auto-selecting '{cell_preferred}'. "
                    f"Set seeding_type='{cell_preferred}' explicitly to silence this warning.",
                    UserWarning,
                    stacklevel=3,
                )
                seeding_type = cell_preferred.lower().strip()

        # Spatial Seeding
        sublattice_ids = None
        use_habit_plane = bool(config.habit_plane and str(config.habit_plane).strip() not in ("100", "{100}"))

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
        elif seeding_type in ("orthogonal", "orthogonal_supercell") or use_habit_plane:
            net = getattr(cell, "parent_network", "pcu")
            if seeding_type in ("diamond", "cubic_diamond"):
                net = "dia"
            elif seeding_type in ("bcc", "body_centered", "body_centered_cubic"):
                net = "bcc"
            elif seeding_type in ("cartesian", "pcu", "sc"):
                net = "pcu"
            centers, frames, indices, sublattice_ids = seed_orthogonal_lattice(
                repeats=config.grid_size,
                pitch=pitch,
                network=net,
                plane=config.habit_plane,
                origin=config.origin,
            )
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
        elif seeding_type in ("bcc", "body_centered", "body_centered_cubic"):
            centers, frames, indices, sublattice_ids = seed_bcc_lattice(
                repeats=config.grid_size,
                cell_size=pitch,
                origin=config.origin,
            )
        else:
            # Default: Cartesian grid
            centers, frames, indices = seed_cartesian_lattice(
                repeats=config.grid_size,
                pitch=pitch,
                origin=config.origin,
            )

        # Apply global lattice rotation if specified (rotation_matrix, orientation, or Euler angles)
        rot_spec = config.rotation_matrix if config.rotation_matrix is not None else config.orientation
        if rot_spec is not None or config.euler_angles_deg is not None:
            from .seeding import resolve_orientation_matrix
            R_rot = resolve_orientation_matrix(rot_spec, euler_angles_deg=config.euler_angles_deg)
            orig = np.asarray(config.origin, dtype=np.float64)
            centers = (R_rot @ (centers - orig).T).T + orig
            frames = np.array([R_rot @ f for f in frames], dtype=np.float64)

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

        # Apply per-site properties if configured
        if config.site_props is not None:
            for p in particles:
                props = config.site_props(p.center)
                if props is None:
                    continue
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
                    rad = np.radians(float(tw_deg))
                    c_th, s_th = np.cos(rad), np.sin(rad)
                    R_twist = np.array([
                        [c_th, -s_th, 0.0],
                        [s_th, c_th, 0.0],
                        [0.0, 0.0, 1.0],
                    ], dtype=np.float64)
                    p.rotation = R_twist @ p.rotation

        has_variable_radii = (gradient is not None) or any(p.wire_radius is not None for p in particles)
        strut_r_input: float | list[float] = (
            [p.effective_wire_radius(fallback=wire_r) for p in particles]
            if has_variable_radii
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
                    cell_index=tuple(p.metadata.get("grid_index", getattr(p, "cell_index", (0, 0, 0)))),
                )
                rings.append(r_obj)

        ring_clearance_info = None
        if check_clearance and len(rings) > 1:
            from .clearance import measure_ring_lattice_clearance
            ring_clearance_info = measure_ring_lattice_clearance(
                rings,
                min_clearance=config.min_clearance,
                check_linking=True,
            )
            min_clr = min(min_clr, ring_clearance_info["global_min_clearance"])
            clearance_valid = bool(min_clr >= config.min_clearance and len(ring_clearance_info["violations"]) == 0)

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
            "dimensioning": dimensioning_result,
            "ring_clearance": ring_clearance_info,
            "has_frame": frame_mesh is not None,
            "is_watertight": bool(mesh.is_watertight),
            "num_vertices": len(mesh.vertices),
            "num_faces": len(mesh.faces),
            "clearance_report": clr_report,
            "preflight_warning": preflight_warning,
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
