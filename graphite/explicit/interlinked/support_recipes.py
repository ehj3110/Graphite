"""
Graphite Explicit Interlinked — PAM Support Recipe Store & Custom Prototype Engine

Provides:
- get_available_support_recipes: Catalog of standard printability support recipes for PAM unit cells.
- export_seed_cell_stl: Seed cell exporter generating clean, isolated central prototypes.
- apply_support_recipe: Algorithmic synthesis of sacrificial support structures (e.g. 0.25mm vertical pin bridges).
- create_custom_supported_particle: Custom supported particle wrapper for seamless multi-body instancing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence, Any
import numpy as np
import trimesh
import manifold3d

from .particle import ParticleGeometry, InterlinkedParticle
from .cell import InterlinkedRegistry
from .generator import _particles_to_combined_mesh
from .writer_3mf import solidify_particle_prototype
from graphite.explicit.geometry_module import (
    build_clean_miter_truss,
    _manifold_cylinder_between,
    _trimesh_to_manifold,
    _manifold_to_trimesh,
)

DEFAULT_SUPPORT_RECIPES: list[str] = [
    "None (Unprinted / Free)",
    "Vertical Pin Bridges",
    "Base Plate Breakaway Pins",
]


def get_available_support_recipes(cell_key: str = "c6tt") -> list[str]:
    """
    Return available support recipe names for a given interlinked cell.

    Parameters
    ----------
    cell_key : str
        Canonical identifier of the interlinked cell (e.g. 'c6tt', 'd4tet').

    Returns
    -------
    list[str]
        Available support recipe names.
    """
    return list(DEFAULT_SUPPORT_RECIPES)


def export_seed_cell_stl(
    cell_key: str,
    pitch: float,
    wire_radius: float,
    output_path: Path | str,
) -> Path:
    """
    Generate a single central seed particle at origin [0, 0, 0] for `cell_key`
    using InterlinkedRegistry or cells.py, export it as a clean binary STL to
    `output_path`, and return the path.

    Parameters
    ----------
    cell_key : str
        Canonical cell identifier (e.g. 'c6tt', 'd4tet', 'euro_4in1').
    pitch : float
        Unit cell pitch / lattice spacing in mm.
    wire_radius : float
        Strut or wire cross-sectional radius in mm.
    output_path : Path | str
        Destination file path for binary STL export.

    Returns
    -------
    Path
        Path to the exported binary STL file.
    """
    cell_cls = InterlinkedRegistry.get(cell_key)
    cell = cell_cls()
    particles = cell.instantiate_site(
        grid_index=(0, 0, 0),
        site_origin=np.zeros(3, dtype=np.float64),
        cell_pitch=float(pitch),
        id_start=0,
    )
    if not particles:
        raise ValueError(f"Cell '{cell_key}' produced zero particles at origin site.")

    # Select the central seed particle (closest to [0, 0, 0])
    centers = np.array([p.center for p in particles], dtype=np.float64)
    dists = np.linalg.norm(centers, axis=1)
    seed_idx = int(np.argmin(dists))
    seed_particle = particles[seed_idx]

    # Solidify using standard prototype instancing
    mesh = _particles_to_combined_mesh([seed_particle], wire_radius=float(wire_radius))
    if mesh is None or len(mesh.faces) == 0:
        mesh = solidify_particle_prototype(seed_particle.geometry, strut_radius=float(wire_radius))
        if not np.allclose(seed_particle.transform[:3, :3], np.eye(3)):
            mesh.apply_transform(seed_particle.transform)

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(out_p), file_type="stl")
    return out_p


def apply_support_recipe(
    prototype: ParticleGeometry,
    recipe_name: str,
    wire_radius: float,
) -> ParticleGeometry:
    """
    Synthesize support structures onto a ParticleGeometry prototype according to a named recipe.

    Supported Recipes:
    - "Vertical Pin Bridges":
      Synthesizes sacrificial vertical pins (diameter 0.25 mm) connecting bottom-most
      nodes of the prototype to the bottom Z plane of the particle bounding box,
      returning a combined ParticleGeometry.
    - "None" or unrecognized:
      Returns the original prototype unchanged.

    Parameters
    ----------
    prototype : ParticleGeometry
        Canonical particle geometry prototype centered at local origin.
    recipe_name : str
        Name of the recipe to apply.
    wire_radius : float
        Wire or strut radius in mm.

    Returns
    -------
    ParticleGeometry
        Combined particle geometry with supports, or original prototype if unchanged.
    """
    norm_recipe = str(recipe_name).strip().lower().replace("_", " ")

    # If recipe is None or unrecognized, return prototype unchanged
    if "none" in norm_recipe or "vertical pin" not in norm_recipe:
        return prototype

    nodes = np.asarray(prototype.nodes, dtype=np.float64)
    struts = np.asarray(prototype.struts, dtype=np.int64)

    if len(nodes) == 0:
        return prototype

    z_vals = nodes[:, 2]
    z_min = float(np.min(z_vals))
    bottom_mask = np.abs(z_vals - z_min) < 1e-4
    bottom_indices = np.where(bottom_mask)[0]

    if len(bottom_indices) == 0:
        return prototype

    # Bottom Z plane of the particle bounding box
    wr = float(wire_radius) if float(wire_radius) > 0.0 else 0.25
    z_plane = z_min - wr
    if "solid_mesh" in prototype.metadata and isinstance(prototype.metadata["solid_mesh"], trimesh.Trimesh):
        z_mesh_min = float(prototype.metadata["solid_mesh"].bounds[0, 2])
        z_plane = min(z_plane, z_mesh_min)

    # For planar shapes (e.g. flat rings where all nodes share z_min), select 4 symmetric pins
    if len(bottom_indices) > 8:
        step = max(1, len(bottom_indices) // 4)
        selected_indices = bottom_indices[::step][:4]
    else:
        selected_indices = bottom_indices

    # Synthesize sacrificial vertical pin nodes and struts (diameter 0.25 mm -> radius 0.125 mm)
    pin_radius = 0.125
    pin_diameter = 0.25

    new_node_coords = [[nodes[i, 0], nodes[i, 1], z_plane] for i in selected_indices]
    pin_nodes = np.array(new_node_coords, dtype=np.float64)
    combined_nodes = np.vstack([nodes, pin_nodes])

    pin_struts = np.array(
        [[b_idx, len(nodes) + k] for k, b_idx in enumerate(selected_indices)],
        dtype=np.int64,
    )
    combined_struts = np.vstack([struts, pin_struts]) if len(struts) > 0 else pin_struts

    new_bounding_radius = float(np.max(np.linalg.norm(combined_nodes, axis=1)))

    meta = dict(prototype.metadata)
    meta["support_recipe"] = recipe_name
    meta["pin_diameter"] = pin_diameter
    meta["pin_radius"] = pin_radius
    meta["pin_struts"] = list(range(len(struts), len(combined_struts)))
    meta["pin_nodes"] = list(range(len(nodes), len(combined_nodes)))
    meta["bottom_z_plane"] = z_plane

    radii = np.full(len(combined_struts), wr, dtype=np.float64)
    radii[len(struts):] = pin_radius
    meta["strut_radii"] = radii

    # Pre-solidify supported prototype solid mesh
    try:
        if prototype.geometry_type.lower() in ("ring", "torus") or "major_radius" in prototype.metadata:
            base_mesh = solidify_particle_prototype(prototype, strut_radius=wr)
            pin_cyls = []
            for k, b_idx in enumerate(selected_indices):
                cyl = _manifold_cylinder_between(
                    combined_nodes[b_idx],
                    combined_nodes[len(nodes) + k],
                    radius=pin_radius,
                )
                if cyl is not None:
                    pin_cyls.append(cyl)
            if pin_cyls:
                pin_manifold = manifold3d.Manifold.compose(pin_cyls)
                base_manifold = _trimesh_to_manifold(base_mesh)
                united = manifold3d.Manifold.compose([base_manifold, pin_manifold])
                meta["solid_mesh"] = _manifold_to_trimesh(united)
            else:
                meta["solid_mesh"] = base_mesh
        else:
            meta["solid_mesh"] = build_clean_miter_truss(combined_nodes, combined_struts, radii)
    except Exception:
        pass

    return ParticleGeometry(
        nodes=combined_nodes,
        struts=combined_struts,
        bounding_radius=new_bounding_radius,
        geometry_type=prototype.geometry_type,
        metadata=meta,
    )


def create_custom_supported_particle(
    custom_mesh: trimesh.Trimesh,
    base_particle: InterlinkedParticle,
) -> InterlinkedParticle:
    """
    Wraps a user-imported supported mesh as the geometry of the particle so that
    it instances seamlessly into generate_interlinked_lattice.

    Parameters
    ----------
    custom_mesh : trimesh.Trimesh
        Supported mesh centered at local particle origin.
    base_particle : InterlinkedParticle
        Base particle providing ID, SE(3) transform, and metadata context.

    Returns
    -------
    InterlinkedParticle
        New particle instance holding the custom supported geometry.
    """
    if not isinstance(custom_mesh, trimesh.Trimesh):
        raise TypeError(f"custom_mesh must be a trimesh.Trimesh, got {type(custom_mesh)}")

    meta = dict(base_particle.geometry.metadata)
    meta["solid_mesh"] = custom_mesh
    meta["custom_supported"] = True

    # Maintain wireframe skeleton for clearance/collision verification
    if len(base_particle.geometry.nodes) > 0 and len(base_particle.geometry.struts) > 0:
        nodes = base_particle.geometry.nodes.copy()
        struts = base_particle.geometry.struts.copy()
    else:
        nodes = np.asarray(custom_mesh.vertices, dtype=np.float64)
        struts = np.asarray(custom_mesh.edges_unique, dtype=np.int64)

    r_mesh = (
        float(np.max(np.linalg.norm(custom_mesh.vertices, axis=1)))
        if len(custom_mesh.vertices) > 0
        else 0.0
    )
    bounding_radius = max(float(base_particle.geometry.bounding_radius), r_mesh)

    geom = ParticleGeometry(
        nodes=nodes,
        struts=struts,
        bounding_radius=bounding_radius,
        geometry_type="custom_supported",
        metadata=meta,
    )

    return InterlinkedParticle(
        particle_id=base_particle.particle_id,
        geometry=geom,
        transform=base_particle.transform.copy(),
        sublattice_id=base_particle.sublattice_id,
        cell_index=base_particle.cell_index,
        metadata={**base_particle.metadata, "custom_supported": True},
    )
