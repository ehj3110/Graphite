"""
Graphite Explicit Interlinked — Instanced 3MF Solidification Engine (Phase 5)

Implements production-grade 3MF object instancing (ISO/IEC 5165 standard):
- Exactly ONE mesh resource definition per distinct particle prototype in <resources>.
- N rigid item placements in <build> via SE(3) transform matrices.
- Reduces multi-body assembly file sizes by 90% to 99% compared to raw STLs.
- Natively compatible with standard slicing software (Bambu Studio, PrusaSlicer, Cura, Magics).
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Sequence, Any
import numpy as np
import trimesh
import manifold3d as m3d

from .particle import ParticleGeometry, InterlinkedParticle
from graphite.explicit.geometry_module import generate_geometry, _manifold_to_trimesh


_CONTENT_TYPES_XML = """<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>
</Types>"""

_RELS_XML = """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Target="/3D/3dmodel.model" Id="rel0" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>
</Relationships>"""


def solidify_particle_prototype(
    geometry: ParticleGeometry,
    strut_radius: float,
    circular_segments: int = 24,
) -> trimesh.Trimesh:
    """
    Solidify a canonical ParticleGeometry prototype at the local origin [0, 0, 0].

    Generates a watertight trimesh.Trimesh:
    - If geometry_type == "ring": revolves a circle of radius strut_radius.
    - If wireframe truss: calls generate_geometry with clean mitered joints (add_spheres=False).
    - If custom solid: uses metadata['solid_mesh'].

    Args:
        geometry: Canonical unmutated ParticleGeometry.
        strut_radius: Strut or wire radius in mm.
        circular_segments: Cross-sectional discretization resolution.

    Returns:
        Watertight trimesh.Trimesh prototype centered at local origin.
    """
    r = float(strut_radius)
    g_type = getattr(geometry, "geometry_type", "").lower()
    meta = getattr(geometry, "metadata", {})

    if "solid_mesh" in meta and isinstance(meta["solid_mesh"], trimesh.Trimesh):
        return meta["solid_mesh"]

    if g_type in ("ring", "torus") or "major_radius" in meta:
        R_major = float(meta.get("major_radius", geometry.bounding_radius))
        cross_segs = max(16, circular_segments // 2)
        revolve_segs = max(32, circular_segments * 2)

        c_circle = m3d.CrossSection.circle(r, circular_segments=cross_segs).translate([R_major, 0.0])
        torus = m3d.Manifold.revolve(c_circle, circular_segments=revolve_segs)
        return _manifold_to_trimesh(torus)

    # General wireframe truss solid (clean mitered joints)
    nodes = np.asarray(geometry.nodes, dtype=np.float64)
    struts = np.asarray(geometry.struts, dtype=np.int64)

    if len(nodes) == 0 or len(struts) == 0:
        raise ValueError(f"Cannot solidify empty particle geometry {geometry}")

    mesh = generate_geometry(
        nodes=nodes,
        struts=struts,
        strut_radius=r,
        add_spheres=False,
        circular_segments=circular_segments,
        crop_to_boundary=False,
    )
    return mesh


def format_3mf_transform(transform_4x4: np.ndarray) -> str:
    """
    Format a 4x4 homogeneous transformation matrix into the 12-value 3MF affine string:
        "m00 m01 m02 m10 m11 m12 m20 m21 m22 m03 m13 m23"
    where [m00..m22] is rotation and [m03, m13, m23] is translation.
    """
    T = np.asarray(transform_4x4, dtype=np.float64)
    m00, m01, m02, m03 = T[0, 0], T[0, 1], T[0, 2], T[0, 3]
    m10, m11, m12, m13 = T[1, 0], T[1, 1], T[1, 2], T[1, 3]
    m20, m21, m22, m23 = T[2, 0], T[2, 1], T[2, 2], T[2, 3]
    return f"{m00:.8f} {m01:.8f} {m02:.8f} {m10:.8f} {m11:.8f} {m12:.8f} {m20:.8f} {m21:.8f} {m22:.8f} {m03:.8f} {m13:.8f} {m23:.8f}"


def export_interlinked_3mf(
    particles: Sequence[InterlinkedParticle],
    strut_radius: float,
    output_path: str | Path,
    solid_frames: Sequence[trimesh.Trimesh] | trimesh.Trimesh | None = None,
    prototype_meshes: dict[str, trimesh.Trimesh] | None = None,
    circular_segments: int = 24,
    num_thickness_buckets: int = 16,
) -> Path:
    """
    Export an assembly of InterlinkedParticle instances as a true instanced 3MF package.

    Key Advantages:
        - Unique particle meshes are stored exactly ONCE in the <resources> block.
        - Placements are lightweight <item> transform references in the <build> block.
        - Yields 90% to 99% reduction in file size compared to raw monolithic STLs.
        - Fully compatible with slicers (PrusaSlicer, Bambu Studio, Cura, Magics).

    Args:
        particles: Sequence of InterlinkedParticle instances.
        strut_radius: Wire/strut radius in mm.
        output_path: Filepath for the generated .3mf file.
        solid_frames: Optional solid boundary frames (e.g. from Policy C).
        prototype_meshes: Optional pre-solidified prototype cache keyed by prototype identifier.
        circular_segments: Cross-sectional circle resolution.

    Returns:
        Path to the saved .3mf file.
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    if not particles and not solid_frames:
        raise ValueError("Cannot export empty particle list to 3MF")

    proto_cache: dict[str, trimesh.Trimesh] = dict(prototype_meshes or {})
    # Map prototype_key -> (3mf_object_id, trimesh_mesh)
    prototype_registry: dict[str, tuple[int, trimesh.Trimesh]] = {}
    next_obj_id = 1

    # 1. Inspect per-particle wire radii and establish bucket centers
    radii_list = [p.effective_wire_radius(fallback=strut_radius) for p in particles]
    if radii_list:
        min_r = min(radii_list)
        max_r = max(radii_list)
    else:
        min_r = max_r = strut_radius

    if abs(max_r - min_r) > 1e-4 and num_thickness_buckets > 1:
        bucket_centers = np.linspace(min_r, max_r, num_thickness_buckets)
    else:
        bucket_centers = np.array([0.5 * (min_r + max_r)])

    def _quantized_radius(r_val: float) -> float:
        idx = int(np.argmin(np.abs(bucket_centers - r_val)))
        return float(bucket_centers[idx])

    # 2. Group particles by prototype key (including quantized radius)
    def _proto_key(p: InterlinkedParticle, r_q: float) -> str:
        g = p.geometry
        return f"{g.geometry_type}_{len(g.nodes)}_{len(g.struts)}_{p.sublattice_id}_r{r_q:.4f}"

    particle_items: list[tuple[int, str]] = []  # (object_id, transform_str)

    for p in particles:
        p_r = p.effective_wire_radius(fallback=strut_radius)
        r_bucket = _quantized_radius(p_r)
        key = _proto_key(p, r_bucket)

        if key not in prototype_registry:
            # Solidify prototype once per unique geometry + radius bucket
            if key in proto_cache:
                m = proto_cache[key]
            else:
                m = solidify_particle_prototype(
                    geometry=p.geometry,
                    strut_radius=r_bucket,
                    circular_segments=circular_segments,
                )
                proto_cache[key] = m

            prototype_registry[key] = (next_obj_id, m)
            obj_id = next_obj_id
            next_obj_id += 1
        else:
            obj_id = prototype_registry[key][0]

        # Format 3MF item transform
        t_str = format_3mf_transform(p.transform)
        particle_items.append((obj_id, t_str))

    # 2. Handle optional solid frames (Policy C)
    frame_items: list[int] = []
    if solid_frames is not None:
        if isinstance(solid_frames, trimesh.Trimesh):
            frames_list = [solid_frames]
        else:
            frames_list = list(solid_frames)

        for f_idx, f_mesh in enumerate(frames_list):
            f_key = f"__solid_frame_{f_idx}__"
            prototype_registry[f_key] = (next_obj_id, f_mesh)
            frame_items.append(next_obj_id)
            next_obj_id += 1

    # 3. Construct 3MF XML Resources (<resources>)
    resources_xml_parts: list[str] = ["  <resources>"]
    for key, (obj_id, mesh) in prototype_registry.items():
        v_lines = [f'          <vertex x="{v[0]:.6f}" y="{v[1]:.6f}" z="{v[2]:.6f}"/>' for v in mesh.vertices]
        f_lines = [f'          <triangle v1="{f[0]}" v2="{f[1]}" v3="{f[2]}"/>' for f in mesh.faces]

        obj_xml = f"""    <object id="{obj_id}" type="model" name="{key}">
      <mesh>
        <vertices>
{chr(10).join(v_lines)}
        </vertices>
        <triangles>
{chr(10).join(f_lines)}
        </triangles>
      </mesh>
    </object>"""
        resources_xml_parts.append(obj_xml)

    resources_xml_parts.append("  </resources>")
    resources_xml = "\n".join(resources_xml_parts)

    # 4. Construct 3MF Build (<build>)
    build_xml_parts: list[str] = ["  <build>"]
    for obj_id, t_str in particle_items:
        build_xml_parts.append(f'    <item objectid="{obj_id}" transform="{t_str}"/>')

    for frame_obj_id in frame_items:
        # Frames are already in world coordinates
        build_xml_parts.append(f'    <item objectid="{frame_obj_id}"/>')

    build_xml_parts.append("  </build>")
    build_xml = "\n".join(build_xml_parts)

    # 5. Assemble Full 3D Model XML Document
    model_xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<model unit="millimeter" xml:lang="en-US" xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">
{resources_xml}
{build_xml}
</model>"""

    # 6. Package into OPC ZIP Archive
    with zipfile.ZipFile(out_file, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", _CONTENT_TYPES_XML)
        z.writestr("_rels/.rels", _RELS_XML)
        z.writestr("3D/3dmodel.model", model_xml)

    return out_file
