import os
import time
import itertools

import gmsh
import numpy as np
import trimesh
import manifold3d

from graphite.explicit import generate_topology


def _now():
    return time.time()


def _print_timer(label: str, t0: float):
    print(f"    -> {label} in {time.time() - t0:.3f} seconds.")


def _rotation_matrix_from_z(vec: np.ndarray) -> np.ndarray:
    """Build transform that rotates +Z to vec direction."""
    length = np.linalg.norm(vec)
    if length <= 0:
        return np.eye(4)

    v = vec / length
    z = np.array([0.0, 0.0, 1.0], dtype=float)
    axis = np.cross(z, v)
    axis_norm = np.linalg.norm(axis)
    dot = float(np.clip(np.dot(z, v), -1.0, 1.0))

    if axis_norm < 1e-12:
        if dot < 0:
            return trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0])
        return np.eye(4)

    axis /= axis_norm
    angle = np.arccos(dot)
    return trimesh.transformations.rotation_matrix(angle, axis)


def generate_base_scaffold(stl_path: str, target_size: float = 25.0):
    """Run working GMSH pipeline and extract linear tet scaffold."""
    print("\n" + "=" * 72)
    print("BASE SCAFFOLD: GMSH Conformal Tet Pipeline")
    print(f"Input STL: {stl_path}")
    print(f"Target Size: {target_size} mm")
    print("=" * 72)

    if not os.path.exists(stl_path):
        raise FileNotFoundError(f"Could not find STL: {stl_path}")

    t_total = _now()
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.option.setNumber("Mesh.MeshSizeMin", target_size * 0.9)
    gmsh.option.setNumber("Mesh.MeshSizeMax", target_size * 1.1)
    gmsh.option.setNumber("Mesh.Algorithm3D", 1)

    try:
        t0 = _now()
        print("[+] STEP 1: Merge STL")
        gmsh.merge(stl_path)
        gmsh.model.geo.synchronize()
        _print_timer("Merged STL", t0)

        t0 = _now()
        print("[+] STEP 2: Classify surfaces + create geometry + define volume")
        gmsh.model.mesh.classifySurfaces(np.pi, True, True, np.pi)
        gmsh.model.mesh.createGeometry()

        surfaces = gmsh.model.getEntities(2)
        surface_tags = [tag for dim, tag in surfaces]
        if not surface_tags:
            raise RuntimeError("No model surfaces found after classification.")

        surface_loop = gmsh.model.geo.addSurfaceLoop(surface_tags)
        volume_tag = gmsh.model.geo.addVolume([surface_loop])
        gmsh.model.geo.synchronize()
        volumes = gmsh.model.getEntities(3)
        if not volumes:
            raise RuntimeError("No 3D volume entities created.")
        print(
            f"    -> Built volume {volume_tag} from {len(surface_tags)} surfaces "
            f"({len(volumes)} volume entities total)."
        )
        _print_timer("Created CAD + volume", t0)

        t0 = _now()
        print("[+] STEP 3: Generate 3D mesh")
        gmsh.model.mesh.generate(3)
        _print_timer("Generated 3D mesh", t0)

        t0 = _now()
        print("[+] STEP 4: Extract nodes, tets, and boundary triangles")
        node_tags, coords, _ = gmsh.model.mesh.getNodes()
        nodes = np.asarray(coords, dtype=float).reshape(-1, 3)
        # Map gmsh node tags to contiguous 0..N-1 indices
        tag_to_idx = {int(tag): i for i, tag in enumerate(node_tags)}

        elem_types_3d, _, elem_node_tags_3d = gmsh.model.mesh.getElements(dim=3)
        tet_type = 4  # linear tetrahedron
        if tet_type not in list(elem_types_3d):
            raise RuntimeError(f"No linear tetrahedra (type {tet_type}) in extracted 3D elements.")
        tet_i = list(elem_types_3d).index(tet_type)
        tet_tags = np.asarray(elem_node_tags_3d[tet_i], dtype=np.int64).reshape(-1, 4)
        tets = np.vectorize(tag_to_idx.get, otypes=[np.int64])(tet_tags)

        elem_types_2d, _, elem_node_tags_2d = gmsh.model.mesh.getElements(dim=2)
        tri_type = 2  # linear triangle
        if tri_type in list(elem_types_2d):
            tri_i = list(elem_types_2d).index(tri_type)
            tri_tags = np.asarray(elem_node_tags_2d[tri_i], dtype=np.int64).reshape(-1, 3)
            surface_faces = np.vectorize(tag_to_idx.get, otypes=[np.int64])(tri_tags)
        else:
            surface_faces = np.empty((0, 3), dtype=np.int64)
        _print_timer("Extracted scaffold arrays", t0)

        print(
            f"[OK] Scaffold: nodes={len(nodes)}, tets={len(tets)}, "
            f"surface_faces={len(surface_faces)}"
        )
        _print_timer("Total scaffold pipeline", t_total)
        return nodes, tets, surface_faces
    finally:
        gmsh.finalize()


def extract_tet_struts(tets: np.ndarray) -> np.ndarray:
    """Extract all unique edges from linear tetrahedra."""
    pairs = np.array(list(itertools.combinations(range(4), 2)), dtype=np.int64)  # 6 edges
    edges = np.sort(tets[:, pairs], axis=2).reshape(-1, 2)
    return np.unique(edges, axis=0)


def generate_kagome_dual_struts(nodes: np.ndarray, tets: np.ndarray, surface_faces: np.ndarray, target_size: float):
    """Use project's conformal Kagome + surface dual logic."""
    nodes_out, struts = generate_topology(
        nodes,
        elements=tets,
        surface_faces=surface_faces,
        type="kagome",
        include_surface_cage=True,
        target_element_size=target_size,
        merge_short_struts=False,
    )
    return nodes_out, struts


def sweep_to_manifold(nodes: np.ndarray, struts: np.ndarray, radius: float = 0.5, sections: int = 12):
    """Sweep struts to cylinders and boolean-union with manifold3d."""
    t0 = _now()
    manifolds = []
    skipped_zero = 0
    skipped_invalid = 0

    for a, b in np.asarray(struts, dtype=np.int64):
        p1 = nodes[a]
        p2 = nodes[b]
        vec = p2 - p1
        length = float(np.linalg.norm(vec))
        if length < 1e-9:
            skipped_zero += 1
            continue

        cyl = trimesh.creation.cylinder(radius=radius, height=length, sections=sections)
        mat = _rotation_matrix_from_z(vec)
        mat[:3, 3] = (p1 + p2) * 0.5
        cyl.apply_transform(mat)

        try:
            m = manifold3d.Manifold(
                manifold3d.Mesh(
                    vert_properties=np.asarray(cyl.vertices, dtype=np.float32),
                    tri_verts=np.asarray(cyl.faces, dtype=np.uint32),
                )
            )
            manifolds.append(m)
        except Exception:
            skipped_invalid += 1
            continue

    if not manifolds:
        raise RuntimeError("No valid struts could be swept into manifold cylinders.")

    print(
        f"    -> Swept {len(manifolds)} cylinders "
        f"(skipped_zero={skipped_zero}, skipped_invalid={skipped_invalid})."
    )

    u0 = _now()
    combined = manifolds[0]
    for m in manifolds[1:]:
        combined = combined + m
    _print_timer("Boolean union completed", u0)

    mesh_raw = combined.to_mesh()
    verts = np.asarray(mesh_raw.vert_properties)
    faces = np.asarray(mesh_raw.tri_verts)
    if verts.ndim == 1:
        verts = verts.reshape(-1, 3)
    if faces.ndim == 1:
        faces = faces.reshape(-1, 3)
    solid = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
    _print_timer("Sweep + union total", t0)
    return solid


def main():
    stl_path = "top_part_new_BMeshRepaired_Repaired.stl"
    target_size = 12.5
    radius = 0.5

    nodes, tets, surface_faces = generate_base_scaffold(stl_path, target_size=target_size)

    # Lattice 1: direct tet-edge lattice
    print("\n" + "=" * 72)
    print("LATTICE 1: Simple Tetrahedral Edge Lattice")
    print("=" * 72)
    t0 = _now()
    tet_struts = extract_tet_struts(tets)
    print(f"[+] Unique tet struts: {len(tet_struts)}")
    tet_solid = sweep_to_manifold(nodes, tet_struts, radius=radius)
    out1 = "top_part_new_Tet_Lattice.stl"
    tet_solid.export(out1)
    print(f"[OK] Exported: {out1}")
    _print_timer("Lattice 1 total", t0)

    # Lattice 2: conformal kagome + surface dual
    print("\n" + "=" * 72)
    print("LATTICE 2: Conformal Kagome + Surface Dual")
    print("=" * 72)
    t0 = _now()
    kag_nodes, kag_struts = generate_kagome_dual_struts(
        nodes, tets, surface_faces, target_size=target_size
    )
    print(f"[+] Kagome/dual nodes: {len(kag_nodes)} | struts: {len(kag_struts)}")
    kag_solid = sweep_to_manifold(kag_nodes, kag_struts, radius=radius)
    out2 = "top_part_new_Conformal_Kagome.stl"
    kag_solid.export(out2)
    print(f"[OK] Exported: {out2}")
    _print_timer("Lattice 2 total", t0)


if __name__ == "__main__":
    main()

