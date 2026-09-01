import math
import os
import time
from pathlib import Path

import manifold3d
import numpy as np
import trimesh
from scipy.spatial.distance import cdist


def _tet_faces():
    """Returns the vertex indices for the 4 faces of a tetrahedron."""
    return [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]


def apply_kagome_rule(tet_coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Nodes: 4 face centroids.
    Struts: complete graph on 4 nodes -> 6 segments.
    """
    coords = np.asarray(tet_coords, dtype=np.float64)
    if coords.shape != (4, 3):
        raise ValueError("tet_coords must have shape (4, 3).")

    nodes = np.array([coords[list(face)].mean(axis=0) for face in _tet_faces()])
    struts = np.array(
        [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)],
        dtype=np.int64,
    )
    return nodes, struts


def generate_fcc_tetrahedra_array(grid=(2, 1, 1), scale=20.0):
    """Generates FCC tetrahedra across a unit-cell grid."""
    all_tets = []
    for x in range(grid[0]):
        for y in range(grid[1]):
            for z in range(grid[2]):
                offset = np.array([x, y, z], dtype=np.float64) * scale

                corners = (
                    np.array(
                        [
                            (0, 0, 0),
                            (1, 0, 0),
                            (1, 1, 0),
                            (0, 1, 0),
                            (0, 0, 1),
                            (1, 0, 1),
                            (1, 1, 1),
                            (0, 1, 1),
                        ],
                        dtype=np.float64,
                    )
                    * scale
                    + offset
                )

                faces = {
                    "xy": np.array([0.5, 0.5, 0], dtype=np.float64) * scale + offset,
                    "XY": np.array([0.5, 0.5, 1], dtype=np.float64) * scale + offset,
                    "xz": np.array([0.5, 0, 0.5], dtype=np.float64) * scale + offset,
                    "XZ": np.array([0.5, 1, 0.5], dtype=np.float64) * scale + offset,
                    "yz": np.array([0, 0.5, 0.5], dtype=np.float64) * scale + offset,
                    "YZ": np.array([1, 0.5, 0.5], dtype=np.float64) * scale + offset,
                }

                all_tets.append([corners[0], faces["xy"], faces["xz"], faces["yz"]])
                all_tets.append([corners[1], faces["xy"], faces["xz"], faces["YZ"]])
                all_tets.append([corners[2], faces["xy"], faces["XZ"], faces["YZ"]])
                all_tets.append([corners[3], faces["xy"], faces["XZ"], faces["yz"]])
                all_tets.append([corners[4], faces["XY"], faces["xz"], faces["yz"]])
                all_tets.append([corners[5], faces["XY"], faces["xz"], faces["YZ"]])
                all_tets.append([corners[6], faces["XY"], faces["XZ"], faces["YZ"]])
                all_tets.append([corners[7], faces["XY"], faces["XZ"], faces["yz"]])

    return np.array(all_tets)


def create_cylinder_manifold(p1, p2, radius=0.3, sections=8):
    vec = p2 - p1
    length = np.linalg.norm(vec)
    if length < 1e-6:
        return None

    cyl = trimesh.creation.cylinder(radius=radius, height=length, sections=sections)
    z_axis = np.array([0, 0, 1], dtype=np.float64)
    vec_norm = vec / length
    axis = np.cross(z_axis, vec_norm)
    angle = np.arccos(np.clip(np.dot(z_axis, vec_norm), -1.0, 1.0))
    if np.linalg.norm(axis) > 1e-6:
        mat = trimesh.transformations.rotation_matrix(angle, axis / np.linalg.norm(axis))
    elif vec_norm[2] < 0:
        mat = trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0])
    else:
        mat = np.eye(4)
    mat[:3, 3] = p1 + vec / 2.0
    cyl.apply_transform(mat)

    try:
        return manifold3d.Manifold(
            manifold3d.Mesh(
                vert_properties=np.array(cyl.vertices, dtype=np.float32),
                tri_verts=np.array(cyl.faces, dtype=np.uint32),
            )
        )
    except Exception:
        return None


def build_global_kagome_graph(grid_dims, scale):
    tets = generate_fcc_tetrahedra_array(grid=grid_dims, scale=scale)

    all_nodes = []
    all_struts = []
    node_offset = 0
    for tet in tets:
        nodes, struts = apply_kagome_rule(tet)
        all_nodes.extend(nodes)
        all_struts.extend((struts + node_offset).tolist())
        node_offset += len(nodes)

    nodes = np.asarray(all_nodes, dtype=np.float64)
    struts = [tuple(map(int, s)) for s in all_struts]

    print(f"    Graph base: nodes={len(nodes)} internal_struts={len(struts)}")
    print("    Building bridge struts via global distance matrix...")
    t0 = time.time()
    bridge_target = scale / 3.0
    tolerance = bridge_target * 0.1
    dists = cdist(nodes, nodes)
    bridge_struts = []
    i_idx, j_idx = np.where(
        (dists > bridge_target - tolerance) & (dists < bridge_target + tolerance)
    )
    for i, j in zip(i_idx, j_idx):
        if i < j:
            bridge_struts.append((int(i), int(j)))
    print(f"    Bridges added: {len(bridge_struts)} in {time.time() - t0:.3f}s")

    struts.extend(bridge_struts)
    return nodes, struts


def _load_part_mesh():
    candidates = [
        Path("test_parts/top_part_new.stl"),
        Path("outputs/top_part_new_BMeshRepaired_Repaired.stl"),
        Path("top_part_new_BMeshRepaired_Repaired.stl"),
        Path("archive_test_files/top_part_new_BMeshRepaired_Repaired.stl"),
        Path("archive_test_files/Part2_Adapter_Repaired.stl"),
    ]
    for p in candidates:
        if p.exists():
            mesh = trimesh.load_mesh(str(p), process=True)
            if isinstance(mesh, trimesh.Scene):
                mesh = mesh.dump(concatenate=True)
            extents = np.asarray(mesh.bounds[1] - mesh.bounds[0], dtype=np.float64)
            # Skip tiny placeholder meshes that are not valid production parts.
            if float(np.max(extents)) < 5.0:
                print(f"[!] Skipping tiny candidate: {p} extents={extents}")
                continue
            return p, mesh
    raise FileNotFoundError("Could not locate a repaired source STL.")


def main():
    print("=" * 70)
    print("GRIDDED SUPERCELL TRIMMER")
    print("=" * 70)

    t0 = time.time()
    mesh_path, mesh = _load_part_mesh()
    print(f"[+] Loaded part: {mesh_path}")
    print(f"    Watertight={mesh.is_watertight}  Faces={len(mesh.faces)}")

    scale = 15.0
    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    extents = bounds[1] - bounds[0]
    grid_dims = tuple(max(1, int(math.ceil(float(e) / scale))) for e in extents)
    print(f"[+] Bounds extents: {extents}")
    print(f"[+] Unit cell scale: {scale}")
    print(f"[+] Computed grid dims: {grid_dims}")

    tg = time.time()
    nodes, struts = build_global_kagome_graph(grid_dims=grid_dims, scale=scale)
    print(f"[+] Graph generation total: {time.time() - tg:.3f}s")

    grid_center = nodes.mean(axis=0)
    nodes = nodes - grid_center + np.asarray(mesh.centroid, dtype=np.float64)
    print(f"[+] Centered graph to mesh centroid in {time.time() - tg:.3f}s")

    tt = time.time()
    if mesh.is_watertight:
        inside_mask = np.asarray(mesh.contains(nodes), dtype=bool)
    else:
        # Fallback for non-watertight meshes: trim by axis-aligned bounds.
        mins = bounds[0]
        maxs = bounds[1]
        inside_mask = np.all((nodes >= mins) & (nodes <= maxs), axis=1)
        print("[!] Mesh is non-watertight; using AABB containment fallback.")
    core_struts = []
    boundary_struts = []
    for s, e in struts:
        s_in = inside_mask[s]
        e_in = inside_mask[e]
        if s_in and e_in:
            core_struts.append((s, e))
        elif s_in or e_in:
            boundary_struts.append((s, e))
    print(f"[+] 1D trim completed in {time.time() - tt:.3f}s")
    print(f"    core_struts={len(core_struts)} boundary_struts={len(boundary_struts)}")

    part_manifold = manifold3d.Manifold(
        manifold3d.Mesh(
            vert_properties=np.asarray(mesh.vertices, dtype=np.float32),
            tri_verts=np.asarray(mesh.faces, dtype=np.uint32),
        )
    )

    core_solid = None
    t_core = time.time()
    if core_struts:
        core_parts = [create_cylinder_manifold(nodes[s], nodes[e], radius=0.3) for s, e in core_struts]
        core_parts = [m for m in core_parts if m is not None]
        if core_parts:
            core_solid = manifold3d.Manifold.compose(core_parts)
    print(f"[+] Core sweep+union: {time.time() - t_core:.3f}s")

    boundary_solid = None
    t_bound = time.time()
    if boundary_struts:
        bound_parts = [create_cylinder_manifold(nodes[s], nodes[e], radius=0.3) for s, e in boundary_struts]
        bound_parts = [m for m in bound_parts if m is not None]
        if bound_parts:
            boundary_solid = manifold3d.Manifold.compose(bound_parts)
            print("    Running boundary CSG intersection...")
            boundary_solid = boundary_solid ^ part_manifold
    print(f"[+] Boundary sweep+CSG: {time.time() - t_bound:.3f}s")

    if core_solid is not None and boundary_solid is not None:
        final_solid = core_solid + boundary_solid
    elif core_solid is not None:
        final_solid = core_solid
    elif boundary_solid is not None:
        final_solid = boundary_solid
    else:
        raise RuntimeError("No geometry generated from trimmed struts.")

    t_export = time.time()
    out_mesh = final_solid.to_mesh()
    result = trimesh.Trimesh(
        vertices=np.asarray(out_mesh.vert_properties).reshape(-1, 3),
        faces=np.asarray(out_mesh.tri_verts).reshape(-1, 3),
        process=False,
    )
    os.makedirs("outputs", exist_ok=True)
    out_path = "outputs/Adapter_Gridded_SuperCell.stl"
    result.export(out_path)
    print(f"[+] Exported: {out_path}")
    print(f"[+] Export time: {time.time() - t_export:.3f}s")
    print(f"[+] Total runtime: {time.time() - t0:.3f}s")


if __name__ == "__main__":
    main()
