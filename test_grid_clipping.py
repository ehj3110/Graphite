import numpy as np
import trimesh
from scipy.spatial import cKDTree
from graphite.explicit import generate_geometry


def generate_global_kagome_lattice(grid_size_mm, cell_size_mm):
    """Generates a continuous Kagome lattice over a global volume."""
    print("1. Generating Global FCC Node Grid...")
    L = cell_size_mm
    N = int(np.ceil(grid_size_mm / L))

    # 1. Generate all FCC nodes in the global space
    fcc_nodes = []
    for ix in range(N + 1):
        for iy in range(N + 1):
            for iz in range(N + 1):
                x, y, z = ix * L, iy * L, iz * L
                fcc_nodes.append([x, y, z])
                # Add face-centered nodes
                if ix < N and iy < N:
                    fcc_nodes.append([x + L / 2, y + L / 2, z])
                if ix < N and iz < N:
                    fcc_nodes.append([x + L / 2, y, z + L / 2])
                if iy < N and iz < N:
                    fcc_nodes.append([x, y + L / 2, z + L / 2])

    fcc_nodes = np.array(fcc_nodes)

    print("2. Finding Equilateral Tetrahedral/Octahedral Faces...")
    a = L * np.sqrt(2) / 2.0
    tree = cKDTree(fcc_nodes)
    pairs = tree.query_pairs(r=a + 1e-3)

    # Build a fast neighbor lookup
    neighbors = {i: set() for i in range(len(fcc_nodes))}
    for i, j in pairs:
        dist = np.linalg.norm(fcc_nodes[i] - fcc_nodes[j])
        if abs(dist - a) < 1e-3:
            neighbors[i].add(j)
            neighbors[j].add(i)

    # Find triangles (cliques of size 3)
    faces = []
    for i in range(len(fcc_nodes)):
        for j in neighbors[i]:
            if j > i:
                shared = neighbors[i].intersection(neighbors[j])
                for k in shared:
                    if k > j:
                        faces.append((i, j, k))

    print(f"   -> Found {len(faces)} perfect faces.")

    print("3. Calculating True Kagome Nodes and Connecting Struts...")
    kagome_nodes = np.array([np.mean(fcc_nodes[list(face)], axis=0) for face in faces])

    # Map edges to faces to quickly find adjacent faces
    edge_to_faces = {}
    for face_idx, face in enumerate(faces):
        edges = [(face[0], face[1]), (face[1], face[2]), (face[0], face[2])]
        for e in edges:
            e_sorted = tuple(sorted(e))
            if e_sorted not in edge_to_faces:
                edge_to_faces[e_sorted] = []
            edge_to_faces[e_sorted].append(face_idx)

    kagome_struts = []
    for _, f_list in edge_to_faces.items():
        if len(f_list) == 2:  # Two faces sharing an edge
            kagome_struts.append([f_list[0], f_list[1]])

    return kagome_nodes, kagome_struts


def run_global_clipping():
    cell_size = 6.0
    cube_size = 20.0
    radius = 0.3

    nodes, struts = generate_global_kagome_lattice(grid_size_mm=24.0, cell_size_mm=cell_size)

    print(f"4. Sweeping {len(struts)} continuous struts using Native Manifold3D Union...")
    import manifold3d

    manifolds = []
    for start_idx, end_idx in struts:
        p1 = nodes[start_idx]
        p2 = nodes[end_idx]
        vec = p2 - p1
        length = np.linalg.norm(vec)
        if length < 1e-6:
            continue

        cyl = trimesh.creation.cylinder(radius=radius, height=length, sections=12)
        z_axis = np.array([0, 0, 1])
        vec_norm = vec / length
        axis = np.cross(z_axis, vec_norm)
        angle = np.arccos(np.clip(np.dot(z_axis, vec_norm), -1.0, 1.0))

        if np.linalg.norm(axis) > 1e-6:
            axis = axis / np.linalg.norm(axis)
            mat = trimesh.transformations.rotation_matrix(angle, axis)
        elif vec_norm[2] < 0:
            mat = trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0])
        else:
            mat = np.eye(4)

        mat[:3, 3] = p1 + vec / 2.0
        cyl.apply_transform(mat)

        # Convert IMMEDIATELY to Manifold3D object
        try:
            cyl_man = manifold3d.Manifold(manifold3d.Mesh(
                vert_properties=np.array(cyl.vertices, dtype=np.float32),
                tri_verts=np.array(cyl.faces, dtype=np.uint32)
            ))
            manifolds.append(cyl_man)
        except Exception:
            pass  # Skip invalid micro-cylinders if any exist

    if not manifolds:
        print("   -> CRITICAL FAILURE: No valid manifold cylinders were created.")
        return

    print("   -> Performing true Boolean Union on all struts (This melts away self-intersections)...")
    # Using the native + operator for Boolean Union
    lattice_manifold = manifolds[0]
    for m in manifolds[1:]:
        lattice_manifold = lattice_manifold + m

    print(f"5. Creating {cube_size}mm Bounding Cube for CSG Clipping...")
    try:
        cube_manifold = manifold3d.Manifold.cube(size=[cube_size, cube_size, cube_size], center=True)
        cube_manifold = cube_manifold.translate([12.0, 12.0, 12.0])
    except Exception:
        # Fallback if native cube fails
        cube_mesh = trimesh.creation.box(extents=[cube_size, cube_size, cube_size])
        cube_mesh.apply_translation([12.0, 12.0, 12.0])
        cube_manifold = manifold3d.Manifold(manifold3d.Mesh(
            vert_properties=np.array(cube_mesh.vertices, dtype=np.float32),
            tri_verts=np.array(cube_mesh.faces, dtype=np.uint32)
        ))

    # --- DIAGNOSTIC CHECKS ---
    vol_lattice = lattice_manifold.volume()
    vol_cube = cube_manifold.volume()
    print(f"   -> Pristine Lattice Volume: {vol_lattice:.2f}")
    print(f"   -> Cube Volume: {vol_cube:.2f}")

    print("6. Performing Boolean Intersection (Cookie Cutter)...")
    # Using the native ^ operator for Boolean Intersection
    clipped_manifold = lattice_manifold ^ cube_manifold

    result_vol = clipped_manifold.volume()
    print(f"   -> Clipped Result Volume: {result_vol:.2f}")

    if result_vol < 0.1:
        print("   -> CRITICAL FAILURE: Intersection is still empty.")
        return

    # --- SAFE TRIMESH CONVERSION ---
    clipped_m_mesh = clipped_manifold.to_mesh()
    verts = np.array(clipped_m_mesh.vert_properties)
    faces = np.array(clipped_m_mesh.tri_verts)
    if verts.ndim == 1:
        verts = verts.reshape(-1, 3)
    if faces.ndim == 1:
        faces = faces.reshape(-1, 3)

    clipped_mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    output_file = "Global_Clipped_20mm_Kagome.stl"
    clipped_mesh.export(output_file)
    print(f"Successfully exported {output_file}! (Faces: {len(clipped_mesh.faces)})")


if __name__ == "__main__":
    run_global_clipping()

