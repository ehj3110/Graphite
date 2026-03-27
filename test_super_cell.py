import numpy as np
import trimesh
import itertools


def generate_true_kagome_super_cell():
    print("Generating True Face-Centroid Kagome Super-Cell (10x10x10mm)...")

    L = 10.0  # Voxel Size
    radius = 0.3  # Strut Thickness
    a = L * np.sqrt(2) / 2.0  # The strut length of the invisible FCC scaffold

    # 1. Define the 14 Nodes of the FCC Scaffold
    fcc_nodes = np.array([
        [0, 0, 0], [L, 0, 0], [0, L, 0], [L, L, 0],
        [0, 0, L], [L, 0, L], [0, L, L], [L, L, L],
        [L / 2, L / 2, 0], [L / 2, L / 2, L],
        [L / 2, 0, L / 2], [L / 2, L, L / 2],
        [0, L / 2, L / 2], [L, L / 2, L / 2]
    ])

    # 2. Find all equilateral triangular faces in the scaffold
    print("Identifying tetrahedral/octahedral faces...")
    faces = []
    tol = 1e-4
    for tri in itertools.combinations(range(14), 3):
        p1, p2, p3 = fcc_nodes[tri[0]], fcc_nodes[tri[1]], fcc_nodes[tri[2]]
        d12 = np.linalg.norm(p1 - p2)
        d23 = np.linalg.norm(p2 - p3)
        d31 = np.linalg.norm(p3 - p1)

        # If all three sides equal the scaffold edge length, it's a valid face
        if abs(d12 - a) < tol and abs(d23 - a) < tol and abs(d31 - a) < tol:
            faces.append(tri)

    # 3. KAGOME MATH: Nodes are the exact center (centroid) of each face
    print(f"Calculating Kagome nodes from {len(faces)} faces...")
    kagome_nodes = np.array([np.mean(fcc_nodes[list(face)], axis=0) for face in faces])

    # 4. Connect nodes: Struts connect face centroids that share an edge
    print("Connecting adjacent face centroids...")
    kagome_struts = []
    for i in range(len(faces)):
        for j in range(i + 1, len(faces)):
            # If two triangular faces share exactly 2 nodes, they share an edge
            shared_nodes = set(faces[i]).intersection(set(faces[j]))
            if len(shared_nodes) == 2:
                kagome_struts.append([i, j])

    # 5. Sweep the KAGOME struts into 3D cylinders
    meshes = []
    print(f"Sweeping {len(kagome_struts)} true Kagome struts...")
    for start_idx, end_idx in kagome_struts:
        p1 = kagome_nodes[start_idx]
        p2 = kagome_nodes[end_idx]

        vec = p2 - p1
        length = np.linalg.norm(vec)
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
        meshes.append(cyl)

    print("Combining Kagome geometry...")
    if meshes:
        combined = trimesh.util.concatenate(meshes)
        output_file = "Single_True_Kagome_SuperCell.stl"
        combined.export(output_file)
        print(f"Exported perfectly to {output_file}!")
    else:
        print("Error: No struts generated.")


if __name__ == "__main__":
    generate_true_kagome_super_cell()

