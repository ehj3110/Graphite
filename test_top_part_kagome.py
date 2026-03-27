import numpy as np
import trimesh
import os
import time
from graphite.explicit import generate_conformal_scaffold, generate_topology, generate_geometry

# Try to import the heavy-duty repair library
try:
    import pymeshfix
    HAS_MESHFIX = True
except ImportError:
    HAS_MESHFIX = False


def profile_kagome_generation():
    target = 25.0
    file_name = "test_parts/top_part_new.stl"
    if not os.path.exists(file_name):
        file_name = "top_part_new.stl"

    print(f"--- PROFILING RUN: {file_name} (Target: {target}mm Kagome) ---")

    # 1. Loading & Processing
    t0 = time.time()
    try:
        mesh = trimesh.load(file_name)
    except Exception as e:
        print(f"Failed to load file: {e}. Please ensure '{file_name}' is in the correct directory.")
        return

    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)

    print("Applying basic trimesh repairs (normals, winding)...")
    trimesh.repair.fix_normals(mesh)
    trimesh.repair.fix_inversion(mesh)
    trimesh.repair.fix_winding(mesh)

    # Aggressive Repair for Intersecting Faces
    if HAS_MESHFIX:
        print("Attempting aggressive mesh repair (pymeshfix) to resolve intersecting faces...")
        try:
            v = np.asarray(mesh.vertices, dtype=np.float64, order="C")
            f = np.asarray(mesh.faces, dtype=np.int32, order="C")
            vclean, fclean = pymeshfix.clean_from_arrays(v, f)
            mesh_process = trimesh.Trimesh(vertices=vclean, faces=fclean, process=True)
            print("pymeshfix repair successful!")
        except Exception as e:
            print(f"pymeshfix failed, falling back to basic trimesh: {e}")
            mesh_process = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=True)
    else:
        print("pymeshfix not found (run 'pip install pymeshfix'). Proceeding with basic mesh...")
        mesh_process = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=True)

    t_load = time.time() - t0
    print(f"[Timer] STL Loading & Repair: {t_load:.3f} seconds")
    print(f"        -> Watertight: {mesh_process.is_watertight}")

    # 2. Scaffold Generation (GMSH)
    t0 = time.time()
    print(f"Generating Linear Scaffold ({target}mm)...")
    scaffold = generate_conformal_scaffold(mesh_process, target_element_size=target, element_order=1)
    nodes, tets, surface_faces = scaffold.nodes, scaffold.elements, scaffold.surface_faces
    t_mesh = time.time() - t0
    print(f"[Timer] GMSH Meshing: {t_mesh:.3f} seconds")

    # 3. Topology Extraction (Kagome + Surface Dual)
    t0 = time.time()
    print("Extracting Kagome Topology...")
    nodes_out, struts = generate_topology(
        nodes,
        elements=tets,
        surface_faces=surface_faces,
        type="kagome",
        include_surface_cage=True,
        target_element_size=target,
        merge_short_struts=False
    )
    t_topo = time.time() - t0
    print(f"[Timer] Topology Extraction ({len(struts)} total struts): {t_topo:.3f} seconds")

    # 4. Geometry Generation (Manifold3D CSG)
    t0 = time.time()
    print("Sweeping struts into solid geometry (using 0.5mm radius)...")
    solid = generate_geometry(nodes_out, struts, strut_radius=0.5)
    t_geom = time.time() - t0
    print(f"[Timer] CSG Solidification (Manifold3D): {t_geom:.3f} seconds")

    # 5. Export
    t0 = time.time()
    output_name = "Top_Part_New_25mm_Kagome_Profiled.stl"
    solid.export(output_name)
    t_export = time.time() - t0
    print(f"[Timer] STL Export: {t_export:.3f} seconds")

    # Total
    total_time = t_load + t_mesh + t_topo + t_geom + t_export
    print(f"\n--- TOTAL TIME: {total_time:.3f} seconds ---")


if __name__ == "__main__":
    profile_kagome_generation()

