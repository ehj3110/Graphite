import time
import os
import gmsh
import numpy as np


def test_meshing_pipeline(stl_path, target_size=25.0):
    print(f"\n{'='*60}")
    print(f" PIPELINE STRESS TEST: Conformal Tet Meshing")
    print(f" Target File: {stl_path}")
    print(f" Target Element Size: {target_size}mm")
    print(f"{'='*60}\n")

    if not os.path.exists(stl_path):
        print(f"[-] CRITICAL ERROR: Could not find {stl_path}")
        return

    t_total_start = time.time()

    # Initialize GMSH and force it to be extremely loud
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)  # 1 = Verbose output to terminal

    # Set target sizes
    gmsh.option.setNumber("Mesh.MeshSizeMin", target_size * 0.9)
    gmsh.option.setNumber("Mesh.MeshSizeMax", target_size * 1.1)

    # Optional: Use the robust Frontal-Delaunay algorithm for 3D
    gmsh.option.setNumber("Mesh.Algorithm3D", 1)

    try:
        print("[+] STEP 1: Loading and merging STL...")
        t0 = time.time()
        gmsh.merge(stl_path)
        gmsh.model.geo.synchronize()
        print(f"    -> Done in {time.time() - t0:.3f} seconds.\n")

        print("[+] STEP 2: Classifying surfaces and creating topological CAD geometry...")
        t0 = time.time()
        gmsh.model.mesh.classifySurfaces(np.pi, True, True, np.pi)
        gmsh.model.mesh.createGeometry()

        # --- THE MISSING LINK: DEFINING THE VOLUME ---
        print("    -> Gathering generated surfaces to define 3D volume...")
        surfaces = gmsh.model.getEntities(2)
        surface_tags = [tag for dim, tag in surfaces]

        if len(surface_tags) > 0:
            # Group all surfaces into a closed loop
            surface_loop = gmsh.model.geo.addSurfaceLoop(surface_tags)
            # Define the solid volume inside that loop
            volume = gmsh.model.geo.addVolume([surface_loop])
            gmsh.model.geo.synchronize()
            print(f"    -> Successfully stitched {len(surface_tags)} surfaces into Volume Tag: {volume}")
        else:
            print("    [-] ERROR: No surfaces were generated!")

        # Verify the volume exists
        volumes = gmsh.model.getEntities(3)
        print(f"    -> Total 3D Volumes ready for meshing: {len(volumes)}")
        if len(volumes) == 0:
            raise RuntimeError("Failed to create a 3D volume entity.")

        print(f"    -> Done in {time.time() - t0:.3f} seconds.\n")

        print("[+] STEP 3: Generating 3D Tetrahedral Mesh (Watch GMSH logs below)...")
        print("-" * 40)
        t0 = time.time()
        gmsh.model.mesh.generate(3)
        print("-" * 40)
        print(f"    -> 3D Meshing finished in {time.time() - t0:.3f} seconds.\n")

        print("[+] STEP 4: Extracting Nodes and Elements...")
        t0 = time.time()
        nodeTags, coord, _ = gmsh.model.mesh.getNodes()
        nodes = np.array(coord).reshape(-1, 3)

        elemTypes, elemTags, elemNodeTags = gmsh.model.mesh.getElements(dim=3)
        if len(elemTypes) > 0:
            # Tet elements are typically type 4 in GMSH
            elements = np.array(elemNodeTags[0]).reshape(-1, 4) - 1
            element_count = len(elements)
        else:
            elements = []
            element_count = 0
        print(f"    -> Extraction done in {time.time() - t0:.3f} seconds.\n")

        print(f"\n{'='*60}")
        print(f" TEST RESULTS: SUCCESS")
        print(f" Nodes Generated:    {len(nodes)}")
        print(f" Elements Generated: {element_count}")
        print(f" Total Pipeline Time: {time.time() - t_total_start:.3f} seconds")
        print(f"{'='*60}\n")

    except Exception as e:
        print(f"\n{'='*60}")
        print(f" TEST RESULTS: CRITICAL FAILURE")
        print(f" Error: {e}")
        print(f"{'='*60}\n")
    finally:
        gmsh.finalize()


if __name__ == "__main__":
    target_file = "top_part_new_BMeshRepaired_Repaired.stl"
    test_meshing_pipeline(target_file, target_size=25.0)

