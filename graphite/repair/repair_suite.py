"""
Graphite Repair - Repair Suite

This module provides a dual-phase mesh repair pipeline using Trimesh and PyMeshLab. 
Phase 1 ("Gentle Triage") attempts lightweight topological fixes (degenerate face removal, 
hole filling). Phase 2 ("Fail-Safe") uses Screened Poisson Surface Reconstruction 
for aggressive rebuilding of heavily intersecting or non-manifold meshes.
"""
import os
import trimesh
import pymeshlab
import numpy as np

def get_boundary_edges(ms):
    """Safely extracts the number of open boundary edges."""
    try:
        out = ms.get_topological_measures()
        return out.get('boundary_edges', -1)
    except Exception:
        return -1

def get_connected_components(ms):
    """Safely extracts the number of connected components (shells)."""
    try:
        out = ms.get_topological_measures()
        return out.get('connected_components_number', -1)
    except Exception:
        return -1

def get_non_two_manifold_edges(ms):
    """Safely extracts count of non-two-manifold edges (MeshLab definition)."""
    try:
        out = ms.get_topological_measures()
        return out.get('non_two_manifold_edges', -1)
    except Exception:
        return -1

def get_non_two_manifold_vertices(ms):
    """Safely extracts count of non-two-manifold vertices (MeshLab definition)."""
    try:
        out = ms.get_topological_measures()
        return out.get('non_two_manifold_vertices', -1)
    except Exception:
        return -1


def get_pymeshlab_health(file_path):
    """Safely extracts topological metrics using PyMeshLab."""
    try:
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(file_path)
        out = ms.get_topological_measures()
        nm_edges = out.get('non_two_manifold_edges', -1)

        # Check intersections
        ms.compute_selection_by_self_intersections_per_face()
        # The number of selected faces is the number of intersections
        # PyMeshLab doesn't return this directly in topological_measures, so we extract it:
        # We can try to get selection count, but safely we'll just check if applying deletion changes face count.
        # For pure diagnostics, we rely on Trimesh for standard counts, and PyMeshLab for intersections.
        return nm_edges
    except Exception as e:
        return -1


def check_intersections_pymeshlab(file_path):
    """Returns the exact number of self-intersecting faces."""
    try:
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(file_path)
        initial_faces = ms.current_mesh().face_number()
        ms.compute_selection_by_self_intersections_per_face()
        ms.meshing_remove_selected_faces()
        final_faces = ms.current_mesh().face_number()
        return initial_faces - final_faces
    except Exception:
        return -1


def print_health_report(file_path, stage="Diagnostic"):
    """
    Print a diagnostic health report of the mesh including topological issues.

    Parameters
    ----------
    file_path : str
        Path to the STL file to inspect.
    stage : str, optional
        Label for the report (e.g., 'Initial', 'Final'), by default "Diagnostic".
    """
    print(f"\n--- {stage} Health Report ---")
    try:
        mesh = trimesh.load_mesh(file_path)
        if isinstance(mesh, trimesh.Scene):
             mesh = mesh.dump(concatenate=True)
        print(f"  Vertices:            {len(mesh.vertices)}")
        print(f"  Faces:               {len(mesh.faces)}")
        print(f"  Watertight:          {mesh.is_watertight}")
    except Exception:
        print("  Basic Stats:         [Failed to load in Trimesh]")

    nm_edges = get_pymeshlab_health(file_path)
    intersections = check_intersections_pymeshlab(file_path)
    try:
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(file_path)
        boundary_edges = get_boundary_edges(ms)
        shells = get_connected_components(ms)
        nm_vertices = get_non_two_manifold_vertices(ms)
    except Exception:
        boundary_edges, shells, nm_vertices = -1, -1, -1

    print(f"  Shells (Components): {shells if shells != -1 else '[Uncomputable]'}")
    print(f"  Boundary Edges:      {boundary_edges if boundary_edges != -1 else '[Uncomputable]'}")
    print(f"  Non-Manifold Edges:  {nm_edges if nm_edges != -1 else '[Uncomputable]'}")
    print(f"  Non-Manifold Verts:  {nm_vertices if nm_vertices != -1 else '[Uncomputable]'}")
    print(f"  Intersecting Faces:  {intersections if intersections != -1 else '[Uncomputable]'}")
    print("-------------------------------\n")


def repair_trimesh_gentle(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """
    In-memory Phase 1 gentle triage on a Trimesh object.

    Attempts to fix duplicate vertices, degenerate faces, duplicate faces,
    unreferenced vertices, incorrect winding, inverted normals, and simple holes
    without altering the outer bounding shape.
    """
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        return mesh

    m = mesh.copy()

    # Step 1: Deduplicate identical vertices so topological edge sharing is established
    try:
        m.merge_vertices(merge_tex=True, merge_norm=True)
    except Exception:
        pass

    # Step 2: Remove degenerate and duplicate faces
    try:
        m.update_faces(m.nondegenerate_faces())
    except Exception:
        pass
    try:
        m.update_faces(m.unique_faces())
    except Exception:
        pass

    # Step 3: Remove unreferenced vertices
    try:
        m.remove_unreferenced_vertices()
    except Exception:
        pass

    # Step 4: Fix small holes, winding consistency, and normal orientation
    try:
        trimesh.repair.fill_holes(m)
    except Exception:
        pass
    try:
        trimesh.repair.fix_winding(m)
    except Exception:
        pass
    try:
        trimesh.repair.fix_inversion(m)
    except Exception:
        pass
    try:
        trimesh.repair.fix_normals(m)
    except Exception:
        pass

    # Step 5: If still not watertight, attempt gentle vertex snapping on tight tolerance
    if not m.is_watertight:
        span = float(np.max(m.extents)) if m.extents is not None else 0.0
        if span > 0.0:
            try:
                m.merge_vertices(merge_tex=True, merge_norm=True, digits_vertex=4)
            except Exception:
                pass
            try:
                trimesh.repair.fill_holes(m)
            except Exception:
                pass
            try:
                trimesh.repair.fix_winding(m)
            except Exception:
                pass
            try:
                trimesh.repair.fix_inversion(m)
            except Exception:
                pass
            try:
                trimesh.repair.fix_normals(m)
            except Exception:
                pass

    try:
        m.remove_unreferenced_vertices()
    except Exception:
        pass

    return m


def repair_trimesh_poisson(mesh: trimesh.Trimesh, depth: int = 10) -> trimesh.Trimesh:
    """
    Phase 2 in-memory wrapper for Screened Poisson reconstruction.
    """
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        in_path = os.path.join(tmpdir, "raw.stl")
        out_path = os.path.join(tmpdir, "repaired.stl")
        mesh.export(in_path)
        run_poisson_reconstruction(in_path, out_path, depth=depth)
        repaired = trimesh.load(out_path, force="mesh", process=True)
        if isinstance(repaired, trimesh.Scene):
            repaired = repaired.dump(concatenate=True)
        return repaired


def repair_trimesh(
    mesh: trimesh.Trimesh,
    *,
    mode: str = "gentle",
    escalate_to_full: bool = True,
    poisson_depth: int = 10,
) -> tuple[trimesh.Trimesh, list[str]]:
    """
    Repair a trimesh.Trimesh object using gentle triage and optional Poisson fail-safe.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Input mesh.
    mode : {"gentle", "full", "none"}, optional
        Repair intensity, by default "gentle".
    escalate_to_full : bool, optional
        Whether to escalate to Poisson reconstruction if gentle triage fails
        to achieve watertightness, by default True.
    poisson_depth : int, optional
        Octree depth for Screened Poisson reconstruction, by default 10.

    Returns
    -------
    tuple[trimesh.Trimesh, list[str]]
        Repaired mesh and diagnostic log notes.
    """
    notes: list[str] = []
    norm_mode = str(mode).strip().lower()
    if norm_mode == "explicit":
        norm_mode = "gentle"
    elif norm_mode == "implicit":
        norm_mode = "full"

    if norm_mode == "none":
        return mesh, notes

    m = repair_trimesh_gentle(mesh)
    notes.append("Applied gentle triage (dedup, winding, normals, hole filling).")

    should_escalate = (norm_mode == "full") or (escalate_to_full and not m.is_watertight)
    if should_escalate:
        try:
            notes.append(f"Deploying Screened Poisson reconstruction (depth={poisson_depth}).")
            m = repair_trimesh_poisson(m, depth=poisson_depth)
            notes.append("Completed Screened Poisson reconstruction.")
        except Exception as exc:
            notes.append(f"Poisson reconstruction failed ({exc}); retaining gentle triage mesh.")

    return m, notes


def run_gentle_triage(input_path, output_path):
    """
    Phase 1: Lightweight Trimesh cleanup.

    Attempts to fix degenerate faces, duplicate faces, unreferenced vertices, 
    incorrect winding, inverted normals, and simple holes without altering 
    the core geometry significantly.

    Parameters
    ----------
    input_path : str
        Path to the input STL.
    output_path : str
        Path to save the gently repaired STL.
    """
    print("[+] Phase 1: Running Gentle Triage (Trimesh)...")
    mesh = trimesh.load_mesh(input_path)
    repaired = repair_trimesh_gentle(mesh)
    repaired.export(output_path)


def run_poisson_reconstruction(input_path, output_path, depth=10):
    """
    Phase 2: The Industrial Fail-Safe (Screened Poisson + Aggressive Loop).

    Uses PyMeshLab to wrap the mesh in a continuous scalar field and extract 
    a single, watertight isosurface. Follows up with aggressive non-manifold 
    removal and component purging.

    Parameters
    ----------
    input_path : str
        Path to the input STL.
    output_path : str
        Path to save the heavily reconstructed STL.
    depth : int, optional
        Octree depth for Screened Poisson reconstruction, by default 10.
    """
    print(f"[!] Phase 2: Initiating Screened Poisson Reconstruction (Depth: {depth})...")

    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(input_path)

    # Normals: PyMeshLab API varies by build/version.
    try:
        if hasattr(ms, "meshing_recompute_vertex_normals"):
            ms.meshing_recompute_vertex_normals()
        elif hasattr(ms, "compute_normal_per_vertex"):
            ms.compute_normal_per_vertex()
        elif hasattr(ms, "compute_normal_polygon_mesh_per_face"):
            ms.compute_normal_polygon_mesh_per_face()
        elif hasattr(ms, "compute_normal_per_face"):
            ms.compute_normal_per_face()
    except Exception:
        pass

    # 1) Generate the continuous surface
    ms.generate_surface_reconstruction_screened_poisson(depth=depth, preclean=True)

    print("    -> Purging Poisson splatter and enforcing single-shell...")

    # 2) STRICT SINGLE SHELL POLICY:
    # Remove small disconnected components first (ghost sheets), then attempt to keep only the dominant shell.
    try:
        ms.meshing_remove_connected_component_by_face_number(mincomponentsize=5000)
    except Exception as e:
        print(f"    -> Small-component purge skipped: {e}")

    # If multiple shells remain, crank threshold to keep only the largest (heuristic).
    try:
        topo = ms.get_topological_measures()
        shells = topo.get("connected_components_number", -1)
        faces = topo.get("faces_number", -1)
        if shells and shells > 1 and faces and faces > 0:
            # Keep only components close to the full mesh size (drops internal shells/bubbles).
            ms.meshing_remove_connected_component_by_face_number(
                mincomponentsize=max(5000, int(0.90 * faces))
            )
    except Exception:
        pass

    try:
        ms.meshing_remove_duplicate_faces()
    except Exception:
        pass
    try:
        ms.meshing_remove_unreferenced_vertices()
    except Exception:
        pass

    print("    -> NUKING non-manifold geometry, sealing holes, and unifying normals...")

    # 3) Aggressive iterative fix loop until Trimesh says watertight (or we hit max loops).
    max_loops = 6
    tmp_check = output_path + ".tmp_check.stl"
    tried_invert = False
    for i in range(max_loops):
        # Merge coincident/near-coincident vertices so the mesh is actually stitched.
        # (STL can carry fully de-duplicated geometry, but some reconstruction outputs
        # can leave per-face disconnected vertices unless we explicitly merge.)
        try:
            ms.meshing_merge_close_vertices()
        except Exception:
            pass

        # Nuke non-manifold geometry (edges + vertices)
        try:
            ms.meshing_repair_non_manifold_edges(method="Remove Faces")
        except TypeError:
            try:
                ms.meshing_repair_non_manifold_edges()
            except Exception:
                pass
        except Exception:
            pass

        try:
            ms.meshing_repair_non_manifold_vertices()
        except Exception:
            pass

        # Close holes aggressively
        try:
            ms.meshing_close_holes(maxholesize=10000)
        except Exception:
            pass

        # Unify normals / orientation
        try:
            ms.meshing_re_orient_faces_coherently()
        except Exception:
            pass
        try:
            ms.meshing_re_orient_faces_by_geometry()
        except Exception:
            pass
        try:
            if hasattr(ms, "compute_normal_per_vertex"):
                ms.compute_normal_per_vertex()
        except Exception:
            pass

        # Save and validate with Trimesh's strict check
        ms.save_current_mesh(tmp_check)
        try:
            m = trimesh.load_mesh(tmp_check)
            if isinstance(m, trimesh.Scene):
                m = m.dump(concatenate=True)
            wt = bool(m.is_watertight)
        except Exception:
            wt = False

        bounds = get_boundary_edges(ms)
        shells = get_connected_components(ms)
        nm_e = get_non_two_manifold_edges(ms)
        nm_v = get_non_two_manifold_vertices(ms)
        print(
            f"    -> Loop {i+1}/{max_loops}: watertight={wt} "
            f"boundary_edges={bounds} shells={shells} nm_edges={nm_e} nm_verts={nm_v}"
        )

        if wt:
            break

        # If still not watertight, try a single global invert once (sometimes orientation is flipped).
        if not tried_invert:
            try:
                ms.meshing_invert_face_orientation()
                tried_invert = True
            except Exception:
                tried_invert = True

    # Final save
    try:
        if os.path.exists(tmp_check):
            os.replace(tmp_check, output_path)
        else:
            ms.save_current_mesh(output_path)
    except Exception:
        ms.save_current_mesh(output_path)


def repair_stl(file_path):
    """
    Execute the full repair suite on an STL file.

    Evaluates mesh health before and after. Always attempts Phase 1 (triage). 
    If the mesh remains non-watertight or has intersections, it escalates 
    to Phase 2 (Poisson reconstruction).

    Parameters
    ----------
    file_path : str
        Path to the STL file to repair. The output will be saved with a 
        `_Repaired.stl` suffix.
    """
    print(f"\n========================================")
    print(f" Initiating Clean Repair Suite for: {os.path.basename(file_path)}")
    print(f"========================================")

    if not os.path.exists(file_path):
        print(f"[-] CRITICAL ERROR: File '{file_path}' not found!")
        return

    print_health_report(file_path, stage="Initial")

    base, ext = os.path.splitext(file_path)
    temp_path = f"{base}_temp.stl"
    final_path = f"{base}_Repaired.stl"

    # Always run gentle triage first
    run_gentle_triage(file_path, temp_path)

    # Check if triage was enough
    mesh = trimesh.load_mesh(temp_path)
    nm_edges = get_pymeshlab_health(temp_path)
    intersections = check_intersections_pymeshlab(temp_path)

    if mesh.is_watertight and nm_edges == 0 and intersections == 0:
        print("[+] Phase 1 Successful! Mesh is perfectly clean.")
        os.rename(temp_path, final_path)
    else:
        print(f"[-] Phase 1 Insufficient (Intersections: {intersections}, NM Edges: {nm_edges}).")
        # Deploy Poisson Reconstruction
        run_poisson_reconstruction(temp_path, final_path, depth=10)
        if os.path.exists(temp_path):
            os.remove(temp_path)

    print_health_report(final_path, stage="Final")
    print(f"[+] Output saved to: {final_path}")


if __name__ == "__main__":
    # Point directly to the BMesh Repaired file
    target_file = "top_part_new_BMeshRepaired.stl"
    repair_stl(target_file)

