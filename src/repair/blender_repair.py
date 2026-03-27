import bpy
import os
import sys
import addon_utils


def clean_scene():
    """
    Clear the current scene without resetting Blender preferences.

    IMPORTANT: We avoid `read_factory_settings()` because it can also clear/disable
    bundled operators/add-ons (including the 3D-Print Toolbox ops) in headless runs.
    """
    # Deselect and delete all objects in the current scene
    try:
        bpy.ops.object.select_all(action="SELECT")
        bpy.ops.object.delete(use_global=False, confirm=False)
    except Exception:
        pass

    # Purge orphaned data blocks best-effort
    for coll in (bpy.data.meshes, bpy.data.materials, bpy.data.images, bpy.data.textures):
        try:
            for datablock in list(coll):
                coll.remove(datablock)
        except Exception:
            pass


def _enable_3d_print_toolbox():
    # Blender add-on module name varies by version/build.
    candidates = [
        "object_print3d_utils",  # common official add-on name
        "mesh_print3d_toolbox",  # seen in some forks/builds
    ]
    for addon_name in candidates:
        # Try operator-based enabling first (more reliable in headless builds).
        try:
            if hasattr(bpy.ops.wm, "addon_enable"):
                bpy.ops.wm.addon_enable(module=addon_name)
            else:
                addon_utils.enable(addon_name, default_set=True, persistent=True)
            print(f"[+] Enabled 3D-Print Toolbox add-on: {addon_name}")
            # Confirm operators actually registered.
            ops = [op for op in dir(bpy.ops.mesh) if "print3d_" in op]
            if ops:
                print(f"[+] print3d ops available: {len(ops)}")
                return True
            print("[-] Warning: add-on enabled but print3d ops not registered yet.")
        except Exception as e:
            print(f"[-] Warning: could not enable {addon_name}: {e}")
    return False


def _try_op(op, label: str, **kwargs):
    """Run a bpy operator best-effort (headless-safe)."""
    try:
        res = op(**kwargs) if kwargs else op()
        return res
    except Exception as e:
        print(f"      (warning) {label} skipped: {type(e).__name__}: {e}")
        return None


def repair_stl_with_bmesh(file_path):
    print(f"\n========================================")
    print(f" Initiating BMesh + 3D-Print Toolbox Repair for: {os.path.basename(file_path)}")
    print(f"========================================")

    if not os.path.exists(file_path):
        print(f"[-] CRITICAL ERROR: File '{file_path}' not found!")
        return

    clean_scene()

    print("[+] Importing STL...")
    try:
        bpy.ops.import_mesh.stl(filepath=file_path)
    except AttributeError:
        bpy.ops.wm.stl_import(filepath=file_path)

    obj = bpy.context.selected_objects[0]
    bpy.context.view_layer.objects.active = obj

    # --- THE BMESH REPAIR SEQUENCE ---
    toolbox_ok = _enable_3d_print_toolbox()

    print("[+] Entering Edit Mode...")
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')

    print("[i] Available print3d mesh ops:", [op for op in dir(bpy.ops.mesh) if "print3d_" in op])

    print("[+] Triangulating Faces...")
    bpy.ops.mesh.quads_convert_to_tris(quad_method='BEAUTY', ngon_method='BEAUTY')

    print("[+] Welding initial duplicate vertices...")
    bpy.ops.mesh.remove_doubles(threshold=0.001)

    if toolbox_ok:
        # 3D-Printing Toolbox: checks + non-manifold cleanup.
        # Note: Blender exposes many "check_*" operators and a smaller set of "clean_*" operators.
        print("[+] 3D-Print Toolbox: Running full checks...")
        _try_op(getattr(bpy.ops.mesh, "print3d_check_all"), "print3d_check_all")

        print("[+] 3D-Print Toolbox: Cleaning non-manifold geometry...")
        _try_op(getattr(bpy.ops.mesh, "print3d_clean_non_manifold"), "print3d_clean_non_manifold")

        # Re-run key checks so the console report is up to date.
        print("[+] 3D-Print Toolbox: Re-checking intersections / degenerate faces...")
        _try_op(getattr(bpy.ops.mesh, "print3d_check_intersect"), "print3d_check_intersect")
        _try_op(getattr(bpy.ops.mesh, "print3d_check_degenerate"), "print3d_check_degenerate")
    else:
        # Fallback: BMesh intersect cut (may miss some cases, but still helps).
        print("[+] Slicing Self-Intersecting Faces (BMesh Intersect fallback)...")
        _try_op(
            bpy.ops.mesh.intersect,
            "mesh.intersect",
            mode="SELECT_UNSELECT",
            separate_mode="NONE",
        )

    print("[+] Welding newly created intersection seams...")
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.remove_doubles(threshold=0.001)

    print("[+] Recalculating Normals Outside...")
    bpy.ops.mesh.normals_make_consistent(inside=False)

    print("[+] Returning to Object Mode...")
    bpy.ops.object.mode_set(mode='OBJECT')

    # --- EXPORT ---
    base, ext = os.path.splitext(file_path)
    out_path = f"{base}_ToolboxRepaired.stl"

    print(f"[+] Exporting repaired STL to: {out_path}...")
    try:
        bpy.ops.export_mesh.stl(filepath=out_path, use_selection=True)
    except AttributeError:
        bpy.ops.wm.stl_export(filepath=out_path, export_selected_objects=True)

    print("[+] BMesh Repair Complete!")
    print("========================================\n")


if __name__ == "__main__":
    if len(sys.argv) > 1 and ".stl" in sys.argv[-1].lower():
        target_file = sys.argv[-1]
    else:
        target_file = "top_part_new.stl"

    repair_stl_with_bmesh(target_file)

