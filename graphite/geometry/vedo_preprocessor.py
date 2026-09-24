"""
Graphite Geometry - Vedo Preprocessor

Interactive boundary condition (BC) picker using Vedo. Allows the user to select
fixed and load faces on an STL mesh, define bolt zones, and export the selections
to a JSON manifest for topology optimization.
"""
import hashlib
import json
import os
from pathlib import Path

import trimesh
import vedo
import numpy as np

class BCPreprocessor:
    """
    Interactive STL face picking for BC manifests.

    Parameters
    ----------
    file_path : str | Path
        Path to the STL file.
    manifest_rel : str | Path | None
        Relative path (from cwd) for ``bc_manifest.json`` output. Default: sandbox manifest.
    enable_symmetry_face_mode : bool
        If True, enables **Symmetry** picking (toggle with **S**): each click runs a coplanar
        face sweep and stores face IDs in ``symmetry_elements``. Original **[P]** coplanar
        behavior is disabled in this mode to avoid conflicting shortcuts.
    """

    def __init__(
        self,
        file_path,
        manifest_rel=None,
        enable_symmetry_face_mode: bool = False,
    ):
        self.file_path = Path(file_path)
        self.tm_mesh = None
        self.checksum = None
        self.manifest_rel = Path(
            manifest_rel or "experiments/scikit_topt_sandbox/bc_manifest.json"
        )
        self.enable_symmetry_face_mode = enable_symmetry_face_mode
        
    def _compute_checksum(self):
        sha256_hash = hashlib.sha256()
        with open(self.file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def run(self):
        """
        Launch the interactive Vedo 3D viewer to pick boundary condition faces.

        The selected fixed, load, and symmetry faces are serialized into a JSON 
        manifest along with the mesh's SHA256 checksum for reproducibility.

        Returns
        -------
        dict
            The serialized boundary condition manifest containing indices and coordinates.
        """
        print(f"Loading {self.file_path} with process=False to maintain absolute index integrity...")
        # Mandatory: process=False to prevent re-indexing
        self.tm_mesh = trimesh.load(self.file_path, process=False)
        self.checksum = self._compute_checksum()
        
        print(f"Mesh loaded: {len(self.tm_mesh.vertices)} vertices, {len(self.tm_mesh.faces)} faces.")
        
        faces_arr = np.asarray(self.tm_mesh.faces, dtype=np.int64)
        face_normals = np.asarray(self.tm_mesh.face_normals, dtype=float)
        face_centers = np.asarray(self.tm_mesh.triangles_center, dtype=float)

        # State Management
        state = {
            "fixed_indices": set(),
            "load_indices": set(),
            "bolt_centroid": None,
            "current_cell_id": None,
            "current_pt": None,
            "coplanar_mode": False,
            "symmetry_mode": False,
            "coplanar_highlight": set(),
            "symmetry_nodes": set(),
            "symmetry_elements": set(),
        }

        # Convert to vedo Mesh
        vmesh = vedo.Mesh([self.tm_mesh.vertices, self.tm_mesh.faces])
        
        # Visual Enhancements
        vmesh.compute_normals().flat() # Flat shading
        vmesh.linecolor("black").linewidth(0.5) # Simulate silhouette / wireframe overlay
        
        # Color: 0=Grey, 1=fixed, 2=load, 3=current, 4=coplanar highlight, 5=symmetry (accumulated)
        colors = np.zeros(vmesh.ncells, dtype=np.uint8)
        vmesh.celldata["StateColor"] = colors
        vmax_color = 5 if self.enable_symmetry_face_mode else 4
        vmesh.cmap("Pastel1", "StateColor", on="cells", vmin=0, vmax=vmax_color)
        
        vedo.settings.enable_default_keyboard_callbacks = False

        plt = vedo.Plotter(title="Vedo Boundary Condition Pre-Processor", axes=4)
        
        # HUD Text Actor
        hud = vedo.Text2D("", pos="top-left", font="VictorMono", c="black", bg="white", alpha=0.8, s=0.8)
        
        bolt_sphere = vedo.Sphere(r=1.5, c="magenta").alpha(0)

        def coplanar_faces_and_nodes(
            seed_cell_id: int,
            tol_dot: float = 1e-4,
            tol_plane: float = 1e-4,
        ):
            """Faces sharing the same plane as *seed_cell_id* (parallel normal, on-plane distance)."""
            n1 = face_normals[seed_cell_id]
            n1 = n1 / (np.linalg.norm(n1) + 1e-30)
            p1 = face_centers[seed_cell_id]
            normal_alignment = np.abs(face_normals @ n1)
            plane_distance = np.abs((face_centers - p1) @ n1)
            mask = (normal_alignment >= (1.0 - tol_dot)) & (plane_distance <= tol_plane)
            coplanar_ids = np.nonzero(mask)[0].astype(np.int64)
            node_ids = np.unique(faces_arr[coplanar_ids].reshape(-1)).astype(np.int64)
            return coplanar_ids, node_ids

        def redraw_hud():
            c = vmesh.celldata["StateColor"]
            c[:] = 0
            if self.enable_symmetry_face_mode:
                for sf in state["symmetry_elements"]:
                    c[int(sf)] = 5
            for hf in state["coplanar_highlight"]:
                c[int(hf)] = 4
            for f_id in state["fixed_indices"]:
                c[f_id] = 1
            for l_id in state["load_indices"]:
                c[l_id] = 2

            if state["current_cell_id"] is not None:
                c[state["current_cell_id"]] = 3

            vmesh.dataset.GetCellData().GetArray("StateColor").Modified()
            
            # Update bolt
            if state["bolt_centroid"] is not None:
                bolt_sphere.pos(state["bolt_centroid"]).alpha(1)
            else:
                bolt_sphere.alpha(0)
            
            pt_str = f"({state['current_pt'][0]:.2f}, {state['current_pt'][1]:.2f}, {state['current_pt'][2]:.2f})" if state['current_pt'] else "None"
            bolt_str = f"({state['bolt_centroid'][0]:.2f}, {state['bolt_centroid'][1]:.2f}, {state['bolt_centroid'][2]:.2f})" if state['bolt_centroid'] else "None"
            
            sym_face_line = (
                f"Symmetry Faces (coplanar): {len(state['symmetry_elements'])}\n"
                if self.enable_symmetry_face_mode
                else ""
            )
            mode_help = (
                "[S]ymmetry mode — click face → coplanar sweep (tight plane)\n"
                "[F]ix | [L]oad | [B]olt | [C]lear face label\n"
                if self.enable_symmetry_face_mode
                else (
                    "[P] Coplanar select (click face → plane)\n"
                    "[F]ix | [L]oad | [B]olt | [C]lear face label\n"
                )
            )
            coplanar_or_sym = (
                f"Symmetry Mode: {'ON' if state['symmetry_mode'] else 'OFF'}\n"
                if self.enable_symmetry_face_mode
                else f"Coplanar Mode: {'ON' if state['coplanar_mode'] else 'OFF'}\n"
            )
            text = (
                "--- Graphite Selection Manager ---\n"
                f"Current Highlight: Face {state['current_cell_id'] if state['current_cell_id'] is not None else 'None'}\n"
                f"Cursor Coord: {pt_str}\n"
                f"Fixed Faces: {len(state['fixed_indices'])}\n"
                f"Load Faces: {len(state['load_indices'])}\n"
                f"{sym_face_line}"
                f"Symmetry Nodes: {len(state['symmetry_nodes'])}\n"
                f"{coplanar_or_sym}"
                f"Bolt Centroid: {bolt_str}\n"
                "----------------------------------\n"
                f"{mode_help}"
                "Press 'Q' to save and exit."
            )
            hud.text(text)
            plt.render()

        def on_click(evt):
            if not getattr(evt, "actor", None) or evt.actor is not vmesh or getattr(evt, "picked3d", None) is None:
                state["current_cell_id"] = None
                state["current_pt"] = None
                state["coplanar_highlight"] = set()
                redraw_hud()
                return
                
            pt = evt.picked3d
            cell_id = evt.actor.closest_point(pt, return_cell_id=True)

            if cell_id is None or cell_id < 0:
                return

            state["current_cell_id"] = cell_id
            state["current_pt"] = list(pt)

            if self.enable_symmetry_face_mode and state["symmetry_mode"]:
                cp_faces, node_ids = coplanar_faces_and_nodes(
                    int(cell_id), tol_dot=1e-6, tol_plane=1e-5
                )
                state["coplanar_highlight"] = set(int(x) for x in cp_faces.tolist())
                state["symmetry_elements"].update(int(x) for x in cp_faces.tolist())
                state["symmetry_nodes"].update(int(v) for v in node_ids.tolist())
                print(
                    f"[Symmetry plane] +{len(cp_faces)} faces "
                    f"(total symmetry faces {len(state['symmetry_elements'])}, "
                    f"nodes {len(state['symmetry_nodes'])})"
                )
            elif not self.enable_symmetry_face_mode and state["coplanar_mode"]:
                cp_faces, node_ids = coplanar_faces_and_nodes(int(cell_id))
                state["coplanar_highlight"] = set(int(x) for x in cp_faces.tolist())
                state["symmetry_nodes"].update(int(v) for v in node_ids.tolist())
                print(
                    f"[Coplanar] {len(node_ids)} nodes from {len(cp_faces)} faces "
                    f"(symmetry total {len(state['symmetry_nodes'])})"
                )
            else:
                state["coplanar_highlight"] = set()

            redraw_hud()

        def on_key(evt):
            key = evt.keypress.lower()
            if key == 'f' and state["current_cell_id"] is not None:
                state["load_indices"].discard(state["current_cell_id"])
                state["fixed_indices"].add(state["current_cell_id"])
                redraw_hud()
            elif key == 'l' and state["current_cell_id"] is not None:
                state["fixed_indices"].discard(state["current_cell_id"])
                state["load_indices"].add(state["current_cell_id"])
                redraw_hud()
            elif key == 'b' and state["current_pt"] is not None:
                state["bolt_centroid"] = state["current_pt"]
                redraw_hud()
            elif key == 'c' and state["current_cell_id"] is not None:
                state["fixed_indices"].discard(state["current_cell_id"])
                state["load_indices"].discard(state["current_cell_id"])
                redraw_hud()
            elif key == 'p' and not self.enable_symmetry_face_mode:
                state["coplanar_mode"] = not state["coplanar_mode"]
                if not state["coplanar_mode"]:
                    state["coplanar_highlight"] = set()
                redraw_hud()
            elif key == 's' and self.enable_symmetry_face_mode:
                state["symmetry_mode"] = not state["symmetry_mode"]
                if not state["symmetry_mode"]:
                    state["coplanar_highlight"] = set()
                redraw_hud()
            elif key == 'q':
                plt.close()

        plt.add_callback('mouse click', on_click)
        plt.add_callback('key press', on_key)
        
        plt.add(vmesh, hud, bolt_sphere)
        redraw_hud()
        print("\nLaunching Vedo... Press Q when finished to save manifest.")
        plt.show()

        # Serialize
        fixed_nodes = sorted(
            {
                int(v)
                for fid in state["fixed_indices"]
                for v in self.tm_mesh.faces[int(fid)]
            }
        )
        load_nodes = sorted(
            {
                int(v)
                for fid in state["load_indices"]
                for v in self.tm_mesh.faces[int(fid)]
            }
        )
        symmetry_nodes = sorted(int(v) for v in state["symmetry_nodes"])
        symmetry_elements = sorted(int(x) for x in state["symmetry_elements"])

        manifest = {
            "fixed_indices": sorted(list(state["fixed_indices"])),
            "load_indices": sorted(list(state["load_indices"])),
            "bolt_centroid": state["bolt_centroid"],
            "fixed_nodes": fixed_nodes,
            "load_nodes": load_nodes,
            "symmetry_nodes": symmetry_nodes,
            "mesh_checksum": self.checksum,
            "file_path": str(self.file_path.name),
        }
        if self.enable_symmetry_face_mode:
            manifest["symmetry_elements"] = symmetry_elements

        out_file = Path.cwd() / self.manifest_rel
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w") as f:
            json.dump(manifest, f, indent=4)
            
        print(f"\nManifest successfully saved to {out_file}")
        return manifest

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = "test_parts/J_hook_Updated.STL"
        
    preprocessor = BCPreprocessor(path)
    preprocessor.run()
