# Implementation Plan: Dual-EDT Boundary-Driven Grading

## Objective
Implement a "Control Surfaces" grading mode where the user defines Start Surface(s) and End Surface(s). The engine will calculate the true Euclidean distance from both boundaries and smoothly interpolate pore size and solid fraction between them using a blending weight $W = D_A / (D_A + D_B)$.

## Step 1: Create the Math Engine (`graphite/implicit/boundary_graded.py`)
1. Create a new file in `graphite/implicit/` called `boundary_graded.py`.
2. **Imports:** `numpy`, `trimesh`, `marching_cubes`, `edt` (from scipy.ndimage), `evaluate_tpms` (from graphite.math.tpms), and `voxelize_mesh_and_edt` (from graphite.geometry.masking).
3. **Function Signature:** `def generate_boundary_graded_lattice(stl_path, lattice_type='Gyroid', start_surfaces=[0], end_surfaces=[1], pore_sizes=[2.0, 5.0], solid_fractions=[0.4, 0.15], resolution=0.25, center_origin=False, output_path=None):`
4. **Internal Logic:**
   - **Voxelization:** Load the mesh and run `voxelize_mesh_and_edt` to get the base grid.
   - **Distance A ($D_A$):** Extract the `start_surfaces` faces. Scatter a dense point cloud on them. Map the points to the voxel grid indices. Create a boolean mask and run `edt` (just like the localized shelling trick) to generate the $D_A$ distance field.
   - **Distance B ($D_B$):** Repeat the exact same point-cloud EDT process for the `end_surfaces` to generate the $D_B$ distance field.
   - **The Blending Weight ($W$):** Calculate `W = D_A / (D_A + D_B + 1e-8)`. Ensure it is clipped between 0.0 and 1.0. Apply the smoothstep function: `W = 3*W**2 - 2*W**3`.
   - **Interpolation:** Calculate `L_grid = pore_sizes[0] * (1-W) + pore_sizes[1] * W` and `SF_grid = solid_fractions[0] * (1-W) + solid_fractions[1] * W`. Calculate `K_grid = 2*np.pi / np.maximum(L_grid, 0.001)`.
   - **Extraction:** Evaluate TPMS, apply boolean intersection with `cad_sdf`, run marching cubes, shift vertices, center if requested, and return/export the mesh.

## Step 2: Update the UI (`app.py`)
1. **Step 4 (Grading Options):**
   - Add `"Boundary-Driven (Dual-EDT)"` to the `grading_mode` selectbox.
   - Add UI inputs that only appear for this mode:
     - A text input for "Start Surface IDs (comma separated)" (e.g., "0, 1"). Parse into a list of integers: `st.session_state.params['bound_start']`.
     - A text input for "End Surface IDs (comma separated)" (e.g., "5"). Parse into a list of integers: `st.session_state.params['bound_end']`.
     - A text input for "Pore Sizes (Start, End) in mm". Parse into a list of floats.
     - A text input for "Solid Fractions (Start, End)". Parse into a list of floats.
   - Ensure the "Launch Surface Picker" button is highly visible here so users can pop open PyVista to find their face IDs.
2. **Step 6 (Routing):**
   - Import `generate_boundary_graded_lattice`.
   - Add an `elif grading_mode == 'Boundary-Driven (Dual-EDT)':` block that calls the new engine with the user's parameters.