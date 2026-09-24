# Streamlit Workflow (Current)

This document captures the current, production Streamlit app flow in `app.py`, with emphasis on the finalized Step 4 implicit field controls.

## Wizard Steps

The app now uses a 6-step workflow:

1. **Step 1: Geometry Selection**
   - Primitive, Custom STL, or ASTM entry path.
2. **Step 2: Lattice Selection**
   - Implicit TPMS or Explicit fast path selection.
3. **Step 3: Sizing and Density**
   - **Density control** (either/or): **Solid Fraction** or **Wall Thickness (mm)**.
   - Base size mode: pore size or unit cell size.
   - Wall thickness sets physical strut width; solid fraction sets a volume-fraction target (with optional calibration in Step 4).
4. **Step 4: Boundary and Field Control**
   - Field grading controls for implicit TPMS.
5. **Step 5: Shelling and Export Modes**
   - Core, skin, or combined output controls.
6. **Step 6: Execution**
   - Resolution, output base name, export format (STL / STEP / both), generation run.

Legacy Step 3a (A/B legacy grading panel) has been removed.

## Step 4 Finalized Behavior

Step 4 is designed to support stable editing under Streamlit reruns while retaining fast iteration.

- **Grade lattice fields** controls whether the implicit backend uses:
  - `grading_mode = "Field Controls"` when enabled
  - `grading_mode = "Uniform"` when disabled
- **Density grading** (solid fraction or wall thickness, matching Step 3) and **unit-cell grading** can be enabled independently.
- **Coordinate modes**:
  - `Cartesian` (multi-axis 6-column table)
  - `Cylindrical Radius`
  - `Spherical Radius`

### Origin handling

Gradient and phase origin options:

- `Center of Mass` (default)
- `Bounding Box Center`
- `Manual`

The selected origin is used by radial/cylindrical coordinate fields and phase-aligned grading behavior.

### Density control (Step 3)

- Choose **Solid Fraction** or **Wall Thickness (mm)** — not both at once.
- Once pore size or unit cell size is set, the other density knob defines strut width vs volume fraction.
- Wall thickness mode maps to TPMS threshold using local period `L` (≈ `tau = w * k / 2` for sheet networks).
- Solid fraction mode can use **Calibrate target solid fraction to TPMS tau** in Step 4 (quantile mapping).

### Control point editing model

Control-point editors use a staged model for stability:

- A **draft** table state is edited in the UI.
- A **committed** table state is only updated on table-level **Apply**.
- Endpoints are auto-locked on apply:
  - First row position = axis/coordinate minimum
  - Last row position = axis/coordinate maximum

This is intentionally retained to avoid value-loss and double-entry issues seen with direct hot-path table mutation.

### Cartesian table format

When coordinate mode is `Cartesian`, each row contains:

- `X Position (mm)`, `X Value`
- `Y Position (mm)`, `Y Value`
- `Z Position (mm)`, `Z Value`

Users can add interior rows; endpoint rows remain min/max locked on apply.

### Preview

Step 4 includes a right-side preview panel:

- **Update Preview** generates a center slice binary image.
- Pixel size is fixed at **10 microns (0.01 mm)**.
- Background is black, lattice voxels are white.
- Slice is cropped around the part with bounded padding and capped pixel count for responsiveness.

## Removed Debug/Legacy Controls

The following temporary debugging UI has been removed:

- Step 4 debug mode toggle
- Debug trace clear button
- In-app debug event panel

Step 4 now focuses on production controls only.

## Output Path Convention

App-generated outputs are written to:

- `outputs/App outputs/<YYYY-MM-DD>/`

This applies to generated STL/STEP outputs and keeps runs grouped by date.

### Step 6 export formats (implicit TPMS)

- **STL only** — default; fastest path for printing.
- **STEP only** — faceted solid via Gmsh (may take longer).
- **STL + STEP** — writes both `<base>.stl` and `<base>.step`.

Implicit runs use an **auto-generated base name**:

`GeometryType_LatticeType_SizeXxSizeYxSizeZ_PoreOrUC_WTorSF_Grading`

Example: `Primitive_Cube_Gyroid_20p0x10p0x5p0_P5p0_SF0p33_Uniform.stl`

Optional **Export parameter manifest (.txt)** writes all implicit settings (explicit engine fields are omitted). You can still enable **Use custom base name** to override auto naming.

STEP is a tessellated import body, not a parametric feature tree.

## Backend Integration (Implicit Field Controls)

When Step 4 grading is enabled and execution runs through the implicit path, `app.py` routes to:

- `graphite.implicit.field_driven.generate_field_driven_lattice`

with committed field settings, including:

- grading toggles
- coordinate mode
- control points (scalar or Cartesian)
- solid-fraction calibration toggle
- field origin

**Lofted grading (experimental):** For a **single spine direction** (e.g. Cartesian Y on a part lying on its side), enable unit-cell grading with 1D control points along that coordinate. This is the implicit counterpart to explicit `taper_along` hex scaffolds and generalizes osteochondral Z-profiles to non-parallel ends. See [LOFTED_GRADING.md](LOFTED_GRADING.md).

## Explicit hex (not in UI yet)

Graphite’s **default hex mesh** is **Conformal Dual** (Route 3). It is validated in scripts but **not** wired into the Streamlit wizard today — the app still exposes legacy tet and old Gmsh/cropped hex paths in Advanced mode.

When hex is added to the UI, route Step 6 through:

- `generate_conformed_hex_scaffold(..., conformal_dual_mode=True, cull_mostly_external_hexes=True)`
- `synthesize_conformal_dual_lattice`
- `generate_geometry(..., crop_to_boundary=True)`

See [CONFORMAL_DUAL_HEX.md](CONFORMAL_DUAL_HEX.md).

## Practical Notes

- Table-level Apply is expected behavior and should be used after editing control-point values.
- For non-graded runs, disabling **Grade lattice fields** returns execution to uniform controls from Step 3.
- If geometry is missing in Step 4, origin-dependent and preview features are gated until geometry is available.
