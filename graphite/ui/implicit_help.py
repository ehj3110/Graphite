"""Structured per-section help & workflow documentation for Graphite Implicit Studio."""

from __future__ import annotations

_STYLE_PREFIX = """
<style>
  body { color: #e2e8f0; font-family: "Segoe UI", sans-serif; font-size: 13px; line-height: 1.48; }
  h3 { color: #00f5d4; margin-top: 0; margin-bottom: 8px; }
  h4 { color: #38bdf8; margin-top: 12px; margin-bottom: 4px; }
  p { margin: 6px 0; }
  ul { margin: 4px 0 8px 18px; padding-left: 0; }
  li { margin-bottom: 5px; }
  b { color: #f8fafc; }
  code { background-color: #1e293b; color: #38bdf8; padding: 1px 4px; border-radius: 3px; }
  .note { background-color: #0f172a; border-left: 3px solid #fbbf24; padding: 8px 10px; margin-top: 8px; border-radius: 4px; }
</style>
"""


def _wrap(body_html: str) -> str:
    return f"<html><head>{_STYLE_PREFIX}</head><body>{body_html}</body></html>"


SECTION_HELP_DOCS: dict[str, tuple[str, str]] = {
    "quick_start": (
        "How to Use Graphite Implicit Studio",
        _wrap(
            """
            <h3>Quick-Start Workflow</h3>
            <p>Graphite Implicit Studio keeps viewport interaction fast by deferring full high-resolution 3D meshing until final export.</p>
            <ul>
              <li><b>1. CAD Geometry &amp; Health:</b> Choose a built-in Primitive or load any <code>.stl</code> file, optionally center at <code>(0, 0, 0)</code>, and repair non-watertight shells in memory.</li>
              <li><b>2. Architecture &amp; Shelling:</b> Choose a TPMS family or <code>Woodpile</code>, and check <b>Generate Skin</b> only if you want a solid outer shell (internal-only fillet keeps the outer CAD surface flush).</li>
              <li><b>3. 4-Way Sizing &amp; Calibration:</b> Pick any <b>two distinct drivers</b> from the two dropdowns. Those two inputs unlock while the other two stay grayed out and autofill dynamically.</li>
              <li><b>4. Unified Functional Grading:</b> Keep <b>Uniform</b> (with optional Euler rotation) or apply spatial ramps, Multi-Knot tables, or Piecewise Discrete Bands.</li>
              <li><b>5. Surface Micro-Features (Optional):</b> Preview analytical surface textures in the 2D slice or explicit micropillars on the 3D surface.</li>
              <li><b>6. Export:</b> Click <b>Generate Production Lattice</b> to run full-resolution multi-threaded Flying Edges + Exact CAD Trim and write <code>.stl</code>, <code>.3mf</code>, and reproducibility manifests.</li>
            </ul>
            <h4>Viewport Controls &amp; Shortcuts</h4>
            <ul>
              <li><b>Rotate:</b> Left-click and drag (Trackball orbit).</li>
              <li><b>Pan:</b> Middle-drag or <code>Shift</code> + Left-drag.</li>
              <li><b>Zoom:</b> Mouse wheel or Right-drag.</li>
              <li><b>Views:</b> <code>1</code> Iso, <code>2</code> Top, <code>3</code> Front, <code>4</code> Side, <code>R</code> Reset camera, <code>Ctrl+P</code> Snapshot PNG.</li>
            </ul>
            <div class="note">
              <b>Why Previews Stay Fast:</b> <b>2D Surface (Ultra-Sharp)</b> projects exact contours directly onto your CAD skin over an opaque gray backdrop, and <b>3D Flying Edges (Coarse)</b> caps previews at ~350k voxels. Full voxel density runs only on export.
            </div>
            """
        ),
    ),
    "cad_geometry": (
        "1. CAD Geometry & Health",
        _wrap(
            """
            <h3>1. CAD Geometry &amp; Health</h3>
            <p>Defines the bounding solid volume trimmed around your implicit lattice.</p>
            <ul>
              <li><b>Primitive:</b> Generates exact analytical shapes (<code>Cube</code>, <code>Sphere</code>, <code>Cylinder</code>, <code>Torus</code>, <code>Tube</code>) sized by <b>X / Y / Z (mm)</b>.</li>
              <li><b>STL Fixture / File:</b> Pick any fixture from <code>test_parts/</code> or browse to any external <code>.stl</code> mesh.</li>
              <li><b>Center CAD at Origin (0, 0, 0):</b> Translates the part center to world origin so symmetric Z/radial grading profiles align cleanly around <code>0 mm</code>.</li>
              <li><b>Watertight Health &amp; Repair:</b> Displays face count, physical bounds, and manifold status. If an imported STL has holes or open boundary edges, click <b>Repair</b> for non-destructive in-memory cleanup before slicing or meshing.</li>
            </ul>
            """
        ),
    ),
    "architecture_shelling": (
        "2. Architecture & Shelling",
        _wrap(
            """
            <h3>2. Architecture &amp; Shelling</h3>
            <ul>
              <li><b>Architecture:</b> Choose among 7 sheet TPMS structures (<code>Gyroid</code>, <code>Diamond</code>, <code>Schwarz-P</code>, <code>Schwarz-Diamond</code>, <code>Neovius</code>, <code>Lidinoid</code>, <code>Split-P</code>) or Cartesian bar/rod <code>Woodpile</code>.</li>
              <li><b>True Shifted Woodpile</b> (Woodpile only): Alternates stacked 0&deg;/90&deg; cylindrical struts across adjacent Z-layers instead of intersecting bars in the same plane.</li>
              <li><b>Generate Skin:</b> When unchecked, exports bare conformal lattice (<code>core</code> mode). Checking <b>Generate Skin</b> enables <code>combined</code> mode and reveals shelling parameters:
                <ul>
                  <li><b>Skin (mm):</b> Solid outer wall thickness measured inward from the CAD surface.</li>
                  <li><b>Fillet (mm):</b> Smooth blend radius applied <b>strictly at the internal junction</b> between lattice walls and inner skin so outer CAD walls stay flat without TPMS ridges.</li>
                  <li><b>Blend Method:</b> Smooth-min kernel (<code>polynomial</code>, <code>exponential</code>, or <code>chamfer</code>).</li>
                </ul>
              </li>
            </ul>
            <div class="note">
              <b>Exact CAD Trim:</b> Always enabled by default in Graphite Implicit (<code>exact_cad_trim=True</code>). Manifold 2-solid CAD meshes automatically run sub-voxel B-Rep Boolean trimming on production export for crisp exterior corners.
            </div>
            """
        ),
    ),
    "four_way_sizing": (
        "3. 4-Way Sizing & Calibration",
        _wrap(
            """
            <h3>3. 4-Way Sizing &amp; Calibration</h3>
            <p>All four lattice dimensions are mathematically coupled:</p>
            <ul>
              <li><b>Drivers Dropdowns:</b> Both dropdowns can pick any of the four metrics. Until <b>two distinct drivers</b> are selected, all four inputs remain grayed out at studio defaults (<code>1.500 mm</code> cell, <code>0.800 mm</code> / 800 &micro;m pore, <code>0.330</code> SF, <code>0.500 mm</code> wall). Once two drivers are chosen, those two boxes unlock while the other two stay grayed out and autofill live.</li>
              <li><b>Unit Cell L (mm):</b> Periodic spatial repeat wavelength <code>L = 2&pi;/k</code>.</li>
              <li><b>Pore Size (MIS, mm):</b> Estimated Maximum Inscribed Sphere pore diameter inside the open void channels.</li>
              <li><b>Solid Fraction (0&ndash;1):</b> Volume fraction of solid material inside the lattice domain (e.g. <code>0.33</code> = 33% dense / 67% porous).</li>
              <li><b>Wall Thickness (mm):</b> Characteristic sheet wall or strut thickness.</li>
              <li><b>Voxel Pitch &amp; Auto Nyquist:</b> Automatically chooses fine production sampling to resolve at least ~3 voxels across the thinnest wall and &ge;16 voxels per unit cell period (uncheck to set pitch manually).</li>
              <li><b>Calibrate Exact (LUT / Newton):</b> Runs numerical calibration against the unit cell geometry to converge the true MIS pore diameter and exact volume fraction quantile.</li>
            </ul>
            <div class="note">
              <b>Nominal/Average vs. Uniform Wall Thickness:</b> Standard TPMS implicit equations (<code>|F(x)| &le; &tau;</code>) define thickness via a constant scalar threshold <code>&tau; = w&middot;&pi;/L</code>. Because <code>||&nabla;F||</code> varies slightly across a periodic TPMS unit cell, local wall thickness varies naturally around its nominal/average target (e.g. saddle regions vs. flat inflections). When truly uniform physical thickness is required, select <b>Multi-Knot &rarr; Normalized SDF Blend</b> (divides by analytical <code>||&nabla;F||</code> for unit gradient) or use <b>Woodpile</b>.
            </div>
            """
        ),
    ),
    "functional_grading": (
        "4. Unified Functional Grading",
        _wrap(
            """
            <h3>4. Unified Functional Grading</h3>
            <ul>
              <li><b>Apply Grading Preset...:</b> One-click biomedical/engineering templates (<i>Osteochondral Z-Loft</i>, <i>Cube Thirds Stepped</i>, <i>Radial Vascular Scaffold</i>) scaled to your active CAD bounds.</li>
              <li><b>Uniform:</b> Constant cell size and threshold throughout the part, with optional X/Y/Z Euler rotation angles (&deg;) to orient lattice planes.</li>
              <li><b>Linear / Radial Ramp:</b> Smoothly interpolates Solid Fraction, Cell Size (Chirp via integrated phase), or Both along <code>Z</code>, <code>X</code>, <code>Y</code>, or <code>Radial</code> distance.</li>
              <li><b>Multi-Knot Control Table:</b> Specify exact <code>Pos (mm)</code>, <code>L (mm)</code>, <code>w (mm)</code>, and <code>SF</code> knots along Cartesian, Cylindrical, or Spherical coordinates. Choose <b>Normalized SDF Blend</b> (blends metric-normalized SDFs across zones with uniform wall thickness) or <b>Jacobian Phase Integration</b> (continuous phase chirp).</li>
              <li><b>Piecewise Discrete Bands:</b> Sharp stepped Z-zones with customizable breaks; keep <b>Phase-Continuous W</b> checked to eliminate seams across zone boundaries.</li>
            </ul>
            """
        ),
    ),
    "micro_features": (
        "5. Surface Micro-Features (Optional)",
        _wrap(
            """
            <h3>5. Surface Micro-Features</h3>
            <ul>
              <li><b>Enable Procedural Surface Texture:</b> Superimposes high-frequency micro-relief (<code>microgrooves</code>, <code>bumps</code>, <code>knurling</code>, or <code>spinodal</code>) onto the solid field. Tune Amplitude, Wavelength, waveform Profile (<code>sine</code>/<code>triangle</code>/<code>square</code>), and Displacement mode (<code>centered</code>, <code>emboss</code>, <code>engrave</code>). Visible live in the <b>2D Cross-Section Slice</b> tab.</li>
              <li><b>Enable Explicit Micropillar Forest:</b> Seeds oriented cylindrical micro-pillars over the lattice surface (Poisson-disk or random spacing, <code>all</code>, <code>internal_only</code>, or <code>outer_only</code>). Diameter, Height, and Spacing are specified in <code>&micro;m</code>, with optional upward overhang filtering for printability. Previewed as lightweight amber glyphs in 3D before final union.</li>
            </ul>
            """
        ),
    ),
    "telemetry": (
        "Field & Voxel Telemetry",
        _wrap(
            """
            <h3>Field &amp; Voxel Telemetry</h3>
            <p>Live resource forecast for the <b>Production Synthesis</b> export grid based on your active CAD bounding box and <b>Voxel Pitch</b>:</p>
            <ul>
              <li><b>Grid:</b> Exact <code>nx &times; ny &times; nz</code> voxel dimensions and total sample count.</li>
              <li><b>Est. Memory &amp; Pitch:</b> Peak RAM estimate for scalar arrays plus physical voxel size in millimeters. Warnings appear automatically for multi-million-voxel grids.</li>
              <li><b>Profile L / SF:</b> Minimum-to-maximum unit cell size and solid fraction range across the current grading configuration.</li>
            </ul>
            """
        ),
    ),
    "calibration_status": (
        "Calibration & Pore Check",
        _wrap(
            """
            <h3>Calibration &amp; Pore Check</h3>
            <ul>
              <li><b>Status Readout:</b> Summarizes whether values come from the fast Analytical LUT or exact Newton MIS solver (showing converged <code>L</code>, measured MIS pore, residual error in mm, iterations, and threshold <code>&tau;</code>).</li>
              <li><b>Verify MIS Pore Size on Exported Mesh:</b> When checked, measures true 3D Euclidean distance transform pore diameter directly on the final exported solid mesh after synthesis and prints the measured MIS pore and confidence in the summary.</li>
            </ul>
            """
        ),
    ),
    "production_synthesis": (
        "Production Synthesis",
        _wrap(
            """
            <h3>Production Synthesis</h3>
            <ul>
              <li><b>Generate Production Lattice:</b> Runs full-resolution evaluation in a background worker thread using <b>Flying Edges</b> isosurface extraction (plus Exact CAD Trim and optional skin/micro-features) without freezing the UI.</li>
              <li><b>Refresh 3D Footprint &amp; Slice:</b> Manually forces an immediate refresh of the 3D surface preview and 2D slice view.</li>
              <li><b>Progress Bar &amp; Log:</b> Displays step-by-step stage timings, face counts, and Manifold3D trim telemetry.</li>
            </ul>
            """
        ),
    ),
    "export_ledger": (
        "Export & Recipe Ledger",
        _wrap(
            """
            <h3>Export &amp; Recipe Ledger</h3>
            <ul>
              <li><b>Output Formats:</b> Export watertight <code>.STL</code>, <code>.3MF</code> (enabled when <code>lxml</code> is installed), and <code>JSON &amp; Manifest</code> parameter logs under <code>outputs/App outputs/YYYY-MM-DD/</code>.</li>
              <li><b>Custom Output Stem:</b> Optional filename prefix; leaves timestamped structured naming when blank.</li>
              <li><b>Load Generated 3D Mesh in Viewport:</b> Kept OFF by default so heavy million-triangle exports write directly to disk without stalling GPU memory. Check only when you want to inspect the full production triangles in 3D.</li>
              <li><b>Recipes &amp; Batch Queue:</b> Save/load complete studio states as portable <code>.json</code> recipes, run multi-recipe batch queues sequentially (<code>Run Batch Queue...</code>), open the output folder, or save an extra mesh copy elsewhere.</li>
            </ul>
            """
        ),
    ),
}
