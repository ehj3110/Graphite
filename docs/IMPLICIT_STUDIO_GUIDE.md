# Graphite Implicit Studio — Lab Workstation Guide

Graphite Implicit Studio is a standalone native PySide6 + VTK desktop workbench focused exclusively on implicit TPMS, graded/piecewise scaffolds, woodpile lattices, surface micro-features, calibration, and print-ready STL/3MF export.

## Quick Lab Machine Setup

Create an isolated workstation environment containing only Implicit Studio dependencies (no FEA, LBM, PyTorch, Trame, or PyVista bloat):

```powershell
python scripts\setup_implicit_studio_env.py --create-venv .venv_studio
```

Include optional `.3MF` support (`lxml`):

```powershell
python scripts\setup_implicit_studio_env.py --create-venv .venv_studio --with-3mf
```

Verify any lab machine before launching:

```powershell
.venv_studio\Scripts\python.exe scripts\setup_implicit_studio_env.py --check
```

## Launching

- **Windows one-click launcher:** double-click [launch_implicit_studio.bat](../launch_implicit_studio.bat) (or create a Desktop shortcut pointing to it)
- **Python CLI:**
  - GUI: `.venv_studio\Scripts\python.exe scripts\launch_implicit_gui.py` (or `python -m graphite.ui`)
  - Single recipe (headless): `.venv_studio\Scripts\python.exe scripts\launch_implicit_gui.py --recipe outputs\my_recipe.json`
  - Batch recipe queue: `.venv_studio\Scripts\python.exe scripts\launch_implicit_gui.py --batch r1.json r2.json --output-dir outputs\batch`

## Deploying to Another Lab Computer

**Recommended workflow: Clone Graphite + isolated `.venv_studio` (instead of a manual stripped folder or separate repo):**
1. On the target lab workstation, clone the `Graphite` repository (so updates only require `git pull`, without maintaining a separate fork/mirror).
2. Run the minimal environment installer from the repo root once:
   `python scripts\setup_implicit_studio_env.py --create-venv .venv_studio`
   (This installs only Implicit Studio's lightweight runtime dependencies and excludes heavy FEA/LBM/PyTorch/Trame stacks.)
3. Create a Windows Desktop shortcut to `launch_implicit_studio.bat` — the batch script changes into the repository root (`%~dp0`) and launches `.venv_studio\Scripts\python.exe scripts\launch_implicit_gui.py`.

*(Why not a stripped subset folder or second GitHub repo? Implicit Studio imports core geometry, math, boolean blending, Flying Edges extraction, and I/O modules directly from `graphite/`. A separate repo would drift out of sync whenever upstream math or bug fixes land. Only consider a standalone release/PyInstaller bundle later if you need distribution to non-Python end users.)*

## Keyboard Shortcuts

- `1` — Isometric preset view
- `2` — Top (`+Z`) view
- `3` — Front (`-Y`) view
- `4` — Side (`+X`) view
- `R` — Reset padded camera around CAD bounds
- `Ctrl+P` — Save window PNG snapshot to `outputs/implicit_studio/snapshots/`

## Core Workflow

1. **CAD Geometry & Health:** Choose a parametric primitive or load any `.stl` fixture (with optional **Center CAD at Origin**). Non-watertight STL shells can be repaired non-destructively in memory via **Repair**.
2. **Architecture & Shelling:** Pick any of the 7 TPMS equations or `Woodpile`. Check **Generate Skin** when an outer shell is needed (`combined` mode); checking it reveals `Shell / Blend` (`Skin`, `Fillet`) and `Blend Method`, applying smooth fillets strictly at the internal lattice-to-inner-skin junction so the outer CAD wall stays flush. **Exact CAD Trim** (`exact_cad_trim=True`) is always enabled by default in Graphite Implicit.
3. **4-Way Sizing & Calibration:** Select any two distinct driving parameters across both dropdowns (`Unit Cell L`, `Pore Size (MIS)`, `Solid Fraction`, `Wall Thickness`). All four boxes start locked (`1.50 mm` cell, `800 um` / `0.80 mm` pore, `33%` SF, `0.50 mm` wall) until two drivers are chosen; the two unchosen boxes remain grayed out and autofill automatically via the 6-pair analytical solver alongside **Auto Nyquist** voxel pitch, or click **Calibrate Exact (LUT / Newton)** for exact pore convergence.
4. **Unified Functional Grading:** Switch cleanly between `Uniform`, `Linear / Radial Ramp`, `Multi-Knot Control Table` (`Normalized SDF Blend` or `Jacobian Phase Integration` across Cartesian, Cylindrical, or Spherical axes), and `Piecewise Discrete Bands` (phase-continuous).
5. **Surface Micro-Features:** Toggle analytical Surface Texture modulation or Explicit Micropillar Forest anchors (`#fbbf24` 3D preview glyphs) before running final volume synthesis.
6. **Interactive Viewport, Cutaway & Export:** Left-click drag rotates smoothly with Trackball camera controls. Default **2D Surface (Ultra-Sharp)** renders crisp CAD-surface contours over opaque gray CAD, while optional **3D Flying Edges (Coarse)** (`~350k` voxel preview cap) previews 3D walls without slowing the UI. **Live Cutaway** uses GPU clipping during slider drag and seals closed solid caps on release; **2D Cross-Section Slice** samples a cached CAD SDF for sub-100 ms slicing alongside the live **1D Spatial Grading Profile** chart (`L`, `w`, `SF`). Heavy full-resolution **Flying Edges** extraction is deferred to **Generate Production Lattice** (or **Run Batch Queue...**), exporting watertight `.stl` / `.3mf` + `_recipe.json` + `_parameters.txt`.
