# Comprehensive Retrospective: 2D Planar & Conformal Lattice Engine

This document provides a complete technical retrospective of the 2D planar, graded, and conformal lattice engine developed in `graphite.explicit.surface_lattice`. It covers design motivations, architectural decisions, what worked, what failed, root causes of subtle bugs, and engineering rationales.

---

## Table of Contents
1. [Core Purpose & Scope](#1-core-purpose--scope)
2. [What Worked & Key Capabilities Delivered](#2-what-worked--key-capabilities-delivered)
3. [What Did Not Work & Issues Encountered](#3-what-did-not-work--issues-encountered)
4. [Why We Do Specific Things (Engineering Rationales)](#4-why-we-do-specific-things-engineering-rationales)
5. [The Sine Wave Thermoforming Compensation Case Study](#5-the-sine-wave-thermoforming-compensation-case-study)
6. [Architecture & File Reference](#6-architecture--file-reference)

---

## 1. Core Purpose & Scope

### 1.1 Beyond Metamaterials
While initial research focused on auxetics and chiral metamaterials, the engine is fundamentally a **generalized 2D planar and surface-conformal lattice pipeline**. It unifies:
- **Topology generators**: Regular grids, triangles, Kagome, chiral (tri-chiral, tetra-chiral, anti-chiral), and re-entrant auxetic honeycombs.
- **Grading engines**: Spatial strut width $w(x, y)$, out-of-plane thickness $t_z(x, y)$, and spatially variable cell pitch $L(x, y)$ under constant solid-fraction constraints.
- **Surface sweeper backends**:
  - `plate`: 2D planar buffering + orthogonal Z-extrusion.
  - `cylinder` / `tube`: Circumferential cylindrical conformal unrolling.
  - `mesh`: Arbitrary curved manifold surfaces with surface-normal bar sweeps.

---

## 2. What Worked & Key Capabilities Delivered

### 2.1 Integer Whole-Unit-Cell Tessellation
- **The problem with generic boundary clipping**: Slicing periodic lattices with bounding boxes (`box.intersection(lattice)`) caused dangling strut remnants, half-split chiral rings, and stress concentrations along specimen borders.
- **The solution**: Switched to explicit whole-cell grid loops (`n_circumferential` columns $\times$ `m_vertical` rows) where unit cells are parameterized from their geometric centers.
- **Outcome**: Completely eliminated hanging struts and half-cut nodes on open boundaries while maintaining exact periodic boundary registration for cylindrical wraps.

### 2.2 Continuous Trapezoidal Strut Tapering
- **The problem with midpoint grading**: Setting a strut's width based solely on its midpoint $w = w(\mathbf{x}_{\text{mid}})$ produces piecewise constant rectangular bars with abrupt, jagged steps at cell nodes when gradients are steep.
- **The solution**: Implemented true trapezoidal tapering in `sweep_planar_2d`:
  - Widths evaluated at endpoints: $w_1 = w(\mathbf{p}_1)$, $w_2 = w(\mathbf{p}_2)$.
  - Strut polygon generated as a directed linear trapezoid:
    $$\mathbf{n} = \frac{[-v_y, v_x]}{\|\mathbf{v}\|}, \quad \mathbf{p}_{1, \pm} = \mathbf{p}_1 \pm \mathbf{n} \frac{w_1}{2}, \quad \mathbf{p}_{2, \pm} = \mathbf{p}_2 \pm \mathbf{n} \frac{w_2}{2}$$
  - Circular disc pads added at segment endpoints (`Point(p).buffer(w/2)`) to ensure seamless union without degenerate knife-edge gaps.
- **Outcome**: Silky smooth transitions across large width gradients without meshing artifacts.

### 2.3 Variable Cell Sizing with Constant Solid Fraction
- Enabled multi-scale cell sizing where cell pitch $L(x, y)$ varies across space (e.g. $4\text{ mm} \to 12\text{ mm}$).
- Derived and implemented the analytical relation for constant solid-fraction $\phi^*$:
  - Square grid: $\phi \approx \frac{2 w}{L} - \left(\frac{w}{L}\right)^2 \implies w(L) = L \left(1 - \sqrt{1 - \phi^*}\right)$
  - Triangular / Kagome: $\phi \approx \sqrt{3} \frac{w}{L} \implies w(L) = \frac{\phi^*}{\sqrt{3}} L$
- Allows graded pore sizes while maintaining identical material density across the entire part.

### 2.4 Maximal Inscribed Circle (MIC) Pore Sizing
- Surgeons and scaffold designers specify pore clearance via the Maximal Inscribed Circle diameter ($D_{\text{pore}}$) rather than bounding box pitch.
- Closed-form formulas map target pore diameter $D_{\text{pore}}$ and strut width $t_{\text{strut}}$ directly to cell pitch $a$ and height $b$, ensuring biocompatible tissue ingrowth specifications (e.g., $2.0\text{ mm}$ clear pores with $0.4\text{ mm}$ walls) are met exactly.

### 2.5 1D Developable Curvature Mapping & Inverse Forming Engine
- Implemented exact numerical curvature $\kappa(x)$ and arc length $s(x)$ integration for 1D sine wave profiles ($z(x) = A \sin(2\pi x / \lambda)$).
- Provided `SineCurvatureMapper` caching 1-period arc lengths with cubic spline inversion $x(s)$, mapping flat sheet coordinates $s = X_{\text{flat}}$ directly to local curvature $\kappa(s)$.
- Supported outer-fiber strain compensation via internal angle modulation $\theta_0(s)$ and strut stiffening $t_{\text{strut}}(s)$.

---

## 3. What Did Not Work & Issues Encountered

### 3.1 "The Sheet Looks Exactly the Same" (Visual Imperceptibility & Curvature Symmetry)
- **The Observation**: When comparing the uncompensated flat sheet with the compensated flat sheet, both looked virtually identical in rendered PNGs.
- **Root Cause 1: Sine Wave Curvature Symmetry**:
  - The sine profile $z(x) = A \sin(2\pi x / \lambda)$ has identical peak curvature magnitudes at crests ($x = \lambda/4$) and troughs ($x = 3\lambda/4$), namely $\kappa_{\max} = \frac{0.8\pi^2}{\lambda}$.
  - The curvature drops to zero at inflection points ($x = 0, \lambda/2, \lambda$).
  - Because crests and troughs both experience peak outer-fiber bending strain, the compensation pattern is completely symmetric across the wave.
- **Root Cause 2: Physically Small Strain in Benchmark Geometry**:
  - With benchmark parameters $\lambda = 100\text{ mm}$, $A = 20\text{ mm}$, and sheet thickness $t_s = 2.0\text{ mm}$, peak curvature is $\kappa_{\max} \approx 0.079\text{ mm}^{-1}$.
  - The outer-fiber strain is:
    $$\epsilon_{x,\max} = \kappa_{\max} \cdot \frac{t_s}{2} = 0.07896 \times 1.0\text{ mm} \approx 7.90\%$$
  - The required angular pre-closing is $\Delta \theta = \beta_{\text{deg}} \cdot \epsilon_x \approx 4.81^\circ$ (changing from $20.0^\circ$ to $15.19^\circ$).
  - On a $5.5\text{ mm}$ unit cell, a $4.8^\circ$ angle shift changes the waist width by only **$0.46\text{ mm}$** (from $1.57\text{ mm}$ to $2.03\text{ mm}$).
  - In a 1200-pixel wide render of a 132 mm sheet containing 24 cell columns, each cell occupies ~45 pixels, and a $0.46\text{ mm}$ shift is **barely 3 pixels wide**. To the human eye, this is imperceptible without extreme zoom or color-coded diagnostic overlays.
- **Root Cause 3: Flat Sheet vs. Formed Shape**:
  - Rendering the unbent flat sheet hides the actual physical deformation. What the user actually cares about is what happens **after** it is thermoformed over the cylinder/sine tool. Without a 3D deformed preview showing uniform pores vs. distorted pores, the flat sheet appears un-graduated.

### 3.2 Left-Side Disconnection Bug on Open Plates
- **The Issue**: In open-boundary flat plates (`periodic_x=False`), cells on the leftmost boundary appeared disconnected.
- **Root Cause**: The horizontal waist connector loop was originally written assuming cylindrical periodic wrapping (`periodic_x=True`), bridging cell $i$ to cell $i+1$. In open mode, when boundary conditions were checked, indexing offsets caused the leftmost connector to be dropped or misaligned with adjacent columns.
- **Fix**: Decoupled waist coordinates $u_{\text{wr}, i}$ of column $i$ and $u_{\text{wl}, i+1}$ of column $i+1$, adding explicit guards `if periodic_x or (i < cols - 1): raw_segments.append((p_wr, p_next_wl))`.

### 3.3 "Tri-Zone Top as Only Nodes" Issue
- **The Issue**: In an exploratory script, a tri-zone lattice rendered with disconnected circular nodes floating in space without connecting struts.
- **Root Cause**: The user's requested solid fraction or strut width was too small relative to the circular node radius $r_{\text{node}}$. When the in-plane strut width buffer was computed, the struts were clipped or failed to bridge the distance $L - 2r_{\text{node}}$, leaving only isolated circular islands.
- **Rule Established**: Always evaluate connectivity geometrically: strut length must exceed node overlap ($L > 2 r_{\text{node}}$), and strut width must be clamped above the minimum printability limit ($w_{\min} \ge 0.4\text{ mm}$).

### 3.4 Writing Ad-Hoc Scripts Instead of Using the Unified Engine
- **The Issue**: Early iterations spawned separate one-off generator scripts that reinvented wireframe tessellation, leading to inconsistent behaviors and unmaintained code paths.
- **Strict Rule**: All 2D planar, cylindrical, and surface lattice generation **must** go through `generate_surface_lattice(...)` in `graphite/explicit/surface_lattice/pipeline.py`. No standalone procedural loops.

---

## 4. Why We Do Specific Things (Engineering Rationales)

| Design Choice | Rationale |
|---------------|-----------|
| **Integer Unit Cell Tessellation (No Box Clipping)** | Preserves structural integrity, prevents dangling struts, and eliminates stress concentrations along boundaries. |
| **Trapezoidal Tapering with Round Joint Discs** | Eliminates jagged stepping across width gradients; joint discs prevent knife-edge non-manifold facets during boolean union. |
| **Developable Flat Arc Length ($S \approx 132\text{ mm} > \lambda$)** | A sheet bent without membrane stretching preserves its neutral axis arc length. Printing a sheet of length $\lambda = 100\text{ mm}$ would fail to cover a sine period of wavelength $100\text{ mm}$ because the true developable length is $132.07\text{ mm}$. |
| **Waist Clamping Bounds ($w_{\text{waist}} \in [0.10 a, 0.85 a]$)** | Clamping prevents self-intersecting diamond degenerate struts when $\theta$ is too large, and prevents struts from merging into flat horizontal bars when $\theta$ is too small. |
| **Maximal Inscribed Circle (MIC) Metric** | Clinical tissue ingrowth and biological fluid transport depend strictly on unobstructed spherical/circular clearance, not strut-to-strut centerlines. |
| **Watertight Manifold Volume Check (`if mesh.volume < 0`)** | Shapely polygon extrusions can yield clockwise-wound 2D contours depending on geometry, producing negative volume / inverted normals. The engine automatically detects and flips face normals to guarantee outward-pointing, watertight 2-manifold STL exports. |

---

## 5. The Sine Wave Thermoforming Case Study

### 5.1 Geometry & Governing Equations
- **Sine Surface**: $z(x) = A \sin\left(\frac{2\pi x}{\lambda}\right)$ with $A = 20\text{ mm}, \lambda = 100\text{ mm}$.
- **Extrusion Thickness**: $t_s = 2.0\text{ mm}$.
- **Arc Length Integral**: $s(x) = \int_0^x \sqrt{1 + \left(\frac{2\pi A}{\lambda}\cos\left(\frac{2\pi u}{\lambda}\right)\right)^2} du$. Total developable period: $S \approx 132.07\text{ mm}$.
- **Local Curvature**: $\kappa(x) = \frac{|z''(x)|}{[1 + (z'(x))^2]^{3/2}}$, peaking at $0.07896\text{ mm}^{-1}$.
- **Outer Fiber Strain**: $\epsilon_x(s) = \kappa(s) \cdot \frac{t_s}{2} \in [0\%, 7.9\%]$.

### 5.2 Inverse Compensation Functions
1. **Re-entrant Angle Modulation**:
   $$\theta_0(s) = \theta^* - \beta \left(\frac{t_s}{2}\right) \kappa(s)$$
   - At inflection ($\kappa = 0$): $\theta_0 = 20.0^\circ$
   - At crest / trough ($\kappa = \kappa_{\max}$): $\theta_0 = 15.19^\circ$
2. **Strut Thickness Grading**:
   $$t_{\text{strut}}(s) = t_{\text{base}} \left(1 + \gamma \frac{\kappa(s)}{\kappa_{\max}}\right)$$
   - Grades from $0.80\text{ mm}$ up to $0.92\text{ mm}$ (+15%) at peak curvature zones to flexurally stiffen the re-entrant hinges.

---

## 6. Architecture & File Reference

| Module / Script | Description |
|-----------------|-------------|
| [`graphite/explicit/surface_lattice/grading.py`](../graphite/explicit/surface_lattice/grading.py) | Mathematical foundation: `compute_sine_curvature`, `SineCurvatureMapper`, constant solid fraction formulas, and spatial grading evaluators. |
| [`graphite/explicit/surface_lattice/unit_cells.py`](../graphite/explicit/surface_lattice/unit_cells.py) | Whole-cell tessellators: `tessellate_reentrant_domain`, `tessellate_chiral_domain`, `tessellate_graded_cell_domain`. Dynamic per-column angle & waist width modulation. |
| [`graphite/explicit/surface_lattice/sweepers.py`](../graphite/explicit/surface_lattice/sweepers.py) | Solidification backends: `sweep_planar_2d` (with continuous trapezoidal tapering), `sweep_cylindrical_prism`, and `sweep_surface_normal_bars`. |
| [`graphite/explicit/surface_lattice/pipeline.py`](../graphite/explicit/surface_lattice/pipeline.py) | Primary entrypoint: `generate_surface_lattice(...)`. Dispatches surface types, handles `sine_compensation` parameter bundle, and produces watertight 3D meshes. |
| [`tests/test_surface_lattice.py`](../tests/test_surface_lattice.py) | Core unit tests for chiral, re-entrant, cylindrical wrapping, and CAD collar fusion. |
| [`tests/test_surface_lattice_grading.py`](../tests/test_surface_lattice_grading.py) | Unit tests for sine curvature mapping, arc length roundtrips, angle compensation, and variable cell sizing. |
| [`scripts/case_studies/sine_wave_forming_compensation.py`](../scripts/case_studies/sine_wave_forming_compensation.py) | Case study runnable generating STLs and high-resolution diagnostic profile plots. |
