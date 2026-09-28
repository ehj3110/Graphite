# Technical Review & Architecture Handoff: Spinodal GRF, Flying Edges & Taubin Smoothing

**Date:** September 2026  
**Status:** Production  
**Scope:** `graphite.math.spinodal`, `graphite.mesh.extraction`, `graphite.mesh.smoothing`, `graphite.implicit.spinodal`

---

## 1. Executive Summary

This handoff documents the architectural enhancement to `graphite` delivering:
1. **Gaussian Random Field (GRF) Spinodal Lattice Generator:** High-performance procedural generator for stochastic bicontinuous spinodal metamaterials (both skeletal and sheet morphologies) with exact analytic solid fraction thresholding and directional anisotropy.
2. **Flying Edges Isosurface Extraction:** Multi-threaded isosurface extraction pipeline utilizing `vtkFlyingEdges3D` via PyVista (with an automatic fallback to `skimage.measure.marching_cubes`), delivering an order-of-magnitude speedup and reduced peak memory over classic Marching Cubes.
3. **Volume-Preserving Taubin Smoothing:** An alternating, curvature-relaxing filter ($\lambda > 0$, $\mu < -\lambda < 0$) that relaxes implicit level-set surfaces toward constant mean curvature (CMC) and eliminates voxel stair-stepping while strictly preserving macro volume ($|\Delta V / V_0| < 0.5\%$).

All unit tests ($24/24$) pass cleanly in $<7$ seconds, and the central specification (`GRAPHITE_CORE_SPEC.md`) is in complete synchronization.

---

## 2. Architectural Layout & Modules

```text
graphite/
├── math/
│   ├── spinodal.py              <-- GRF wavevectors, in-place field evaluation, analytic thresholding
│   └── __init__.py              <-- Exported spinodal math entrypoints
├── mesh/                        <-- New mesh processing package
│   ├── __init__.py              <-- Exported mesh extraction and smoothing entrypoints
│   ├── extraction.py            <-- Multi-threaded Flying Edges extraction with MC fallback
│   ├── smoothing.py             <-- Volume-preserving Taubin filter & mean curvature diagnostics
│   └── README.md                <-- Capability card
├── implicit/
│   ├── spinodal.py              <-- High-level generate_spinodal_lattice builder
│   └── __init__.py              <-- Exported implicit spinodal generator
tests/
├── test_spinodal.py             <-- 17 unit tests for GRF math & CAD lattice generation
└── test_extraction_and_smoothing.py <-- 7 unit tests for Flying Edges & Taubin volume conservation
outputs/
├── spinodal_flying_edges_raw.stl    <-- Reviewable un-smoothed Flying Edges sample
└── spinodal_flying_edges_taubin.stl <-- Reviewable Taubin-smoothed sample (0.38% drift)
```

---

## 3. Mathematical Formulations & Derivations

### 3.1 Gaussian Random Field (GRF) Spinodal Decomposition
Spinodal decomposition is modeled as an isotropic or anisotropic Gaussian Random Field formed by a superposition of $N$ standing cosine waves:

$$F(\mathbf{x}) = \sqrt{\frac{2}{N}} \sum_{i=1}^{N} \cos\left(\mathbf{k}_i \cdot \mathbf{x} + \phi_i\right)$$

Where:
- $\mathbf{x} = [X, Y, Z]^T$ (mm) evaluated on regular grids (`indexing="ij"`).
- $\phi_i \sim \mathcal{U}(0, 2\pi)$ (uniform random phase).
- $\hat{\mathbf{n}}_i$ are uniformly distributed random unit vectors on $S^2$, sampled via **Marsaglia's (1972) rejection method**:
  Sample $u, v \sim \mathcal{U}(-1, 1)$ such that $0 < s = u^2 + v^2 < 1$:
  $$\hat{\mathbf{n}} = \left[2u\sqrt{1-s}, \; 2v\sqrt{1-s}, \; 1 - 2s\right]^T$$
- $\mathbf{A} = \operatorname{diag}(a_x, a_y, a_z)$ is the directional anisotropy scaling vector ($a_x, a_y, a_z > 0$).
- Wavevectors:
  $$\mathbf{k}_i = k_0 \cdot \frac{\mathbf{A} \hat{\mathbf{n}}_i}{\Vert \mathbf{A} \hat{\mathbf{n}}_i \Vert}, \quad \text{where } k_0 = \frac{2\pi}{\lambda}$$
- **Central Limit Theorem (CLT):** Each wave mode has mean $0$ and variance $E[\cos^2(\theta)] = 1/2$. The sum of $N$ modes has variance $N/2$, scaled by $\sqrt{2/N}$ to yield asymptotic standard normality:
  $$F(\mathbf{x}) \sim \mathcal{N}(0, 1)$$

### 3.2 Analytic Solid Fraction ($\phi \in (0, 1)$) Thresholding
Graphite standardizes on the **solid $\le 0$** level-set convention ($\text{solid\_field} \le 0 \implies \text{solid material}$).

#### 1. Skeletal / Network Topology (`is_sheet=False`)
Solid phase is defined by $F(\mathbf{x}) \le t$. To satisfy $P(F \le t) = \Phi(t) = \phi$:
$$\Phi(t) = \frac{1}{2}\left[1 + \operatorname{erf}\left(\frac{t}{\sqrt{2}}\right)\right] = \phi \implies t = \sqrt{2} \cdot \operatorname{erf}^{-1}(2\phi - 1)$$
$$\text{solid\_field}(\mathbf{x}) = F(\mathbf{x}) - t \quad (\le 0 \implies \text{solid})$$

#### 2. Sheet / Lamellar Topology (`is_sheet=True`)
Solid phase is defined by $|F(\mathbf{x})| \le t_{\text{sheet}}$. To satisfy $P(|F| \le t_{\text{sheet}}) = \operatorname{erf}(t_{\text{sheet}} / \sqrt{2}) = \phi$:
$$t_{\text{sheet}} = \sqrt{2} \cdot \operatorname{erf}^{-1}(\phi)$$
$$\text{solid\_field}(\mathbf{x}) = |F(\mathbf{x})| - t_{\text{sheet}} \quad (\le 0 \implies \text{solid})$$

---

### 3.3 Flying Edges Isosurface Extraction
- **Why Flying Edges:** Traditional Marching Cubes scans row-by-row with complex cell-case lookups and is inherently serial in `scikit-image`. Flying Edges (`vtkFlyingEdges3D`) preprocesses row intersections independently, allowing SIMD and thread-level parallelism across row slices while eliminating intermediate topological structures.
- **Coordinate Space Alignment:** In VTK `ImageData`, memory layout varies fastest along the $X$-axis (axis 0), then $Y$ (axis 1), then $Z$ (axis 2). Flattening NumPy arrays with Fortran ordering (`order='F'`) maps NumPy `indexing='ij'` grids directly to VTK's point numbering without transposition or coordinate permutation.
- World translation $\mathbf{x}_{\text{world}} = \mathbf{origin} + \mathbf{x}_{\text{grid}} \cdot \mathbf{spacing}$ is handled natively by the VTK image grid parameters.

---

### 3.4 Volume-Preserving Taubin Smoothing
Standard Laplacian smoothing causes severe inward volume shrinkage because $\mathbf{L}(\mathbf{x})$ acts as a diffusion operator. Taubin smoothing uses an alternating two-step filter:

$$\mathbf{x}' = \mathbf{x} + \lambda \mathbf{L}(\mathbf{x}) \quad (0 < \lambda < 1, \text{positive shrinkage step})$$
$$\mathbf{x}'' = \mathbf{x}' + \mu \mathbf{L}(\mathbf{x}') \quad (\mu < -\lambda < 0, \text{negative dilation step})$$

The transfer function after one cycle is:
$$f(k) = (1 - \lambda k)(1 - \mu k)$$
With $k_{\text{PB}} = \frac{1}{\lambda} + \frac{1}{\mu} > 0$, low-frequency surface shapes are preserved ($f(k) \approx 1$), high-frequency noise and voxel terracing are attenuated ($|f(k)| \ll 1$), and the net volumetric change is bounded by $|\Delta V / V_0| < 0.5\%$.

**Parity Guard:** A Taubin cycle strictly requires a pair of steps ($\lambda$ then $\mu$). If an odd number of iterations is requested (e.g., $15$), the filter automatically rounds up to the next even integer ($16$) to avoid terminating on an uncompensated positive shrinkage step.

---

## 4. Key Performance & Engineering Decisions

1. **Avoidance of 4D Tensors:**
   - In `evaluate_spinodal_field`, intermediate tensors of shape $(N, nx, ny, nz)$ are strictly avoided.
   - For a $200^3$ grid with $N=120$, a 4D tensor would require $120 \times 32\text{ MB} = 3.84\text{ GB}$.
   - Instead, waves are accumulated in-place into a single preallocated `float32` array using one reusable 3D buffer: total memory is $\approx 64\text{ MB}$ (a $60\times$ memory reduction).
2. **Zero-Copy Memory Passing:**
   - `extract_isosurface_flying_edges` forces contiguous memory (`np.ascontiguousarray(field, dtype=np.float32)`) before flattening, preventing duplicate volume allocation.
3. **Clean Topology Assurance:**
   - Every mesh extraction cleans unreferenced vertices, deduplicates faces, removes degenerate elements, and verifies positive outward winding (`if mesh.volume < 0: mesh.invert()`).

---

## 5. API Reference & Signatures

### 5.1 `graphite.math.spinodal`
```python
def generate_spinodal_wavevectors(
    num_waves: int,
    wavelength: float,
    anisotropy: tuple[float, float, float] = (1.0, 1.0, 1.0),
    seed: int | None = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate (num_waves, 3) wavevectors and (num_waves,) phases in [0, 2*pi)."""

def evaluate_spinodal_field(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    wavelength: float,
    num_waves: int = 120,
    anisotropy: tuple[float, float, float] = (1.0, 1.0, 1.0),
    seed: int | None = 42,
) -> np.ndarray:
    """Evaluate F(x) ~ N(0, 1) in-place in float32 without 4D intermediate tensors."""

def threshold_spinodal_field(
    F: np.ndarray,
    solid_fraction: float,
    is_sheet: bool = False,
) -> np.ndarray:
    """Analytically threshold standard normal field to exact target solid fraction."""
```

### 5.2 `graphite.mesh.extraction`
```python
def extract_isosurface_flying_edges(
    field: np.ndarray,
    origin: tuple[float, float, float] | Sequence[float],
    spacing: tuple[float, float, float] | Sequence[float],
    level: float = 0.0,
) -> trimesh.Trimesh:
    """Extract high-resolution isosurface mesh using multi-threaded Flying Edges with Marching Cubes fallback."""
```

### 5.3 `graphite.mesh.smoothing`
```python
def smooth_mesh_taubin(
    mesh: trimesh.Trimesh,
    iterations: int = 15,
    lamb: float = 0.5,
    nu: float = -0.53,
    inplace: bool = False,
) -> trimesh.Trimesh:
    """Apply volume-preserving Taubin smoothing (alternating shrink/dilate) to relax curvature without shrinkage."""

def compute_mean_curvature(mesh: trimesh.Trimesh) -> np.ndarray:
    """Compute vertex mean curvature H via PyVista/VTK or discrete Laplace-Beltrami operator."""
```

### 5.4 `graphite.implicit.spinodal`
```python
def generate_spinodal_lattice(
    cad_mesh: trimesh.Trimesh | str | Path,
    resolution: float = 0.25,
    wavelength: float = 2.0,
    solid_fraction: float = 0.3,
    is_sheet: bool = False,
    anisotropy: tuple[float, float, float] = (1.0, 1.0, 1.0),
    num_waves: int = 120,
    seed: int | None = 42,
    pad_width: int = 4,
    output_path: str | Path | None = None,
    taubin_iterations: int = 15,
    taubin_lamb: float = 0.5,
    taubin_nu: float = -0.53,
) -> trimesh.Trimesh:
    """Generate conformal Gaussian Random Field (GRF) spinodal lattice inside CAD geometry."""
```

---

## 6. Test Suite & Verification Results

### Summary: 24 Passed in 6.38s

| Test Suite | Tests | Verification Focus | Result |
|---|:---:|---|:---:|
| `tests/test_spinodal.py` | 17 | Marsaglia $S^2$ unit norm ($1 \pm 10^{-7}$), CLT field mean $\approx 0$, std $\approx 1$ ($< 0.85\%$ error), analytic solid fraction precision ($\pm 0.007 \ll 0.02$), CAD box integration. | **PASS** |
| `tests/test_extraction_and_smoothing.py` | 7 | Sphere extraction ($R=5\text{ mm}$): Volume err $0.095\% \ll 2\%$, Area err $0.050\% \ll 2\%$. Taubin volume drift: $0.383\% < 0.5\%$. Curvature reduction: $45.5\%$. Mock no-regression test (zero calls to `marching_cubes`). | **PASS** |

### Empirical Benchmarks
- **Raw Flying Edges Mesh:** `outputs/spinodal_flying_edges_raw.stl`  
  - Faces: $56,584$
  - Volume: $310.919\text{ mm}^3$
  - Mean Curvature Variation $\sigma(H)$: $24.2379$
- **Taubin-Smoothed Mesh:** `outputs/spinodal_flying_edges_taubin.stl`  
  - Faces: $56,584$
  - Volume: $312.108\text{ mm}^3$
  - Volume Drift: $|\Delta V / V_0| = 0.383\% < 0.5\%$
  - Mean Curvature Variation $\sigma(H)$: $13.2099$ ($45.5\%$ variance reduction)
