# Smooth Boolean Operators (R-Functions) & Implicit Skin Blending Handoff

This document details the design, mathematical formulation, code architecture, and validation of the smooth Boolean blending subsystem implemented in `graphite`.

---

## 1. Executive Summary & Problem Statement

Prior implicit lattice trimming in `graphite` relied exclusively on non-differentiable hard Boolean operators:
- Domain clipping: $\text{final\_field} = \max(\text{lattice\_field}, \text{cad\_sdf})$
- Skin combination: $\text{final\_field} = \min(\text{core\_sdf}, \text{skin\_sdf})$

Hard $\min$ and $\max$ operators introduce discontinuous normal transitions ($C^0$ continuity) at the intersection boundary. In vat photopolymerization (DLP/SLA) and powder bed fusion (PBF) additive manufacturing, sharp re-entrant corners at lattice-wall interfaces cause:
1. **Severe Stress Concentrations:** Notch effects that initiate premature fatigue delamination under mechanical loads.
2. **Resin Entrapment:** High surface-tension cusps that trap viscous unpolymerized resin during cleaning and post-curing.

To eliminate these failure modes, we implemented a suite of $C^1$ and $C^2$ continuous **smooth Boolean blending operators (R-functions)** directly into the level-set pipeline.

---

## 2. Mathematical Formulations & Level-Set Conventions

In `graphite`, solid material is defined as the sublevel set $\{ \mathbf{x} \in \mathbb{R}^3 \mid f(\mathbf{x}) \le 0 \}$. The boundary surface is the zero isosurface $f(\mathbf{x}) = 0$.

### 2.1 Set Operations Under $f \le 0$
- **Union ($A \cup B$):** A point is solid if it is inside $A$ or inside $B$.
  $$\operatorname{smin}(a, b, r)$$
- **Intersection ($A \cap B$):** A point is solid if it is inside $A$ and inside $B$.
  $$\operatorname{smax}(a, b, r) = -\operatorname{smin}(-a, -b, r) \quad \text{(Exact De Morgan Duality)}$$
- **Difference ($A \setminus B$):** A point is inside $A$ and outside $B$ (where complement $\bar{B}$ is $-b \le 0$).
  $$\operatorname{smooth\_difference}(a, b, r) = \operatorname{smax}(a, -b, r)$$

### 2.2 Implemented Blending Methods

Let $r > 0$ be the smoothing / fillet radius. When $r \le 10^{-6}$, all operators immediately short-circuit to exact `np.minimum` / `np.maximum`.

#### 1. Polynomial Formulation (Default, $C^1$ Continuous)
Based on quadratic Bézier blending (Inigo Quilez):
- **Smooth Minimum:**
  $$h = \operatorname{clip}\left(0.5 + 0.5 \frac{b - a}{r}, 0.0, 1.0\right)$$
  $$\operatorname{smin}_{\text{poly}}(a, b, r) = a \cdot h + b \cdot (1 - h) - r \cdot h \cdot (1 - h)$$
- **Smooth Maximum:**
  $$h = \operatorname{clip}\left(0.5 + 0.5 \frac{a - b}{r}, 0.0, 1.0\right)$$
  $$\operatorname{smax}_{\text{poly}}(a, b, r) = a \cdot h + b \cdot (1 - h) + r \cdot h \cdot (1 - h)$$
- **Fillet Offset:** At $a = b = 0$, $\operatorname{smax}(0, 0, r) = +r/4$, shifting the corner inward by $0.25r$ to create a quadratic fillet.

#### 2. Circular Formulation ($C^1$ Exact Circular Fillet)
Constructs an exact circular arc transition tangent to the boundary surfaces:
- **Smooth Minimum:**
  $$h = \operatorname{clip}\left(\frac{r - |a - b|}{r}, 0.0, 1.0\right)$$
  $$\operatorname{smin}_{\text{circ}}(a, b, r) = \min(a, b) - 0.5 r \left(1.0 - \sqrt{\max\left(1.0 - h^2, 0.0\right)}\right)$$
- **Smooth Maximum:**
  $$\operatorname{smax}_{\text{circ}}(a, b, r) = \max(a, b) + 0.5 r \left(1.0 - \sqrt{\max\left(1.0 - h^2, 0.0\right)}\right)$$
- **Fillet Offset:** At $a = b = 0$, $\operatorname{smax}(0, 0, r) = +0.5r$.

#### 3. Exponential Formulation ($C^\infty$ Continuous LogSumExp)
Based on the smooth LogSumExp softmin/softmax formulation, rendered unconditionally numerically stable:
- **Smooth Minimum:**
  $$\operatorname{smin}_{\text{exp}}(a, b, r) = \min(a, b) - r \cdot \ln\left(1 + e^{-\frac{|a - b|}{r}}\right) = \min(a, b) - r \cdot \operatorname{log1p}\left(e^{-\frac{|a - b|}{r}}\right)$$
- **Smooth Maximum:**
  $$\operatorname{smax}_{\text{exp}}(a, b, r) = \max(a, b) + r \cdot \operatorname{log1p}\left(e^{-\frac{|a - b|}{r}}\right)$$
- **Fillet Offset:** At $a = b = 0$, $\operatorname{smax}(0, 0, r) = +r \ln(2) \approx +0.6931r$.
- **Numerical Stability:** The factoring of $\min(a, b)$ and $\max(a, b)$ with `np.log1p(np.exp(-abs(a-b)/r))` ensures that even extreme values like $\pm 1000.0$ never overflow or produce `NaN`/`Inf`.

---

## 3. Code Architecture & Integration Points

| File | Role | Changes Made |
| :--- | :--- | :--- |
| `graphite/math/boolean.py` | Mathematical core | Implemented `smooth_min`, `smooth_max`, and `smooth_difference` supporting `"polynomial"`, `"circular"`, and `"exponential"`. |
| `graphite/math/__init__.py` | Package export | Exported `smooth_min`, `smooth_max`, and `smooth_difference` in `__all__`. |
| `graphite/implicit/blending.py` | High-level implicit pipeline | Implemented `blend_lattice_with_skin(lattice_field, cad_sdf, skin_thickness, blend_radius, method)`. |
| `graphite/implicit/conformal.py` | Conformal TPMS engine | Added `blend_radius: float = 0.0` and `blend_method: str = "polynomial"`. Applied `smooth_max` for core clipping and `smooth_min` for combined lattice+skin union. |
| `graphite/implicit/spinodal.py` | Spinodal GRF engine | Added `blend_radius: float = 0.0` and `blend_method: str = "polynomial"`. Applied `smooth_max` for CAD boundary trimming. |
| `graphite/implicit/__init__.py` | Package export | Exported `blend_lattice_with_skin` in `__all__`. |
| `scripts/export_core_spec.py` | Architectural contract gate | Added new symbols to `required_symbols` list. |
| `GRAPHITE_CORE_SPEC.md` | Single source of truth | Documented smooth Boolean formulas, signatures, and array contracts. |
| `tests/test_boolean.py` | Automated test suite | 9 test cases covering asymptotic convergence, analytical offsets, De Morgan duality, numerical stability, and end-to-end lattice meshing. |

---

## 4. Verification & Validation

### 4.1 Unit Test Suite (`pytest tests/test_boolean.py`)
- **Asymptotic Convergence:** Verified that for random 2D/3D grids, smooth operators converge to exact `np.minimum` / `np.maximum` as $r \to 0$ ($< 10^{-3}$ at $r=10^{-4}$ and $< 10^{-12}$ at $r=0$).
- **Fillet Offset Precision:** Verified exact analytical shifts along $a = b = 0$:
  - Polynomial: $+0.25r$ / $-0.25r$
  - Exponential: $+r\ln(2)$ / $-r\ln(2)$
  - Circular: $+0.5r$ / $-0.5r$
- **De Morgan Duality:** Confirmed `smax(a, b, r) == -smin(-a, -b, r)` within floating-point tolerance across all methods.
- **Extreme Range Stability:** Verified evaluation on $\pm 1000.0$ arrays produces zero `NaN` or `Inf`.
- **Lattice Skin Meshing:** Verified `blend_lattice_with_skin` extracts a watertight manifold mesh with Flying Edges.
- **Full Suite Status:** 33/33 tests passing across `test_boolean.py`, `test_spinodal.py`, and `test_extraction_and_smoothing.py`.

### 4.2 Spec & Contract Compliance
- `python scripts/export_core_spec.py --check` passes with return code 0:
  ```text
  [OK] Specification GRAPHITE_CORE_SPEC.md is fully verified and consistent with graphite/ source tree.
  ```

### 4.3 Demonstration Artifacts Generated Under `outputs/`
1. `outputs/conformal_filleted_cube.stl`:
   - Conformal Gyroid lattice in 20 mm cube, $L = 8.0\,\text{mm}$, solid fraction $= 0.35$, shell thickness $= 1.5\,\text{mm}$, `blend_radius=0.6mm`.
   - Verified mesh: 309,704 faces, watertight manifold (`watertight=True`, boundary edges $= 0$).
2. `outputs/spinodal_filleted_cube.stl`:
   - Conformal GRF spinodal lattice in 20 mm cube, $\lambda = 5.0\,\text{mm}$, solid fraction $= 0.35$, `blend_radius=0.7mm`, 10 Taubin smoothing iterations.
   - Verified mesh: 100,012 faces, watertight manifold (`watertight=True`).
