# TPMS Solid-Fraction Graded Ramping: Quantile-Mapping Architecture

## 1. Problem Overview & Finding

When functionally grading Triply Periodic Minimal Surface (TPMS) sheet lattices from a porous interior (e.g. 20%–25% solid fraction) to solid rims (100% solid fraction), a naive linear interpolation of the implicit isosurface threshold $\tau$:
$$\tau(y) = \tau_{\text{base}} + (\tau_{\text{target}} - \tau_{\text{base}}) \times r(y)$$
fails dramatically due to two interrelated phenomena:

1. **Finite Domain Bounds ($\tau_{\max} < \infty$):**
   The implicit function $|F(U, V, W)|$ has a finite mathematical maximum over its periodic domain. For the Split-P surface ($F = 1.1 t_1 - 0.2 t_2 - 0.4 t_3$), $\max(|F|) \approx 1.810$. If $\tau_{\text{target}}$ is arbitrarily set higher (e.g., $2.55$), the condition $|F| \le \tau$ is satisfied across 100% of the unit cell volume as soon as $\tau(y) \ge 1.810$. Under linear threshold interpolation, this threshold is reached at $r(y) \approx 0.674$, causing the model to become **100% completely solid across the outer 33% of its length**, choking off all pores prematurely.

2. **Non-Linear Cumulative Distribution Function (S-Curve CDF):**
   The probability distribution of $|F|$ across the 3D unit cell is non-uniform. Its cumulative distribution $CDF(\tau) = P(|F| \le \tau) = \phi$ has an S-curve profile:
   - For Split-P:
     - $\tau(20\%) = 0.2755$
     - $\tau(50\%) = 0.6880$ ($\Delta = 0.4125$ for 30% volume change $\to 0.0137/\%$)
     - $\tau(90\%) = 1.3641$ ($\Delta = 0.6761$ for 40% volume change $\to 0.0169/\%$)
     - $\tau(100\%) = 1.8100$ ($\Delta = 0.4459$ for the final 10% volume change $\to 0.0446/\%$, >2.5× steeper!)
   - When $\tau$ is ramped linearly, the volume fraction $\phi$ grows much faster than linearly. At halfway along the ramp ($r = 0.5$), the solid fraction already reaches **91.9%** instead of the expected 60%.

---

## 2. Quantitative Comparison: Threshold-Linear vs. True Quantile Linear $\phi$

Comparison along a 20% $\to$ 100% linear ramp ($r=0.0$ center, $r=1.0$ rim):

| Ramp Factor $r$ | Desired Linear Solid-Fraction $\phi$ | Old Buggy $\tau$-Interpolation ($\tau_{\max}=2.55$) | True Quantile-Mapped Linear $\phi$ |
| :---: | :---: | :---: | :---: |
| **0.0 (Center)** | **20.0%** | **20.0%** | **20.0%** |
| **0.2** | 36.0% | **53.0%** | 36.0% |
| **0.4** | 52.0% | **81.7%** | 52.0% |
| **0.5 (Halfway)** | **60.0%** | **91.9%** (Nearly closed) | **60.0%** (Wide open pores) |
| **0.6** | 68.0% | **98.0%** | 68.0% |
| **0.67** | 73.6% | **100.0%** (Premature solid closure) | 73.6% |
| **0.8** | 84.0% | **100.0%** (Solid blank tube) | 84.0% |
| **0.9** | 92.0% | **100.0%** (Solid blank tube) | 92.0% |
| **1.0 (Rim)** | **100.0%** | **100.0%** | **100.0%** (Flush solid boundary) |

---

## 3. Mathematical Formulation: True Quantile Mapping

To guarantee that the physical solid fraction $\phi(x, y, z)$ follows the intended linear ramp, we decouple the spatial profile from the implicit function:

1. **Define the Target Spatial Solid-Fraction Field $\phi(y)$:**
   For a ramp starting at $y_{\text{start}}$ and ending at $y_{\text{end}}$ with relative coordinate $y_{\text{rel}} = (y - y_{\text{start}}) / H$:
   - Full-length ramp:
     $$r(y) = \text{clip}\left(2 \cdot |y_{\text{rel}} - 0.5|, 0.0, 1.0\right)$$
     $$\phi(y) = \phi_{\text{center}} + (\phi_{\text{rim}} - \phi_{\text{center}}) \cdot r(y)$$
   - Outer 25% ramp:
     $$r(y) = \begin{cases} 0 & 0.25 \le y_{\text{rel}} \le 0.75 \\ \frac{0.25 - y_{\text{rel}}}{0.25} & y_{\text{rel}} < 0.25 \\ \frac{y_{\text{rel}} - 0.75}{0.25} & y_{\text{rel}} > 0.75 \end{cases}$$
     $$\phi(y) = \phi_{\text{center}} + (\phi_{\text{rim}} - \phi_{\text{center}}) \cdot r(y)$$

2. **Sample the Unit-Cell Distribution:**
   Evaluate $|F(U, V, W)|$ over a fine discretization of the periodic unit cell $[0, 2\pi)^3$ and sort:
   $$\mathbf{v} = \text{sort}\left(\{|F(\mathbf{u}_i)\}_{i=1}^N\right)$$
   $$\mathbf{p} = \text{linspace}(0.0, 1.0, N)$$

3. **Invert via 1D Interpolation (Quantile Function $Q(\phi)$):**
   $$\tau(y) = Q(\phi(y)) = \text{interp}\left(\phi(y), \mathbf{p}, \mathbf{v}\right)$$

4. **Isosurface Extraction:**
   Marching cubes extracts the level set:
   $$S = \{(x, y, z) \mid |F(x, y, z)| - \tau(y) \le 0\}$$

This guarantees that at every cross-section $y$, the fraction of voxels satisfying $|F| \le \tau(y)$ matches $\phi(y)$ exactly.
