# Lattice Math Architecture

This document defines the lattice-gradient math architecture that should guide Graphite integration.

It is intentionally complementary to existing documents:
- `docs/Implicit_TPMS_Architecture.md` (broad implicit engine + file manifest)
- `Oldinfo/README.md` (historical Split-P exploration + pitfalls)
- `tests/test_splitp_comparisons.py` and `tests/test_matlab_debug.py` (current reference implementations of comparison math)

## 1. Architectural Goal

The objective is to generate continuously varying TPMS lattices (especially Split-P) with smooth, singularity-free pore-size gradients and stable phase behavior across the full domain.

The project is explicitly abandoning Basic Chirping (direct frequency multiplication) as the primary generation model because it introduces:
- phase-shear artifacts in aggressive gradients,
- local history rewriting (phase at a point ignores traversed frequency history),
- brittle behavior around steep control-point transitions.

Basic chirping may still be useful as a debug baseline, but not as the canonical engine for production implicit generation.

## 2. The Primary Engine: Jacobian Integration (Implicit Generation)

### Core math

Let local angular frequency be:

\[
\omega(z) = \frac{2\pi}{L(z)}
\]

with \(L(z)\) defined by user control points.

Instead of direct substitution \(W(z)=\omega(z)\,z\), we define phase via accumulated path integral:

\[
W(z) = \int_0^z \omega(t)\,dt
\]

In implementation this is computed numerically with cumulative trapezoidal integration on a dense 1D axis:
- `z_1d = linspace(0, H, N)`
- `omega_1d = 2*pi / L(z_1d)`
- `W_int_1d = cumulative_trapezoid(omega_1d, z_1d, initial=0)`
- interpolate `W_int_1d` onto runtime `Z`.

For Split-P:
- \(U = \omega(z)\,x\)
- \(V = \omega(z)\,y\)
- \(W = W_{\text{int}}(z)\)
- evaluate Split-P trigonometric terms using \((U,V,W)\).

### Odometer analogy

Treat \(W(z)\) like an odometer: each infinitesimal segment contributes phase based on the local frequency at that segment, and the total phase is the cumulative sum of all prior segments. This preserves lattice spatial history and prevents discontinuous rewrites when \(L(z)\) changes abruptly.

### Role in Graphite

This Jacobian-integrated phase engine is the core method for native implicit TPMS generation in Graphite (field-first, not mesh-warp-first), including future UI-driven multi-knot gradient definitions.

**Lofted grading (experimental):** Production use today is **1D along a single spine** (osteochondral Z, or field-driven Cartesian X/Y/Z). Explicit hex uses the same idea via slice-lofted scaffolds (`taper_along`). See `docs/LOFTED_GRADING.md`.

## 3. The Secondary Engine: Explicit Mesh Warping (Legacy MATLAB Method)

### What the legacy script did

The legacy MATLAB workflow deformed existing geometry coordinates directly:

\[
Z_{\text{new}} = Z_{\text{old}} \cdot S(Z_{\text{old}})
\]

then applied XY scaling based on an axial rule (historically at times on stretched coordinates, later corrected to original coordinates in debugging).

This is a rubber-sheet coordinate transform on an already tessellated mesh.

### Why it can fail (runaway acceleration / folding)

When scaling is tied to transformed coordinates, the map can self-amplify and push coordinates beyond expected bounds quickly (e.g., short target heights stretching to much larger heights). Even with crop/flip workarounds, explicit warping remains sensitive to:
- initial tessellation density,
- base unit-cell count in seed geometry,
- non-invertible or near-folding regions from aggressive scale curves.

A representative limitation: stretching a mesh that starts with only ~2.5 pores across height cannot create high-fidelity gradients over large final heights without aliasing and geometric distortion.

### Role in Graphite

Explicit coordinate-warping math should be reserved for a future Graphite tool focused on stretching imported STL/mesh assets. It should not be the foundation for native TPMS synthesis from scratch.

## 4. Implementation Next Steps

We are **not** building a new implicit engine. The existing core remains in place:
- TPMS evaluator family (`graphite/math/tpms.py`),
- CAD SDF boolean masking / conformal clipping (`graphite/geometry/masking.py`),
- dual-EDT distance-field workflows (`graphite/implicit/boundary_graded.py`).

The Jacobian work is a targeted replacement of the basic chirp-style phase construction in existing generators.

- [ ] Keep current TPMS + CAD SDF + marching-cubes architecture unchanged.
- [ ] Add Jacobian phase utility (`omega(d)` -> integrated `W(d)`), then map dense fields with `np.interp`.
- [ ] Use integrated phase as the primary gradient-direction phase (`W`) in existing scripts (e.g. osteochondral, boundary-driven).
- [ ] Use local frequency for orthogonal phases (`U = X * omega_local`, `V = Y * omega_local`).
- [ ] Retain current sheet thresholding, CAD boolean intersection, and meshing pipeline.
- [ ] Add UI controls for gradient knots and enforce monotonic/validation rules where required.
- [ ] Keep Basic Chirp as optional debug/reference mode only.
- [ ] Keep Legacy Explicit Warp as a separate imported-mesh tool path (not implicit core).
- [ ] Add regression plots/tests comparing Basic vs Jacobian vs Legacy at fixed slices.
