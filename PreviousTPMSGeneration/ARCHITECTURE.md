# Previous TPMS Architecture (Legacy Experimental Stack)

## 1) Project Overview

`PreviousTPMSGeneration` captures the earlier R&D phase for generating graded TPMS scaffolds, with a strong focus on Split-P sheet lattices for implant-style geometries.

The experimental target was a graded pore strategy (commonly 200-800 um) at approximately 33% solid fraction, with workflows for:

- flat and angled cylinders,
- two-zone angled gradients,
- chirped single-field TPMS,
- two-field sigmoid morphing,
- and Voronoi/IDW blended graded fields.

This stack is script-driven (no GUI orchestrator) and primarily implicit/signed-field based.

## 2) Directory Structure

Focused on the legacy package content and scripts currently present.

```text
PreviousTPMSGeneration/
├── ARCHITECTURE.md
├── Experimental/
│   ├── README.md
│   ├── gradient_cylinder.py
│   ├── angled_cylinder_generator.py
│   ├── sigmoidal_morph_generator.py
│   ├── tpms_implant_generator.py
│   └── Implant_generator
└── General/
    ├── GeminiReference.py
    ├── General_Lattice_Generator.py
    └── gyroid_compare_cube.py
```

### Module / Script Notes

- `Experimental/README.md`
  - Main narrative of the experimental campaign.
  - Documents method evolution, math rationale, outputs, and limitations.

- `Experimental/gradient_cylinder.py`
  - Primary script for graded angled TPMS outputs.
  - Provides:
    - `generate_flat()`
    - `generate_angled()`
    - `generate_angled_twostage()`
  - Uses Split-P sheet field, smoothstep interpolation, tau calibration, and STL export.

- `Experimental/angled_cylinder_generator.py`
  - Earlier standalone angled-cylinder chirped workflow.
  - Uses angled top geometry mask plus smoothstep-driven k(z) chirp.

- `Experimental/sigmoidal_morph_generator.py`
  - Side-by-side comparison of:
    - chirped single-field method,
    - sigmoid blend between two Split-P fields.
  - Includes the corrected absolute-field blend logic.

- `Experimental/tpms_implant_generator.py`
  - Voronoi/IDW style graded scaffold generator.
  - Builds seed cloud and blends local Split-P contributions using nearest-neighbor weighting.

- `Experimental/Implant_generator`
  - Reference legacy script used in the workflow lineage (as noted in README).

- `General/GeminiReference.py`
  - Gyroid reference generator with pore-size and solid-fraction parameterization and marching cubes export.

## 3) Legacy Generation Pipeline (Graded + Angled TPMS)

The legacy scripts follow a common implicit-field pipeline.

### A) Parameterization

Typical configurable inputs include:

- geometry dimensions (cylinder radius, height, angled top parameters),
- voxel resolution (`RES_XY`, `RES_Z`),
- target volume fraction (`TARGET_VF`, commonly 0.33),
- pore-size boundary values (e.g., 800 um bottom to 200 um top),
- transition model (smoothstep or sigmoid),
- gradient mode (single-stage, two-stage, axis-based, or seed-driven).

### B) Grid + Geometry Mask

Scripts build a structured 3D voxel grid with `numpy.meshgrid`.

- flat and prism/cube variants use axis-aligned masks,
- angled cylinder variants define `z_top(x)` and mask out points above the angled cut plane,
- outside-region voxels are forced to void by assigning large field values before extraction.

### C) TPMS Field Evaluation (Split-P Dominant)

Most experimental scripts use Split-P equation terms (`T1 + T2 + T3`) and evaluate either:

1. **single chirped field** with spatially varying `k`, or
2. **two-field morph** blended by a weight map, or
3. **IDW Voronoi blend** of local seed-centered fields.

For graded/angled designs:

- smoothstep `W(t) = 3t^2 - 2t^3` is heavily used to avoid abrupt boundary slopes,
- chirped workflows log-interpolate frequency/pore progression,
- angled workflows use column-local normalization (e.g., `t = z / z_top(x)`) to enforce target pore values on both bottom and angled top surfaces.

### D) Tau Calibration (Volume Fraction Control)

A calibration pass sweeps candidate `tau` values and selects the value minimizing error to target solid fraction:

- sheet interpretation uses `F_abs = abs(F)` and keeps regions where `F_abs < tau`,
- this ties wall thickness directly to target VF and tau (not an independent wall-thickness knob).

### E) Surface Extraction + Export

After masking and tau calibration:

1. run marching cubes (`skimage.measure.marching_cubes`),
2. apply coordinate shifts if needed (e.g., centering x/y),
3. export STL via `numpy-stl`.

Some scripts also export half-geometry variants for cross-section inspection.

## 4) Current Legacy Capabilities (As Implemented)

- Split-P sheet TPMS generation with tau-based VF targeting.
- Flat-top and angled-top cylinder generation.
- Two-stage angled gradient construction (top zone + lower zone).
- Chirped gradient generation using spatially varying `k`.
- Sigmoid two-field morphing with corrected absolute-field blending.
- Modifier-like geometric control in angled scripts via plane-based masks (`z_top` and derived planes).
- Voronoi/IDW nearest-seed blended graded scaffolds (`cKDTree` + inverse-distance weighting).
- Full and half STL exports for inspection in selected scripts.

## 5) Implicit vs Explicit Segmentation (Legacy Context)

The legacy TPMS stack is almost entirely **implicit**:

- scalar-field construction,
- mask composition,
- iso-surface extraction with marching cubes.

There is no dedicated explicit node/strut topology engine in this legacy folder. The workflow centers on continuous fields rather than discrete beam graphs.

---

## Notes on Scope Accuracy

This document is based on:

- `PreviousTPMSGeneration/Experimental/README.md`,
- and currently present scripts in `PreviousTPMSGeneration/Experimental` and `PreviousTPMSGeneration/General`.

It intentionally describes only implemented/recorded behavior in those files and does not introduce future features.

