# Modular SC Conformal Engine

> Status: implemented (Aug 2026). SC conformal is **rule-driven** via `rule_name` on
> `generate_conformal_lattice`; boolean crop is a **debug/fast** path only.
>
> **Pivot (target architecture):** cage morph via nearest-point / face-normal is
> not the long-term conformal authority. See
> [SURFACE_FIRST_DUAL_TRIM.md](SURFACE_FIRST_DUAL_TRIM.md) for surface-first dual
> + volumetric trim + gated stitch.

## Pipeline

1. VF-gate SC hex scaffold (`conformal_core.cull_hex_elements`)
2. **Morph hex cage** — identify exposed faces by face adjacency (exactly one owner), project **all** exposed corners onto the CAD bidirectionally (algebraic sphere if `sphere_center`/`sphere_radius` are set; else mesh closest-point), then relax the scaffold (`morph_hex_scaffold`; default Laplacian with exposed nodes fixed, or experimental `relax_mode="radial_equalize"` on spheres)
3. **Stamp** volume topology into deformed bricks (`generate_hex_topology`) — kelvin/tesseract use trilinear hex8 maps, so the *entire* unit cell morphs with the cage
4. Surface skin from `skin_mode` on the deformed scaffold
5. Sweep / export

`mode="boolean"` skips steps 2–4 and may ∩ CAD for quick debug crops.

Interiors are never CAD-projected as free snaps; they move only because their parent hex cage deformed.

## Boundary projection and transition-layer relaxation

The active 50% VF cull remains authoritative. Boundary detection does not use
node valency:

1. Gather all 6 ordered quad faces from every retained hex.
2. Sort each face's four global node IDs to form an orientation-independent key.
3. Count identical keys; a face with exactly one owner is exposed.
4. Project every unique node referenced by those exposed faces. This includes
   re-entrant stair-step corners that may have valency 6.

Sphere projection is algebraic:

```python
delta = exposed_points - sphere_center
projected = sphere_center + delta * (sphere_radius / norm(delta))
```

It is bidirectional: outside nodes move inward and inside nodes move outward.
The 20 mm sphere diagnostic found 74 exposed nodes: 48 outside, 20 inside, and
6 already on the surface. Maximum inward travel was 2.247 mm; maximum outward
travel was 2.929 mm. The outward move is therefore the larger distortion source.
The 50% face-centroid VF gate does **not** guarantee corner travel <= 0.5 cell;
corners can be farther from the surface than the sampled face centroids.

After projection, Jacobi Laplacian relaxation runs on the shared hex-corner
edge graph:

- depth 0: projected boundary, fixed (conformity is exact);
- depth >= 1: full relaxation.

The purpose of the relaxation is **radial** redistribution: projection
compresses or stretches the outermost cell layer radially, and relaxation lets
successive interior layers absorb that displacement instead of trapping it at
the surface. No rigid-core depth cap is needed — a deep interior node with a
symmetric Cartesian neighborhood self-limits, because the average of its
neighbors is its own position, so unconstrained iteration cannot make the core
drift. `relax_layers` remains available to freeze nodes deeper than N layers
(default `None` = relax everything interior); `relax_iterations` (default 15)
and `relax_alpha` (default 0.5) control convergence.

Diagnostic generator:
`scripts/generate_sphere_octahedral_snap_diagnostics.py`. It exports culled
unsnapped, inward-only, and outward-only octahedral meshes without relaxation
or final Boolean clipping, so projection effects remain directly visible.

### Experimental: `relax_mode="radial_equalize"` (sphere-only)

Laplacian with a hard-pinned shell saturates quickly on coarse spheres (few
interior DOFs). An optional morph mode trades core compression for outer
expansion:

- **Objective:** spring mostly-radial scaffold edges toward a shared rest length
  \(L_r = R / \max(\text{depths})\); weak tangential springs fight surface bunching.
- **Boundary:** exposed nodes stay **fixed** at their projected sphere
  positions (no tangential slide). Interior travel is capped at
  `0.5 * cell_size`.
- **API:** pass `relax_mode="radial_equalize"` with `sphere_center` /
  `sphere_radius`. Use ~`relax_iterations=300` and `relax_alpha=0.2` (step size).
  Default remains `relax_mode="laplacian"`.

Compare script (grid + octahedral, quality table + cutaways):
`scripts/generate_sphere_radial_equalize_compare.py` → `test_parts/sphere_sc_20mm/`.

```python
res = generate_conformal_lattice(
    cad_filepath=sphere,
    cell_size=4.0,
    strut_radius=0.25,
    lattice_type="SC",
    rule_name="octahedral",
    volume_fraction_threshold=0.5,
    sphere_center=(0.0, 0.0, 0.0),
    sphere_radius=10.0,
    relax_mode="radial_equalize",
    relax_iterations=300,
    relax_alpha=0.2,
    skip_sweep=True,
)
```

## Key modules

| Module | Role |
|---|---|
| [`hex_topology_module.py`](../graphite/explicit/hex_topology_module.py) | Rules + `conform_dofs` / `skin_mode` policy |
| [`conformal_core.py`](../graphite/explicit/conformal_core.py) | Cull, DOF map, depths, skin merge |
| [`conformal_generator.py`](../graphite/explicit/conformal_generator.py) | Orchestrator (`lattice_type="SC"`, `rule_name=...`) |
| [`hex_surface_dual.py`](../graphite/explicit/hex_surface_dual.py) | Boundary-quad dual primitives |

## API

```python
from graphite.explicit import generate_conformal_lattice

res = generate_conformal_lattice(
    cad_filepath=mesh,
    cell_size=12.7,
    strut_radius=1.0,
    lattice_type="SC",
    rule_name="star",       # any registered hex rule
    mode="conformal",       # or "boolean" for debug
    skip_sweep=True,
    volume_fraction_threshold=0.5,
)
# res["nodes_relaxed"], res["volume_struts"], res["skin_struts"] / cyan_struts
```

Bracket batch: `scripts/generate_uniform_bracket_lattice.py` (default conformal; `--boolean` for debug).

## Surface skin modes

| Mode | Rules | Behavior |
|---|---|---|
| `face_centroid_dual` | octahedral, hex_dual, hex_face_dual | Integer dual cage on boundary face centroids |
| `face_local_rule` | star, octet | Face centroid + spokes to the 4 corners (not opposite-corner diagonals) |
| `corner_edge_cage` | grid, tesseract | Exterior SC hex edges only |
| `kelvin_face_bridge` | kelvin, kelvin14 | Corresponding Kelvin face nodes on shared exterior side walls of neighbors (top↔bottom of cell above, left↔right of cell beside) |
| `none` | a15_kagome | Deferred |

### Surface-cage solid geometry

SC volume struts remain cylindrical. Surface-dual / cage struts default to a
uniform rectangular section (`surface_cage_profile="rectangular"`):

1. Resample each cage bar into `n_segments` stations and project them onto the
   CAD. Cage nodes are hex face centroids, so on curved CAD they sit *below*
   the surface (measured up to 0.97 mm inside on the Toros). Projecting first
   is what keeps the bar registered to the surface.
2. Loft a rectangular section along the projected path, holding the inner face
   a constant `surface_cage_thickness` below the surface at every station and
   using a constant in-surface `surface_cage_width`. Width defaults to the
   cylindrical volume-strut diameter (`2 * strut_radius`); thickness defaults
   to half that width. Because the floor follows the surface rather than a
   single flat plane, wall thickness stays uniform along the bar instead of
   picking up the local chord sag as extra depth.
3. Grow the section outward by `surface_cage_normal_oversize` (default 0.25 mm)
   so the exterior is guaranteed to poke through.
4. Union the oversized cage with the cylindrical core.
5. Boolean-intersect the complete solid with the CAD, producing a flush exterior
   interface.

This generalizes the baseball square-cage oversize + Boolean-trim recipe to
arbitrary watertight CAD. `surface_cage_profile="cylindrical"` preserves the
legacy surface-strut solidification. Detection and conversion of volume struts
that happen to run close to the surface is intentionally deferred.

## Notes

- **Kelvin**: volume nodes are not at hex corners (~0.5 cell from corners). Conformal DOFs come from skin-appended exterior corner nodes; volume stretches via cage morph. Skin bridges corresponding face nodes between adjacent cells.
- **a15_kagome**: `skin_mode=none` for now (shared-edge dual deferred). Corner ironing against hex scaffold is not yet matched to the dense tet-subcell graph.
