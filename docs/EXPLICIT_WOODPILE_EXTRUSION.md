# Explicit woodpile / cross-hatch extrusion

Graphite builds piecewise woodpile and cross-hatch lattices with two mesh backends:

| Backend | Module | When to use |
|---------|--------|-------------|
| **`extrude`** (default) | `graphite/explicit/woodpile_extrude.py` | Production STLs — sharp square struts, 10–100× fewer faces, watertight by construction |
| **`implicit`** | `graphite/implicit/piecewise_woodpile.py` | Reference meshes, rounded MC struts, `invert_solids`, legacy union path |

Both paths share the same **spec**, **band metadata**, and **anchor / layer-continuity** rules in `graphite/math/woodpile_anchor.py`.

---

## API

| Symbol | Location | Role |
|--------|----------|------|
| `WoodpileLatticeSpec` | `graphite/implicit/woodpile_input.py` | Spec dataclass (`generator` defaults to `"extrude"`) |
| `build_piecewise_woodpile_mesh()` | same | Dispatches to extrude or implicit MC |
| `repair_woodpile_mesh()` | same | Post-export trimesh repair (both backends) |

**Deprecated aliases** (emit `DeprecationWarning`):

- `WoodpileImplicitSpec` → `WoodpileLatticeSpec`
- `repair_implicit_woodpile_mesh` → `repair_woodpile_mesh`
- `scripts/generate_piecewise_woodpile_implicit.py` → `scripts/generate_piecewise_woodpile.py`
- `woodpile_piecewise_cylinder_union` / `--union` / `combine_mode: union`

---

## Usage

### Canonical CLI

```powershell
# Extrude (default)
python scripts/generate_piecewise_woodpile.py

# Implicit MC reference
python scripts/generate_piecewise_woodpile.py --generator implicit --resolution-mm 0.015
```

### 1 mm³ cube case study

```powershell
python scripts/generate_cube_1mm_woodpile.py --match-splitp-pores
python scripts/generate_cube_1mm_woodpile.py --match-splitp-pores --generator implicit
```

### YAML (`generate_lattice.py`)

```yaml
engine_type: Implicit (Woodpile)
grading_mode: Piecewise Z bands
generator: extrude   # default; omit or set implicit for MC
diameter_mm: 2.0
z_breaks_mm: [0.0, 2.23, 2.31, 2.7]
pore_mm: [0.8, 0.4, 0.2]
anchor_mode: center_void
alternate_band_orientation: true
```

See `example_woodpile_piecewise_2mm.yaml`.

---

## Extrusion engine

**`graphite/explicit/woodpile_extrude.py`**

- 2D bar grid → Shapely union → Z extrude → `manifold3d` slab union
- Uniform and piecewise cross-hatch (box + cylinder)
- True woodpile half-pitch shifts (`true_woodpile=True`)
- Cylinder clip QC (`verify_cylinder_clip`)
- Global layer-index continuity at piecewise grade interfaces

Validation artifacts: `outputs/explicit_woodpile/P0` … `P4`. Phase spike scripts archived under `scripts/explicit_woodpile/archive/`.

---

## Extrude constraints

Validated in `WoodpileLatticeSpec.validate()`:

- `invert_solids` not supported
- `combine_mode: union` not supported
- `resolution_mm` ignored (no marching cubes)

---

## Decision guide

| Need | Use |
|------|-----|
| Printable STL, FEA, Vocal flow mesh | **`extrude`** (default) |
| Rounded struts matching legacy implicit STLs | **`implicit`** |
| True woodpile with half-pitch shifts | either; extrude validated @ 100% field agreement (P4) |
| `invert_solids` / per-band union | **`implicit` only** |
| Sub-µm corner fidelity studies | **`implicit`** at fine `resolution_mm` |

---

## Remaining cleanup (future)

| Item | Status |
|------|--------|
| Rename `WoodpileLatticeSpec` / canonical CLI | **Done** |
| Default `generator: extrude` | **Done** |
| Deprecate union path + old script names | **Done** (warnings) |
| Archive P0–P4 spike scripts | **Done** |
| Retire `uniform_woodpile.py` slab MC | Open — still used by `generate_three_cylinder_lattices_user_spec.py` |
| Streamlit woodpile generator toggle | Not started |
| Regenerate case-study STLs with `_extrude` stems | Optional — run case-study CLI when ready |

### Keep indefinitely

- `graphite/math/woodpile.py` — analytical field, slice QC, calibration
- `graphite/math/woodpile_anchor.py` — anchor, layer parity, interface QC
- `graphite/implicit/piecewise_woodpile.py` — implicit reference meshes
