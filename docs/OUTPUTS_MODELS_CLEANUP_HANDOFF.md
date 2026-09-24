# outputs/models/ cleanup handoff

> **Doc role:** Ops guardrails (not lattice algorithms). Agents: [AGENTS.md](../AGENTS.md); only open when cleaning `outputs/models/`.

Guidance for **repo-wide** cleanup under `outputs/models/` (separate from the Mirae prism folder guide).

**Approved next step:** **Option 2 — Recommended Default** (legacy + rebuildable subdirectories).  
**Must preserve:** `outputs/models/user_spec_rect_prism_3x1p5x5mm/` (Aristo V4 production artifacts — see [folder CLEANUP_GUIDE](../outputs/models/user_spec_rect_prism_3x1p5x5mm/CLEANUP_GUIDE.md)).

---

## Option summary

| Option | Scope | ~Space recovered | Risk |
|--------|--------|------------------|------|
| **1 — Safe & legacy** | Unreferenced files + obsolete backups (e.g. `Warped_Cube_Kagome_6mm.stl` ~959 MB) | ~1.01 GB | Low |
| **2 — Recommended** | Option 1 + delete `piecewise_slab_union_test/` + `user_spec_cylinders_3x5mm/` | ~1.89 GB | Low–medium (rebuildable) |
| **3 — Maximum** | Keep only active gridding trimmer I/O (e.g. `Adapter_Gridded_SuperCell.stl`, `top_part_new_Conformal_Kagome.stl`) | ~2.86 GB | High — confirm trimmer workflow first |

**User decision:** Proceed with **Option 2**. Another agent may execute deletion; this doc records intent and guardrails.

---

## Do not delete (Option 2)

| Path / pattern | Reason |
|----------------|--------|
| **`user_spec_rect_prism_3x1p5x5mm/`** entire folder | Active Aristo V4 geometry + 1N/5N/10N sweep — [CLEANUP_GUIDE](../outputs/models/user_spec_rect_prism_3x1p5x5mm/CLEANUP_GUIDE.md) |
| Gridding trimmer inputs/outputs (if still in use) | e.g. `Adapter_Gridded_SuperCell.stl`, `top_part_new_Conformal_Kagome.stl`, `top_part_new_Tet_Lattice.stl` — verify against `scripts/gridded_supercell_trimmer.py` before any Option 3-style purge |
| Any STL explicitly referenced by a script you still run | Grep repo before delete |

---

## Safe to delete under Option 2 (in addition to Option 1)

### Subdirectories (rebuildable)

| Directory | Rebuild with |
|-----------|--------------|
| `outputs/models/piecewise_slab_union_test/` | `scripts/archive/tests_diagnostics/test_piecewise_slab_union_woodpile.py` |
| `outputs/models/user_spec_cylinders_3x5mm/` | `scripts/generate_three_cylinder_lattices_user_spec.py` |

Documented in [PIECEWISE_PRISM_LATTICE_GENERATION.md](PIECEWISE_PRISM_LATTICE_GENERATION.md).

### Option 1 examples (typical legacy — confirm unreferenced)

- `Warped_Cube_Kagome_6mm.stl` (~959 MB) — if no script grep hits
- Obsolete calibration sweeps, duplicate backups, superseded test arrays (see external audit `models_cleanup_analysis.md` if available)

**Always grep before deleting:**

```bash
rg -l "FILENAME_OR_STEM" --glob "*.{py,md,json,yaml}"
```

---

## Pre-cleanup checklist

1. **Commit documentation first** (see below) so cleanup rationale survives disk deletes.
2. Confirm **`user_spec_rect_prism_3x1p5x5mm/`** is excluded from all delete globs.
3. Grep for paths inside `piecewise_slab_union_test/` and `user_spec_cylinders_3x5mm/` beyond the rebuild scripts above.
4. After Option 2, spot-check:
   - `python scripts/run_mirae_lattice_slab_v4_aristo.py` (or 1N-only dry path) if V4 folder intact
   - Any gridding trimmer script you rely on weekly

---

## Git / documentation protocol

`outputs/` is typically **gitignored** — deleted STLs will not live in history.

**Before mass delete, commit to git:**

- `docs/ARISTO.md`, `docs/ARISTO_MESHING.md`, `docs/ARISTO_REGRESSION_BASELINE.md`
- `docs/OUTPUTS_MODELS_CLEANUP_HANDOFF.md` (this file)
- `outputs/models/user_spec_rect_prism_3x1p5x5mm/CLEANUP_GUIDE.md`
- `outputs/models/user_spec_rect_prism_3x1p5x5mm/MIRAE_LATTICE_FEA_WORKLOG.md`
- Any Aristo code changes on the branch

That preserves **why** artifacts existed even after local files are gone.

---

## Git workspace state (March 2026)

The repo working tree is **large and mixed** — not suitable for a single commit. Approximate counts on `main`:

| State | ~Count | Examples |
|-------|--------|----------|
| Modified (tracked) | 29 | `app.py`, `graphite/explicit/*`, `docs/README.md` |
| Deleted (tracked) | 150 | Root sandboxes (`Supercell_Modules/`, `Hex_Sandbox/`), moved docs |
| Untracked (`??`) | 298 | `graphite/aristo/`, `docs/history/`, most of `outputs/` |

**You can and should commit in slices** via `git add <paths>` — no need to commit everything at once.

### Recommended commit order

| Phase | What to stage | Purpose |
|-------|---------------|---------|
| **0 — Hygiene** | `.gitignore` (add `outputs/`, `*.bak.py`) | Stop untracked output noise after disk cleanup |
| **1 — Aristo** | `graphite/aristo/`, Aristo scripts, `docs/ARISTO*.md`, optional prism markdown guides | Coherent FEA/mesh story (~20–40 files) |
| **2 — Docs reorg** | `docs/history/`, `docs/research/`, new `docs/*.md`, staged root `.md` deletions | Moves, not accidental deletes |
| **3 — Explicit hex** | `graphite/explicit/`, related tests/docs | Separate reviewable chunk |
| **4 — Sandbox removal** | `git add -u` for deleted module trees | Confirm no imports remain; run tests |
| **5 — Remainder** | `app.py`, implicit engine, other scripts | WIP integration |

**Do not commit:** `outputs/**/*.stl`, `*.vtu`, most PNGs, `*.bak.py`, large binaries. Prefer gitignoring `outputs/` rather than versioning regenerable artifacts.

**Optional branches:** `feat/aristo`, `chore/docs-reorg`, `chore/remove-sandboxes` — merge as separate PRs.

### Phase 1 staging example (Aristo slice)

```powershell
git add graphite/aristo/
git add scripts/gmsh_lattice_mesh.py scripts/mesh_tpms_lattice_gmsh.py
git add scripts/run_mirae_lattice_slab_v4_aristo.py scripts/fix_lattice_stl_slivers.py
git add requirements-aristo.txt
git add docs/ARISTO.md docs/ARISTO_MESHING.md docs/ARISTO_REGRESSION_BASELINE.md
git add docs/ARISTO_ACCURACY_PATHS.md docs/OUTPUTS_MODELS_CLEANUP_HANDOFF.md
git add docs/README.md
# Optional small text artifacts:
git add outputs/models/user_spec_rect_prism_3x1p5x5mm/CLEANUP_GUIDE.md
git add outputs/models/user_spec_rect_prism_3x1p5x5mm/MIRAE_LATTICE_FEA_WORKLOG.md

git commit -m "feat(aristo): P1 Delaunay lattice meshing, flat-top BC, and docs"
```

Use `git add -p` when a file mixes unrelated edits (e.g. partial `docs/README.md`).

### Suggested `.gitignore` additions (Phase 0)

```
outputs/
*.bak.py
*_module.bak.py
```

Exception: if you intentionally version small markdown guides under `outputs/models/.../`, force-add them after ignoring `outputs/`:

```powershell
git add -f outputs/models/user_spec_rect_prism_3x1p5x5mm/CLEANUP_GUIDE.md
```

### Order of operations (cleanup + git)

1. **Phase 0 + 1 git commits** (docs + Aristo code)  
2. **Option 2 disk cleanup** (this handoff) — no git impact if `outputs/` ignored  
3. **Phases 2–4** commits as you consolidate the rest of the workspace  
4. **Regression:** `python scripts/run_mirae_lattice_slab_v4_aristo.py` vs [ARISTO_REGRESSION_BASELINE.md](ARISTO_REGRESSION_BASELINE.md)

---

## Mirae prism folder status

**Tiers A + B already completed** by user (superseded V4 outputs, V2/V3/V6/V7 Mirae STLs, logs).

**Still keep in `user_spec_rect_prism_3x1p5x5mm/`:**

- `Mirae_LatticeSlab_V4_fixed.stl`
- `Mirae_LatticeSlab_V4_fixed_{1N,5N,10N}_aristo_report.json` + `.vtu`
- Optional: PNGs, `Mirae_LatticeSlab_V4_fixed.adaptive_mesh_report.json`
- `CLEANUP_GUIDE.md`, `MIRAE_LATTICE_FEA_WORKLOG.md`

**Optional Tier C (Mirae-only, not part of Option 2):** V5 branch (~500 MB) if abandoning dense-mesh experiment — see prism CLEANUP_GUIDE Tier C.

---

## Related docs

- [ARISTO_REGRESSION_BASELINE.md](ARISTO_REGRESSION_BASELINE.md) — re-verify V4 after any broad cleanup
- [PIECEWISE_PRISM_LATTICE_GENERATION.md](PIECEWISE_PRISM_LATTICE_GENERATION.md) — Split-P / cylinder output conventions
- [docs/README.md](README.md) — documentation index
