# Streamlit App Revamp Plan

> **Doc role:** Planning only. Agents: do not implement from this file unless tasked. Start at [AGENTS.md](../AGENTS.md); current UI behavior is [STREAMLIT_WORKFLOW.md](STREAMLIT_WORKFLOW.md).

Status: **planning only** — not implemented.  
Date: Aug 2026  
Companion: [STREAMLIT_WORKFLOW.md](STREAMLIT_WORKFLOW.md) (current behavior),
[SURFACE_FIRST_DUAL_TRIM.md](SURFACE_FIRST_DUAL_TRIM.md) §4 (explicit sequential UI),
[SC_CONFORMAL_FUTURE_WORK.md](SC_CONFORMAL_FUTURE_WORK.md),
[SC_CONFORMAL_THREE_WAY_COMPARE.md](SC_CONFORMAL_THREE_WAY_COMPARE.md).

## 1. One-line diagnosis

**`app.py` is a polished implicit TPMS wizard with a likely-broken legacy
explicit stub.** Scripts and `graphite/` packages already ship A15 Kagome,
modular SC hex, V1/V2/V3 conformal compares, woodpile, Dual/legacy hex,
Aristo FEA, Vocal LBM, and topology optimization — almost none of that is
honestly reachable from the UI.

---

## 2. What the app is today

Single-file Streamlit wizard (`app.py`, ~1.7k lines), six steps:

| Step | Role | Actually used by |
|------|------|------------------|
| 1 Geometry | Custom STL / ASTM / Primitive | Both engines (ASTM: *not wired*) |
| 2 Lattice | Implicit TPMS vs Explicit - Fast (Delaunay) | Branch point |
| 3 Sizing & density | SF vs wall thickness, pore/UC | Implicit only (explicit still walks through) |
| 4 Boundary & field | Field grading + slice preview | Implicit only |
| 5 Shelling | Core / skin / combined | Implicit only |
| 6 Execute | Resolution, naming, Generate | Both (different backends) |

**Implicit path (production quality):** Uniform + Field Controls →
`generate_conformal_lattice` / `generate_field_driven_lattice`. Solid
fraction vs wall thickness, calibration, origin modes, STL/STEP, manifests.
This is the part of the app that matches Graphite.

**Explicit path (stale):** Calls
`graphite.explicit.generate_conformal_scaffold` → `generate_topology` →
`generate_geometry`. But `__init__.generate_conformal_scaffold` now forwards
to A15 with `skip_sweep=True` and returns a **dict**, not the
`ScaffoldResult(nodes, elements, surface_faces)` the app expects. Spinner
copy still says “GMSH + manifold3d”. Topology labels (“Standard Tet”,
“Surface Cage / Dual”) do not match A15 / SC modular naming. Default
`explicit_topology = "Kagome Surface Dual"` is not even offered in the radio.

**Docs drift:** `STREAMLIT_WORKFLOW.md` / README still describe Advanced mode,
YAML config upload, and Conformal Dual wiring that are **absent** from
current `app.py`. Dead grading modes (Variable Porosity, Chirped,
Osteochondral, Boundary-Driven) remain as unreachable `elif` branches.

---

## 3. Capability gap (Graphite today vs Streamlit)

### Exposed well
- Implicit TPMS: Gyroid, Diamond, Schwarz-P/D, Neovius, Split-P
- Uniform + field-driven grading (Cartesian / cylindrical / spherical)
- Density control (solid fraction XOR wall thickness)
- Shell modes, STL/STEP, parameter manifests, dated output folders
- Center-slice field preview (Step 4)

### Exists in code, missing or broken in UI

| Capability | Where | App |
|---|---|---|
| A15 conformal Kagome | `a15_conformal`, `generate_conformal_lattice(lattice_type='A15')` | Broken / mislabeled “Delaunay” |
| Modular SC hex | `conformal_generator` / `conformal_core`, `rule_name` | None |
| Compare V1 closest morph | `compare_v1_closest/` | None |
| Compare V2 surface-first | `compare_v2_surface_first/` (+ stitch gates) | Doc’d for future Streamlit only |
| Compare V3 hybrid | `compare_v3_hybrid/` | None |
| Hex rules (grid, octahedral, kelvin, …) | `hex_rules.py` | None |
| Woodpile (implicit) | `generate_lattice.py` + woodpile modules | None |
| Conformal Dual (legacy Route 3) | `legacy_gmsh` / Dual docs | Docs claim wire-up; not in app |
| Mesh repair as UI step | `mesh_repair.repair_cad_mesh` | Scripts only |
| Lidinoid / multi-zonal / chirped / osteochondral | Implicit packages | Removed from UI or never offered |
| Aristo FEA | `graphite/aristo/` | Separate requirements; no app |
| Vocal LBM | `graphite/lbm/` | Separate venv; no app |
| Topology optimization | `graphite/topt/`, `sandbox_app.py` | Separate sandbox |

### UX gaps even on the working implicit path
- Draft/commit table Apply friction (necessary for Streamlit stability, still confusing)
- No 3D lattice preview — only a 2D field slice
- Long runs: spinner only, no stage progress / cancel
- Right-rail dumps raw `params` (noisy recipe view)
- ASTM option shown then `st.stop()`s
- Downloads tied to current session run
- Streamlit not pinned in a root requirements file

---

## 4. Cleaner UI options (keep vs replace)

Stay honest: **no Gradio / NiceGUI / React product UI exists in the repo.**
Dash appears only as an Open3D transitive dependency — not a foundation.
Streamlit is the declared product UI (`streamlit run app.py`) but is not
pinned at the root.

### Option A — Stay on Streamlit, modernize (recommended near-term)

**Why:** Implicit wizard already works; coworkers know it; lowest migration
cost. Streamlit 1.30+ multipage + `st.navigation` + `st.status` /
`st.fragment` / `@st.cache_data` cover most pain points if we stop trying to
stuff every engine into one linear 6-step script.

**Shape:**
- Multipage app under `ui/` (or `pages/`): Home · Implicit TPMS · Explicit
  Conformal · Compare Lab · Settings.
- Engine-aware flows (explicit never walks density-grading-shell steps).
- Background / staged jobs with `st.status` + stage logs.
- Pin `streamlit` (+ `streamlit-plotly` or pyvista/trame for 3D if needed) in
  a root requirements.

**Limits:** Heavy interactive tables and true cancelable long jobs remain
awkward; draft/commit patterns likely stay for field tables.

### Option B — Thin FastAPI + lightweight frontend (recommended mid-term if jobs grow)

**Why:** Conformal V2/V3 and solidify jobs are multi-minute, multi-stage.
A job queue (FastAPI + RQ/Celery or even a simple process pool) with a thin
UI (HTMX, React, or even Streamlit as a client) matches how the scripts
already work.

**Shape:** API wraps `generate_conformal_lattice`, `generate_compare_v*`,
implicit generators; UI polls job status and serves preview artifacts
(slice PNG, colored GLB, report JSON).

**Cost:** Real engineering; only worth it once Explicit Conformal is a
first-class product path, not a lab script.

### Option C — Gradio / NiceGUI / Solara

Possible for demos, but **do not** introduce a second product UI unless
Streamlit is being abandoned. Gradio is great for ML demos; NiceGUI is nicer
for long-lived desktop-like apps; neither is already in the stack.

### Option D — Keep TO / Aristo / Vocal as separate apps forever

Do **not** fold FEA / LBM / TO into the lattice wizard. The existing
`experiments/scikit_topt_sandbox/sandbox_app.py` pattern is correct: sister
apps, shared packages.

**Recommendation:** Option A now (fix honesty + multipage + explicit
branch). Re-evaluate Option B when surface-first SC + solidify becomes the
default coworker path and jobs regularly exceed ~2–3 minutes.

---

## 5. Target information architecture

```text
Graphite UI
├── Home                 — what Graphite is; pick a workflow; recent outputs
├── Implicit TPMS        — current Steps 1–6, cleaned (coworkers / print path)
│     optional: Woodpile sub-mode
├── Explicit Conformal   — A15 | Modular SC | Surface-first (V2) | Experimental
│     sequential params per SURFACE_FIRST_DUAL_TRIM §4
├── Compare Lab          — run V1/V2/V3 on same CAD; side-by-side metrics/renders
└── Settings / Health    — dependency check, output root, advanced toggles
```

Simulation suites (Aristo, Vocal, TO) stay out.

### Explicit Conformal sequential flow (engine-aware)

1. **Geometry** — STL + optional `repair_cad_mesh` preview (watertight? holes?).
2. **Family** — A15 Kagome · Modular SC (V1 morph) · Surface-first SC (V2) ·
   Hybrid (V3, experimental).
3. **Volume grid** — cell size (anisotropic), VF threshold, `rule_name`, strut
   radius. Preview: culled unsnapped scaffold.
4. **Surface / dual** (V2/V3 only) — dual recipe, cage params. Preview: dual
   alone + overlaid on CAD.
5. **Stitch gates** (V2/V3) — distance ≤ L, angle ≤ 60°, orphan policy.
   Preview: pass/fail colored candidates, counts.
6. **Execute** — staged progress, QA report (stranded nodes, stitch rejects,
   strut length hist), STL + manifest.

Never show Steps 3–5 of the *implicit* wizard on this branch.

### Implicit flow (keep, tidy)

Keep the 6-step wizard, but:
- Remove ASTM until implemented, or implement it.
- Delete dead grading `elif`s or restore them deliberately under Advanced.
- Add Woodpile as a sibling under Implicit, not a fake TPMS type.
- Replace raw `params` dump with a readable recipe card.
- Add optional 3D / GLB preview after generate.

---

## 6. Phased delivery plan

### Phase 0 — Honesty & repair (1–2 days)
- Pin Streamlit in root requirements.
- Fix or **remove** the Explicit Delaunay path; do not leave a broken Generate
  button. Preferred fix: call
  `generate_conformal_lattice(..., lattice_type='A15'|'SC')` like
  `scripts/generate_lattice.py`.
- Hide ASTM; delete or quarantine dead grading branches.
- Align `STREAMLIT_WORKFLOW.md` / README with `app.py` (no Advanced claims
  that do not exist).
- Fix spinner / labels (no “GMSH” on GMSH-free paths).

### Phase 1 — Multipage shell + Implicit polish (few days)
- Split `app.py` into `ui/Home.py`, `ui/implicit_wizard.py`, shared
  `ui/state.py` / `ui/io.py`.
- Recipe card, health check page, recent-outputs browser.
- `st.status` stages for implicit generate.
- Keep field-table draft/commit (it works); document Apply in UI copy.

### Phase 2 — Explicit Conformal product path (1–2 weeks)
- Wire A15 one-click and Modular SC (`rule_name`, cell, VF, strut).
- Stage previews: cull-before-snap, post-morph, solid.
- QA metrics panel (from compare report keys + future colorized GLB —
  see future-work C1).
- Optional: stair-step gate / surface-relax toggles as Advanced.

### Phase 3 — Surface-first + Compare Lab (after V2 API stabilizes)
- Implement Streamlit mapping from `SURFACE_FIRST_DUAL_TRIM.md` §4.
- Compare Lab: same CAD → V1/V2/V3 → shared metrics dashboard.
- Unify compare pipelines behind one interface (future-work C5).

### Phase 4 — Jobs & previews (as needed)
- If runs stay long: background worker + artifact store (Option B lite).
- Color-coded diagnostic exports; 3D mesh preview.
- Parameter sweep harness behind Compare Lab.

---

## 7. Design principles for the revamp

1. **Engine-aware navigation** — never show irrelevant controls.
2. **Stage previews before commit** — especially for conformal (cull → dual →
   stitch).
3. **Honest labels** — “Surface-first dual + trim”, not “fully conformal
   volume”.
4. **Scripts remain source of truth** — UI wraps the same public APIs;
   no divergent generation logic in `app.py`.
5. **Sister apps for simulation** — lattice UI does not grow FEA/LBM/TO.
6. **Docs match code** — every UI Mode / Advanced claim either ships or is
   removed from docs in the same PR.

---

## 8. Success criteria

- A coworker can generate a print-ready implicit TPMS without touching
  field-table internals they do not need.
- Explicit Generate produces a real A15 or SC lattice (no API mismatch).
- Explicit users never see TPMS grading / shell steps.
- V2 path (when wired) collects grid → dual → stitch with orphan/reject
  counts visible.
- README / STREAMLIT_WORKFLOW describe only features that exist.
- Streamlit is pinned and `streamlit run` works from a documented env.

## 9. Explicit non-goals (for this revamp)

- Replacing Streamlit with React in Phase 0–2.
- Merging Aristo / Vocal / TO into the lattice wizard.
- Re-implementing generation inside the UI layer.
- Perfecting surface relaxation / ARAP before the UI can even call SC
  modular (do those in the engine; expose toggles later).
