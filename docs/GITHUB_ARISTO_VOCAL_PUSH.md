# GitHub push — Aristo + Vocal experimental work

**Branch:** `cleanup/phase-0-1`  
**Purpose:** Cross-machine handoff (desktop with 32 GB VRAM) including Vocal LBM caches and Aristo FEA artifacts.

## What is tracked

### Code
- `graphite/aristo/` — FEA pipeline, cross-section viz, BCs
- `graphite/lbm/` — Vocal (Lettuce LBM), cache, warm-start
- `graphite/case_studies/cube_1mm/` — 1 mm³ cube specs, generators, runners
- Supporting implicit/woodpile modules for cube geometry
- Tier-1 scripts: `run_vocal.py`, `run_aristo_fea.py`, `plot_cube_1mm_*`, cascade scripts, etc.
- Docs: `ARISTO.md`, `VOCAL.md`, `CUBE_1MM_*`, `VOCAL_CASCADE_WARMSTART.md`

### Data (large files via Git LFS)
- `outputs/vocal/cache/` — full LBM caches (`f.pkl`, `velocity.npy`, `metrics.json`, …)
- `outputs/case_studies/cube_1mm/` — STLs, FEA VTUs (`fea/`), figures (`figures/`), fluid metrics

### Not tracked
- `outputs/case_studies/cube_1mm/archive/` — superseded sweeps
- `outputs/models/`, `optimization/runs/`, `.venv_torch/`
- `.cursor/`

## New machine setup

```powershell
git clone https://github.com/ehj3110/Graphite.git
cd Graphite
git checkout cleanup/phase-0-1
git lfs pull

python -m venv .venv
.\.venv\Scripts\pip install -r requirements-aristo.txt

python -m venv .venv_torch
.\.venv_torch\Scripts\pip install torch lettuce scipy numpy matplotlib
# see docs/VOCAL.md for full Vocal venv notes
```

## Regenerate figures (no re-simulation)

```powershell
python scripts/plot_cube_1mm_aristo_vocal_comparison.py --plot-only
.\.venv_torch\Scripts\python.exe scripts/plot_cube_1mm_vocal_wss_comparison.py
```

## n=256 Vocal refine (desktop)

```powershell
.\.venv_torch\Scripts\python.exe scripts/run_cube_1mm_vocal_refine_n256_linear_splitp.py --force-rerun
```

See `docs/CUBE_1MM_VOCAL_N256_HANDOFF.md`.
