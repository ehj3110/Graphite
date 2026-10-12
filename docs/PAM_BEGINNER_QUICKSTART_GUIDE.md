# Graphite Interlinked & PAMs — Beginner User Guide & Quickstart

Welcome! This guide is for mechanical designers, engineers, and researchers who want to generate **3D-printable interlinked metamaterials** and **chainmail fabrics** in Graphite without needing to understand crystallography, space groups, or dislocation mathematics.

---

## 1. What are Interlinked Metamaterials?

In standard 3D-printed lattices, all struts are welded together at rigid nodal joints. 

In **interlinked / polycatenated metamaterials (PAMs)**, every ring or cage is an **independent, unbonded moving body**. They interlock through each other's open windows like links in a chain:
* **Print-in-Place:** Fabricated in a single build with zero assembly.
* **Non-Bonded Motion:** Individual links slide and rotate within a strictly positive clearance gap ($\Delta > 0$).
* **Kinematic Properties:** Produce flexible fabrics, compliant shock absorbers, and crash cushions that absorb energy by inter-cage friction and jamming rather than brittle strut buckling.

---

## 2. 3-Line Quickstart

You can generate a print-ready, watertight assembly in three lines of Python:

```python
from graphite.explicit.interlinked import quick_lattice

# Fill a 40 x 40 x 40 mm volume with compliant diamond cages
lattice = quick_lattice(preset="shock_absorber", bounding_box=(40, 40, 40))

# Export an ultra-compact, instanced 3MF package for your slicer
lattice.export("my_lattice.3mf")
```

Open `my_lattice.3mf` directly in **Bambu Studio**, **PrusaSlicer**, **Cura**, or **Formlabs PreForm**.

---

## 3. Presets & Cell Cheat Sheet

Select a preset or cell depending on what you are designing:

| Preset Name | Cell Tag | Geometry | Dimension | Ideal Application |
|---|---|---|:---:|---|
| `"shock_absorber"` | `d4tet` | Regular Tetrahedra (Diamond) | 3D | Maximum compliance, sports padding, helmet liners, energy dissipation. |
| `"impact_cushion"` | `c6tt` | Truncated Tetrahedra (Simple Cubic) | 3D | Higher load-bearing capacity, crash boxes, heavy impact cushions. |
| `"cube_cages"` | `d4cube` | Wireframe Cubes (Diamond) | 3D | Orthogonal kinematics, shear-compliant mechanisms. |
| `"chainmail"` | `euro_4in1` | European 4-in-1 Rings | 2D | Wearable fabrics, flexible curtains, protective blankets. |
| `"space_fabric"` | `nasa_hexagon` | NASA JPL Spiral Rings | 2D | Snag-resistant deployable space fabrics, flexible antenna covers. |
| `"octahedra"` | `j4oct` | Alternating Octahedra | 3D | Multi-axis kinematic compliance. |

---

## 4. Setting Dimensions for Your 3D Printer

Two numbers dictate whether an interlinked lattice prints successfully:
1. **Strut Outer Diameter ($d$):** The thickness of each strut beam (e.g., $0.75\,\text{mm}$).
2. **Clearance Gap ($\Delta$):** The physical empty air gap between adjacent links (e.g., $0.20\,\text{mm}$ or $200\,\mu\text{m}$).

### Recommended Starting Parameters by 3D Printing Technology:

| Printer Technology | Materials | Strut Diameter ($d$) | Minimum Clearance ($\Delta$) | Recommended Preset |
|---|---|:---:|:---:|---|
| **Metal LPBF / SLM** | Ti-6Al-4V, AlSi10Mg, 316L | $0.50\text{--}0.80\,\text{mm}$ | $0.15\text{--}0.25\,\text{mm}$ | `"shock_absorber"` (`d4tet`) |
| **Resin SLA / DLP** | Tough / Engineering Resins | $0.60\text{--}1.00\,\text{mm}$ | $0.20\text{--}0.30\,\text{mm}$ | `"impact_cushion"` (`c6tt`) |
| **Polymer SLS / MJF** | PA12 (Nylon), TPU | $0.75\text{--}1.50\,\text{mm}$ | $0.30\text{--}0.50\,\text{mm}$ | `"chainmail"` (`euro_4in1`) |
| **Filament FDM / FFF** | PLA, PETG, TPU (0.4 nozzle) | $1.50\text{--}3.00\,\text{mm}$ | $0.40\text{--}0.80\,\text{mm}$ | `"chainmail"` (`euro_4in1`) |

```python
from graphite.explicit.interlinked import quick_lattice

# Custom tuned for a Formlabs Form 4 resin printer (clearance = 250 um)
lattice = quick_lattice(
    cell="d4tet",
    bounding_box=(50, 50, 25),      # 50 x 50 x 25 mm envelope
    strut_diameter=0.80,            # 800 um thick struts
    clearance=0.25,                 # 250 um gap between parts
    add_frame=True,                 # Solid perimeter border to hold for tensile testing
)
lattice.export("formlabs_test.3mf")
```

---

## 5. Generating 2D Flexible Chainmail Fabrics

To create flat textiles and flexible sheets:

```python
from graphite.explicit.interlinked import quick_fabric

# Make a 80 x 80 mm patch of NASA space fabric
fabric = quick_fabric(
    pattern="space_fabric",
    width=80.0,
    length=80.0,
    strut_diameter=0.75,
    clearance=0.25,
)
fabric.export("nasa_space_fabric.3mf")
```

---

## 6. No-Code Command-Line Utilities

You don't even have to write Python code to design, inspect, or manufacture interlinked lattices. Graphite includes three turnkey command-line utilities:

### Tool 1: Turnkey Lattice Generator (`scripts/generate_pam.py` or `python -m graphite.explicit.interlinked`)
Generate ready-to-print `.3mf` files and matte preview renders directly from your terminal:
```bash
# 1. Generate a 40 x 40 x 40 mm shock absorber lattice (default 750 um strut, 200 um clearance):
python scripts/generate_pam.py --preset shock_absorber --box 40 40 40 --out outputs/my_shock.3mf

# 2. Generate a 60 x 60 mm flexible space fabric sheet:
python scripts/generate_pam.py --preset space_fabric --width 60 --length 60 --out outputs/my_fabric.3mf

# 3. Fast DfAM preflight dry run (calculates link count and outer cage sizes in 0.05s without generating meshes):
python scripts/generate_pam.py --preset shock_absorber --box 50 50 50 --dry-run

# 4. Generate with a solid handling frame for tensile grips and save a preview image:
python scripts/generate_pam.py --cell d4tet --box 40 40 20 --frame --preview outputs/specimen.png

# 5. List all available presets and applications:
python scripts/generate_pam.py --list-presets
```

### Tool 2: Dimension & Sizing Calculator (`scripts/dimension_pam.py`)
Want to know what cell pitch or outer cage diameter you'll get before building?
```bash
# Calculate pitch and cage diameter for a 750 um strut with 200 um clearance:
python scripts/dimension_pam.py --cell d4tet --wire-diameter 0.75 --clearance 0.20
```
Output:
```text
=== PAM Dimensioning Result: D-4-TET (DIA) ===
  Unit Cell Pitch (a):        9.3320 mm
  Cage Outer Diameter (D):    6.8919 mm
  Wire Diameter (d):          0.7500 mm (750.0 um)
  Surface Clearance (Delta):  0.2000 mm
  Solid Volume Fraction (Vf): 14.68%
  Jamming Limit (Vf_max):     23.56%
  Feasible Non-Colliding:     YES
```

Compare all 11 cell types side-by-side:
```bash
python scripts/dimension_pam.py --compare --clearance 0.20 --vf 0.08
```

### Tool 3: High-Resolution 3D Visualizer (`scripts/pam_gallery.py`)
Generate clean, non-glare, matte scientific render cards of any cell with internal cutaways:
```bash
# Render a standard 3D specimen:
python scripts/pam_gallery.py --cell d4tet

# Render an internal cross-section cutaway (revealing internal links):
python scripts/pam_gallery.py --cell d4tet --cutaway
```
Images are automatically saved to `outputs/pam_gallery_<cell>.png`.

---

## 7. Troubleshooting & Frequently Asked Questions

### Q: Why did my print fuse into a solid block?
**A:** Your printer's laser or nozzle spot size is wider than the requested clearance gap. Increase `clearance` (e.g. from `0.20` to `0.35` mm) or reduce your exposure/flow rate.

### Q: Why is my STL file 500 MB, but my 3MF file is only 30 KB?
**A:** Standard monolithic STLs duplicate geometry for every single link. Graphite's **instanced 3MF exporter** stores the unique cage geometry once in `<resources>` and uses lightweight transformation matrices in `<build>`—giving you a **98%+ reduction in file size** while opening instantly in slicers. Always prefer `.3mf`!

### Q: Why did boundary links fall off the edges of my print?
**A:** In an interlinked lattice, edge links only have neighbors on one side. If they aren't gripped, they will fall free like loose chain links. Pass `add_frame=True` in `quick_lattice` to automatically weld a solid CAD border around the outer edges.

### Q: What does "Jamming Limit Exceeded" mean?
**A:** If you request a solid density higher than the cell's physical geometry allows (e.g. asking for 30% density on a cell whose jamming limit is 23%), the struts would have to intersect each other. The engine protects you from wasted print jobs by raising a DfAM warning.
