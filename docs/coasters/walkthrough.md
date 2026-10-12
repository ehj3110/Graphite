# Coaster Deliverable Figures — Final Combined Edition

All coaster families have been consolidated into **4 combined deliverable figures** (`1800 x 1350 px`) in [`outputs/Coasters/`](file:///c:/Users/ehunt/OneDrive/Documents/Python%20Scripts/Graphite/outputs/Coasters/), provided in **three color/style editions**:
- **Solid Black (`#141416`)**: `*_previews.png`
- **Solid Lighter Blue (`#3b66b0`, `RGB(59, 102, 176)`)**: `*_previews_blue.png`
- **Original Blue Outline (`#1e3a8a`, `RGB(30, 58, 138)`, `linewidth=1.2`)**: `*_previews_outline.png`

---

## Key Specifications Applied

- **Zero Drop Shadows**: All drop shadows have been completely removed (`#FFFFFF` pure white canvas and interior cells).
- **Exact 2D Cross-Sections**: Re-generated directly from the true 2D cross-section polygons (`build_c15_v1/v2/v3`, `build_a15_v1/v2/v3`, `build_voronoi_polygon`, `build_explicit_polygon`, and exact `z_mm` slice level-sets of the 3D TPMS coasters).
- **10% Side Margins & $1.2 \times D$ Center-to-Center Spacing**:
  - 2-row figures (`C15/A15` and `Voroni`) use $D = 482\text{ px}$, giving exact $10\%$ left/right margins ($180.2\text{ px}$ on an $1800\text{ px}$ canvas) and center-to-center spacing $\text{pitch} = 1.2 \times D = 578.4\text{ px}$ both horizontally and vertically.
  - 3-row figures (`Explicit` and `TPMS`) use $\text{pitch} = 1.2 \times D$ in both directions ($D = 390\text{ px}$ for `Explicit` and $D = 360\text{ px}$ for `TPMS`) so all 3 rows plus labels fit cleanly within the $1350\text{ px}$ height.
- **25% Reduced Coaster Labels**: Coaster titles are centered directly above each coaster at `27 pt` (`25 pt` for 3-row figures), with main figure headers at `44 pt`.

---

## Combined Deliverable Figures (Outline, Lighter Blue & Black)

````carousel
![C15/A15 — Original Blue Outline (#1e3a8a)](C:\Users\ehunt\.gemini\antigravity\brain\ec7d534e-8069-4ab3-a613-837f39a53621\c15_a15_previews_outline.png)
<!-- slide -->
![Voroni — Original Blue Outline (#1e3a8a)](C:\Users\ehunt\.gemini\antigravity\brain\ec7d534e-8069-4ab3-a613-837f39a53621\voroni_previews_outline.png)
<!-- slide -->
![Explicit — Original Blue Outline (#1e3a8a)](C:\Users\ehunt\.gemini\antigravity\brain\ec7d534e-8069-4ab3-a613-837f39a53621\explicit_previews_outline.png)
<!-- slide -->
![TPMS — Original Blue Outline (#1e3a8a)](C:\Users\ehunt\.gemini\antigravity\brain\ec7d534e-8069-4ab3-a613-837f39a53621\tpms_previews_outline.png)
<!-- slide -->
![C15/A15 — Solid Lighter Blue (#3b66b0)](C:\Users\ehunt\.gemini\antigravity\brain\ec7d534e-8069-4ab3-a613-837f39a53621\c15_a15_previews_blue.png)
<!-- slide -->
![Voroni — Solid Lighter Blue (#3b66b0)](C:\Users\ehunt\.gemini\antigravity\brain\ec7d534e-8069-4ab3-a613-837f39a53621\voroni_previews_blue.png)
<!-- slide -->
![Explicit — Solid Lighter Blue (#3b66b0)](C:\Users\ehunt\.gemini\antigravity\brain\ec7d534e-8069-4ab3-a613-837f39a53621\explicit_previews_blue.png)
<!-- slide -->
![TPMS — Solid Lighter Blue (#3b66b0)](C:\Users\ehunt\.gemini\antigravity\brain\ec7d534e-8069-4ab3-a613-837f39a53621\tpms_previews_blue.png)
<!-- slide -->
![C15/A15 — Solid Black (#141416)](C:\Users\ehunt\.gemini\antigravity\brain\ec7d534e-8069-4ab3-a613-837f39a53621\c15_a15_previews.png)
<!-- slide -->
![Voroni — Solid Black (#141416)](C:\Users\ehunt\.gemini\antigravity\brain\ec7d534e-8069-4ab3-a613-837f39a53621\voroni_previews.png)
<!-- slide -->
![Explicit — Solid Black (#141416)](C:\Users\ehunt\.gemini\antigravity\brain\ec7d534e-8069-4ab3-a613-837f39a53621\explicit_previews.png)
<!-- slide -->
![TPMS — Solid Black (#141416)](C:\Users\ehunt\.gemini\antigravity\brain\ec7d534e-8069-4ab3-a613-837f39a53621\tpms_previews.png)
````

---

## Output File Inventory

| Figure Title | Outline (`#1e3a8a`) | Lighter Solid Blue (`#3b66b0`) | Solid Black (`#141416`) |
| :--- | :--- | :--- | :--- |
| **C15/A15** | [`c15_a15_previews_outline.png`](file:///c:/Users/ehunt/OneDrive/Documents/Python%20Scripts/Graphite/outputs/Coasters/c15_a15_previews_outline.png) | [`c15_a15_previews_blue.png`](file:///c:/Users/ehunt/OneDrive/Documents/Python%20Scripts/Graphite/outputs/Coasters/c15_a15_previews_blue.png) | [`c15_a15_previews.png`](file:///c:/Users/ehunt/OneDrive/Documents/Python%20Scripts/Graphite/outputs/Coasters/c15_a15_previews.png) |
| **Voroni** | [`voroni_previews_outline.png`](file:///c:/Users/ehunt/OneDrive/Documents/Python%20Scripts/Graphite/outputs/Coasters/voroni_previews_outline.png) | [`voroni_previews_blue.png`](file:///c:/Users/ehunt/OneDrive/Documents/Python%20Scripts/Graphite/outputs/Coasters/voroni_previews_blue.png) | [`voroni_previews.png`](file:///c:/Users/ehunt/OneDrive/Documents/Python%20Scripts/Graphite/outputs/Coasters/voroni_previews.png) |
| **Explicit** | [`explicit_previews_outline.png`](file:///c:/Users/ehunt/OneDrive/Documents/Python%20Scripts/Graphite/outputs/Coasters/explicit_previews_outline.png) | [`explicit_previews_blue.png`](file:///c:/Users/ehunt/OneDrive/Documents/Python%20Scripts/Graphite/outputs/Coasters/explicit_previews_blue.png) | [`explicit_previews.png`](file:///c:/Users/ehunt/OneDrive/Documents/Python%20Scripts/Graphite/outputs/Coasters/explicit_previews.png) |
| **TPMS** | [`tpms_previews_outline.png`](file:///c:/Users/ehunt/OneDrive/Documents/Python%20Scripts/Graphite/outputs/Coasters/tpms_previews_outline.png) | [`tpms_previews_blue.png`](file:///c:/Users/ehunt/OneDrive/Documents/Python%20Scripts/Graphite/outputs/Coasters/tpms_previews_blue.png) | [`tpms_previews.png`](file:///c:/Users/ehunt/OneDrive/Documents/Python%20Scripts/Graphite/outputs/Coasters/tpms_previews.png) |
