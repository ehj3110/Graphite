# Metamaterial Heteroepitaxy Interface Catalog (Stage 2)

> Quantitative evaluation and ranking of multi-network metamorphic interfaces.
> Generated using the automated crystallographic engine (`plan_heteroepitaxial_interface`),
> audited via **Manifold3D exact CSG booleans**, **multi-face penetration checks ($N_A \ge 2, N_B \ge 2$)**,
> and ranked by the **Quantitative Interface Quality Index ($\mathcal{Q}_{\text{interface}}$)**.

## Interface Leaderboard

| Rank | System | Habit Plane | Mode | Dimensions (mm) | Bodies | Inter-Zone Links | Penetration Depth (min / avg) | Clearance (µm) | $\mathcal{Q}_{\text{interface}}$ | Status |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **#1** | **D4-CUBE -> D4-TET on {111}** | {111} | `continuous` | 28.7x26.3x23.3 | 31 | 5/5 | 3 / 3.00 | +381.4 | **0.951** | 🟩 PASS |
| **#2** | **C-6-TT -> B-8-CO on {100}** | {100} | `discrete` | 27.6x27.6x44.2 | 17 | 4/4 | 4 / 4.00 | +183.7 | **0.868** | 🟩 PASS |
| **#3** | **C-6-TT -> D-4-TET on {100}** | {100} | `discrete` | 41.5x41.5x25.2 | 41 | 5/5 | 2 / 2.00 | +191.4 | **0.807** | 🟩 PASS |
| **#4** | **D4-CUBE -> D4-TET on {110}** | {110} | `discrete` | 20.5x18.8x31.8 | 26 | 5/5 | 3 / 3.00 | +104.7 | **0.805** | 🟩 PASS |
| **#5** | **D4-TET -> D4-TET on {111}** | {111} | `continuous` | 28.7x27.3x21.0 | 31 | 0/5 | 0 / 0.00 | +-64.7 | **0.000** | 🟥 FAIL |
| **#6** | **C-6-TT -> C-6-TT on {100}** | {100} | `discrete` | 27.6x27.6x46.0 | 16 | 12/4 | 2 / 2.75 | +-71.1 | **0.000** | 🟥 FAIL |

## Detailed Analysis & Structural Rationales

### #1. D4-CUBE -> D4-TET on {111} (Score: $\mathcal{Q} = 0.951$)
- **Identifier:** `catalog_1_d4cube_to_d4tet_111`
- **Habit Plane:** {111}
- **Assembly Mode:** `continuous`
- **Strategy:** Diamond {111} Glide-Set Shockley Interface
- **Boundary Interlinking:** 5 / 5 possible catenations (100.0%)
- **Penetration Depth:** Minimum 3 faces punctured per link, average 3.00 faces (satisfies $N \ge 2$ standard).
- **Physical Surface Clearance:** $+381.4\,\mu\text{m}$ (zero collisions via Manifold3D exact CSG boolean audit).
- **Assets:**
  - 3MF Package: [`catalog_1_d4cube_to_d4tet_111.3mf`](catalog_1_d4cube_to_d4tet_111.3mf)
  - Solid STL: [`catalog_1_d4cube_to_d4tet_111.stl`](catalog_1_d4cube_to_d4tet_111.stl)
  - Visual Render: ![D4-CUBE -> D4-TET on {111}](catalog_1_d4cube_to_d4tet_111.png)

### #2. C-6-TT -> B-8-CO on {100} (Score: $\mathcal{Q} = 0.868$)
- **Identifier:** `catalog_3_c6tt_to_b8co_100`
- **Habit Plane:** {100}
- **Assembly Mode:** `discrete`
- **Strategy:** Canonical CSL Match theta=0.0deg
- **Boundary Interlinking:** 4 / 4 possible catenations (100.0%)
- **Penetration Depth:** Minimum 4 faces punctured per link, average 4.00 faces (satisfies $N \ge 2$ standard).
- **Physical Surface Clearance:** $+183.7\,\mu\text{m}$ (zero collisions via Manifold3D exact CSG boolean audit).
- **Assets:**
  - 3MF Package: [`catalog_3_c6tt_to_b8co_100.3mf`](catalog_3_c6tt_to_b8co_100.3mf)
  - Solid STL: [`catalog_3_c6tt_to_b8co_100.stl`](catalog_3_c6tt_to_b8co_100.stl)
  - Visual Render: ![C-6-TT -> B-8-CO on {100}](catalog_3_c6tt_to_b8co_100.png)

### #3. C-6-TT -> D-4-TET on {100} (Score: $\mathcal{Q} = 0.807$)
- **Identifier:** `catalog_4_c6tt_to_d4tet_100`
- **Habit Plane:** {100}
- **Assembly Mode:** `discrete`
- **Strategy:** Canonical 45-deg CSL Commensurate Domain Match
- **Boundary Interlinking:** 5 / 5 possible catenations (100.0%)
- **Penetration Depth:** Minimum 2 faces punctured per link, average 2.00 faces (satisfies $N \ge 2$ standard).
- **Physical Surface Clearance:** $+191.4\,\mu\text{m}$ (zero collisions via Manifold3D exact CSG boolean audit).
- **Assets:**
  - 3MF Package: [`catalog_4_c6tt_to_d4tet_100.3mf`](catalog_4_c6tt_to_d4tet_100.3mf)
  - Solid STL: [`catalog_4_c6tt_to_d4tet_100.stl`](catalog_4_c6tt_to_d4tet_100.stl)
  - Visual Render: ![C-6-TT -> D-4-TET on {100}](catalog_4_c6tt_to_d4tet_100.png)

### #4. D4-CUBE -> D4-TET on {110} (Score: $\mathcal{Q} = 0.805$)
- **Identifier:** `catalog_2_d4cube_to_d4tet_110`
- **Habit Plane:** {110}
- **Assembly Mode:** `discrete`
- **Strategy:** Canonical CSL Match theta=0.0deg
- **Boundary Interlinking:** 5 / 5 possible catenations (100.0%)
- **Penetration Depth:** Minimum 3 faces punctured per link, average 3.00 faces (satisfies $N \ge 2$ standard).
- **Physical Surface Clearance:** $+104.7\,\mu\text{m}$ (zero collisions via Manifold3D exact CSG boolean audit).
- **Assets:**
  - 3MF Package: [`catalog_2_d4cube_to_d4tet_110.3mf`](catalog_2_d4cube_to_d4tet_110.3mf)
  - Solid STL: [`catalog_2_d4cube_to_d4tet_110.stl`](catalog_2_d4cube_to_d4tet_110.stl)
  - Visual Render: ![D4-CUBE -> D4-TET on {110}](catalog_2_d4cube_to_d4tet_110.png)

### #5. D4-TET -> D4-TET on {111} (Score: $\mathcal{Q} = 0.000$)
- **Identifier:** `catalog_5_d4tet_to_d4tet_111`
- **Habit Plane:** {111}
- **Assembly Mode:** `continuous`
- **Strategy:** Diamond {111} Glide-Set Shockley Interface
- **Boundary Interlinking:** 0 / 5 possible catenations (0.0%)
- **Penetration Depth:** Minimum 0 faces punctured per link, average 0.00 faces (satisfies $N \ge 2$ standard).
- **Physical Surface Clearance:** $+-64.7\,\mu\text{m}$ (zero collisions via Manifold3D exact CSG boolean audit).
- **Assets:**
  - 3MF Package: [`catalog_5_d4tet_to_d4tet_111.3mf`](catalog_5_d4tet_to_d4tet_111.3mf)
  - Solid STL: [`catalog_5_d4tet_to_d4tet_111.stl`](catalog_5_d4tet_to_d4tet_111.stl)
  - Visual Render: ![D4-TET -> D4-TET on {111}](catalog_5_d4tet_to_d4tet_111.png)

### #6. C-6-TT -> C-6-TT on {100} (Score: $\mathcal{Q} = 0.000$)
- **Identifier:** `catalog_6_c6tt_to_c6tt_100`
- **Habit Plane:** {100}
- **Assembly Mode:** `discrete`
- **Strategy:** Canonical CSL Match theta=0.0deg
- **Boundary Interlinking:** 12 / 4 possible catenations (300.0%)
- **Penetration Depth:** Minimum 2 faces punctured per link, average 2.75 faces (satisfies $N \ge 2$ standard).
- **Physical Surface Clearance:** $+-71.1\,\mu\text{m}$ (zero collisions via Manifold3D exact CSG boolean audit).
- **Assets:**
  - 3MF Package: [`catalog_6_c6tt_to_c6tt_100.3mf`](catalog_6_c6tt_to_c6tt_100.3mf)
  - Solid STL: [`catalog_6_c6tt_to_c6tt_100.stl`](catalog_6_c6tt_to_c6tt_100.stl)
  - Visual Render: ![C-6-TT -> C-6-TT on {100}](catalog_6_c6tt_to_c6tt_100.png)
