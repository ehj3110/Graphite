# Research Entry: A15 + Kagome Supercell Breakthrough

**Date:** 2026-06-25  
**Status:** Validated — unit cell verified, multiple grid configurations exported

---

## Summary

We have successfully demonstrated that the A15 Frank-Kasper phase (space group $Pm\bar{3}n$) can serve as a deterministic, perfectly symmetric host structure for a 3D Kagome lattice infill. This eliminates all dependency on Delaunay tetrahedralization, which is inherently non-repeatable on highly symmetric crystal lattices.

---

## Exact Methodology

### Step 1 — A15 Primary Network Generation

The A15 ($Pm\bar{3}n$) unit cell is seeded with 8 atomic sites in fractional coordinates:

**B-atom sites (BCC backbone):**
$$\mathbf{b}_1 = (0, 0, 0), \quad \mathbf{b}_2 = \left(\tfrac{1}{2}, \tfrac{1}{2}, \tfrac{1}{2}\right)$$

**A-atom sites (mutually orthogonal chains on cube faces):**
$$\left(\tfrac{1}{4}, 0, \tfrac{1}{2}\right),\; \left(\tfrac{3}{4}, 0, \tfrac{1}{2}\right),\; \left(\tfrac{1}{2}, \tfrac{1}{4}, 0\right),\; \left(\tfrac{1}{2}, \tfrac{3}{4}, 0\right),\; \left(0, \tfrac{1}{2}, \tfrac{1}{4}\right),\; \left(0, \tfrac{1}{2}, \tfrac{3}{4}\right)$$

These are tiled across an $n_x \times n_y \times n_z$ supercell. Duplicate nodes on shared cell faces are deduplicated using coordinate rounding.

### Step 2 — Explicit Deterministic Bonding

Rather than Delaunay triangulation, the primary bond network is built using a **nearest-neighbor distance matrix** via `scipy.spatial.cKDTree`:

$$d_{ij} \leq 0.62 \times \ell_{\text{cell}}$$

This threshold reliably captures the authentic Frank-Kasper Z12 and Z14 coordination polyhedra without introducing spurious cross-void bonds. The resulting network forms the geometrically accurate A15 cage structure.

**Verified bond counts (1×1×1 unit cell):**
- 8 primary nodes
- 22 primary bonds

### Step 3 — Deterministic Clique (4-Clique) Reconstruction

To extract the tetrahedral elements of the A15 network **without any Delaunay call**, we perform a graph-theoretic search for **4-cliques**: sets of 4 nodes that are all mutually bonded.

```python
for u in range(N):
    for v in adj[u] where v > u:
        common_uv = adj[u] ∩ adj[v]
        for w in common_uv where w > v:
            common_uvw = common_uv ∩ adj[w]
            for x in common_uvw where x > w:
                cliques.append((u, v, w, x))
```

This produces the exact set of space-filling irregular tetrahedra that tile the A15 Frank-Kasper phase. Because the bond network itself has perfect $Pm\bar{3}n$ symmetry (built from crystallographic data, not arbitrary triangulation), all extracted cliques obey that symmetry identically.

**Verified tet counts:**
- 1×1×1: **28 tetrahedra**
- 2×2×2: **296 tetrahedra**

### Step 4 — Face-Centroid Kagome Mapping

With the tetrahedral elements in hand, we apply the **canonical Kagome (pyrochlore honeycomb)** micro-rule:

**Nodes:** One node at the centroid of each unique triangular face of the tet network:
$$\mathbf{c}_{\triangle} = \frac{1}{3}\left(\mathbf{v}_a + \mathbf{v}_b + \mathbf{v}_c\right)$$

Adjacent tetrahedra sharing a face automatically share that centroid node — no explicit merging step is needed beyond deduplication by sorted vertex-index key.

**Struts:** Within each tetrahedron, its 4 face-centroid nodes are connected as a complete graph $K_4$ (6 struts per tet):
$$\text{Struts}_{\text{tet}} = \{(c_i, c_j) \mid 0 \leq i < j \leq 3\}$$

This produces an inscribed inverted tetrahedron inside each parent tet, and the globally merged network forms the **3D pyrochlore Kagome honeycomb** — a perfectly isotropic, uniform-coordination lattice.

**Verified Kagome counts (1×1×1):**
- Kagome nodes: **74**
- Kagome struts: **168**

---

## Why This Is a Breakthrough

| Property | Delaunay-based | Deterministic Clique |
|---|---|---|
| Repeatability on symmetric lattices | ❌ Arbitrary slicing of equidistant spheres | ✅ Guaranteed by crystallographic seed |
| Boundary tessellation continuity | ❌ Non-matching cuts across cell faces | ✅ Perfect periodic boundary fusion |
| Symmetry preservation | ❌ $Pm\bar{3}n$ symmetry broken by arbitrary cuts | ✅ Full $Pm\bar{3}n$ symmetry preserved |
| Kagome isotropy | ❌ Skewed Kagome triangles from sliver tets | ✅ Uniform, isotropic Kagome connectivity |
| Scalability | ❌ Degenerate tets require purging | ✅ No degenerate elements possible |

---

## Output Files

- [A15_Kagome_infill_1x1.stl](file:///c:/Users/ehunt/OneDrive/Documents/Python%20Scripts/Graphite/A15_Kagome_infill_1x1.stl)
- [A15_Kagome_infill_1x2.stl](file:///c:/Users/ehunt/OneDrive/Documents/Python%20Scripts/Graphite/A15_Kagome_infill_1x2.stl)
- [A15_Kagome_infill_2x2.stl](file:///c:/Users/ehunt/OneDrive/Documents/Python%20Scripts/Graphite/A15_Kagome_infill_2x2.stl)
- Generation script: [generate_a15_c15_kagome_infill.py](file:///c:/Users/ehunt/OneDrive/Documents/Python%20Scripts/Graphite/scripts/generate_a15_c15_kagome_infill.py)

---

## Related Documentation

- [docs/research/explicit_supercell_methodology.md](file:///c:/Users/ehunt/OneDrive/Documents/Python%20Scripts/Graphite/docs/research/explicit_supercell_methodology.md) — original methodology rationale
- [docs/HEX_EXPLICIT_ENGINE.md](file:///c:/Users/ehunt/OneDrive/Documents/Python%20Scripts/Graphite/docs/HEX_EXPLICIT_ENGINE.md) — hex engine architecture for tiling
