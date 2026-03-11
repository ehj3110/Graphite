# Lattice Architecture

## Overview

This project generates structural lattice geometries by combining two independent axes:

1. **Topology** — the spatial arrangement of seed points that drives the Delaunay tetrahedralization.
2. **Micro-Rule** — the rule that converts each individual tetrahedron into a lattice segment (nodes + struts).

Any topology can be combined with any micro-rule through the `LatticeFactory` interface in `Supercell_Modules/core/factory.py`.

---

## Pipeline

```
Seed Generator  →  Delaunay Triangulation  →  Degeneracy Filter  →  Micro-Rule Application  →  Global Node Merge  →  STL Export
```

1. **Seed Generator** (`core/topologies.py`): Produces a 3D point cloud representing the lattice sites for a given topology.
2. **Delaunay Triangulation** (`scipy.spatial.Delaunay`): Tetrahedralizes the point cloud into a set of non-overlapping tetrahedra.
3. **Degeneracy Filter**: Removes tetrahedra with volume < `1e-8` (coplanar or collinear vertices).
4. **Micro-Rule Application** (`core/lattice_rules.py`): Transforms each clean tetrahedron's 4 vertices into a local set of nodes and struts.
5. **Global Node Merge**: Deduplicates spatially coincident nodes using rounded-coordinate hashing, ensuring the final mesh is manifold and connected.
6. **STL Export**: Renders nodes as spheres and struts as cylinders using `manifold3d`, then exports to STL.

---

## Topologies (`core/topologies.py`)

| Name | Function | Description |
|---|---|---|
| Simple Cubic | `generate_simple_cubic_seeds` | Corner points of a regular cubic grid. |
| BCC | `generate_bcc_seeds` | Cubic corners + body-center points. Most uniform Delaunay tets. |
| A15 | `generate_a15_seeds` | BCC + face-offset points. Models the Pm-3n intermetallic structure. |
| Bitruncated Cubic | `generate_bitruncated_cubic_seeds` | Points at signed permutations of (0, 1, 2). |
| Truncated Oct-Tet | `generate_truncated_oct_tet_seeds` | Hybrid coordinate table: corners + body-center + quarter-offsets. |
| Rhombicuboctahedron | `generate_rhombicuboct_seeds` | Points at signed permutations of (1, 1, 2) + center. |

---

## Micro-Rules (`core/lattice_rules.py`)

Each micro-rule is a pure function that accepts the 4 vertex coordinates of a single tetrahedron and returns `(nodes, struts)` for that element. The factory applies the rule to every tetrahedron in the mesh and merges the results globally.

### Voronoi

```
Nodes:   1 centroid + 4 face-centers  (5 nodes per tet)
Struts:  4 struts connecting centroid → each face-center
```

Each tetrahedron produces a star-shaped local cluster. The centroid acts as a hub node; the four face-centers are the spoke endpoints. Because face-centers are shared between adjacent tetrahedra, the global result is a fully connected Voronoi foam network.

### Kagome

```
Nodes:   4 face-centers  (no centroid)
Struts:  6 struts — every face-center connected to every other face-center
         (forms an inverted tetrahedron inscribed in the original)
```

The struts trace the edges of a smaller tetrahedron whose vertices sit at the face-centers of the parent. Across a BCC Delaunay mesh this produces the classic Kagome / pyrochlore honeycomb.

### Icosahedral

```
Nodes:   6 edge-midpoints
Struts:  12 struts — 3 per original face (connecting the 3 edge-midpoints
         that bound each face)
```

Each face's three edge-midpoints are connected into a triangle. The four triangles together approximate an icosahedral local symmetry. This rule generates high-connectivity, near-isotropic lattice segments.

### Rhombic

```
Nodes:   1 centroid + 4 vertices  (5 nodes per tet)
Struts:  4 struts connecting centroid → each vertex
```

Similar in connectivity to Voronoi but anchored to the original seed vertices rather than face-centers. The centroid-to-vertex struts produce a rhombic local geometry. Across a periodic BCC mesh this generates a lattice closely related to the diamond cubic.

---

## Centroid Dual (`core/topologies.py → generate_centroid_dual`)

The centroid dual is an alternative to the micro-rule approach. Instead of decorating each tetrahedron with internal nodes, it:

1. Computes the **centroid of every tetrahedron** as a new dual node.
2. Connects dual nodes whose parent tetrahedra **share a face**.

This is equivalent to the Kagome micro-rule at the global mesh level and produces the true pyrochlore / Kagome honeycomb when applied to a BCC Delaunay mesh.

---

## File Structure

```
Graphite/
├── LATTICE_ARCHITECTURE.md          ← this file
├── Supercell_Modules/
│   ├── core/
│   │   ├── topologies.py            ← seed generators + centroid dual
│   │   ├── lattice_rules.py         ← 4 micro-rule functions
│   │   └── factory.py               ← LatticeFactory class
│   ├── tests/
│   │   ├── test_supercell.py        ← primary integration test
│   │   └── experimental_archive/   ← retired / exploratory scripts
│   └── output/                      ← generated STL files
```
