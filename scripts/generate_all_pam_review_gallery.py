#!/usr/bin/env python
"""
Generate Review Gallery for All 11 PAM Architectures (Zhou et al., Science 2025).

Outputs for each PAM architecture:
- Solid STL under outputs/pam_review_<cell>.stl
- High-resolution Studio PNG render under outputs/pam_review_<cell>.png
- Copies PNGs into artifact directory for embedding in review report.
"""
from __future__ import annotations

from pathlib import Path
import shutil
import sys
import numpy as np
import pyvista as pv
import trimesh
from scipy.sparse import csgraph, csr_matrix

_REPO_ROOT = Path(r"c:\Users\ehunt\OneDrive\Documents\Python Scripts\Graphite")
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from graphite.explicit.interlinked import (
    InterlinkedRegistry,
    InterlinkedConfig,
    generate_interlinked_lattice,
    InterlinkedParticle,
)
from graphite.explicit.interlinked.generator import (
    _particles_to_combined_mesh,
    solidify_particle_prototype,
)
from graphite.explicit.interlinked.clearance import check_particle_clearance
from graphite.explicit.interlinked.catenation import verify_particle_linking
from graphite.explicit.interlinked.epitaxy import (
    curve_curve_linking_number,
    extract_triangulated_faces,
)

ARTIFACT_DIR = Path(r"C:\Users\ehunt\.gemini\antigravity\brain\c3d5d97b-2716-45e7-97c0-6173eedf89c0")
OUTPUTS_DIR = _REPO_ROOT / "outputs"


def segment_triangle_intersection(p0, p1, v0, v1, v2):
    e1 = v1 - v0
    e2 = v2 - v0
    d = p1 - p0
    h = np.cross(d, e2)
    a = float(np.dot(e1, h))
    if abs(a) < 1e-9:
        return False
    f = 1.0 / a
    s = p0 - v0
    u = f * float(np.dot(s, h))
    if u < 1e-5 or u > 1.0 - 1e-5:
        return False
    q = np.cross(s, e1)
    v = f * float(np.dot(d, q))
    if v < 1e-5 or u + v > 1.0 - 1e-5:
        return False
    t = f * float(np.dot(e2, q))
    return 1e-5 <= t <= 1.0 - 1e-5


def are_particles_linked(pA: InterlinkedParticle, pB: InterlinkedParticle) -> bool:
    nA, sA = pA.global_nodes(), pA.struts
    nB, sB = pB.global_nodes(), pB.struts

    # 1. Closed loop vs closed loop (rings or hexagons)
    if len(sA) == len(nA) and len(sB) == len(nB):
        p0A, p1A = nA, np.roll(nA, -1, axis=0)
        p0B, p1B = nB, np.roll(nB, -1, axis=0)
        lk = curve_curve_linking_number(p0A, p1A, p0B, p1B)
        return abs(lk) >= 0.7

    # 2. Wireframe cages: triangulate faces
    fA = extract_triangulated_faces(nA, sA)
    # Check 6-cycles if Truncated Tetrahedron
    if len(nA) == 12 and len(sA) == 18:
        adj = [set() for _ in range(len(nA))]
        for u, v in sA:
            adj[u].add(v)
            adj[v].add(u)
        hexagons = []
        def dfs(v, start, path):
            if len(path) == 6:
                if start in adj[v]:
                    hexagons.append(tuple(path))
                return
            for nxt in adj[v]:
                if nxt not in path and nxt > path[0]:
                    dfs(nxt, start, path + [nxt])
        for i in range(len(nA)):
            dfs(i, i, [i])
        unique_hex = []
        for h in hexagons:
            if not any(set(uh) == set(h) for uh in unique_hex):
                unique_hex.append(h)
        for h in unique_hex:
            for k in range(1, 5):
                tri = [h[0], h[k], h[k+1]]
                fA = np.vstack([fA, tri]) if len(fA) > 0 else np.array([tri])

    for s in sB:
        p0, p1 = nB[s[0]], nB[s[1]]
        for f in fA:
            if segment_triangle_intersection(p0, p1, nA[f[0]], nA[f[1]], nA[f[2]]):
                return True

    fB = extract_triangulated_faces(nB, sB)
    for s in sA:
        p0, p1 = nA[s[0]], nA[s[1]]
        for f in fB:
            if segment_triangle_intersection(p0, p1, nB[f[0]], nB[f[1]], nB[f[2]]):
                return True

    return False


def render_pam_specimen(
    particles: list[InterlinkedParticle],
    out_png: Path,
    title: str,
    subtitle: str,
    sublattice_colors: dict[str, str],
    camera_pos: list | None = None,
    window_size: tuple[int, int] = (1800, 1200),
) -> None:
    """Render particles with sublattice color differentiation and studio lighting."""
    pv.set_plot_theme("document")
    plotter = pv.Plotter(off_screen=True, window_size=window_size)
    plotter.set_background("#0a0e14", top="#141a24")

    proto_mesh_cache: dict[str, trimesh.Trimesh] = {}

    for idx, p in enumerate(particles):
        r_wire = p.effective_wire_radius(fallback=0.35)
        k = f"{p.geometry.geometry_type}_{len(p.geometry.nodes)}_{len(p.geometry.struts)}_{p.sublattice_id}_r{r_wire:.4f}"
        if k not in proto_mesh_cache:
            proto_mesh_cache[k] = solidify_particle_prototype(
                p.geometry,
                strut_radius=r_wire,
                circular_segments=16,
            )
        base_m = proto_mesh_cache[k]
        V_homo = np.hstack([base_m.vertices, np.ones((len(base_m.vertices), 1), dtype=np.float64)])
        V_w = (p.transform @ V_homo.T).T[:, :3]
        mesh_w = trimesh.Trimesh(vertices=V_w, faces=base_m.faces, process=False)
        pv_m = pv.wrap(mesh_w)

        # Color by sublattice or cluster role
        sub_key = str(p.sublattice_id)
        color = sublattice_colors.get(sub_key, sublattice_colors.get("default", "#3498db"))
        plotter.add_mesh(
            pv_m,
            color=color,
            specular=0.55,
            specular_power=30,
            roughness=0.25,
            smooth_shading=True,
        )

    plotter.enable_anti_aliasing("ssaa")
    light1 = pv.Light(position=(60, -60, 70), focal_point=(10, 10, 10), intensity=0.9)
    light2 = pv.Light(position=(-40, 80, 50), focal_point=(10, 10, 10), intensity=0.45)
    plotter.add_light(light1)
    plotter.add_light(light2)

    if camera_pos is not None:
        plotter.camera_position = camera_pos
    else:
        plotter.view_isometric()
        plotter.reset_camera()
        plotter.camera.azimuth += 15.0
        plotter.camera.elevation += 12.0
        plotter.reset_camera()

    header_text = f"{title}\n{subtitle}" if subtitle else title
    plotter.add_text(
        header_text,
        position="upper_left",
        font_size=12,
        color="#ffffff",
        font="arial",
    )

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plotter.screenshot(str(out_png))
    plotter.close()
    if ARTIFACT_DIR.exists():
        shutil.copy2(out_png, ARTIFACT_DIR / out_png.name)
    print(f"  [OK] Render saved to {out_png}")


PAM_SPECS = [
    {
        "id": "c6tt",
        "name": "C-6-TT",
        "network": "pcu",
        "seeding": "cartesian",
        "grid": (2, 2, 2),
        "pitch": 12.0,
        "wire_r": 0.35,
        "sublattice_colors": {"A": "#00d2d3", "default": "#00d2d3"},
        "desc": "Simple Cubic Truncated Tetrahedron (12v/18e)",
    },
    {
        "id": "d4tet",
        "name": "D-4-TET",
        "network": "dia",
        "seeding": "diamond",
        "grid": (2, 2, 2),
        "pitch": 16.0,
        "wire_r": 0.35,
        "sublattice_colors": {"A": "#ff6b6b", "B": "#4ecdc4", "default": "#ff6b6b"},
        "desc": "Diamond Regular Tetrahedron (4v/6e, Bipartite Inversion)",
    },
    {
        "id": "b8co",
        "name": "B-8-CO",
        "network": "bcc",
        "seeding": "bcc",          # BCC corner+body-center sublattices; 60° [111] twist
        "grid": (2, 2, 2),
        "pitch": 16.0,
        "wire_r": 0.35,
        "sublattice_colors": {"A": "#9b59b6", "B": "#f39c12", "default": "#9b59b6"},
        "desc": "Body-Centered Cubic Cuboctahedron (12v/24e, geometry-fixed)",
    },
    {
        "id": "c6co",
        "name": "C-6-CO",
        "network": "pcu",
        "seeding": "cartesian",
        "grid": (2, 2, 2),
        "pitch": 14.0,
        "wire_r": 0.35,
        "sublattice_colors": {"A": "#e84393", "B": "#fd79a8", "default": "#e84393"},
        "desc": "Simple Cubic Cuboctahedron [EXPERIMENTAL — geometrically frustrated on pcu]",
    },
    {
        "id": "d4cube",
        "name": "D-4-CUBE",
        "network": "dia",
        "seeding": "diamond",
        "grid": (2, 2, 2),
        "pitch": 16.0,
        "wire_r": 0.35,
        "sublattice_colors": {"A": "#e67e22", "B": "#f1c40f", "default": "#e67e22"},
        "desc": "Diamond Cube Wireframe (8v/12e)",
    },
    {
        "id": "d4ct",
        "name": "D-4-CT",
        "network": "dia",
        "seeding": "diamond",
        "grid": (2, 2, 2),
        "pitch": 16.0,
        "wire_r": 0.30,
        "sublattice_colors": {"A": "#3498db", "B": "#2ecc71", "default": "#3498db"},
        "desc": "Diamond Triakis / Catalan Tetrahedron (8v/18e)",
    },
    {
        "id": "d4ico",
        "name": "D-4-ICO",
        "network": "dia",
        "seeding": "diamond",
        "grid": (2, 2, 2),
        "pitch": 18.0,
        "wire_r": 0.30,
        "sublattice_colors": {"A": "#fd79a8", "B": "#a29bfe", "default": "#fd79a8"},
        "desc": "Diamond Regular Icosahedron Wireframe (12v/30e)",
    },
    {
        "id": "j4oct",
        "name": "J-4-OCT",
        "network": "square_2d",
        "seeding": "cartesian",
        "grid": (1, 1, 1),   # Canonical 1+4 planar cross motif (5 catenated octahedra)
        "pitch": 20.0,
        "wire_r": 0.35,
        "sublattice_colors": {
            "center": "#2ecc71", "arm_pos_x": "#27ae60", "arm_neg_x": "#1abc9c",
            "arm_pos_y": "#3498db", "arm_neg_y": "#2980b9", "default": "#2ecc71"
        },
        "desc": "Square-Planar Octahedron Cross (1 center + 4 orthogonal arms, 45-deg twist)",
    },
    {
        "id": "t6hex",
        "name": "T-6-HEX",
        "network": "pcu",
        "seeding": "cartesian",
        "grid": (2, 2, 2),
        "pitch": 16.0,
        "wire_r": 0.35,
        "sublattice_colors": {
            "hex_0": "#a8e6cf", "hex_1": "#dcedc1", "hex_2": "#ffd3b6", "hex_3": "#ffaaa5", "default": "#a8e6cf"
        },
        "desc": "Simple Cubic 4-Hexagon Cluster (6v/6e each)",
    },
    {
        "id": "t6ring",
        "name": "T-6-RING",
        "network": "pcu",
        "seeding": "cartesian",
        "grid": (2, 2, 2),
        "pitch": 18.0,
        "wire_r": 0.35,
        "sublattice_colors": {
            "loop_0": "#f1c40f", "loop_1": "#f39c12", "loop_2": "#e67e22", "loop_3": "#d35400", "default": "#f1c40f"
        },
        "desc": "Simple Cubic Tetrahedral 4-Ring Cluster (Tori)",
    },
    {
        "id": "j4ring",
        "name": "J-4-RING",
        "network": "dia",
        "seeding": "diamond",
        "grid": (1, 1, 1),  # 1 conventional diamond cube = 8 sites * 6 rings = 48 rings
        "pitch": 20.0,
        "wire_r": 0.35,
        "sublattice_colors": {
            "pos_x": "#1abc9c", "neg_x": "#16a085",
            "pos_y": "#2ecc71", "neg_y": "#27ae60",
            "pos_z": "#3498db", "neg_z": "#2980b9",
            "default": "#1abc9c"
        },
        "desc": "Diamond Octahedral 6-Ring Cluster (48 rings in conventional cube)",
    },
]


def main():
    print("=" * 80)
    print("GENERATING EXHAUSTIVE PAM REVIEW SPECIMENS ACROSS ALL 11 ARCHITECTURES")
    print("=" * 80)

    summary_records = []

    for spec in PAM_SPECS:
        c_id = spec["id"]
        c_name = spec["name"]
        print(f"\n--- Processing [{c_name}] ---")

        cfg = InterlinkedConfig(
            cell=c_id,
            seeding_type=spec["seeding"],
            grid_size=spec["grid"],
            pitch=spec["pitch"],
            wire_radius=spec["wire_r"],
        )
        res = generate_interlinked_lattice(cfg, check_clearance=False)
        pts = res.particles
        N = len(pts)

        # Export STL
        stl_path = OUTPUTS_DIR / f"pam_review_{c_id}.stl"
        res.export_stl(stl_path)
        print(f"  [OK] Exported STL to {stl_path} ({len(res.mesh.vertices)} vertices)")

        # Evaluate Physical Clearance
        valid_clr, min_clr, _ = check_particle_clearance(
            pts, strut_radius=spec["wire_r"], min_clearance=0.05
        )

        # Topological Adjacency & Component Analysis via Universal Link Checker
        adj = np.zeros((N, N), dtype=int)
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(pts[i].center - pts[j].center)
                if d > pts[i].bounding_radius + pts[j].bounding_radius + 2.0 * spec["wire_r"]:
                    continue
                res_link = verify_particle_linking(pts[i], pts[j], strut_radius=spec["wire_r"])
                if res_link.is_linked:
                    adj[i, j] = 1
                    adj[j, i] = 1

        n_comp, labels = csgraph.connected_components(csr_matrix(adj), directed=False)
        deg = adj.sum(axis=1)
        n_linked = int(np.count_nonzero(deg > 0))
        n_links = int(adj.sum() // 2)

        # Evaluate bounding radius vs neighbor spacing
        r_bound = pts[0].bounding_radius
        centers = np.array([p.center for p in pts])
        min_neighbor_dist = float("inf")
        for i in range(N):
            dists = [np.linalg.norm(centers[i] - centers[j]) for j in range(N) if i != j]
            if dists:
                min_neighbor_dist = min(min_neighbor_dist, min(dists))

        overlap_ratio = (2.0 * r_bound) / max(1e-6, min_neighbor_dist)

        verdict = (
            "FULLY INTERLINKED"
            if n_comp == 1 and n_linked == N
            else ("PARTIAL LINK (Boundary Open)" if n_linked > 0 else "DISCONNECTED (Zero Links)")
        )

        # Render PNG
        png_path = OUTPUTS_DIR / f"pam_review_{c_id}.png"
        render_pam_specimen(
            particles=pts,
            out_png=png_path,
            title=f"{c_name}: {spec['desc']}",
            subtitle=(
                f"Net: {spec['network']} | Seeding: {spec['seeding']} | Grid: {spec['grid']} | "
                f"Particles: {N} | Links: {n_links} ({n_linked}/{N} linked) | Verdict: {verdict}"
            ),
            sublattice_colors=spec["sublattice_colors"],
        )

        record = {
            "id": c_id,
            "name": c_name,
            "desc": spec["desc"],
            "network": spec["network"],
            "seeding": spec["seeding"],
            "grid": str(spec["grid"]),
            "pitch": spec["pitch"],
            "particles": N,
            "links": n_links,
            "linked_ratio": f"{n_linked}/{N}",
            "components": n_comp,
            "r_bound": r_bound,
            "d_neighbor": min_neighbor_dist,
            "overlap_ratio": overlap_ratio,
            "min_clearance": min_clr,
            "verdict": verdict,
            "stl_file": f"outputs/{stl_path.name}",
            "png_file": f"outputs/{png_path.name}",
        }
        summary_records.append(record)

    print("\n" + "=" * 80)
    print("ALL 11 PAM SPECIMENS GENERATED & AUDITED SUCCESSFULLY")
    print("=" * 80)

    # Print summary table
    print(
        f"\n{'Name':<10} | {'Net':<6} | {'Seeding':<10} | {'Parts':<5} | {'Links':<5} | "
        f"{'Linked':<8} | {'2r/d':<6} | {'Clr(um)':<8} | {'Verdict':<25}"
    )
    print("-" * 95)
    for r in summary_records:
        clr_str = f"{r['min_clearance']*1000:+.1f}" if not np.isinf(r['min_clearance']) else "N/A"
        print(
            f"{r['name']:<10} | {r['network']:<6} | {r['seeding']:<10} | {r['particles']:<5d} | "
            f"{r['links']:<5d} | {r['linked_ratio']:<8} | {r['overlap_ratio']:<6.2f} | "
            f"{clr_str:<8} | {r['verdict']:<25}"
        )

    # Save summary as json for artifact builder
    import json
    json_path = ARTIFACT_DIR / "pam_all_architectures_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_records, f, indent=2)
    print(f"\n[OK] Summary metrics written to {json_path}")


if __name__ == "__main__":
    main()
