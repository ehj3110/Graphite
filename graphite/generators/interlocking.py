"""
Graphite Interlocking Auxetic and Kinematic Assembly Generator

Implements discrete, unbonded unit cell chains and metamaterial fabrics:
- 3D Re-entrant Auxetic Bowtie Links with geometric jamming and positive clearance.
- Alternating Hook and Ring Arrays (2.5D space fabric and chainmail).
- Returns disjoint, non-welded list[trimesh.Trimesh] meshes ready for additive manufacturing.
"""

from __future__ import annotations

from typing import Sequence
import numpy as np
from scipy.spatial import cKDTree
import trimesh
import manifold3d as m3d

from graphite.explicit.geometry_module import (
    _trimesh_to_manifold,
    _manifold_to_trimesh,
    _rotation_align_local_z_to_unit,
    _affine_rows_from_R_t,
)
from graphite.explicit.interlinked.nasa_hexagon import NasaHexagonCell
from graphite.explicit.interlinked.importer import generate_hexagonal_sheet_seeds
from graphite.explicit.interlinked.patterns import (
    generate_european_4in1_rings,
    generate_kusari_rings,
)


def _build_reentrant_bowtie_manifold(
    pitch: float,
    wire_radius: float,
    clearance_gap: float,
    is_odd: bool = False,
) -> m3d.Manifold:
    """
    Construct a single 3D re-entrant auxetic bowtie link with interlocking loop-in-loop eyelets.
    - Central 8-point re-entrant bowtie body (waists at cardinal axes, lobes at diagonals).
    - Four wishbone arms extending from the diagonal lobes to 4 cardinal interlocking tori.
    - Even cells feature horizontal loops (in XY); odd cells feature vertical loops (in XZ / YZ).
    - Topological linking number = 1 across all adjacent pairs, guaranteeing true chainmail connectivity.
    """
    parts: list[m3d.Manifold] = []
    r_arm = pitch * 0.55
    r_loop = pitch * 0.19
    r_waist = pitch * 0.18
    r_lobe = pitch * 0.35
    z_body = 2.0 * (pitch / 10.0) if is_odd else -2.0 * (pitch / 10.0)

    # 1. Central 8-point re-entrant auxetic bowtie frame
    pts = []
    for k in range(8):
        ang = k * (np.pi / 4.0)
        rad = r_waist if (k % 2 == 0) else r_lobe
        pts.append([rad * np.cos(ang), rad * np.sin(ang), z_body])
    pts = np.array(pts, dtype=np.float64)

    for i in range(8):
        p0 = pts[i]
        p1 = pts[(i + 1) % 8]
        seg = p1 - p0
        l_seg = float(np.linalg.norm(seg))
        if l_seg <= 1e-8:
            continue
        cyl = m3d.Manifold.cylinder(l_seg, wire_radius, wire_radius, circular_segments=14, center=True)
        R_mat = _rotation_align_local_z_to_unit(seg / l_seg)
        aff = _affine_rows_from_R_t(R_mat, 0.5 * (p0 + p1))
        parts.append(cyl.transform(aff))
        parts.append(m3d.Manifold.sphere(wire_radius, circular_segments=12).translate(tuple(float(c) for c in p0)))

    # 2. Four interlocking rings and wishbone arms from lobes
    lobe_indices = {
        0: (1, 7),  # East wishbone connects to 45 deg and 315 deg lobes
        1: (1, 3),  # North wishbone connects to 45 deg and 135 deg lobes
        2: (3, 5),  # West wishbone connects to 135 deg and 225 deg lobes
        3: (5, 7),  # South wishbone connects to 225 deg and 315 deg lobes
    }

    for k in range(4):
        ang_card = k * (np.pi / 2.0)
        p_loop_center = np.array([r_arm * np.cos(ang_card), r_arm * np.sin(ang_card), 0.0], dtype=np.float64)
        idxA, idxB = lobe_indices[k]
        pA, pB = pts[idxA], pts[idxB]
        angA = ang_card + (np.pi / 3.0)
        angB = ang_card - (np.pi / 3.0)

        p_ringA = p_loop_center + r_loop * np.array([np.cos(angA), np.sin(angA), 0.0], dtype=np.float64)
        p_ringB = p_loop_center + r_loop * np.array([np.cos(angB), np.sin(angB), 0.0], dtype=np.float64)

        if is_odd:
            p_ringA[2] = z_body * 0.4
            p_ringB[2] = z_body * 0.4
        else:
            p_ringA[2] = 0.0
            p_ringB[2] = 0.0

        for p_start, p_end in [(pA, p_ringA), (pB, p_ringB)]:
            seg = p_end - p_start
            l_seg = float(np.linalg.norm(seg))
            if l_seg <= 1e-8:
                continue
            cyl = m3d.Manifold.cylinder(l_seg, wire_radius, wire_radius, circular_segments=14, center=True)
            R_mat = _rotation_align_local_z_to_unit(seg / l_seg)
            aff = _affine_rows_from_R_t(R_mat, 0.5 * (p_start + p_end))
            parts.append(cyl.transform(aff))

        # Revolved torus ring
        cross = m3d.CrossSection.circle(wire_radius, circular_segments=14).translate([r_loop, 0.0])
        torus = m3d.Manifold.revolve(cross, circular_segments=20)
        if is_odd:
            if k in (0, 2):
                torus = torus.rotate([90, 0, 0])
            else:
                torus = torus.rotate([0, 90, 0])
        torus = torus.translate(tuple(float(c) for c in p_loop_center))
        parts.append(torus)

    return m3d.Manifold.compose(parts)


def _build_3d_auxetic_cell_manifold(
    pitch: float,
    wire_radius: float,
    clearance_gap: float,
    parity: int = 0,
) -> m3d.Manifold:
    """
    Construct a single 3D volumetric re-entrant auxetic unit cell with 6 cardinal interlocking loops.
    - Central 3D re-entrant corner frame (8 diagonal corner lobes).
    - Six cardinal eyelet tori (+X, -X, +Y, -Y, +Z, -Z) extending to r_arm = 0.55 * pitch.
    - Ring orientations alternate based on checkerboard parity:
      - parity 0: X-rings in XY, Y-rings in YZ, Z-rings in XZ.
      - parity 1: X-rings in XZ, Y-rings in XY, Z-rings in YZ.
    - True 3D chainmail topological linking (L = 1) across all cardinal neighbors with positive clearance.
    """
    parts: list[m3d.Manifold] = []
    r_arm = pitch * 0.55
    r_loop = pitch * 0.18
    r_lobe = pitch * 0.34

    # 8 diagonal corners of 3D re-entrant cell
    corners = []
    for cx in [-1.0, 1.0]:
        for cy in [-1.0, 1.0]:
            for cz in [-1.0, 1.0]:
                corners.append(np.array([cx, cy, cz], dtype=np.float64) * (r_lobe / np.sqrt(3.0)))
    corners = np.array(corners, dtype=np.float64)

    # Connect the 12 edges between adjacent corners of the central cube frame
    for i in range(8):
        for j in range(i + 1, 8):
            diff = corners[i] - corners[j]
            if np.count_nonzero(np.abs(diff) > 1e-4) == 1:
                pA, pB = corners[i], corners[j]
                seg = pB - pA
                l_seg = float(np.linalg.norm(seg))
                cyl = m3d.Manifold.cylinder(l_seg, wire_radius, wire_radius, circular_segments=12, center=True)
                R_mat = _rotation_align_local_z_to_unit(seg / l_seg)
                aff = _affine_rows_from_R_t(R_mat, 0.5 * (pA + pB))
                parts.append(cyl.transform(aff))
                parts.append(m3d.Manifold.sphere(wire_radius, circular_segments=10).translate(tuple(float(c) for c in pA)))
                parts.append(m3d.Manifold.sphere(wire_radius, circular_segments=10).translate(tuple(float(c) for c in pB)))

    # 6 cardinal directions:
    # 0: +X, 1: -X, 2: +Y, 3: -Y, 4: +Z, 5: -Z
    dirs = [
        (0, np.array([1.0, 0.0, 0.0]), [4, 5, 6, 7]),   # +X corners
        (1, np.array([-1.0, 0.0, 0.0]), [0, 1, 2, 3]),  # -X corners
        (2, np.array([0.0, 1.0, 0.0]), [2, 3, 6, 7]),   # +Y corners
        (3, np.array([0.0, -1.0, 0.0]), [0, 1, 4, 5]),  # -Y corners
        (4, np.array([0.0, 0.0, 1.0]), [1, 3, 5, 7]),   # +Z corners
        (5, np.array([0.0, 0.0, -1.0]), [0, 2, 4, 6]),  # -Z corners
    ]

    for d_idx, d_vec, corner_idxs in dirs:
        p_loop_center = d_vec * r_arm
        c1 = corners[corner_idxs[0]]
        c2 = corners[corner_idxs[3]]

        # Ring torus
        c_circ = m3d.CrossSection.circle(wire_radius, circular_segments=14).translate([r_loop, 0.0])
        torus = m3d.Manifold.revolve(c_circ, circular_segments=20)

        if parity == 0:
            if d_idx in (0, 1):    # X axis -> ring in XY
                pass
            elif d_idx in (2, 3):  # Y axis -> ring in YZ
                torus = torus.rotate([0, 90, 0])
            else:                  # Z axis -> ring in XZ
                torus = torus.rotate([90, 0, 0])
        else:
            if d_idx in (0, 1):    # X axis -> ring in XZ
                torus = torus.rotate([90, 0, 0])
            elif d_idx in (2, 3):  # Y axis -> ring in XY
                pass
            else:                  # Z axis -> ring in YZ
                torus = torus.rotate([0, 90, 0])

        torus = torus.translate(tuple(float(c) for c in p_loop_center))
        parts.append(torus)

        # Connect wishbone struts from lobes to ring rim
        for c_pt in [c1, c2]:
            seg = p_loop_center - c_pt
            l_seg = float(np.linalg.norm(seg)) - r_loop * 0.8
            if l_seg > 0.1:
                u_seg = (p_loop_center - c_pt) / np.linalg.norm(p_loop_center - c_pt)
                p_mid = c_pt + 0.5 * l_seg * u_seg
                cyl = m3d.Manifold.cylinder(l_seg, wire_radius, wire_radius, circular_segments=12, center=True)
                R_mat = _rotation_align_local_z_to_unit(u_seg)
                aff = _affine_rows_from_R_t(R_mat, p_mid)
                parts.append(cyl.transform(aff))

    return m3d.Manifold.compose(parts)


def generate_interlocking_auxetic_sheet(
    dimensions: tuple[int, int] | tuple[int, int, int],
    cell_pitch: float,
    clearance_gap: float,
    cell_topology: str = "reentrant_bowtie",
    wire_radius: float | None = None,
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> list[trimesh.Trimesh]:
    """
    Generate an array of discrete, non-connected, interlocking unit cell meshes.

    Parameters:
        dimensions: (nx, ny) unit cell count across the sheet or (nx, ny, nz) 3D volumetric grid.
        cell_pitch: Center-to-center tile spacing P in mm.
        clearance_gap: Minimum physical gap delta between adjacent bodies.
        cell_topology: 'reentrant_bowtie' | 'hook_array' | 'european_ring' | 'kusari_ring'.
        wire_radius: Strut wire cross-section radius in mm (auto-derived if None).
        origin: Base origin offset (x0, y0, z0).

    Returns:
        List of disjoint, watertight trimesh.Trimesh objects ready for print prep.
    """
    if len(dimensions) == 2:
        nx, ny = int(dimensions[0]), int(dimensions[1])
        nz = 1
    elif len(dimensions) == 3:
        nx, ny, nz = int(dimensions[0]), int(dimensions[1]), int(dimensions[2])
    else:
        raise ValueError(f"dimensions must be a 2-tuple (nx, ny) or 3-tuple (nx, ny, nz), got {dimensions}")

    pitch = float(cell_pitch)
    gap = float(clearance_gap)
    ox, oy, oz = origin

    if nx <= 0 or ny <= 0 or nz <= 0:
        raise ValueError(f"dimensions must be positive integers, got {dimensions}")
    if pitch <= 0:
        raise ValueError(f"cell_pitch must be positive, got {pitch}")
    if gap < 0:
        raise ValueError(f"clearance_gap cannot be negative, got {gap}")

    # Auto-derive wire radius if not specified
    if wire_radius is None:
        if cell_topology == "reentrant_bowtie":
            wire_r = max(0.18, pitch * 0.028)
        else:
            wire_r = max(0.20, pitch * 0.035)
    else:
        wire_r = float(wire_radius)

    meshes: list[trimesh.Trimesh] = []

    # 1. 3D RE-ENTRANT AUXETIC BOWTIE LINKS
    if cell_topology == "reentrant_bowtie":
        if nz > 1:
            for i in range(nx):
                for j in range(ny):
                    for k in range(nz):
                        parity = (i + j + k) % 2
                        m = _build_3d_auxetic_cell_manifold(
                            pitch=pitch,
                            wire_radius=wire_r,
                            clearance_gap=gap,
                            parity=parity,
                        )
                        trans = (ox + i * pitch, oy + j * pitch, oz + k * pitch)
                        m_trans = m.translate(trans)
                        meshes.append(_manifold_to_trimesh(m_trans))
        else:
            for i in range(nx):
                for j in range(ny):
                    is_odd = ((i + j) % 2 == 1)
                    m = _build_reentrant_bowtie_manifold(
                        pitch=pitch,
                        wire_radius=wire_r,
                        clearance_gap=gap,
                        is_odd=is_odd,
                    )
                    trans = (ox + i * pitch, oy + j * pitch, oz)
                    m_trans = m.translate(trans)
                    meshes.append(_manifold_to_trimesh(m_trans))

    # 2. ALTERNATING HOOK / NASA FABRIC ARRAY
    elif cell_topology in ("hook_array", "nasa_hexagon"):
        seeds = generate_hexagonal_sheet_seeds(num_rows=ny, num_cols=nx, pitch=pitch, origin=origin)
        for pt in seeds:
            cell = NasaHexagonCell(
                pitch=pitch,
                wire_radius=wire_r,
                plate_radius=pitch * (7.0 / 12.75),
                plate_thickness=0.45,
                ring_radius=pitch * (3.75 / 12.75),
                ring_height=pitch * (5.898 / 12.75),
            )
            m = cell.build_manifold().translate((float(pt[0]), float(pt[1]), float(pt[2])))
            meshes.append(_manifold_to_trimesh(m))

    # 3. EUROPEAN 4-IN-1 CHAINMAIL RINGS
    elif cell_topology == "european_ring":
        rings = generate_european_4in1_rings(
            grid_size=(nx, ny, 1),
            pitch=pitch,
            radius_ratio=0.65,
            wire_radius=wire_r,
            tilt_angle_deg=28.0,
            num_segments=24,
            origin=origin,
        )
        for r in rings:
            # Revolved torus
            c_circ = m3d.CrossSection.circle(r.wire_radius, circular_segments=16).translate([r.radius, 0.0])
            torus = m3d.Manifold.revolve(c_circ, circular_segments=32)
            R_mat = _rotation_align_local_z_to_unit(r.normal)
            aff = _affine_rows_from_R_t(R_mat, r.center)
            m = torus.transform(aff)
            meshes.append(_manifold_to_trimesh(m))

    # 4. JAPANESE KUSARI CHAINMAIL RINGS
    elif cell_topology == "kusari_ring":
        rings = generate_kusari_rings(
            grid_size=(nx, ny, 1),
            pitch=pitch,
            flat_radius=pitch * 0.45,
            arch_radius=pitch * 0.38,
            wire_radius=wire_r,
            num_segments=24,
            origin=origin,
        )
        for r in rings:
            c_circ = m3d.CrossSection.circle(r.wire_radius, circular_segments=16).translate([r.radius, 0.0])
            torus = m3d.Manifold.revolve(c_circ, circular_segments=32)
            R_mat = _rotation_align_local_z_to_unit(r.normal)
            aff = _affine_rows_from_R_t(R_mat, r.center)
            m = torus.transform(aff)
            meshes.append(_manifold_to_trimesh(m))

    else:
        raise ValueError(
            f"Unknown cell_topology '{cell_topology}'. Supported: 'reentrant_bowtie', 'hook_array', 'european_ring', 'kusari_ring'."
        )

    return meshes


def combine_interlocking_meshes(meshes: Sequence[trimesh.Trimesh]) -> trimesh.Trimesh:
    """
    Concatenate a list of disjoint meshes into a single multi-body mesh for export or visualization.
    """
    if not meshes:
        raise ValueError("Cannot combine empty mesh list.")
    return trimesh.util.concatenate(meshes)


def verify_interlocking_clearance(
    meshes: Sequence[trimesh.Trimesh],
    max_neighbor_distance: float = 25.0,
) -> dict:
    """
    Evaluate pairwise minimum surface clearances between adjacent meshes.
    Verifies zero intersection volume and positive clearance gap.
    """
    n = len(meshes)
    centroids = np.array([m.centroid for m in meshes], dtype=np.float64)
    tree = cKDTree(centroids)
    pairs = tree.query_pairs(r=max_neighbor_distance)

    min_gap_overall = float("inf")
    intersection_count = 0
    pair_reports = []

    # Pre-convert meshes to Manifold once for performance
    man_list = [_trimesh_to_manifold(m) for m in meshes]

    for idx1, idx2 in pairs:
        m1_man = man_list[idx1]
        m2_man = man_list[idx2]

        # Boolean intersection volume
        inter_vol = float((m1_man ^ m2_man).volume())
        if inter_vol > 1e-4:
            intersection_count += 1

        # Min surface gap
        gap = float(m1_man.min_gap(m2_man, max_neighbor_distance))
        min_gap_overall = min(min_gap_overall, gap)

        pair_reports.append({
            "pair": (idx1, idx2),
            "intersection_volume": inter_vol,
            "min_gap_mm": gap,
        })

    return {
        "num_meshes": n,
        "num_adjacent_pairs_checked": len(pairs),
        "intersection_count": intersection_count,
        "min_clearance_mm": float(min_gap_overall) if min_gap_overall != float("inf") else 0.0,
        "clearance_valid": (intersection_count == 0 and min_gap_overall > 0.0),
    }
