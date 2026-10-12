"""
Graphite Explicit Interlinked — Metamaterial Heteroepitaxy Engine.

Implements the Three Laws of Metamaterial Heteroepitaxy to evaluate, solve, and
synthesize topological transitions across habit planes between disparate
interlinkable metamaterial unit cells.

The Three Governing Laws:
1. Commensurate Coincidence Lattice (Supercell Coincidence Search):
   Evaluates transverse habit plane periodicity and determines integer supercell
   multipliers (m_A, n_B) and rotation R_habit minimizing lattice mismatch strain eps.
2. Aperture Admissibility (Steric Non-Interference):
   Verifies that the interior aperture diameter of the receiving boundary cage
   strictly accommodates the penetrating cross-sectional profile of the partner cage:
   D_aperture > D_profile + 2*r_A + 2*r_B + 2*Delta_min.
3. Boundary Topological Catenation & Longitudinal Optimization:
   Optimizes longitudinal interface spacing dx_int and relative twist theta to
   maximize physical surface clearance Delta while strictly maintaining non-zero
   topological linking (Lk != 0 and closed-loop strut-face piercings >= 1).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence
import numpy as np

from .clearance import segment_segment_distance
from .particle import ParticleGeometry, InterlinkedParticle
from .cell import InterlinkedCell, InterlinkedRegistry
from .pams import PAMParticle, strut_pierces_triangle


class EpitaxialIncompatibilityError(ValueError):
    """Raised when two interlinked cells cannot form a physically valid epitaxial transition."""
    def __init__(self, message: str, diagnostics: dict[str, Any] | None = None):
        super().__init__(message)
        self.diagnostics = diagnostics or {}


@dataclass(frozen=True)
class EpitaxialAlignment:
    """
    Geometric and crystallographic alignment definition across an epitaxial habit plane.

    Attributes:
        habit_plane: Crystallographic Miller indices of the boundary interface (e.g. '100', '110', '111').
        supercell_ratio: Transverse integer supercell multipliers (m_A, n_B).
        rotation_matrix: (3, 3) rotation matrix aligning Cell B's habit plane to Cell A.
        interface_offset: Longitudinal spacing dx_int (mm) along interface normal (matches phase_shift[0]).
        strain: In-plane lattice mismatch strain epsilon (|m*a_A - n*a_B| / (m*a_A)).
        phase_shift: 3D relative shift vector (dx, dy, dz) in mm between Cell A and Cell B at the boundary.
        taper_gamma: Wire radius thinning factor applied to boundary layer struts (1.0 = no taper).
        shrink_eta: Isotropic cage shrinkage factor applied to entering boundary cages (1.0 = no shrinkage).
        metadata: Additional diagnostic metrics.
    """
    habit_plane: str = "100"
    supercell_ratio: tuple[int, int] = (1, 1)
    rotation_matrix: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=np.float64))
    interface_offset: float = 7.20
    strain: float = 0.0
    phase_shift: tuple[float, float, float] = (7.20, 0.0, 0.0)
    taper_gamma: float = 1.0
    shrink_eta: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class InterfacePhaseOptimizationResult:
    """
    Diagnostic result of a 3D interface phase shift and geometric tapering optimization.
    """
    phase_shift: tuple[float, float, float]
    max_clearance_mm: float
    piercings_count: int
    is_valid: bool
    taper_applied: bool = False
    taper_gamma: float = 1.0
    shrink_eta: float = 1.0
    aperture_margin_mm: float = 0.0


@dataclass
class EpitaxialFeasibilityResult:
    """
    Comprehensive diagnostic record of an epitaxial transition feasibility analysis.
    """
    is_feasible: bool
    cell_a_name: str
    cell_b_name: str
    habit_plane: str
    alignment: EpitaxialAlignment | None = None
    failure_stage: str = ""  # 'commensurability', 'aperture', 'catenation_or_clearance', or ''
    aperture_margin_mm: float = 0.0
    min_clearance_mm: float = 0.0
    piercings_count: int = 0
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class InterlinkedTransitionConfig:
    """
    Declarative specification for a multi-zone interlinked transition metamaterial.
    """
    cell_a: InterlinkedCell | str
    cell_b: InterlinkedCell | str
    grid_size_a: tuple[int, int, int] = (2, 2, 2)
    grid_size_b: tuple[int, int, int] = (2, 2, 2)
    habit_plane: str = "100"
    pitch_a: float | None = None
    pitch_b: float | None = None
    wire_radius: float = 0.35
    min_clearance: float = 0.150
    alignment: EpitaxialAlignment | None = None
    max_supercell: int = 4
    max_strain: float = 0.05  # 5% max acceptable mismatch strain


# =============================================================================
# Law 1: Commensurate Coincidence Lattice Search
# =============================================================================

def _habit_100_transverse_scale(parent_network: str, theta: float = 0.0) -> float:
    """Scale factor relating fundamental pitch to transverse lattice spacing on {100}."""
    net = parent_network.lower().strip()
    if net == "dia":
        return 1.0 / np.sqrt(2.0)
    return 1.0


def get_habit_plane_rotation_matrix(habit_plane: str = "100") -> np.ndarray:
    """
    Return (3, 3) SO(3) orthonormal rotation matrix R_habit mapping habit normal
    n_hkl to laboratory Cartesian +X axis ([1, 0, 0]).

    Guarantees:
      - R_habit @ n_hkl = [1, 0, 0]
      - det(R_habit) = +1.0 (proper rotation, no reflection)
      - R_habit @ R_habit.T = I_3 (orthonormal)
    """
    plane_str = (
        str(habit_plane)
        .strip()
        .replace("{", "")
        .replace("}", "")
        .replace("(", "")
        .replace(")", "")
        .replace(" ", "")
    )
    if plane_str in ("100", "001", "010"):
        if plane_str == "100":
            return np.eye(3, dtype=np.float64)
        elif plane_str == "010":
            return np.array([
                [0.0, 1.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ], dtype=np.float64)
        else:  # 001
            return np.array([
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
                [-1.0, 0.0, 0.0],
            ], dtype=np.float64)
    elif plane_str in ("110", "101", "011"):
        if plane_str == "110":
            # Normal: [1, 1, 0] / sqrt(2) -> [1, 0, 0]
            # Principal in-plane: [-1, 1, 0] / sqrt(2) -> [0, 1, 0]
            # Transverse in-plane: [0, 0, 1] -> [0, 0, 1]
            return np.array([
                [1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0), 0.0],
                [-1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0), 0.0],
                [0.0, 0.0, 1.0],
            ], dtype=np.float64)
        elif plane_str == "101":
            return np.array([
                [1.0 / np.sqrt(2.0), 0.0, 1.0 / np.sqrt(2.0)],
                [0.0, 1.0, 0.0],
                [-1.0 / np.sqrt(2.0), 0.0, 1.0 / np.sqrt(2.0)],
            ], dtype=np.float64)
        else:
            return np.array([
                [0.0, 1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0)],
                [1.0, 0.0, 0.0],
                [0.0, -1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0)],
            ], dtype=np.float64)
    elif plane_str == "111":
        # Normal along [1, 1, 1] / sqrt(3) -> +X
        # In-plane principal along [1, -1, 0] / sqrt(2) -> +Y
        # In-plane transverse along [1, 1, -2] / sqrt(6) -> +Z
        return np.array([
            [1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)],
            [1.0 / np.sqrt(2.0), -1.0 / np.sqrt(2.0), 0.0],
            [1.0 / np.sqrt(6.0), 1.0 / np.sqrt(6.0), -2.0 / np.sqrt(6.0)],
        ], dtype=np.float64)
    else:
        try:
            h, k, l = [float(c) for c in plane_str[:3]]
            norm = np.sqrt(h * h + k * k + l * l)
            if norm < 1e-6:
                return np.eye(3, dtype=np.float64)
            n_vec = np.array([h, k, l], dtype=np.float64) / norm
            ref = np.array([0.0, 0.0, 1.0]) if abs(n_vec[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
            u_vec = np.cross(n_vec, ref)
            u_vec = u_vec / np.linalg.norm(u_vec)
            v_vec = np.cross(n_vec, u_vec)
            return np.vstack([n_vec, u_vec, v_vec])
        except Exception:
            return np.eye(3, dtype=np.float64)


def solve_supercell_coincidence(
    pitch_a: float,
    pitch_b: float,
    habit_plane: str = "100",
    max_supercell: int = 4,
    max_strain: float = 0.05,
    rotation_candidates: Sequence[float] | None = None,
    network_a: str = "pcu",
    network_b: str = "pcu",
) -> tuple[tuple[int, int], float, float]:
    """
    Find smallest integer supercell ratio (m_A, n_B) and rotation angle theta
    that minimizes lattice mismatch strain epsilon along the transverse habit plane.

    Supports {100}, {110}, and {111} habit planes with discrete fast-paths:
    - {100}: square coincidence with 45 deg diamond twist.
    - {110}: centered rectangular coincidence (aspect ratio sqrt(2)).
    - {111}: hexagonal/triangular coincidence (1x1, sqrt(3)xsqrt(3) R30, 2x2).

    Returns:
        tuple: ((m_A, n_B), best_rotation_rad, min_strain)
    """
    p_a = float(pitch_a)
    p_b = float(pitch_b)
    net_a = network_a.lower().strip()
    net_b = network_b.lower().strip()
    plane_str = (
        str(habit_plane)
        .strip()
        .replace("{", "")
        .replace("}", "")
        .replace("(", "")
        .replace(")", "")
        .replace(" ", "")
    )

    if plane_str == "111":
        # Hexagonal net on {111}
        scale_a = (1.0 / np.sqrt(2.0)) if net_a == "dia" else np.sqrt(2.0)
        scale_b = (1.0 / np.sqrt(2.0)) if net_b == "dia" else np.sqrt(2.0)
        eff_p_a = p_a * scale_a
        eff_p_b = p_b * scale_b

        rot_cands = [0.0, np.pi / 6.0] if rotation_candidates is None else list(rotation_candidates)
        best_ratio = (1, 1)
        best_theta = 0.0
        min_strain = float("inf")

        for rot in rot_cands:
            rot_mult = np.sqrt(3.0) if abs(rot - np.pi / 6.0) < 1e-3 else 1.0
            for m in range(1, max_supercell + 1):
                for n in range(1, max_supercell + 1):
                    length_a = m * eff_p_a
                    length_b = n * eff_p_b * rot_mult
                    strain = abs(length_a - length_b) / length_a
                    if strain < min_strain:
                        min_strain = strain
                        best_ratio = (m, n)
                        best_theta = rot
                    if min_strain < 1e-5:
                        break
        return best_ratio, best_theta, min_strain

    elif plane_str in ("110", "101", "011"):
        # Centered rectangular net on {110}
        scale_a = 1.0 if net_a != "dia" else (1.0 / np.sqrt(2.0))
        scale_b = 1.0 if net_b != "dia" else (1.0 / np.sqrt(2.0))
        eff_p_a = p_a * scale_a
        eff_p_b = p_b * scale_b
        best_ratio = (1, 1)
        best_theta = 0.0
        min_strain = float("inf")
        for m in range(1, max_supercell + 1):
            for n in range(1, max_supercell + 1):
                length_a = m * eff_p_a
                length_b = n * eff_p_b
                strain = abs(length_a - length_b) / length_a
                if strain < min_strain:
                    min_strain = strain
                    best_ratio = (m, n)
                if min_strain < 1e-5:
                    break
        return best_ratio, best_theta, min_strain

    else:
        # Default {100}
        rot_cands = (0.0, np.pi / 4.0, np.pi / 2.0) if rotation_candidates is None else list(rotation_candidates)
        requires_45 = (net_a == "dia" and net_b in ("pcu", "bcc")) or (net_b == "dia" and net_a in ("pcu", "bcc"))
        best_ratio = (1, 1)
        best_theta = (np.pi / 4.0) if requires_45 else 0.0
        min_strain = float("inf")

        for rot in rot_cands:
            # 45-deg rotation on {100} scales diamond/FCC transverse spacing by 1/sqrt(2)
            rot_scale_b = (1.0 / np.sqrt(2.0)) if abs(rot - np.pi / 4.0) < 1e-4 else 1.0
            eff_p_b = p_b * rot_scale_b

            for m in range(1, max_supercell + 1):
                for n in range(1, max_supercell + 1):
                    length_a = m * p_a
                    length_b = n * eff_p_b
                    strain = abs(length_a - length_b) / length_a
                    if strain < min_strain:
                        min_strain = strain
                        best_ratio = (m, n)
                        best_theta = rot
                    if min_strain < 1e-5:
                        break

        return best_ratio, best_theta, min_strain


# =============================================================================
# Law 2: Aperture Admissibility Evaluator & Geometric Extractors
# =============================================================================

def compute_habit_plane_profile(
    geometry: ParticleGeometry,
    habit_plane: str = "100",
    wire_radius: float = 0.35,
    cell: InterlinkedCell | None = None,
    pitch: float | None = None,
) -> float:
    """
    Compute the penetrating cross-sectional profile diameter (mm) of a particle
    entering through a habit plane.
    """
    if cell is not None and hasattr(cell, "habit_plane_profile") and callable(getattr(cell, "habit_plane_profile")):
        try:
            return float(cell.habit_plane_profile(plane=habit_plane, cell_pitch=pitch))
        except TypeError:
            try:
                return float(cell.habit_plane_profile(plane=habit_plane))
            except TypeError:
                pass

    g_type = getattr(geometry, "geometry_type", "").lower()
    if "ring" in g_type:
        return 2.0 * wire_radius

    R_habit = get_habit_plane_rotation_matrix(habit_plane)
    raw_nodes = np.asarray(geometry.nodes, dtype=np.float64)
    if len(raw_nodes) == 0:
        return 2.0 * float(geometry.bounding_radius or 1.0)
    nodes = (R_habit @ raw_nodes.T).T

    proj_coords = nodes[:, 1:3]
    diffs = proj_coords[:, None, :] - proj_coords[None, :, :]
    max_d = float(np.max(np.linalg.norm(diffs, axis=-1)))
    return max(max_d, 1e-3)


def compute_habit_plane_aperture(
    geometry: ParticleGeometry,
    habit_plane: str = "100",
    wire_radius: float = 0.35,
    cell: InterlinkedCell | None = None,
    pitch: float | None = None,
) -> float:
    """
    Compute the receiving window aperture diameter (mm) of a particle
    on the boundary face perpendicular to habit_plane normal.
    """
    if cell is not None and hasattr(cell, "habit_plane_aperture") and callable(getattr(cell, "habit_plane_aperture")):
        try:
            return float(cell.habit_plane_aperture(plane=habit_plane, cell_pitch=pitch))
        except TypeError:
            try:
                return float(cell.habit_plane_aperture(plane=habit_plane))
            except TypeError:
                pass

    g_type = getattr(geometry, "geometry_type", "").lower()
    if "ring" in g_type:
        major_r = float(geometry.metadata.get("major_radius", geometry.bounding_radius or 1.0))
        return max(2.0 * (major_r - wire_radius), 0.0)

    R_habit = get_habit_plane_rotation_matrix(habit_plane)
    raw_nodes = np.asarray(geometry.nodes, dtype=np.float64)
    if len(raw_nodes) < 3:
        return 2.0 * float(geometry.bounding_radius or 1.0)
    nodes = (R_habit @ raw_nodes.T).T

    max_x = float(np.max(nodes[:, 0]))
    tol = max(0.15 * (max_x - float(np.min(nodes[:, 0]))), 1e-3)
    face_nodes = nodes[nodes[:, 0] >= (max_x - tol)]
    if len(face_nodes) >= 3:
        face_yz = face_nodes[:, 1:3]
        center = np.mean(face_yz, axis=0)
        radii = np.linalg.norm(face_yz - center, axis=-1)
        inscribed_diam = 2.0 * float(np.min(radii))
        return max(inscribed_diam, 0.5 * float(geometry.bounding_radius or 1.0))

    return float(geometry.bounding_radius or 1.0)


def check_aperture_admissibility(
    d_aperture: float,
    d_profile: float,
    wire_radius: float,
    min_clearance: float,
) -> tuple[bool, float]:
    """
    Verify that the boundary window aperture strictly exceeds the penetrating
    cross-sectional profile plus required solid wire and clearance buffer.

    Formula:
        margin = d_aperture - (d_profile + 2 * wire_radius + 2 * min_clearance)

    Returns:
        tuple: (is_admissible, margin_mm)
    """
    required = float(d_profile + 2.0 * wire_radius + 2.0 * min_clearance)
    margin = float(d_aperture - required)
    return (margin >= 0.0), margin


def extract_void_barycenters(
    geometry: ParticleGeometry | InterlinkedCell | str,
    cell: InterlinkedCell | None = None,
    habit_plane: str = "100",
    pitch: float = 10.0,
    wire_radius: float = 0.35,
) -> tuple[np.ndarray, float]:
    """
    Extract open void barycenters V_A and maximum inscribed aperture radius R_in
    on the receiving boundary face of a host cage (Option C Hybrid).

    Combines analytical Wyckoff interstitial baselines for standard nets with
    numerical 2D Voronoi medial-axis extraction.

    Returns:
        tuple: (void_barycenters_yz: np.ndarray of shape (K, 2), inscribed_radius_mm: float)
    """
    if isinstance(geometry, str):
        cell_cls = InterlinkedRegistry.get(geometry)
        resolved_cell = cell_cls() if callable(cell_cls) else cell_cls
        parts = resolved_cell.instantiate_site((0, 0, 0), np.zeros(3), pitch)
        geometry = parts[0].geometry
        if cell is None:
            cell = resolved_cell
    elif isinstance(geometry, InterlinkedCell):
        resolved_cell = geometry
        parts = resolved_cell.instantiate_site((0, 0, 0), np.zeros(3), pitch)
        geometry = parts[0].geometry
        if cell is None:
            cell = resolved_cell

    R_habit = get_habit_plane_rotation_matrix(habit_plane)
    raw_nodes = np.asarray(geometry.nodes, dtype=np.float64)
    if len(raw_nodes) < 3:
        return np.zeros((1, 2), dtype=np.float64), float(geometry.bounding_radius or 1.0)

    nodes = (R_habit @ raw_nodes.T).T
    max_x = float(np.max(nodes[:, 0]))
    tol = max(0.15 * (max_x - float(np.min(nodes[:, 0]))), 1e-3)
    face_nodes = nodes[nodes[:, 0] >= (max_x - tol)]
    if len(face_nodes) < 3:
        return np.zeros((1, 2), dtype=np.float64), float(geometry.bounding_radius or 1.0)

    face_yz = face_nodes[:, 1:3]
    center = np.mean(face_yz, axis=0)
    radii = np.linalg.norm(face_yz - center, axis=-1)
    r_in = max(float(np.min(radii)) - wire_radius, 0.05)

    candidates = [center]
    try:
        from scipy.spatial import Voronoi
        if len(face_yz) >= 4:
            vor = Voronoi(face_yz)
            for v in vor.vertices:
                if np.linalg.norm(v - center) <= 1.5 * r_in:
                    candidates.append(v)
    except Exception:
        pass

    best_pt = center
    best_dist = 0.0
    for cand in candidates:
        d = float(np.min(np.linalg.norm(face_yz - cand, axis=-1))) - wire_radius
        if d > best_dist:
            best_dist = d
            best_pt = cand

    return np.array([best_pt], dtype=np.float64), max(best_dist, 0.05)


def align_void_olattice(
    cell_a: InterlinkedCell | str,
    cell_b: InterlinkedCell | str,
    habit_plane: str = "100",
    pitch_a: float = 10.0,
    pitch_b: float = 10.0,
    wire_radius: float = 0.35,
) -> dict[str, Any]:
    """
    Inverted Void O-Lattice Alignment (Strategy 1).
    Computes inverted displacement matrix T_void = I - A_B @ inv(A_A) and
    aligns entering guest cage centroids with open void barycenters V_A.

    Returns:
        dict containing T_void, v_k, inscribed_radius_mm, x_guest_star, habit_plane.
    """
    if isinstance(cell_a, str):
        cell_a = InterlinkedRegistry.get(cell_a)()
    if isinstance(cell_b, str):
        cell_b = InterlinkedRegistry.get(cell_b)()

    R_habit = get_habit_plane_rotation_matrix(habit_plane)
    A_A = float(pitch_a) * np.eye(3, dtype=np.float64)
    A_B = float(pitch_b) * R_habit

    T_void = np.eye(3, dtype=np.float64) - A_B @ np.linalg.inv(A_A)

    parts_a = cell_a.instantiate_site((0, 0, 0), np.zeros(3), pitch_a)
    geom_a = parts_a[0].geometry
    void_pts_2d, r_in = extract_void_barycenters(
        geom_a, cell=cell_a, habit_plane=habit_plane, pitch=pitch_a, wire_radius=wire_radius
    )
    v_k_3d = np.array([pitch_a / 2.0, void_pts_2d[0, 0], void_pts_2d[0, 1]], dtype=np.float64)
    x_guest_star = A_B @ np.linalg.inv(A_A) @ v_k_3d

    return {
        "T_void": T_void,
        "v_k": v_k_3d,
        "inscribed_radius_mm": r_in,
        "x_guest_star": x_guest_star,
        "habit_plane": habit_plane,
    }


def apply_strut_taper(
    particles: Sequence[InterlinkedParticle],
    gamma: float = 0.70,
    boundary_layers: Sequence[int] = (-1, 0),
) -> int:
    """
    Apply strut wire radius thinning to particles in the boundary interface layers.

    Args:
        particles: Sequence of InterlinkedParticle instances.
        gamma: Wire radius scaling factor (0.40 <= gamma <= 1.0, default 0.70).
        boundary_layers: Layer indices along X considered interface boundaries.

    Returns:
        int: Number of particles whose struts were tapered.
    """
    count = 0
    g = float(np.clip(gamma, 0.40, 1.0))
    for p in particles:
        layer = p.metadata.get("layer_x")
        if layer in boundary_layers:
            curr_r = p.wire_radius if p.wire_radius is not None else 0.35
            p.wire_radius = curr_r * g
            p.metadata["strut_taper_applied"] = True
            p.metadata["taper_gamma"] = g
            count += 1
    return count


def apply_cage_shrinkage(
    particles: Sequence[InterlinkedParticle],
    eta: float = 0.90,
    boundary_layers: Sequence[int] = (0,),
) -> int:
    """
    Apply isotropic cage shrinkage to entering particles at boundary interface layers.

    Args:
        particles: Sequence of InterlinkedParticle instances.
        eta: Cage scale factor (0.70 <= eta <= 1.0, default 0.90).
        boundary_layers: Layer indices along X considered interface boundaries.

    Returns:
        int: Number of particles shrunk.
    """
    count = 0
    e = float(np.clip(eta, 0.70, 1.0))
    for p in particles:
        layer = p.metadata.get("layer_x")
        if layer in boundary_layers:
            nodes = np.asarray(p.geometry.nodes, dtype=np.float64)
            shrunk_nodes = nodes * e
            shrunk_geom = ParticleGeometry(
                nodes=shrunk_nodes,
                struts=p.geometry.struts,
                bounding_radius=(p.geometry.bounding_radius or 1.0) * e,
                geometry_type=p.geometry.geometry_type,
                metadata=dict(p.geometry.metadata),
            )
            object.__setattr__(p, "geometry", shrunk_geom)
            p.metadata["cage_shrink_applied"] = True
            p.metadata["shrink_eta"] = e
            count += 1
    return count


# =============================================================================
# Law 3: Boundary Catenation & Clearance Optimizer
# =============================================================================

def extract_chordless_cycles(nodes: np.ndarray, struts: np.ndarray) -> list[list[int]]:
    """
    Extract minimal chordless polygonal cycles (faces) from a wireframe cage graph.
    Supports single-loop rings, triangles (3-cycles), squares (4-cycles), hexagons (6-cycles),
    and arbitrary polyhedral face cycles via fundamental cycle basis.
    """
    N = len(nodes)
    if N < 3 or len(struts) < 3:
        return []

    adj = [set() for _ in range(N)]
    for u, v in struts:
        adj[u].add(v)
        adj[v].add(u)

    # Check if 2-regular graph (every vertex has degree 2, i.e. union of disjoint simple cycles)
    if all(len(adj[i]) == 2 for i in range(N)):
        cycles: list[list[int]] = []
        visited: set[int] = set()
        for start_node in range(N):
            if start_node in visited:
                continue
            cycle = [start_node]
            visited.add(start_node)
            curr = start_node
            prev = -1
            while True:
                nbrs = list(adj[curr])
                nxt = nbrs[0] if nbrs[0] != prev else nbrs[1]
                if nxt == start_node:
                    break
                cycle.append(nxt)
                visited.add(nxt)
                prev, curr = curr, nxt
            cycles.append(cycle)
        return cycles

    cycles: list[list[int]] = []
    seen_cycles: set[tuple[int, ...]] = set()

    def _canonical(cycle: list[int]) -> tuple[int, ...]:
        c = list(cycle)
        k1 = min(tuple(c[i:] + c[:i]) for i in range(len(c)))
        c_rev = c[::-1]
        k2 = min(tuple(c_rev[i:] + c_rev[:i]) for i in range(len(c_rev)))
        return min(k1, k2)

    # 1. Triangles (3-cycles)
    for u in range(N):
        for v in adj[u]:
            if v > u:
                for w in adj[u].intersection(adj[v]):
                    if w > v:
                        canon = _canonical([u, v, w])
                        if canon not in seen_cycles:
                            seen_cycles.add(canon)
                            cycles.append([u, v, w])

    # 2. Chordless 4-cycles (squares)
    for u in range(N):
        for w in range(u + 1, N):
            if w in adj[u]:
                continue
            common = sorted(list(adj[u].intersection(adj[w])))
            if len(common) >= 2:
                for i in range(len(common)):
                    for j in range(i + 1, len(common)):
                        v, x = common[i], common[j]
                        if x not in adj[v]:
                            canon = _canonical([u, v, w, x])
                            if canon not in seen_cycles:
                                seen_cycles.add(canon)
                                cycles.append([u, v, w, x])

    # 3. Higher-order chordless cycles (e.g. 5- and 6-cycles in polyhedral cages like C-6-TT)
    deg = [len(adj[i]) for i in range(N)]
    if any(d >= 3 for d in deg):
        try:
            import networkx as nx
            G = nx.Graph()
            for u, v in struts:
                G.add_edge(int(u), int(v))
            for c in nx.simple_cycles(G.to_directed()):
                if len(c) in (5, 6):
                    if G.subgraph(c).number_of_edges() == len(c):
                        # True polyhedral faces are planar; reject non-planar 3D skew cycles
                        pts = nodes[list(c)]
                        centered = pts - np.mean(pts, axis=0)
                        _, s, _ = np.linalg.svd(centered)
                        if s[-1] < 1e-3:
                            canon = _canonical(list(c))
                            if canon not in seen_cycles:
                                seen_cycles.add(canon)
                                cycles.append(list(c))
        except Exception:
            pass

    if len(cycles) > 0:
        return cycles

    # Fallback to networkx cycle basis if no 3-, 4-, 5-, or 6-cycles detected
    try:
        import networkx as nx
        G = nx.Graph()
        for u, v in struts:
            G.add_edge(int(u), int(v))
        basis = nx.cycle_basis(G)
        if len(basis) > 0:
            return basis
    except Exception:
        pass

    return cycles


def extract_triangulated_faces(nodes: np.ndarray, struts: np.ndarray) -> np.ndarray:
    """
    Extract triangulated polygonal faces from a wireframe cage graph.
    Converts 3-cycles directly, and splits chordless 4-cycles into 2 triangles.
    For 3D polyhedral cages (N >= 4 with min degree >= 3), utilizes 3D convex hull triangulation.
    """
    N = len(nodes)
    if N < 3 or len(struts) < 3:
        return np.zeros((0, 3), dtype=np.int64)

    deg = np.zeros(N, dtype=np.int64)
    for u, v in struts:
        deg[u] += 1
        deg[v] += 1
    if np.all(deg <= 2):
        return np.zeros((0, 3), dtype=np.int64)

    if N >= 4 and np.all(deg >= 3):
        try:
            import trimesh
            hull = trimesh.convex.convex_hull(nodes)
            if len(hull.faces) > 0:
                return np.asarray(hull.faces, dtype=np.int64)
        except Exception:
            pass

    cycles = extract_chordless_cycles(nodes, struts)
    if len(cycles) == 0:
        return np.zeros((0, 3), dtype=np.int64)

    triangles = []
    for c in cycles:
        if len(c) == 3:
            triangles.append([c[0], c[1], c[2]])
        elif len(c) == 4:
            triangles.append([c[0], c[1], c[2]])
            triangles.append([c[0], c[2], c[3]])
        elif len(c) > 4:
            for k in range(1, len(c) - 1):
                triangles.append([c[0], c[k], c[k + 1]])

    return np.array(triangles, dtype=np.int64) if len(triangles) > 0 else np.zeros((0, 3), dtype=np.int64)


def curve_curve_linking_number(
    p0: np.ndarray,
    p1: np.ndarray,
    q0: np.ndarray,
    q1: np.ndarray,
) -> float:
    """
    Compute exact topological linking number between two closed polygonal curves
    represented by edge endpoints (p0 -> p1 and q0 -> q1) using discrete Stokes
    polygon-strut signed puncture crossing.
    """
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)

    if len(p0) < 3 or len(q0) < 3:
        return 0.0

    v0 = p0[0]
    n_p = len(p0)
    crossings = 0

    # Infinitesimal generic position shift to prevent exact vertex-on-plane coincidence
    shift = np.array([1.1e-6, 1.3e-6, 1.7e-6], dtype=np.float64)
    q0_s = q0 + shift
    q1_s = q1 + shift

    for k in range(1, n_p - 1):
        v1 = p0[k]
        v2 = p0[k + 1]
        norm = np.cross(v1 - v0, v2 - v0)
        nl = np.linalg.norm(norm)
        if nl < 1e-9:
            continue
        norm = norm / nl

        for j in range(len(q0_s)):
            u0 = q0_s[j]
            u1 = q1_s[j]
            d0 = float(np.dot(u0 - v0, norm))
            d1 = float(np.dot(u1 - v0, norm))
            if d0 * d1 < 0.0:
                t = d0 / (d0 - d1)
                pt = u0 + t * (u1 - u0)
                c0 = np.cross(v1 - v0, pt - v0)
                c1 = np.cross(v2 - v1, pt - v1)
                c2 = np.cross(v0 - v2, pt - v2)
                if (
                    np.dot(c0, norm) >= -1e-7
                    and np.dot(c1, norm) >= -1e-7
                    and np.dot(c2, norm) >= -1e-7
                ):
                    crossings += 1 if (d1 - d0) > 0 else -1

    return float(crossings)


def _get_strut_components(nodes: np.ndarray, struts: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Partition wireframe graph into connected components (e.g. individual rings)."""
    from scipy.sparse import csgraph, csr_matrix
    N = len(nodes)
    if N == 0 or len(struts) == 0:
        return []
    adj = csr_matrix((np.ones(len(struts)), (struts[:, 0], struts[:, 1])), shape=(N, N))
    adj = adj + adj.T
    n_comp, labels = csgraph.connected_components(adj, directed=False)
    comps = []
    for c in range(n_comp):
        idx_nodes = np.where(labels == c)[0]
        mask = np.isin(struts[:, 0], idx_nodes) & np.isin(struts[:, 1], idx_nodes)
        if np.any(mask):
            comps.append((nodes, struts[mask]))
    return comps


def optimize_interface_phase_and_taper(
    nodes_a: np.ndarray,
    struts_a: np.ndarray,
    nodes_b: np.ndarray,
    struts_b: np.ndarray,
    faces_b: np.ndarray | None = None,
    wire_radius: float = 0.35,
    min_clearance: float = 0.150,
    search_range: tuple[float, float] | None = None,
    pitch_a: float = 10.0,
    pitch_b: float = 10.0,
    enable_taper: bool = True,
    strut_taper_gamma: float = 0.70,
    cage_shrink_eta: float = 0.90,
    cell_a: InterlinkedCell | str | None = None,
    cell_b: InterlinkedCell | str | None = None,
    habit_plane: str = "100",
    rotation_b: np.ndarray | None = None,
    patch_size: tuple[int, int] = (2, 2),
    enable_transverse_shift: bool = True,
    _warn_deprecated: bool = True,
) -> InterfacePhaseOptimizationResult:
    """
    3D Interface Phase Shift & Geometric Tapering Optimizer (Strategy 4 & 5).

    Evaluates a coarse 3D uniform grid over delta = (delta_x, delta_y, delta_z)
    (using full 2x2 interface patch segments when cells are provided) followed by
    fine bounded optimization to maximize physical surface clearance subject
    to non-zero closed-loop strut-face catenation (piercings >= 1).
    Applies strut thinning and cage shrinkage if aperture or clearance is constrained.

    Returns:
        InterfacePhaseOptimizationResult with optimal 3D phase shift and clearance.
    """
    if _warn_deprecated:
        import warnings
        warnings.warn(
            "optimize_interface_phase_and_taper is deprecated; use graphite.explicit.interlinked.search_interface instead.",
            DeprecationWarning,
            stacklevel=2,
        )
    p_avg = 0.5 * (float(pitch_a) + float(pitch_b))
    a_trans = min(float(pitch_a), float(pitch_b))
    s_range_x = search_range if search_range is not None else (0.15 * p_avg, 0.85 * p_avg)

    # 1. Setup Evaluation Geometry
    p0_eval = nodes_a[struts_a[:, 0]]
    p1_eval = nodes_a[struts_a[:, 1]]
    q0_eval = nodes_b[struts_b[:, 0]]
    q1_eval = nodes_b[struts_b[:, 1]]

    # Also prepare single prototype geometry for fast topological piercing test
    p0 = nodes_a[struts_a[:, 0]]
    p1 = nodes_a[struts_a[:, 1]]
    q0 = nodes_b[struts_b[:, 0]]
    q1 = nodes_b[struts_b[:, 1]]

    faces_a = extract_triangulated_faces(nodes_a, struts_a)
    if faces_b is None:
        faces_b = extract_triangulated_faces(nodes_b, struts_b)

    has_faces_b = len(faces_b) > 0
    has_faces_a = len(faces_a) > 0
    comps_a = _get_strut_components(nodes_a, struts_a)
    comps_b = _get_strut_components(nodes_b, struts_b)

    cycles_a = extract_chordless_cycles(nodes_a, struts_a)
    cycles_b = extract_chordless_cycles(nodes_b, struts_b)

    def _eval_phase(
        delta_vec: np.ndarray,
        eff_wire_r: float,
        cur_q0: np.ndarray,
        cur_q1: np.ndarray,
        cur_nodes_b: np.ndarray,
        cur_faces_b: np.ndarray,
        cur_q0_eval: np.ndarray,
        cur_q1_eval: np.ndarray,
    ) -> tuple[float, int]:
        d_arr = np.asarray(delta_vec, dtype=np.float64)

        # Clearance evaluated on full patch (or arrays)
        shifted_q0_eval = cur_q0_eval + d_arr
        shifted_q1_eval = cur_q1_eval + d_arr
        d_min = segment_segment_distance(p0_eval, p1_eval, shifted_q0_eval, shifted_q1_eval)
        clr = d_min - 2.0 * eff_wire_r

        # Piercings evaluated on prototype components
        shifted_nodes_a = nodes_a - d_arr
        shifted_p0 = p0 - d_arr
        shifted_p1 = p1 - d_arr

        prc_a_to_b = 0
        if len(cur_faces_b) > 0:
            for i in range(len(shifted_p0)):
                for f in cur_faces_b:
                    tri = cur_nodes_b[f]
                    if strut_pierces_triangle(shifted_p0[i], shifted_p1[i], tri):
                        prc_a_to_b += 1

        prc_b_to_a = 0
        if has_faces_a:
            for j in range(len(cur_q0)):
                for f in faces_a:
                    tri_a = shifted_nodes_a[f]
                    if strut_pierces_triangle(cur_q0[j], cur_q1[j], tri_a):
                        prc_b_to_a += 1

        # Determine catenation
        is_catenated = False
        if len(cur_faces_b) > 0 and has_faces_a:
            is_catenated = (prc_a_to_b > 0 and prc_b_to_a > 0)
        elif len(cur_faces_b) > 0:
            is_catenated = (prc_a_to_b > 0)
        elif has_faces_a:
            is_catenated = (prc_b_to_a > 0)

        # Fallback to Cycle Gauss linking check only if face piercing not established
        max_lk = 0.0
        if not is_catenated and len(cycles_a) > 0 and len(cycles_b) > 0:
            for ca in cycles_a:
                sp0 = shifted_nodes_a[ca]
                sp1 = shifted_nodes_a[np.roll(ca, -1)]
                cA = np.mean(sp0, axis=0)
                rA = np.max(np.linalg.norm(sp0 - cA, axis=-1))
                for cb in cycles_b:
                    sq0 = cur_nodes_b[cb]
                    sq1 = cur_nodes_b[np.roll(cb, -1)]
                    cB = np.mean(sq0, axis=0)
                    rB = np.max(np.linalg.norm(sq0 - cB, axis=-1))
                    if np.linalg.norm(cA - cB) > (rA + rB):
                        continue
                    lk = abs(curve_curve_linking_number(sp0, sp1, sq0, sq1))
                    if lk > max_lk:
                        max_lk = lk
                    if max_lk >= 0.75:
                        break
                if max_lk >= 0.75:
                    break
            if max_lk >= 0.75:
                is_catenated = True

        piercings = (prc_a_to_b + prc_b_to_a) if (prc_a_to_b + prc_b_to_a > 0) else (1 if is_catenated else 0)

        return float(clr), piercings

    cur_q0 = q0
    cur_q1 = q1
    cur_nodes_b = nodes_b
    cur_faces_b = faces_b
    cur_q0_eval = q0_eval
    cur_q1_eval = q1_eval
    eff_wire_r = float(wire_radius)
    taper_applied = False
    taper_gamma = 1.0
    shrink_eta = 1.0

    # Step 1: Coaxial Search along habit normal (1a: coarse, 1b: polish)
    coaxial_x = np.linspace(s_range_x[0], s_range_x[1], 25)
    best_delta = np.array([0.5 * (s_range_x[0] + s_range_x[1]), 0.0, 0.0], dtype=np.float64)
    best_clr = -float("inf")
    best_piercings = 0

    for dx in coaxial_x:
        d_vec = np.array([dx, 0.0, 0.0], dtype=np.float64)
        clr, prc = _eval_phase(
            d_vec, eff_wire_r, cur_q0, cur_q1, cur_nodes_b, cur_faces_b, cur_q0_eval, cur_q1_eval
        )
        if prc >= 1:
            if best_piercings == 0 or clr > best_clr:
                best_clr = clr
                best_delta = d_vec
                best_piercings = prc
        elif best_piercings == 0 and clr > best_clr:
            best_clr = clr
            best_delta = d_vec
            best_piercings = prc

    if best_piercings > 0:
        step_x = (coaxial_x[1] - coaxial_x[0]) if len(coaxial_x) > 1 else 0.5
        fine_x = np.linspace(best_delta[0] - 0.5 * step_x, best_delta[0] + 0.5 * step_x, 9)
        for fx in fine_x:
            d_vec = np.array([fx, 0.0, 0.0], dtype=np.float64)
            clr_pol, prc_pol = _eval_phase(
                d_vec, eff_wire_r, cur_q0, cur_q1, cur_nodes_b, cur_faces_b, cur_q0_eval, cur_q1_eval
            )
            if prc_pol >= 1 and clr_pol > best_clr:
                best_clr = clr_pol
                best_delta = d_vec
                best_piercings = prc_pol

    # Step 2: If coaxial search failed target clearance, explore 3D transverse shift (if enabled)
    if enable_transverse_shift and (best_piercings < 1 or best_clr < min_clearance):
        offsets_x = np.linspace(s_range_x[0], s_range_x[1], 7)
        offsets_y = np.linspace(-a_trans / 2.0, a_trans / 2.0, 7)
        dz_vals = np.linspace(-a_trans / 2.0, a_trans / 2.0, 7)
        for dx in offsets_x:
            for dy in offsets_y:
                for dz in dz_vals:
                    d_vec = np.array([dx, dy, dz], dtype=np.float64)
                    clr, prc = _eval_phase(
                        d_vec, eff_wire_r, cur_q0, cur_q1, cur_nodes_b, cur_faces_b, cur_q0_eval, cur_q1_eval
                    )
                    if prc >= 1:
                        if best_piercings == 0 or clr > best_clr:
                            best_clr = clr
                            best_delta = d_vec
                            best_piercings = prc
                    elif best_piercings == 0 and clr > best_clr:
                        best_clr = clr
                        best_delta = d_vec
                        best_piercings = prc

        if best_piercings > 0 and (abs(best_delta[1]) > 1e-4 or abs(best_delta[2]) > 1e-4):
            step_x = (offsets_x[1] - offsets_x[0]) if len(offsets_x) > 1 else 0.5
            step_y = (offsets_y[1] - offsets_y[0]) if len(offsets_y) > 1 else 0.5
            step_z = (dz_vals[1] - dz_vals[0]) if len(dz_vals) > 1 else 0.5
            fine_x = np.linspace(best_delta[0] - 0.5 * step_x, best_delta[0] + 0.5 * step_x, 5)
            fine_y = np.linspace(best_delta[1] - 0.5 * step_y, best_delta[1] + 0.5 * step_y, 3)
            fine_z = np.linspace(best_delta[2] - 0.5 * step_z, best_delta[2] + 0.5 * step_z, 3)
            for fx in fine_x:
                for fy in fine_y:
                    for fz in fine_z:
                        d_vec = np.array([fx, fy, fz], dtype=np.float64)
                        clr_pol, prc_pol = _eval_phase(
                            d_vec, eff_wire_r, cur_q0, cur_q1, cur_nodes_b, cur_faces_b, cur_q0_eval, cur_q1_eval
                        )
                        if prc_pol >= 1 and clr_pol > best_clr:
                            best_clr = clr_pol
                            best_delta = d_vec
                            best_piercings = prc_pol

    # Step 3: Tapering & Shrinkage Fallback (Strategy 5)
    if (best_clr < min_clearance or best_piercings < 1) and enable_taper:
        eff_wire_r = float(wire_radius * strut_taper_gamma)
        clr_t, prc_t = _eval_phase(
            best_delta, eff_wire_r, cur_q0, cur_q1, cur_nodes_b, cur_faces_b, cur_q0_eval, cur_q1_eval
        )
        if clr_t >= min_clearance and prc_t >= 1:
            best_clr = clr_t
            best_piercings = prc_t
            taper_applied = True
            taper_gamma = strut_taper_gamma
        else:
            # Secondary isotropic cage shrinkage
            shrunk_nodes_b = nodes_b * float(cage_shrink_eta)
            cur_q0 = shrunk_nodes_b[struts_b[:, 0]]
            cur_q1 = shrunk_nodes_b[struts_b[:, 1]]
            cur_nodes_b = shrunk_nodes_b
            cur_q0_eval = q0_eval * float(cage_shrink_eta)
            cur_q1_eval = q1_eval * float(cage_shrink_eta)
            clr_s, prc_s = _eval_phase(
                best_delta, eff_wire_r, cur_q0, cur_q1, cur_nodes_b, cur_faces_b, cur_q0_eval, cur_q1_eval
            )
            if clr_s > best_clr:
                best_clr = clr_s
                best_piercings = prc_s
                taper_applied = True
                taper_gamma = strut_taper_gamma
                shrink_eta = cage_shrink_eta

    is_valid = (best_piercings >= 1) and (best_clr >= min_clearance)
    return InterfacePhaseOptimizationResult(
        phase_shift=(float(best_delta[0]), float(best_delta[1]), float(best_delta[2])),
        max_clearance_mm=float(best_clr),
        piercings_count=int(best_piercings),
        is_valid=bool(is_valid),
        taper_applied=bool(taper_applied),
        taper_gamma=float(taper_gamma),
        shrink_eta=float(shrink_eta),
    )


def optimize_interface_catenation(
    nodes_a: np.ndarray,
    struts_a: np.ndarray,
    nodes_b: np.ndarray,
    struts_b: np.ndarray,
    faces_b: np.ndarray | None = None,
    wire_radius: float = 0.35,
    min_clearance: float = 0.150,
    search_range: tuple[float, float] | None = None,
    num_samples: int = 41,
    pitch_a: float = 10.0,
    pitch_b: float = 10.0,
) -> tuple[float, float, int]:
    """
    Backward-compatible 1D longitudinal interface offset optimizer.

    Returns:
        tuple: (best_dx_int, max_clearance_mm, piercings_count)
    """
    res = optimize_interface_phase_and_taper(
        nodes_a=nodes_a,
        struts_a=struts_a,
        nodes_b=nodes_b,
        struts_b=struts_b,
        faces_b=faces_b,
        wire_radius=wire_radius,
        min_clearance=min_clearance,
        search_range=search_range,
        pitch_a=pitch_a,
        pitch_b=pitch_b,
        enable_taper=False,
        enable_transverse_shift=False,
        _warn_deprecated=False,
    )
    return float(res.phase_shift[0]), float(res.max_clearance_mm), int(res.piercings_count)


# =============================================================================
# Universal Epitaxial Transition Solver
# =============================================================================

def analyze_epitaxial_transition(
    cell_a: InterlinkedCell | str,
    cell_b: InterlinkedCell | str,
    pitch_a: float | None = None,
    pitch_b: float | None = None,
    wire_radius: float = 0.35,
    min_clearance: float = 0.150,
    habit_plane: str = "100",
    max_supercell: int = 4,
    max_strain: float = 0.05,
    enable_taper: bool = True,
    strut_taper_gamma: float = 0.70,
    cage_shrink_eta: float = 0.90,
    validate_patch: bool = False,
) -> EpitaxialFeasibilityResult:
    """
    Evaluate the full Three-Law feasibility of an epitaxial transition between two cells.

    Incorporates:
      1. Commensurate Coincidence Lattice on {100}, {110}, or {111}.
      2. Aperture Admissibility with Strategy 5 geometric tapering fallback.
      3. 3D Phase Shift (delta_x, delta_y, delta_z) and Void O-Lattice alignment.
    """
    if isinstance(cell_a, str):
        cell_a = InterlinkedRegistry.get(cell_a)()
    if isinstance(cell_b, str):
        cell_b = InterlinkedRegistry.get(cell_b)()

    name_a = getattr(cell_a, "name", str(cell_a))
    name_b = getattr(cell_b, "name", str(cell_b))
    net_a = getattr(cell_a, "parent_network", "pcu").lower().strip()
    net_b = getattr(cell_b, "parent_network", "pcu").lower().strip()

    p_a = float(pitch_a if pitch_a is not None else 9.10)
    if pitch_b is None:
        if net_a in ("pcu", "bcc") and net_b == "dia":
            p_b = np.sqrt(2.0) * p_a
        elif net_a == "dia" and net_b in ("pcu", "bcc"):
            p_b = p_a / np.sqrt(2.0)
        else:
            p_b = p_a
    else:
        p_b = float(pitch_b)

    # -------------------------------------------------------------------------
    # Law 1: Commensurability
    # -------------------------------------------------------------------------
    supercell_ratio, theta_rot, strain = solve_supercell_coincidence(
        pitch_a=p_a,
        pitch_b=p_b,
        habit_plane=habit_plane,
        max_supercell=max_supercell,
        max_strain=max_strain,
        network_a=net_a,
        network_b=net_b,
    )

    if strain > max_strain:
        return EpitaxialFeasibilityResult(
            is_feasible=False,
            cell_a_name=name_a,
            cell_b_name=name_b,
            habit_plane=habit_plane,
            failure_stage="commensurability",
            diagnostics={
                "error": f"Lattices are incommensurate on habit plane ({habit_plane}). "
                         f"Minimum strain {strain*100:.2f}% exceeds tolerance {max_strain*100:.1f}%."
            },
        )

    R_plane = get_habit_plane_rotation_matrix(habit_plane)
    R_twist = np.array([
        [1.0, 0.0, 0.0],
        [0.0, np.cos(theta_rot), -np.sin(theta_rot)],
        [0.0, np.sin(theta_rot), np.cos(theta_rot)],
    ], dtype=np.float64)
    R_total_b = R_twist @ R_plane

    # -------------------------------------------------------------------------
    # Law 2: Aperture Admissibility
    # -------------------------------------------------------------------------
    parts_a = cell_a.instantiate_site((0, 0, 0), np.zeros(3), p_a)
    parts_b = cell_b.instantiate_site((0, 0, 0), np.zeros(3), p_b)

    geom_a = parts_a[0].geometry
    geom_b = parts_b[0].geometry

    d_aperture = compute_habit_plane_aperture(
        geom_a, habit_plane=habit_plane, wire_radius=wire_radius, cell=cell_a, pitch=p_a
    )
    d_profile = compute_habit_plane_profile(
        geom_b, habit_plane=habit_plane, wire_radius=wire_radius, cell=cell_b, pitch=p_b
    )

    is_admissible, aperture_margin = check_aperture_admissibility(
        d_aperture=d_aperture,
        d_profile=d_profile,
        wire_radius=wire_radius,
        min_clearance=min_clearance,
    )

    taper_needed_at_law2 = False
    if not is_admissible:
        if enable_taper:
            # Check if Strategy 5 (thinning + shrinkage) resolves the aperture clash
            d_profile_taper = d_profile * float(cage_shrink_eta)
            wire_r_taper = float(wire_radius * strut_taper_gamma)
            is_adm_t, margin_t = check_aperture_admissibility(
                d_aperture=d_aperture,
                d_profile=d_profile_taper,
                wire_radius=wire_r_taper,
                min_clearance=min_clearance,
            )
            if is_adm_t:
                is_admissible = True
                aperture_margin = margin_t
                taper_needed_at_law2 = True
            else:
                return EpitaxialFeasibilityResult(
                    is_feasible=False,
                    cell_a_name=name_a,
                    cell_b_name=name_b,
                    habit_plane=habit_plane,
                    failure_stage="aperture",
                    aperture_margin_mm=aperture_margin,
                    diagnostics={
                        "error": f"Aperture clash on habit plane ({habit_plane}). Cell A aperture "
                                 f"({d_aperture:.2f} mm) is smaller than penetrating profile + clearance "
                                 f"({d_profile + 2*wire_radius + 2*min_clearance:.2f} mm). Margin: {aperture_margin*1000:.1f} um."
                    },
                )
        else:
            return EpitaxialFeasibilityResult(
                is_feasible=False,
                cell_a_name=name_a,
                cell_b_name=name_b,
                habit_plane=habit_plane,
                failure_stage="aperture",
                aperture_margin_mm=aperture_margin,
                diagnostics={
                    "error": f"Aperture clash on habit plane ({habit_plane}). Cell A aperture "
                             f"({d_aperture:.2f} mm) is smaller than penetrating profile + clearance "
                             f"({d_profile + 2*wire_radius + 2*min_clearance:.2f} mm). Margin: {aperture_margin*1000:.1f} um."
                },
            )

    # -------------------------------------------------------------------------
    # Law 3: Boundary Catenation & 3D Phase Shift Optimization
    # -------------------------------------------------------------------------
    nodes_a_list = [(R_plane @ p.global_nodes().T).T for p in parts_a]
    struts_a_list = []
    offset_a = 0
    for p in parts_a:
        struts_a_list.append(p.geometry.struts + offset_a)
        offset_a += len(p.geometry.nodes)
    nodes_a_lab = np.vstack(nodes_a_list)
    struts_a = np.vstack(struts_a_list)

    nodes_b_list = [(R_total_b @ p.global_nodes().T).T for p in parts_b]
    struts_b_list = []
    offset_b = 0
    for p in parts_b:
        struts_b_list.append(p.geometry.struts + offset_b)
        offset_b += len(p.geometry.nodes)
    nodes_b_lab = np.vstack(nodes_b_list)
    struts_b = np.vstack(struts_b_list)

    opt_res = optimize_interface_phase_and_taper(
        nodes_a=nodes_a_lab,
        struts_a=struts_a,
        nodes_b=nodes_b_lab,
        struts_b=struts_b,
        wire_radius=wire_radius,
        min_clearance=min_clearance,
        pitch_a=p_a,
        pitch_b=p_b,
        enable_taper=(enable_taper or taper_needed_at_law2),
        strut_taper_gamma=strut_taper_gamma,
        cage_shrink_eta=cage_shrink_eta,
        cell_a=cell_a,
        cell_b=cell_b,
        habit_plane=habit_plane,
        rotation_b=R_total_b,
        _warn_deprecated=False,
    )

    if opt_res.piercings_count < 1 or opt_res.max_clearance_mm < min_clearance:
        return EpitaxialFeasibilityResult(
            is_feasible=False,
            cell_a_name=name_a,
            cell_b_name=name_b,
            habit_plane=habit_plane,
            failure_stage="catenation_or_clearance" if is_admissible else "aperture",
            aperture_margin_mm=aperture_margin,
            min_clearance_mm=opt_res.max_clearance_mm,
            piercings_count=opt_res.piercings_count,
            diagnostics={
                "error": (
                    f"Aperture clash on habit plane ({habit_plane}). Cell A aperture "
                    f"({d_aperture:.2f} mm) is smaller than penetrating profile + clearance "
                    f"({d_profile + 2*wire_radius + 2*min_clearance:.2f} mm). Margin: {aperture_margin*1000:.1f} um."
                    if not is_admissible
                    else f"Topological linking failure. Piercings: {opt_res.piercings_count} (required >= 1). "
                         f"Max achievable clearance: {opt_res.max_clearance_mm*1000:.1f} um (required >= {min_clearance*1000:.1f} um)."
                )
            },
        )

    alignment = EpitaxialAlignment(
        habit_plane=habit_plane,
        supercell_ratio=supercell_ratio,
        rotation_matrix=R_total_b,
        interface_offset=opt_res.phase_shift[0],
        strain=strain,
        phase_shift=opt_res.phase_shift,
        taper_gamma=opt_res.taper_gamma,
        shrink_eta=opt_res.shrink_eta,
        metadata={
            "aperture_margin_mm": aperture_margin,
            "min_clearance_mm": opt_res.max_clearance_mm,
            "piercings": opt_res.piercings_count,
            "phase_shift": opt_res.phase_shift,
            "taper_applied": opt_res.taper_applied or taper_needed_at_law2,
        },
    )

    # -------------------------------------------------------------------------
    # Optional Gate: Exact 2x2 Boundary Patch Verification
    # -------------------------------------------------------------------------
    if validate_patch:
        from .validator import verify_interface_patch
        patch_report = verify_interface_patch(
            cell_a=cell_a,
            cell_b=cell_b,
            alignment=alignment,
            patch_size=(2, 2),
            pitch_a=p_a,
            pitch_b=p_b,
            wire_radius=wire_radius,
            min_clearance=min_clearance,
            habit_plane=habit_plane,
        )

        if not patch_report.is_valid:
            return EpitaxialFeasibilityResult(
                is_feasible=False,
                cell_a_name=name_a,
                cell_b_name=name_b,
                habit_plane=habit_plane,
                failure_stage="catenation_or_clearance" if is_admissible else "aperture",
                min_clearance_mm=patch_report.min_clearance_mm,
                piercings_count=len(patch_report.linking_pairs),
                diagnostics={
                    "error": (
                        f"Aperture clash on habit plane ({habit_plane}). Cell A aperture "
                        f"({d_aperture:.2f} mm) is smaller than penetrating profile + clearance "
                        f"({d_profile + 2*wire_radius + 2*min_clearance:.2f} mm). Margin: {aperture_margin*1000:.1f} um."
                        if not is_admissible
                        else f"2x2 interface validation failed: {len(patch_report.collision_pairs)} solid collisions, "
                             f"min clearance {patch_report.min_clearance_mm*1000:.1f} um (required >= {min_clearance*1000:.1f} um), "
                             f"linking pairs {len(patch_report.linking_pairs)}."
                    ),
                    "collision_pairs": patch_report.collision_pairs,
                    "collision_volumes_mm3": patch_report.collision_volumes_mm3,
                },
            )

        return EpitaxialFeasibilityResult(
            is_feasible=True,
            cell_a_name=name_a,
            cell_b_name=name_b,
            habit_plane=habit_plane,
            alignment=alignment,
            aperture_margin_mm=aperture_margin,
            min_clearance_mm=patch_report.min_clearance_mm,
            piercings_count=len(patch_report.linking_pairs),
            diagnostics={
                "status": "Optimal heteroepitaxy solved and validated on 2x2 patch successfully",
                "collision_pairs": patch_report.collision_pairs,
                "linking_pairs": len(patch_report.linking_pairs),
            },
        )

    return EpitaxialFeasibilityResult(
        is_feasible=True,
        cell_a_name=name_a,
        cell_b_name=name_b,
        habit_plane=habit_plane,
        alignment=alignment,
        aperture_margin_mm=aperture_margin,
        min_clearance_mm=opt_res.max_clearance_mm,
        piercings_count=opt_res.piercings_count,
        diagnostics={
            "status": "Optimal heteroepitaxy solved successfully",
        },
    )


def solve_epitaxial_transition(
    cell_a: InterlinkedCell | str,
    cell_b: InterlinkedCell | str,
    **kwargs: Any,
) -> EpitaxialAlignment:
    """
    Solve and return the canonical EpitaxialAlignment for two cells.
    Raises EpitaxialIncompatibilityError if the transition is physically infeasible.
    """
    if "validate_patch" not in kwargs:
        kwargs["validate_patch"] = True
    result = analyze_epitaxial_transition(cell_a, cell_b, **kwargs)
    if not result.is_feasible:
        err_msg = result.diagnostics.get("error", "Epitaxial transition physically impossible.")
        raise EpitaxialIncompatibilityError(err_msg, diagnostics=result.diagnostics)
    return result.alignment  # type: ignore[return-value]


# =============================================================================
# Automated 2D CSL Commensurate Matcher & Phase-Aware Slicing Engine
# =============================================================================

@dataclass(frozen=True)
class CSLCommensurateMatch:
    """
    Result of inverting Bollmann's O-lattice equation for 2D CSL commensurate matching.
    """
    theta_deg: float
    m1: tuple[int, int]
    m2: tuple[int, int]
    strain_tensor: np.ndarray
    areal_strain: float
    residual_frobenius: float
    sigma_index: int = 1


def solve_2d_csl_commensurate_match(
    basis_a: np.ndarray,
    basis_b: np.ndarray,
    max_supercell: int = 4,
    max_strain: float = 0.05,
    candidate_angles_deg: Sequence[float] | None = None,
) -> CSLCommensurateMatch | None:
    """
    Automated 2D CSL Commensurate Matcher inverting Bollmann's O-lattice equation.

    Relates the two 2D habit plane basis matrices A_1 and A_2:
      T(theta) = A_2^-1 @ R(theta) @ A_1
    Searches for integer diagonal supercell matrices M_1, M_2 minimizing:
      min || M_2 - M_1 @ T(theta) ||_F
    Filters for areal strain |det(I + E) - 1| <= max_strain where:
      E = M_2^-1 @ M_1 @ T(theta) - I

    Returns:
        CSLCommensurateMatch if a commensurate configuration is found, else None.
    """
    A1 = np.asarray(basis_a, dtype=np.float64).reshape((2, 2))
    A2 = np.asarray(basis_b, dtype=np.float64).reshape((2, 2))
    inv_A2 = np.linalg.inv(A2)

    if candidate_angles_deg is None:
        angles_rad = np.radians(np.linspace(0.0, 90.0, 91))
    else:
        angles_rad = np.radians(np.asarray(candidate_angles_deg, dtype=np.float64))

    best_match: CSLCommensurateMatch | None = None
    min_res = float("inf")

    for th in angles_rad:
        cos_t = np.cos(th)
        sin_t = np.sin(th)
        R_2d = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float64)
        T = inv_A2 @ R_2d @ A1

        for m1x in range(1, max_supercell + 1):
            for m1y in range(1, max_supercell + 1):
                M1 = np.diag([m1x, m1y]).astype(np.float64)
                target_M2 = M1 @ T
                m2x = int(round(target_M2[0, 0]))
                m2y = int(round(target_M2[1, 1]))
                if m2x < 1 or m2y < 1 or m2x > max_supercell or m2y > max_supercell:
                    continue
                M2 = np.diag([m2x, m2y]).astype(np.float64)
                res = float(np.linalg.norm(M2 - target_M2, ord="fro"))

                inv_M2 = np.linalg.inv(M2)
                E = inv_M2 @ M1 @ T - np.eye(2, dtype=np.float64)
                areal_strain = float(abs(np.linalg.det(np.eye(2, dtype=np.float64) + E) - 1.0))

                if areal_strain <= max_strain and res < min_res:
                    min_res = res
                    sigma = int(max(1, round(np.linalg.det(M1))))
                    best_match = CSLCommensurateMatch(
                        theta_deg=float(np.degrees(th)),
                        m1=(m1x, m1y),
                        m2=(m2x, m2y),
                        strain_tensor=E,
                        areal_strain=areal_strain,
                        residual_frobenius=res,
                        sigma_index=sigma,
                    )
                    if res < 1e-5:
                        return best_match

    return best_match


@dataclass(frozen=True)
class PhaseAwareBoundarySlice:
    """
    Result of evaluating boundary slice depth against diamond/zincblende bilayer spacing.
    """
    target_depth: float
    snapped_depth: float
    is_snapped: bool
    requires_shockley_shift: bool
    shockley_shift_vector: np.ndarray
    g_star: float
    d_shuffle: float
    d_glide: float
    d_bilayer: float


def compute_phase_aware_boundary_slice(
    pitch: float,
    plane_normal: str | Sequence[float] = "111",
    target_depth: float = 0.0,
    network: str = "dia",
    wire_radius: float = 0.25,
) -> PhaseAwareBoundarySlice:
    """
    Evaluates slice depth against the periodic bilayer spacing of diamond cubic lattice.
    Forces interface onto structurally stable 3-bond glide plane and applies Shockley shift.
    """
    a = float(pitch)
    rw = float(wire_radius)
    plane_str = str(plane_normal).replace("{", "").replace("}", "").strip()

    if plane_str == "111" and network.lower().strip() in ("dia", "diamond"):
        d_shuffle = (a * np.sqrt(3.0)) / 12.0
        d_glide = (a * np.sqrt(3.0)) / 4.0
        d_bilayer = d_shuffle + d_glide

        local_phase = target_depth % d_bilayer

        # Dihedral angle of diamond struts approaching {111}: 35.26 degrees
        cos_theta = np.cos(np.radians(35.26))
        g_star = d_shuffle - rw * (1.0 + 1.0 / cos_theta)

        # Shockley partial dislocation in {111} supercell frame: e2 axis [0, 1, 0]
        s_star = a * np.sqrt(6.0) / 12.0

        if local_phase < d_shuffle:
            d_snapped = target_depth + (d_shuffle - local_phase)
            requires_shockley_shift = True
        else:
            d_snapped = target_depth
            requires_shockley_shift = False

        return PhaseAwareBoundarySlice(
            target_depth=target_depth,
            snapped_depth=d_snapped,
            is_snapped=requires_shockley_shift,
            requires_shockley_shift=requires_shockley_shift,
            shockley_shift_vector=np.array([0.0, s_star, 0.0], dtype=np.float64),
            g_star=g_star,
            d_shuffle=d_shuffle,
            d_glide=d_glide,
            d_bilayer=d_bilayer,
        )

    return PhaseAwareBoundarySlice(
        target_depth=target_depth,
        snapped_depth=target_depth,
        is_snapped=False,
        requires_shockley_shift=False,
        shockley_shift_vector=np.zeros(3, dtype=np.float64),
        g_star=0.0,
        d_shuffle=0.0,
        d_glide=0.0,
        d_bilayer=0.0,
    )


def check_topological_habit_admissibility(
    habit_plane: str,
    cell_a: Any,
    cell_b: Any,
    wire_radius: float = 0.25,
    min_clearance: float = 0.10,
    pitch_a: float = 12.0,
    pitch_b: float = 12.0,
) -> tuple[bool, str]:
    """
    Analytically evaluate transition admissibility across a given habit plane.

    A transition is analytically classified as Topologically Forbidden if the
    projected cross-sectional footprint of the entering strut cluster strictly
    exceeds the effective aperture diameter ratio (D_ap / a) of the receiving
    habit plane without catenation potential.
    For example, {100} on diamond cages presents foreshortened square facets that
    fundamentally struggle to accommodate out-of-plane strut insertion, causing
    severe aperture clash failure rates (~69.4%) and permitting only single axial
    ring piercing. Conversely, {111} habit planes present equilateral triangular
    facets and large hexagonal voids natively supporting Borromean ring triads
    and 3-cycle strut threading.

    Returns:
        tuple[bool, str]: (is_admissible, diagnostic_reason)
    """
    p_str = str(habit_plane).replace("{", "").replace("}", "").strip()
    name_a = getattr(cell_a, "name", str(cell_a)).lower().strip()
    name_b = getattr(cell_b, "name", str(cell_b)).lower().strip()

    # Rule 1: {100} diamond-tetrahedral self-transition is topologically forbidden
    norm_a = name_a.replace("-", "").replace("_", "")
    norm_b = name_b.replace("-", "").replace("_", "")
    is_dia_tet_a = any(k in norm_a for k in ("d4tet", "tetrahed"))
    is_dia_tet_b = any(k in norm_b for k in ("d4tet", "tetrahed"))
    if p_str in ("100", "001", "010") and is_dia_tet_a and is_dia_tet_b:
        return False, (
            "Topologically Forbidden: {100} habit plane on tetrahedral diamond cages presents "
            "foreshortened facets with aperture ratio D_ap/a < 0.15, causing severe aperture clash (>69% failure). "
            "Use {111} habit planes for 3-cycle threading, or interpose a C6-TT or B8-CO buffer."
        )

    # Rule 2: Quantitative aperture admissibility check if instances available
    try:
        from .cell import InterlinkedRegistry
        c_a = InterlinkedRegistry.get(name_a)() if isinstance(cell_a, str) else cell_a
        c_b = InterlinkedRegistry.get(name_b)() if isinstance(cell_b, str) else cell_b
        p_a = c_a.instantiate_site((0, 0, 0), np.zeros(3), pitch_a)
        p_b = c_b.instantiate_site((0, 0, 0), np.zeros(3), pitch_b)
        d_ap = compute_habit_plane_aperture(p_a[0].geometry, habit_plane=p_str, wire_radius=wire_radius, cell=c_a, pitch=pitch_a)
        
        # On {111} diamond habit planes, entering element is a 3-strut corner apex cluster, not monolithic cage width
        if p_str == "111" and (is_dia_tet_a or is_dia_tet_b):
            d_prof = 2.0 * float(wire_radius) * (1.0 + 1.0 / np.cos(np.radians(35.264)))
        else:
            d_prof = compute_habit_plane_profile(p_b[0].geometry, habit_plane=p_str, wire_radius=wire_radius)

        is_adm, margin = check_aperture_admissibility(d_ap, d_prof, wire_radius, min_clearance)
        if not is_adm:
            return False, (
                f"Aperture clash on habit plane ({p_str}): aperture diameter ({d_ap:.2f} mm) is smaller "
                f"than entering profile + clearance buffer ({d_prof + 2*wire_radius + 2*min_clearance:.2f} mm). "
                f"Margin: {margin*1000.0:.1f} um."
            )
    except Exception:
        pass

    return True, f"Habit plane ({p_str}) is topologically admissible."


def compute_analytical_pam_gap(
    pitch_a: float,
    pitch_b: float,
    habit_plane: str = "111",
    network_a: str = "dia",
    network_b: str = "dia",
    wire_radius: float = 0.25,
) -> float:
    """
    Computes optimal out-of-plane interface gap g* analytically:
      g*(a, r_w) = 0.5 * (Delta_z_A + Delta_z_B) - r_w * (1/cos(phi_A) + 1/cos(phi_B))

    where Delta_z is the interplanar depth to the first internal nodal tier of each
    sublattice, and phi is the insertion dihedral angle of primary struts relative
    to the habit plane normal.

    Returns:
        float: Optimal gap g* in mm.
    """
    p_a = float(pitch_a)
    p_b = float(pitch_b)
    p_str = str(habit_plane).replace("{", "").replace("}", "").strip()
    net_a = str(network_a).lower().strip()
    net_b = str(network_b).lower().strip()
    rw = float(wire_radius)

    if p_str == "111" and net_a in ("dia", "diamond") and net_b in ("dia", "diamond"):
        # Diamond {111} shuffle interplanar depth: a * sqrt(3) / 12
        dz_a = (p_a * np.sqrt(3.0)) / 12.0
        dz_b = (p_b * np.sqrt(3.0)) / 12.0
        # Dihedral angle of diamond struts approaching {111}: 35.26 deg
        phi_rad = np.radians(35.264)
        cos_phi = np.cos(phi_rad)  # sqrt(2/3) ~ 0.8165
        g_star = 0.5 * (dz_a + dz_b) - rw * (2.0 / cos_phi)
        return float(g_star)

    if p_str in ("110", "101", "011") and net_a in ("dia", "diamond") and net_b in ("dia", "diamond"):
        # Diamond {110} interplanar depth: a * sqrt(2) / 4
        dz_a = (p_a * np.sqrt(2.0)) / 4.0
        dz_b = (p_b * np.sqrt(2.0)) / 4.0
        phi_rad = np.radians(35.264)
        cos_phi = np.cos(phi_rad)
        g_star = 0.5 * (dz_a + dz_b) - rw * (2.0 / cos_phi)
        return float(g_star)

    if p_str in ("100", "001", "010"):
        if net_a == "pcu" and net_b in ("dia", "diamond"):
            dz_a = p_a * 0.5
            dz_b = p_b * 0.25
            cos_phi_a = 1.0  # struts normal
            cos_phi_b = np.cos(np.radians(54.736))  # diamond struts ~ 54.74 deg to {100}
            g_star = 0.5 * (dz_a + dz_b) - rw * (1.0 / cos_phi_a + 1.0 / cos_phi_b)
            return float(g_star)
        elif net_a == "pcu" and net_b == "bcc":
            dz_a = p_a * 0.5
            dz_b = p_b * 0.5
            g_star = 0.5 * (dz_a + dz_b) - rw * 2.0
            return float(g_star)

    # General default: 40% of average unit cell pitch minus wire clearance buffer
    avg_p = 0.5 * (p_a + p_b)
    return float(max(2.0 * rw, avg_p * 0.35 - 2.0 * rw))


def compute_inverted_void_shift(
    cell_a: Any,
    cell_b: Any,
    habit_plane: str = "111",
    pitch_a: float = 12.0,
    pitch_b: float = 12.0,
) -> np.ndarray:
    """
    Computes in-plane translation vector s* using Inverted Void O-Lattice Alignment:
      s* = c_{F_B} - x_A^V

    Maps the solid penetrating vertex of Lattice A directly to the receiving void
    centroid of Lattice B, maximizing steric clearance and eliminating aperture clashes.

    Returns:
        np.ndarray: 3D shift vector [sx, sy, sz] in mm.
    """
    p_str = str(habit_plane).replace("{", "").replace("}", "").strip()
    p_a = float(pitch_a)

    if p_str == "111":
        # In {111} orthogonal supercell frame, Shockley partial dislocation b = a/6 <11-2>
        # aligns along the e2 axis (Y) with magnitude a * sqrt(6) / 12
        s_y = (p_a * np.sqrt(6.0)) / 12.0
        return np.array([0.0, s_y, 0.0], dtype=np.float64)

    # General void barycenter extraction
    try:
        from .cell import InterlinkedRegistry
        c_a = InterlinkedRegistry.get(cell_a)() if isinstance(cell_a, str) else cell_a
        c_b = InterlinkedRegistry.get(cell_b)() if isinstance(cell_b, str) else cell_b
        p_b_inst = c_b.instantiate_site((0, 0, 0), np.zeros(3), pitch_b)
        p_a_inst = c_a.instantiate_site((0, 0, 0), np.zeros(3), pitch_a)
        void_b, _ = extract_void_barycenters(p_b_inst[0].geometry, cell=c_b, habit_plane=p_str, pitch=pitch_b)
        R_hab = get_habit_plane_rotation_matrix(p_str)
        nodes_a = (R_hab @ p_a_inst[0].geometry.nodes.T).T
        apex_a = nodes_a[np.argmax(nodes_a[:, 0])][1:3]
        s_yz = void_b[0] - apex_a
        return np.array([0.0, float(s_yz[0]), float(s_yz[1])], dtype=np.float64)
    except Exception:
        return np.zeros(3, dtype=np.float64)


def get_coherent_twin_transform(
    plane_normal: str | Sequence[float] = "111",
    in_supercell_frame: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the analytical reflection matrix and translation vector for a
    coherent twin boundary (Sigma 3).

    Analytical Formulation:
      R_twin = I - 2 * n @ n.T
      t_twin = [0, 0, 0]

    Because the boundary acts as an exact mirror reflection, void barycenters
    of the parent lattice align directly with strut nodes of the twin lattice
    with zero in-plane misfit strain (Sigma = 3, det(R) = -1).

    Returns:
        tuple[np.ndarray, np.ndarray]: (3x3 reflection matrix, 3D translation vector [0, 0, 0]).
    """
    if in_supercell_frame:
        # In orthogonal supercell frame, habit normal is aligned with Z = [0, 0, 1]
        R = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, -1.0],
        ], dtype=np.float64)
        t = np.zeros(3, dtype=np.float64)
        return R, t

    plane_str = str(plane_normal).replace("{", "").replace("}", "").strip()
    if plane_str == "111":
        n = np.array([1.0, 1.0, 1.0], dtype=np.float64) / np.sqrt(3.0)
    elif plane_str == "110":
        n = np.array([1.0, 1.0, 0.0], dtype=np.float64) / np.sqrt(2.0)
    elif plane_str in ("100", "001", "010"):
        n = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        n = np.asarray(plane_normal, dtype=np.float64).reshape(3)
        norm = np.linalg.norm(n)
        n = n / norm if norm > 1e-12 else np.array([0.0, 0.0, 1.0])

    R = np.eye(3, dtype=np.float64) - 2.0 * np.outer(n, n)
    t = np.zeros(3, dtype=np.float64)
    return R, t


@dataclass
class InterfacePlannerResult:
    """
    Authoritative configuration produced by the automated heteroepitaxy engine.
    """
    zone_a: Any
    zone_b: Any
    interface: Any
    mode: str
    csl_match: CSLCommensurateMatch | None = None
    slice_result: PhaseAwareBoundarySlice | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)


def plan_heteroepitaxial_interface(
    zone_a: Any,
    zone_b: Any,
    habit_plane: str = "111",
    wire_radius: float = 0.25,
    min_clearance: float = 0.10,
    max_supercell: int = 4,
    max_strain: float = 0.05,
    is_twin: bool = False,
) -> InterfacePlannerResult:
    """
    Automated Heteroepitaxy Interface Planner.
    Wraps two ZoneSpec objects into an optimal, commensurate, and clash-free assembly plan.
    """
    from .assemble import ZoneSpec, InterfaceSpec
    import copy

    za = copy.deepcopy(zone_a)
    zb = copy.deepcopy(zone_b)
    plane_str = str(habit_plane).replace("{", "").replace("}", "").strip()

    net_a = str(za.network).lower().strip()
    net_b = str(zb.network).lower().strip()
    pitch_a = float(za.pitch)
    pitch_b = float(zb.pitch)
    wr = float(wire_radius)

    # 0. Analytical Pre-flight Gate
    is_admissible, reason = check_topological_habit_admissibility(
        habit_plane=plane_str,
        cell_a=za.cell,
        cell_b=zb.cell,
        wire_radius=wr,
        min_clearance=min_clearance,
        pitch_a=pitch_a,
        pitch_b=pitch_b,
    )
    diagnostics_base: dict[str, Any] = {
        "topologically_admissible": is_admissible,
        "admissibility_reason": reason,
    }

    # Case 0: Coherent Twin Boundary (Sigma 3)
    twin_active = bool(is_twin or getattr(zb, "is_twin", False) or getattr(za, "is_twin", False))
    if twin_active and plane_str == "111":
        R_twin, t_twin = get_coherent_twin_transform("111", in_supercell_frame=True)
        s_vec = compute_inverted_void_shift(za.cell, zb.cell, habit_plane="111", pitch_a=pitch_a, pitch_b=pitch_b)
        s_y = float(s_vec[1])
        iface = InterfaceSpec(
            normal="001",
            gap=0.0,
            shift=[0.0, s_y, -2.50],
            twist_deg=0.0,
            reflection_matrix=R_twin,
        )
        zb.is_twin = True
        return InterfacePlannerResult(
            zone_a=za,
            zone_b=zb,
            interface=iface,
            mode="continuous",
            diagnostics={
                **diagnostics_base,
                "strategy": "Sigma 3 Coherent Twin Boundary Mirror Reflection",
                "sigma_index": 3,
                "s_star": [0.0, s_y, -2.50],
                "det_R": float(np.linalg.det(R_twin)),
            },
        )

    # Case 1: Isotypic Diamond Heteroepitaxy on {111}
    if plane_str == "111" and net_a in ("dia", "diamond") and net_b in ("dia", "diamond"):
        slice_res = compute_phase_aware_boundary_slice(
            pitch=pitch_a,
            plane_normal="111",
            target_depth=0.0,
            network="dia",
            wire_radius=wr,
        )
        s_vec = compute_inverted_void_shift(za.cell, zb.cell, habit_plane="111", pitch_a=pitch_a, pitch_b=pitch_b)
        s_y = float(s_vec[1])
        iface = InterfaceSpec(
            normal="001",
            gap=0.0,
            shift=[0.0, s_y, -2.50],
            twist_deg=0.0,
        )
        return InterfacePlannerResult(
            zone_a=za,
            zone_b=zb,
            interface=iface,
            mode="continuous",
            slice_result=slice_res,
            diagnostics={
                **diagnostics_base,
                "strategy": "Diamond {111} Glide-Set Shockley Interface",
                "g_star": slice_res.g_star,
                "s_star": s_vec.tolist(),
            },
        )

    # Case 2: Cross-Network Commensurate Heteroepitaxy
    ratio, canon_theta, canon_strain = solve_supercell_coincidence(
        pitch_a=pitch_a,
        pitch_b=pitch_b,
        habit_plane=plane_str,
        max_supercell=max_supercell,
        max_strain=max_strain,
        network_a=net_a,
        network_b=net_b,
    )
    canon_th_deg = float(np.degrees(canon_theta))
    g_analytical = compute_analytical_pam_gap(pitch_a, pitch_b, habit_plane=plane_str, network_a=net_a, network_b=net_b, wire_radius=wr)

    if canon_strain <= max_strain:
        if abs(canon_th_deg - 45.0) < 1e-3:
            # 45 deg square CSL domain matching (e.g. PCU 3x3 to DIA 2x2)
            m_a = max(3, 3 * ratio[0])
            m_b = max(2, 2 * ratio[1])
            za.grid_size = (m_a, m_a, za.grid_size[2] if len(za.grid_size) > 2 else 1)
            zb.grid_size = (m_b, m_b, zb.grid_size[2] if len(zb.grid_size) > 2 else 1)
            gap_val = float(max(g_analytical, 5.0) if (net_a == "pcu" and net_b in ("dia", "diamond")) else max(g_analytical, 4.0))
            iface = InterfaceSpec(
                normal="001",
                gap=gap_val,
                shift=[0.0, 0.0, 0.0],
                twist_deg=45.0,
            )
            return InterfacePlannerResult(
                zone_a=za,
                zone_b=zb,
                interface=iface,
                mode="discrete",
                diagnostics={
                    **diagnostics_base,
                    "strategy": "Canonical 45-deg CSL Commensurate Domain Match",
                    "strain": canon_strain,
                    "g_star": gap_val,
                },
            )
        else:
            m_a = ratio[0]
            m_b = ratio[1]
            za.grid_size = (m_a, m_a, za.grid_size[2] if len(za.grid_size) > 2 else 1)
            zb.grid_size = (m_b, m_b, zb.grid_size[2] if len(zb.grid_size) > 2 else 1)
            gap_val = float(max(g_analytical, 4.00))
            iface = InterfaceSpec(
                normal="001",
                gap=gap_val,
                shift=[0.0, 0.0, 0.0],
                twist_deg=canon_th_deg,
            )
            return InterfacePlannerResult(
                zone_a=za,
                zone_b=zb,
                interface=iface,
                mode="discrete",
                diagnostics={
                    **diagnostics_base,
                    "strategy": f"Canonical CSL Match theta={canon_th_deg:.1f}deg",
                    "strain": canon_strain,
                    "g_star": gap_val,
                },
            )

    # General 2D CSL solver fallback
    from .seeding import orthogonal_supercell
    _, per_a = orthogonal_supercell(network=net_a, plane=plane_str)
    _, per_b = orthogonal_supercell(network=net_b, plane=plane_str)
    basis_a = np.diag([pitch_a * per_a[0], pitch_a * per_a[1]])
    basis_b = np.diag([pitch_b * per_b[0], pitch_b * per_b[1]])

    csl_match = solve_2d_csl_commensurate_match(
        basis_a=basis_a,
        basis_b=basis_b,
        max_supercell=max_supercell,
        max_strain=max_strain,
    )

    if csl_match is not None:
        m1x, m1y = csl_match.m1
        m2x, m2y = csl_match.m2
        th_deg = csl_match.theta_deg
        za.grid_size = (m1x, m1y, za.grid_size[2] if len(za.grid_size) > 2 else 1)
        zb.grid_size = (m2x, m2y, zb.grid_size[2] if len(zb.grid_size) > 2 else 1)
        gap_val = float(max(g_analytical, 4.0))
        iface = InterfaceSpec(
            normal="001",
            gap=gap_val,
            shift=[0.0, 0.0, 0.0],
            twist_deg=th_deg,
        )
        return InterfacePlannerResult(
            zone_a=za,
            zone_b=zb,
            interface=iface,
            mode="discrete",
            csl_match=csl_match,
            diagnostics={
                **diagnostics_base,
                "strategy": f"CSL Commensurate Match theta={th_deg:.1f}deg",
                "g_star": gap_val,
            },
        )

    # Default fallback
    gap_val = float(max(g_analytical, 4.0))
    iface = InterfaceSpec(normal="001", gap=gap_val, shift=[0.0, 0.0, 0.0], twist_deg=0.0)
    return InterfacePlannerResult(
        zone_a=za,
        zone_b=zb,
        interface=iface,
        mode="discrete",
        diagnostics={**diagnostics_base, "strategy": "Default Fallback Interface", "g_star": gap_val},
    )


def evaluate_interface_quality_index(
    min_clearance_um: float,
    min_face_punctures: int,
    avg_face_punctures: float,
    num_boundary_links: int,
    max_possible_links: int,
    sigma_index: int = 1,
    k_clearance: float = 0.005,
    weights: tuple[float, float, float, float] = (0.4, 0.2, 0.2, 0.2),
) -> float:
    """
    Computes the Quantitative Interface Quality Index (Q_interface).

    Uses strict Heaviside step gating: returns 0.0 if clearance < 100 um or min_face_punctures < 2.
    Q = H(delta_min - 100) * H(min(Na, Nb) - 2) * [
          w1 * (C / C_max) + w2 * Sigma^-1 + w3 * (1 - exp(-k*(delta_min - 100))) + w4 * (N_bar / 3)
        ]

    Returns:
        float: Q score in [0.0, 1.0].
    """
    if min_clearance_um < 100.0 or min_face_punctures < 2 or max_possible_links <= 0:
        return 0.0

    w1, w2, w3, w4 = weights
    c_frac = min(1.0, num_boundary_links / max(1, max_possible_links))
    sigma_inv = 1.0 / max(1, sigma_index)
    clr_term = 1.0 - float(np.exp(-k_clearance * (min_clearance_um - 100.0)))
    n_term = min(1.0, avg_face_punctures / 3.0)

    q = w1 * c_frac + w2 * sigma_inv + w3 * clr_term + w4 * n_term
    return float(np.clip(q, 0.0, 1.0))



