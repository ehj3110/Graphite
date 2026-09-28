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
        habit_plane: Crystallographic Miller indices of the boundary interface (e.g. '100').
        supercell_ratio: Transverse integer supercell multipliers (m_A, n_B).
        rotation_matrix: (3, 3) rotation matrix aligning Cell B's habit plane to Cell A.
        interface_offset: Longitudinal spacing dx_int (mm) along interface normal.
        strain: In-plane lattice mismatch strain epsilon (|m*a_A - n*a_B| / (m*a_A)).
        metadata: Additional diagnostic metrics.
    """
    habit_plane: str = "100"
    supercell_ratio: tuple[int, int] = (1, 1)
    rotation_matrix: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=np.float64))
    interface_offset: float = 7.20
    strain: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


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

def solve_supercell_coincidence(
    pitch_a: float,
    pitch_b: float,
    habit_plane: str = "100",
    max_supercell: int = 4,
    max_strain: float = 0.05,
    rotation_candidates: Sequence[float] = (0.0, np.pi / 4.0, np.pi / 2.0),
) -> tuple[tuple[int, int], float, float]:
    """
    Find smallest integer supercell ratio (m_A, n_B) and rotation angle theta
    that minimizes lattice mismatch strain epsilon along the transverse habit plane.

    Returns:
        tuple: ((m_A, n_B), best_rotation_rad, min_strain)
    """
    p_a = float(pitch_a)
    p_b = float(pitch_b)
    best_ratio = (1, 1)
    best_theta = 0.0
    min_strain = float("inf")

    for theta in rotation_candidates:
        # Scale effective transverse pitch by rotation geometry if applicable
        # (e.g. 45-deg rotation scales diamond FCC spacing by 1/sqrt(2))
        scale_b = (1.0 / np.sqrt(2.0)) if abs(theta - np.pi / 4.0) < 1e-4 else 1.0
        eff_p_b = p_b * scale_b

        for m in range(1, max_supercell + 1):
            for n in range(1, max_supercell + 1):
                length_a = m * p_a
                length_b = n * eff_p_b
                strain = abs(length_a - length_b) / length_a
                if strain < min_strain:
                    min_strain = strain
                    best_ratio = (m, n)
                    best_theta = theta
                if min_strain < 1e-5:
                    break

    return best_ratio, best_theta, min_strain


# =============================================================================
# Law 2: Aperture Admissibility Evaluator
# =============================================================================

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


# =============================================================================
# Law 3: Boundary Catenation & Clearance Optimizer
# =============================================================================

def optimize_interface_catenation(
    nodes_a: np.ndarray,
    struts_a: np.ndarray,
    nodes_b: np.ndarray,
    struts_b: np.ndarray,
    faces_b: np.ndarray | None = None,
    wire_radius: float = 0.35,
    min_clearance: float = 0.150,
    search_range: tuple[float, float] = (5.0, 9.0),
    num_samples: int = 41,
) -> tuple[float, float, int]:
    """
    Optimize longitudinal interface offset dx_int maximizing physical clearance
    subject to non-zero closed-loop strut-face catenation piercings >= 1.

    Returns:
        tuple: (best_dx_int, max_clearance_mm, piercings_count)
    """
    offsets = np.linspace(search_range[0], search_range[1], num_samples)
    best_dx = search_range[0]
    best_clr = -float("inf")
    best_piercings = 0

    p0 = nodes_a[struts_a[:, 0]]
    p1 = nodes_a[struts_a[:, 1]]

    # Default tetrahedral faces if none provided
    if faces_b is None:
        faces_b = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.int64)

    for dx in offsets:
        # Shift nodes_a by [-dx, 0, 0] relative to nodes_b at origin
        shifted_p0 = p0 + np.array([-dx, 0.0, 0.0])
        shifted_p1 = p1 + np.array([-dx, 0.0, 0.0])

        # 1. Evaluate minimum centerline distance
        q0 = nodes_b[struts_b[:, 0]]
        q1 = nodes_b[struts_b[:, 1]]
        d_min = segment_segment_distance(shifted_p0, shifted_p1, q0, q1)
        clr = d_min - 2.0 * wire_radius

        # 2. Count topological face piercings
        piercings = 0
        for i in range(len(shifted_p0)):
            for f in faces_b:
                tri = nodes_b[f]
                if strut_pierces_triangle(shifted_p0[i], shifted_p1[i], tri):
                    piercings += 1

        # Must have at least 1 piercing to be mechanically catenated
        if piercings >= 1:
            if clr > best_clr:
                best_clr = clr
                best_dx = dx
                best_piercings = piercings

    return float(best_dx), float(best_clr), best_piercings


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
) -> EpitaxialFeasibilityResult:
    """
    Evaluate the full Three-Law feasibility of an epitaxial transition between two cells.
    """
    if isinstance(cell_a, str):
        cell_a = InterlinkedRegistry.get(cell_a)()
    if isinstance(cell_b, str):
        cell_b = InterlinkedRegistry.get(cell_b)()

    name_a = getattr(cell_a, "name", str(cell_a))
    name_b = getattr(cell_b, "name", str(cell_b))

    p_a = float(pitch_a if pitch_a is not None else 9.10)
    p_b = float(pitch_b if pitch_b is not None else np.sqrt(2.0) * p_a)

    # -------------------------------------------------------------------------
    # Law 1: Commensurability
    # -------------------------------------------------------------------------
    supercell_ratio, theta_rot, strain = solve_supercell_coincidence(
        pitch_a=p_a,
        pitch_b=p_b,
        habit_plane=habit_plane,
        max_supercell=max_supercell,
        max_strain=max_strain,
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

    # Rotation matrix around X-axis
    R_habit = np.array([
        [1.0, 0.0, 0.0],
        [0.0, np.cos(theta_rot), -np.sin(theta_rot)],
        [0.0, np.sin(theta_rot), np.cos(theta_rot)],
    ], dtype=np.float64)

    # -------------------------------------------------------------------------
    # Law 2: Aperture Admissibility
    # -------------------------------------------------------------------------
    # Retrieve or estimate face aperture of Cell A and profile of Cell B
    d_aperture = getattr(cell_a, "habit_plane_aperture", lambda plane: 7.54)(habit_plane)
    d_profile = getattr(cell_b, "habit_plane_profile", lambda plane: 5.49)(habit_plane)

    is_admissible, aperture_margin = check_aperture_admissibility(
        d_aperture=d_aperture,
        d_profile=d_profile,
        wire_radius=wire_radius,
        min_clearance=min_clearance,
    )

    if not is_admissible:
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
    # Law 3: Boundary Catenation & Spacing Optimization
    # -------------------------------------------------------------------------
    proto_a = cell_a.basis_particles[0].geometry
    proto_b = cell_b.basis_particles[0].geometry

    # Scale prototypes from canonical unit basis to physical dimensions
    s_a = p_a * float(getattr(cell_a, "size_ratio", 0.879))
    base_size_a = float(proto_a.metadata.get("size", proto_a.metadata.get("edge_length", proto_a.bounding_radius or 1.0)))
    nodes_a = proto_a.nodes * (s_a / max(base_size_a, 1e-6))
    struts_a = proto_a.struts

    L_b = float(getattr(cell_b, "edge_ratio", 0.603)) * p_b
    base_size_b = float(proto_b.metadata.get("edge_length", proto_b.metadata.get("size", proto_b.bounding_radius or 1.0)))
    nodes_b_scaled = proto_b.nodes * (L_b / max(base_size_b, 1e-6))
    nodes_b_rot = (R_habit @ nodes_b_scaled.T).T
    struts_b = proto_b.struts

    best_dx, max_clr, piercings = optimize_interface_catenation(
        nodes_a=nodes_a,
        struts_a=struts_a,
        nodes_b=nodes_b_rot,
        struts_b=struts_b,
        wire_radius=wire_radius,
        min_clearance=min_clearance,
    )

    if piercings < 1 or max_clr < min_clearance:
        return EpitaxialFeasibilityResult(
            is_feasible=False,
            cell_a_name=name_a,
            cell_b_name=name_b,
            habit_plane=habit_plane,
            failure_stage="catenation_or_clearance",
            min_clearance_mm=max_clr,
            piercings_count=piercings,
            diagnostics={
                "error": f"Topological linking failure. Piercings: {piercings} (required >= 1). "
                         f"Max achievable clearance: {max_clr*1000:.1f} um (required >= {min_clearance*1000:.1f} um)."
            },
        )

    alignment = EpitaxialAlignment(
        habit_plane=habit_plane,
        supercell_ratio=supercell_ratio,
        rotation_matrix=R_habit,
        interface_offset=best_dx,
        strain=strain,
        metadata={
            "aperture_margin_mm": aperture_margin,
            "min_clearance_mm": max_clr,
            "piercings": piercings,
        },
    )

    return EpitaxialFeasibilityResult(
        is_feasible=True,
        cell_a_name=name_a,
        cell_b_name=name_b,
        habit_plane=habit_plane,
        alignment=alignment,
        aperture_margin_mm=aperture_margin,
        min_clearance_mm=max_clr,
        piercings_count=piercings,
        diagnostics={"status": "Optimal heteroepitaxy solved successfully"},
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
    result = analyze_epitaxial_transition(cell_a, cell_b, **kwargs)
    if not result.is_feasible:
        err_msg = result.diagnostics.get("error", "Epitaxial transition physically impossible.")
        raise EpitaxialIncompatibilityError(err_msg, diagnostics=result.diagnostics)
    return result.alignment  # type: ignore[return-value]
