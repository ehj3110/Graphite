"""
Quantized Topological Boundary — SC hex VF and boundary-state tagging.

Data layer only: no strut generation or topology stamping.

Estimates true solid volume fraction (VF) of the CAD inside each SC hex via a
shared EDT voxel SDF (`voxelize_mesh_and_edt`), then tags cells as Empty / Half /
Full and orients Half cells with six polarities
(Half_Pos/Neg_X/Y/Z) from the SDF gradient.

Default gate (Task 24 — strict Fully-Inside):
  Empty: VF <= 0.05 | Half: 0.05 < VF < 0.95 | Full: VF >= 0.95
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

import numpy as np
import trimesh

from graphite.explicit.mesh_repair import sanitize_cad_mesh_for_sdf
from graphite.geometry.masking import contains_points_multi_ray, voxelize_mesh_and_edt

# ---------------------------------------------------------------------------
# Public enums / thresholds (strict Fully-Inside gate — Task 24)
# ---------------------------------------------------------------------------
# Empty: VF <= EMPTY_VF_MAX  (almost entirely outside → delete)
# Full:  VF >= FULL_VF_MIN   (almost entirely inside → core)
# Half:  EMPTY_VF_MAX < VF < FULL_VF_MIN  (boundary cut cell)

EMPTY_VF_MAX = 0.05
FULL_VF_MIN = 0.95


class HexFillState(str, Enum):
    EMPTY = "Empty"
    HALF = "Half"
    FULL = "Full"


class HalfOrientation(str, Enum):
    """Which half of the cell contains solid (apex side kept)."""

    HALF_POS_X = "Half_Pos_X"
    HALF_NEG_X = "Half_Neg_X"
    HALF_POS_Y = "Half_Pos_Y"
    HALF_NEG_Y = "Half_Neg_Y"
    HALF_POS_Z = "Half_Pos_Z"
    HALF_NEG_Z = "Half_Neg_Z"


# axis → (pos_tag, neg_tag)
_AXIS_POLARITY: tuple[tuple[HalfOrientation, HalfOrientation], ...] = (
    (HalfOrientation.HALF_POS_X, HalfOrientation.HALF_NEG_X),
    (HalfOrientation.HALF_POS_Y, HalfOrientation.HALF_NEG_Y),
    (HalfOrientation.HALF_POS_Z, HalfOrientation.HALF_NEG_Z),
)


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HexVolumeFractionResult:
    """Per-hex solid volume fraction against a CAD solid."""

    volume_fractions: np.ndarray  # (N,) in [0, 1]
    n_hex: int
    samples_per_axis: int
    voxel_resolution: float
    n_samples_per_hex: int
    method: str = "edt_voxel_trilinear_sample"


@dataclass
class HexBoundaryClassification:
    """VF + Empty/Half/Full tags + Half cut-plane orientations."""

    volume_fractions: np.ndarray
    states: np.ndarray  # object/str, shape (N,)
    orientations: np.ndarray  # object, Half_* or None, shape (N,)
    cut_axes: np.ndarray  # (N,) int: 0/1/2 for Half, -1 otherwise
    cut_normals: np.ndarray  # (N, 3) unit world axis for Half, else 0
    gradient_at_centroid: np.ndarray  # (N, 3) SDF grad estimate
    empty_vf_max: float = EMPTY_VF_MAX
    full_vf_min: float = FULL_VF_MIN
    report: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# SDF field helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _EdtSdfField:
    sdf: np.ndarray
    origin: np.ndarray
    resolution: float
    shape: tuple[int, int, int]

    def sample(self, points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        if pts.size == 0:
            return np.empty(0, dtype=np.float64)
        ind = np.round((pts - self.origin) / self.resolution).astype(np.int64)
        nx, ny, nz = self.shape
        ind[:, 0] = np.clip(ind[:, 0], 0, nx - 1)
        ind[:, 1] = np.clip(ind[:, 1], 0, ny - 1)
        ind[:, 2] = np.clip(ind[:, 2], 0, nz - 1)
        return self.sdf[ind[:, 0], ind[:, 1], ind[:, 2]]

    def gradient(self, points: np.ndarray, *, delta: float | None = None) -> np.ndarray:
        """Central-difference SDF gradient in world space (points toward outside)."""
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        h = float(delta) if delta is not None else float(self.resolution)
        if h <= 0.0:
            raise ValueError("gradient delta must be positive")
        out = np.zeros_like(pts, dtype=np.float64)
        for ax in range(3):
            e = np.zeros(3, dtype=np.float64)
            e[ax] = h
            out[:, ax] = (self.sample(pts + e) - self.sample(pts - e)) / (2.0 * h)
        return out


def build_edt_sdf_field(
    cad_mesh: trimesh.Trimesh,
    resolution: float,
) -> _EdtSdfField:
    """
    Build a padded EDT signed-distance field for the CAD.

    Sanitizes the CAD (fill holes + fix normals) before voxelization.
    Convention matches ``voxelize_mesh_and_edt``: negative inside, positive outside.
    """
    res = float(resolution)
    if res <= 0.0:
        raise ValueError(f"resolution must be positive; got {resolution}")
    cad = sanitize_cad_mesh_for_sdf(cad_mesh)
    _, _, _, cad_sdf, padded_min, _, nx, ny, nz = voxelize_mesh_and_edt(cad, res)
    return _EdtSdfField(
        sdf=np.asarray(cad_sdf, dtype=np.float64),
        origin=np.asarray(padded_min, dtype=np.float64),
        resolution=res,
        shape=(int(nx), int(ny), int(nz)),
    )


def _default_voxel_resolution(
    hex_elems: np.ndarray,
    samples_per_axis: int,
    *,
    resolution: float | None,
) -> float:
    if resolution is not None:
        res = float(resolution)
        if res <= 0.0:
            raise ValueError(f"resolution must be positive; got {resolution}")
        return res
    edges = []
    sample = hex_elems[: min(64, len(hex_elems))]
    for corners in sample:
        # SC edge lengths along the three axes from corner 0.
        edges.append(float(np.linalg.norm(corners[1] - corners[0])))
        edges.append(float(np.linalg.norm(corners[3] - corners[0])))
        edges.append(float(np.linalg.norm(corners[4] - corners[0])))
    min_edge = float(min(edges)) if edges else 1.0
    # Aim for ~1 voxel per sample spacing along the shortest edge.
    return max(min_edge / float(max(samples_per_axis, 1)), 1e-6)


def _trilinear_hex_points(corners: np.ndarray, uvw: np.ndarray) -> np.ndarray:
    """
    Map parametric samples in [0, 1]^3 into a hex brick.

    Corner indexing matches SC background grid / ``hex_rules``:
    0:(0,0,0) 1:(1,0,0) 2:(1,1,0) 3:(0,1,0)
    4:(0,0,1) 5:(1,0,1) 6:(1,1,1) 7:(0,1,1)
    """
    c = np.asarray(corners, dtype=np.float64)
    u = uvw[:, 0:1]
    v = uvw[:, 1:2]
    w = uvw[:, 2:3]
    c000, c100, c110, c010 = c[0], c[1], c[2], c[3]
    c001, c101, c111, c011 = c[4], c[5], c[6], c[7]
    return (
        (1 - u) * (1 - v) * (1 - w) * c000
        + u * (1 - v) * (1 - w) * c100
        + u * v * (1 - w) * c110
        + (1 - u) * v * (1 - w) * c010
        + (1 - u) * (1 - v) * w * c001
        + u * (1 - v) * w * c101
        + u * v * w * c111
        + (1 - u) * v * w * c011
    )


def _unit_cube_sample_grid(samples_per_axis: int) -> np.ndarray:
    n = int(samples_per_axis)
    if n < 2:
        raise ValueError(f"samples_per_axis must be >= 2; got {samples_per_axis}")
    # Cell-centered samples avoid double-counting shared faces across hexes.
    axes = (np.arange(n, dtype=np.float64) + 0.5) / float(n)
    uu, vv, ww = np.meshgrid(axes, axes, axes, indexing="ij")
    return np.column_stack((uu.ravel(), vv.ravel(), ww.ravel()))


# ---------------------------------------------------------------------------
# True VF calculator
# ---------------------------------------------------------------------------


def estimate_hex_volume_fractions(
    cad_mesh: trimesh.Trimesh,
    hex_elems: np.ndarray,
    *,
    samples_per_axis: int = 8,
    resolution: float | None = None,
    sdf_field: _EdtSdfField | None = None,
    inside_eps: float = 0.0,
) -> HexVolumeFractionResult:
    """
    Estimate solid volume fraction of ``cad_mesh`` inside each SC hex.

    Uses a one-shot EDT voxel SDF (``voxelize_mesh_and_edt``), then dense
    trilinear samples inside each hex. VF = fraction of samples with SDF <=
    ``inside_eps`` (negative SDF = inside).

    Parameters
    ----------
    cad_mesh :
        Watertight CAD solid.
    hex_elems :
        ``(N, 8, 3)`` hex corner coordinates (SC ordering).
    samples_per_axis :
        Samples along each parametric axis (total ``n**3`` per hex).
    resolution :
        EDT voxel pitch in mm. Default: ``min_edge / samples_per_axis``.
    sdf_field :
        Optional prebuilt field (reuse across calls).
    inside_eps :
        Treat SDF <= eps as inside (default 0).
    """
    elems = np.asarray(hex_elems, dtype=np.float64)
    if elems.ndim != 3 or elems.shape[1:] != (8, 3):
        raise ValueError(f"hex_elems must have shape (N, 8, 3); got {elems.shape}")
    n_hex = int(elems.shape[0])
    if n_hex == 0:
        return HexVolumeFractionResult(
            volume_fractions=np.empty(0, dtype=np.float64),
            n_hex=0,
            samples_per_axis=int(samples_per_axis),
            voxel_resolution=float(resolution or 0.0),
            n_samples_per_hex=0,
        )

    cad = sanitize_cad_mesh_for_sdf(cad_mesh)
    res = _default_voxel_resolution(elems, samples_per_axis, resolution=resolution)
    field = sdf_field if sdf_field is not None else build_edt_sdf_field(cad, res)
    uvw = _unit_cube_sample_grid(samples_per_axis)
    n_samp = int(uvw.shape[0])
    vfs = np.zeros(n_hex, dtype=np.float64)
    eps = float(inside_eps)

    # Spot-check EDT vs multi-ray vote at the CAD centroid; if they disagree,
    # fall back to multi-ray VF for the whole grid (avoids hollow-core EDT).
    use_ray_vf = False
    try:
        c = np.asarray(cad.centroid, dtype=np.float64).reshape(1, 3)
        sdf_c = float(field.sample(c)[0])
        ray_in = bool(contains_points_multi_ray(cad, c)[0])
        edt_in = sdf_c <= float(eps)
        use_ray_vf = bool(ray_in != edt_in)
    except Exception:
        use_ray_vf = False

    for i, corners in enumerate(elems):
        pts = _trilinear_hex_points(corners, uvw)
        if use_ray_vf:
            inside = contains_points_multi_ray(cad, pts)
            vfs[i] = float(np.count_nonzero(inside)) / float(n_samp)
        else:
            sdf = field.sample(pts)
            vfs[i] = float(np.count_nonzero(sdf <= eps)) / float(n_samp)

    return HexVolumeFractionResult(
        volume_fractions=vfs,
        n_hex=n_hex,
        samples_per_axis=int(samples_per_axis),
        voxel_resolution=float(field.resolution),
        n_samples_per_hex=n_samp,
        method=(
            "multi_ray_vote_sample"
            if use_ray_vf
            else "edt_voxel_trilinear_sample"
        ),
    )


# ---------------------------------------------------------------------------
# Orientation for Half cells
# ---------------------------------------------------------------------------


def _dominant_axis_from_vector(vec: np.ndarray) -> int:
    v = np.asarray(vec, dtype=np.float64).reshape(3)
    return int(np.argmax(np.abs(v)))


def _face_normal_vote_axis(
    cad_mesh: trimesh.Trimesh,
    corners: np.ndarray,
    sdf_field: _EdtSdfField,
    *,
    n_probe: int = 27,
) -> np.ndarray:
    """
    Average CAD triangle normals at near-surface probes inside the hex.

    Fallback when the SDF gradient is weak (flat / far from surface).
    """
    uvw = _unit_cube_sample_grid(max(3, int(round(n_probe ** (1.0 / 3.0)))))
    pts = _trilinear_hex_points(corners, uvw)
    sdf = sdf_field.sample(pts)
    # Prefer probes near the zero level-set.
    band = np.abs(sdf)
    med = float(np.median(band)) if len(band) else 0.0
    near = band <= max(med, float(sdf_field.resolution))
    if not np.any(near):
        near = np.ones(len(pts), dtype=bool)
    probe = pts[near]
    try:
        _closest, _dist, tid = trimesh.proximity.ProximityQuery(cad_mesh).on_surface(
            probe
        )
        normals = np.asarray(cad_mesh.face_normals[tid], dtype=np.float64)
    except Exception:
        return np.zeros(3, dtype=np.float64)
    mean_n = normals.mean(axis=0)
    nrm = float(np.linalg.norm(mean_n))
    if nrm < 1e-12:
        return np.zeros(3, dtype=np.float64)
    return mean_n / nrm


def _axis_straddle_signed(
    corners: np.ndarray,
    sdf_field: _EdtSdfField,
    *,
    samples_per_face: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Per-axis straddle strength and signed outside direction.

    ``sign[ax] > 0`` means the +face has higher SDF (more outside) than the
    −face, so the solid lies toward −axis.
    """
    n = int(samples_per_face)
    t = (np.arange(n, dtype=np.float64) + 0.5) / float(n)
    scores = np.zeros(3, dtype=np.float64)
    signs = np.zeros(3, dtype=np.float64)
    for ax in range(3):
        uu, vv = np.meshgrid(t, t, indexing="ij")
        flat = uu.size
        uvw_lo = np.zeros((flat, 3), dtype=np.float64)
        uvw_hi = np.zeros((flat, 3), dtype=np.float64)
        other = [0, 1, 2]
        other.remove(ax)
        uvw_lo[:, other[0]] = uu.ravel()
        uvw_lo[:, other[1]] = vv.ravel()
        uvw_lo[:, ax] = 0.0
        uvw_hi[:, other[0]] = uu.ravel()
        uvw_hi[:, other[1]] = vv.ravel()
        uvw_hi[:, ax] = 1.0
        sdf_lo = float(
            sdf_field.sample(_trilinear_hex_points(corners, uvw_lo)).mean()
        )
        sdf_hi = float(
            sdf_field.sample(_trilinear_hex_points(corners, uvw_hi)).mean()
        )
        scores[ax] = abs(sdf_hi - sdf_lo)
        signs[ax] = 1.0 if sdf_hi >= sdf_lo else -1.0
    return scores, signs


def estimate_half_cut_orientation(
    cad_mesh: trimesh.Trimesh,
    corners: np.ndarray,
    sdf_field: _EdtSdfField,
    *,
    grad_weak_rel: float = 0.15,
) -> tuple[HalfOrientation, np.ndarray, np.ndarray]:
    """
    Choose a six-way polarity tag for one half-filled hex.

    ``safe_signed_distance`` / EDT SDF is **positive outside**. The gradient
    therefore points toward empty space. Keep the apex on the **solid** side:

      grad_x > 0  → outside is +X → solid is −X → ``Half_Neg_X``
      grad_x < 0  → outside is −X → solid is +X → ``Half_Pos_X``

    (same for Y/Z). Dominant axis from |grad|; weak grad falls back to CAD
    face-normal vote, then signed face-SDF straddle.
    """
    c = np.asarray(corners, dtype=np.float64)
    centroid = c.mean(axis=0)
    edge = float(
        np.mean(
            [
                np.linalg.norm(c[1] - c[0]),
                np.linalg.norm(c[3] - c[0]),
                np.linalg.norm(c[4] - c[0]),
            ]
        )
    )
    delta = max(0.25 * edge, float(sdf_field.resolution))
    grad = sdf_field.gradient(centroid.reshape(1, 3), delta=delta)[0]
    g_abs = np.abs(grad)
    g_norm = float(np.linalg.norm(grad))

    axis = _dominant_axis_from_vector(grad)
    # Outside direction (positive SDF): prefer raw gradient.
    outside_comp = float(grad[axis])

    if g_norm < 1e-12 or (g_abs.max() < grad_weak_rel * max(g_abs.sum(), 1e-12)):
        n_vote = _face_normal_vote_axis(cad_mesh, c, sdf_field)
        if float(np.linalg.norm(n_vote)) > 1e-12:
            axis = _dominant_axis_from_vector(n_vote)
            outside_comp = float(n_vote[axis])
        else:
            scores, signs = _axis_straddle_signed(c, sdf_field)
            axis = int(np.argmax(scores))
            outside_comp = float(signs[axis])

    # outside_comp > 0 → outside along +axis → keep negative apex
    if outside_comp >= 0.0:
        orient = _AXIS_POLARITY[axis][1]  # Neg
        normal_sign = 1.0
    else:
        orient = _AXIS_POLARITY[axis][0]  # Pos
        normal_sign = -1.0

    normal = np.zeros(3, dtype=np.float64)
    normal[axis] = normal_sign  # points toward outside
    return orient, normal, grad


# ---------------------------------------------------------------------------
# State classifier
# ---------------------------------------------------------------------------


def classify_hex_boundary_states(
    cad_mesh: trimesh.Trimesh,
    hex_elems: np.ndarray,
    *,
    volume_fractions: np.ndarray | None = None,
    samples_per_axis: int = 8,
    resolution: float | None = None,
    empty_vf_max: float = EMPTY_VF_MAX,
    full_vf_min: float = FULL_VF_MIN,
    sdf_field: _EdtSdfField | None = None,
) -> HexBoundaryClassification:
    """
    Tag each SC hex as Empty / Half / Full from true VF; orient Half cells.

    Strict Fully-Inside thresholds (Task 24 defaults):
      - Empty: VF <= ``empty_vf_max`` (0.05) — delete almost-outside cells
      - Half:  ``empty_vf_max`` < VF < ``full_vf_min`` — boundary cut cells
      - Full:  VF >= ``full_vf_min`` (0.95) — retain only nearly-solid core

    ``full_vf_min`` may be 1.0 to disable Full (all non-empty cells are Half).

    Half orientation uses SDF gradient polarity (positive outside):
      ``grad_axis > 0`` → ``Half_Neg_*`` (solid on the negative side)
      ``grad_axis < 0`` → ``Half_Pos_*``
    """
    elems = np.asarray(hex_elems, dtype=np.float64)
    if elems.ndim != 3 or elems.shape[1:] != (8, 3):
        raise ValueError(f"hex_elems must have shape (N, 8, 3); got {elems.shape}")
    n_hex = int(elems.shape[0])

    lo = float(empty_vf_max)
    hi = float(full_vf_min)
    if not (0.0 <= lo < hi <= 1.0):
        raise ValueError(
            f"Need 0 <= empty_vf_max < full_vf_min <= 1; got {empty_vf_max}, {full_vf_min}"
        )

    res = _default_voxel_resolution(elems, samples_per_axis, resolution=resolution)
    cad = sanitize_cad_mesh_for_sdf(cad_mesh)
    field = sdf_field if sdf_field is not None else build_edt_sdf_field(cad, res)

    if volume_fractions is None:
        vf_result = estimate_hex_volume_fractions(
            cad,
            elems,
            samples_per_axis=samples_per_axis,
            resolution=field.resolution,
            sdf_field=field,
        )
        vfs = vf_result.volume_fractions
        vf_meta = {
            "samples_per_axis": vf_result.samples_per_axis,
            "voxel_resolution": vf_result.voxel_resolution,
            "n_samples_per_hex": vf_result.n_samples_per_hex,
            "vf_method": vf_result.method,
        }
    else:
        vfs = np.asarray(volume_fractions, dtype=np.float64).reshape(-1)
        if vfs.shape[0] != n_hex:
            raise ValueError(
                f"volume_fractions length {vfs.shape[0]} != n_hex {n_hex}"
            )
        vf_meta = {"vf_method": "caller_supplied"}

    states = np.empty(n_hex, dtype=object)
    orientations = np.empty(n_hex, dtype=object)
    cut_axes = np.full(n_hex, -1, dtype=np.int32)
    cut_normals = np.zeros((n_hex, 3), dtype=np.float64)
    grads = np.zeros((n_hex, 3), dtype=np.float64)

    n_empty = n_half = n_full = 0
    orient_counts = {o.value: 0 for o in HalfOrientation}
    _cut_axis_of = {
        HalfOrientation.HALF_POS_X.value: 0,
        HalfOrientation.HALF_NEG_X.value: 0,
        HalfOrientation.HALF_POS_Y.value: 1,
        HalfOrientation.HALF_NEG_Y.value: 1,
        HalfOrientation.HALF_POS_Z.value: 2,
        HalfOrientation.HALF_NEG_Z.value: 2,
    }

    for i in range(n_hex):
        vf = float(vfs[i])
        # Empty: VF <= lo | Full: VF >= hi | Half: lo < VF < hi
        if vf <= lo:
            states[i] = HexFillState.EMPTY.value
            orientations[i] = None
            n_empty += 1
            continue
        if vf >= hi:
            states[i] = HexFillState.FULL.value
            orientations[i] = None
            n_full += 1
            continue

        states[i] = HexFillState.HALF.value
        orient, normal, grad = estimate_half_cut_orientation(
            cad_mesh, elems[i], field
        )
        orientations[i] = orient.value
        cut_axes[i] = _cut_axis_of[orient.value]
        cut_normals[i] = normal
        grads[i] = grad
        orient_counts[orient.value] += 1
        n_half += 1

    report = {
        **vf_meta,
        "n_hex": n_hex,
        "n_empty": int(n_empty),
        "n_half": int(n_half),
        "n_full": int(n_full),
        "orientation_counts": orient_counts,
        "empty_vf_max": lo,
        "full_vf_min": hi,
        "boundary_policy": "strict_fully_inside",
    }
    return HexBoundaryClassification(
        volume_fractions=vfs,
        states=states,
        orientations=orientations,
        cut_axes=cut_axes,
        cut_normals=cut_normals,
        gradient_at_centroid=grads,
        empty_vf_max=lo,
        full_vf_min=hi,
        report=report,
    )


def tag_sc_hex_boundary_states(
    cad_mesh: trimesh.Trimesh,
    hex_elems: np.ndarray,
    **kwargs,
) -> HexBoundaryClassification:
    """Alias for ``classify_hex_boundary_states`` (pipeline-friendly name)."""
    return classify_hex_boundary_states(cad_mesh, hex_elems, **kwargs)


# Type alias for callers that inject custom SDF samplers later.
SdfSampler = Callable[[np.ndarray], np.ndarray]
