"""
Mesh export utilities — STL (trimesh), 3MF (trimesh), and faceted STEP (Gmsh mesh-to-CAD).

STEP output builds a tessellated solid via Gmsh surface classification; it is
not an analytic NURBS B-rep suitable for parametric feature editing.

Gmsh uses Python signal handlers and must run in the main interpreter thread.
Streamlit (and other hosts) execute user code on a worker thread, so STEP export
automatically spawns ``python -m graphite.io.step_export_worker`` in that case.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import trimesh

# Default cap for STEP conversion (very large lattices can exhaust RAM/time).
_DEFAULT_MAX_FACES_FOR_STEP = 5_000_000


@dataclass(frozen=True)
class StepExportOptions:
    """Options for Gmsh-based mesh → STEP conversion."""

    feature_angle_deg: float = 45.0
    max_faces: int | None = _DEFAULT_MAX_FACES_FOR_STEP
    repair_mesh: bool = True
    geometry_tolerance: float | None = None
    silent_gmsh: bool = True


@dataclass
class MeshHealthReport:
    """Detailed topological and geometric health report for a mesh."""

    vertex_count: int = 0
    face_count: int = 0
    euler_characteristic: int = 0
    is_watertight: bool = False
    boundary_edges: int = 0
    boundary_loops: int = 0
    non_manifold_edges: int = 0
    non_manifold_vertices: int = 0
    self_intersections: int | None = None
    is_export_ready: bool = False
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class MeshRepairConfig:
    """Options for automated mesh repair and validation prior to export."""

    auto_repair: bool = True
    repair_mode: Literal["gentle", "full", "explicit", "implicit", "none"] = "gentle"
    escalate_to_full: bool = False
    poisson_depth: int = 10
    max_non_manifold_edges: int = 0
    max_boundary_loops: int = 0
    check_intersections: bool = False
    log_health: bool = True


@dataclass
class ExportResult:
    """Paths and metadata from ``export_mesh``."""

    paths_written: list[Path] = field(default_factory=list)
    face_count: int = 0
    vertex_count: int = 0
    watertight: bool = False
    step_notes: list[str] = field(default_factory=list)
    health_report: MeshHealthReport | None = None


def formats_from_request(export_format: str) -> tuple[str, ...]:
    """
    Parse UI/config export format strings.

    Accepts: ``stl``, ``step``, ``3mf``, ``both`` (case-insensitive), comma-separated
    combinations like ``stl,3mf``, or labels like ``STL only``, ``STEP only``, ``3MF only``.
    """
    key = str(export_format).strip().lower()
    if "," in key:
        parts = [p.strip() for p in key.split(",") if p.strip()]
        res: list[str] = []
        for p in parts:
            res.extend(formats_from_request(p))
        return tuple(dict.fromkeys(res))
    if key in ("stl", "stl only"):
        return ("stl",)
    if key in ("step", "step only", "stp"):
        return ("step",)
    if key in ("3mf", "3mf only"):
        return ("3mf",)
    if key in ("both", "stl + step", "stl+step", "stl and step"):
        return ("stl", "step")
    raise ValueError(
        f"Unknown export_format {export_format!r}; use 'stl', 'step', '3mf', 'both', or comma-separated list."
    )


def resolve_export_formats(
    output_path: str | Path | None,
    export_formats: Sequence[str] | str | None = None,
) -> tuple[str, ...]:
    """Resolve export formats from explicit request and/or output path suffix."""
    if export_formats is not None:
        if isinstance(export_formats, str):
            return formats_from_request(export_formats)
        normalized = tuple(str(f).lower() for f in export_formats)
        for fmt in normalized:
            if fmt not in ("stl", "step", "3mf"):
                raise ValueError(f"Unsupported export format: {fmt!r}")
        return normalized
    if output_path is None:
        return ()
    suffix = Path(output_path).suffix.lower()
    if suffix in (".step", ".stp"):
        return ("step",)
    if suffix == ".3mf":
        return ("3mf",)
    return ("stl",)


def _output_stem(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in (".stl", ".step", ".stp", ".3mf"):
        return path.stem
    return path.name


def _step_export_in_subprocess() -> bool:
    """True when Gmsh must not run in the current thread (e.g. Streamlit)."""
    return threading.current_thread() is not threading.main_thread()


def compute_mesh_health(
    mesh: trimesh.Trimesh,
    *,
    check_intersections: bool = False,
) -> MeshHealthReport:
    """
    Compute topological and manifold metrics for a mesh prior to serialization.

    Evaluates:
    - vertex and face counts
    - Euler characteristic: chi = V - E + F
    - Watertight status
    - Boundary edge and boundary loop counts
    - Non-manifold edge count
    - Non-manifold vertex count
    - Optional self-intersections (via PyMeshLab)
    """
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)

    v_count = int(len(mesh.vertices))
    f_count = int(len(mesh.faces))

    if v_count == 0 or f_count == 0:
        return MeshHealthReport(
            vertex_count=v_count,
            face_count=f_count,
            euler_characteristic=0,
            is_watertight=False,
            boundary_edges=0,
            boundary_loops=0,
            non_manifold_edges=0,
            non_manifold_vertices=0,
            self_intersections=0 if check_intersections else None,
            is_export_ready=False,
            notes=["Empty mesh"],
        )

    # Edge analysis via numpy
    edges = np.sort(mesh.edges, axis=1)
    unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
    e_unique_count = int(len(unique_edges))

    # Euler characteristic chi = V - E_unique + F
    euler_char = int(v_count - e_unique_count + f_count)

    # Boundary edges (count == 1)
    boundary_edge_mask = counts == 1
    boundary_edges_count = int(np.sum(boundary_edge_mask))

    # Boundary loops via connected components on boundary edges
    boundary_loops_count = 0
    if boundary_edges_count > 0:
        b_edges = unique_edges[boundary_edge_mask]
        adj: dict[int, list[int]] = {}
        for u, v in b_edges:
            u_i, v_i = int(u), int(v)
            adj.setdefault(u_i, []).append(v_i)
            adj.setdefault(v_i, []).append(u_i)
        visited: set[int] = set()
        for node in adj:
            if node not in visited:
                boundary_loops_count += 1
                q = [node]
                visited.add(node)
                while q:
                    curr = q.pop()
                    for nei in adj[curr]:
                        if nei not in visited:
                            visited.add(nei)
                            q.append(nei)

    # Non-manifold edges (incident to > 2 faces)
    non_manifold_edges_count = int(np.sum(counts > 2))

    # Non-manifold vertices (pinch vertices)
    non_manifold_verts_count = 0
    used_pymeshlab = False
    try:
        import pymeshlab
        ms = pymeshlab.MeshSet()
        pm = pymeshlab.Mesh(
            vertex_matrix=np.asarray(mesh.vertices, dtype=np.float64),
            face_matrix=np.asarray(mesh.faces, dtype=np.int32),
        )
        ms.add_mesh(pm)
        meas = ms.get_topological_measures()
        nm_v = meas.get("non_two_manifold_vertices", -1)
        if nm_v != -1:
            non_manifold_verts_count = int(nm_v)
            used_pymeshlab = True
    except Exception:
        used_pymeshlab = False

    if not used_pymeshlab:
        vf = mesh.vertex_faces
        for v_idx in range(v_count):
            f_ids = vf[v_idx]
            f_ids = f_ids[f_ids >= 0]
            if len(f_ids) <= 1:
                continue
            sub_faces = mesh.faces[f_ids]
            opp_edges: list[tuple[int, int]] = []
            for face in sub_faces:
                other = [int(idx) for idx in face if idx != v_idx]
                if len(other) == 2:
                    opp_edges.append((min(other), max(other)))
            v_adj: dict[int, list[int]] = {}
            v_nodes: set[int] = set()
            for a, b in opp_edges:
                v_adj.setdefault(a, []).append(b)
                v_adj.setdefault(b, []).append(a)
                v_nodes.add(a)
                v_nodes.add(b)
            v_visited: set[int] = set()
            comps = 0
            for n in v_nodes:
                if n not in v_visited:
                    comps += 1
                    q = [n]
                    v_visited.add(n)
                    while q:
                        curr = q.pop()
                        for nei in v_adj[curr]:
                            if nei not in v_visited:
                                v_visited.add(nei)
                                q.append(nei)
            if comps > 1:
                non_manifold_verts_count += 1

    intersections = None
    if check_intersections:
        try:
            from graphite.repair.repair_suite import check_intersections_pymeshlab
            with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as tmp:
                tmp_stl = tmp.name
            try:
                mesh.export(tmp_stl)
                intersections = check_intersections_pymeshlab(tmp_stl)
            finally:
                if os.path.exists(tmp_stl):
                    os.remove(tmp_stl)
        except Exception:
            intersections = None

    is_watertight = bool(mesh.is_watertight and boundary_edges_count == 0)
    is_export_ready = (
        is_watertight
        and boundary_edges_count == 0
        and boundary_loops_count == 0
        and non_manifold_edges_count == 0
        and non_manifold_verts_count == 0
        and (intersections is None or intersections == 0)
    )

    notes: list[str] = []
    if not is_watertight:
        notes.append(f"Not watertight ({boundary_edges_count} boundary edges in {boundary_loops_count} loops)")
    if non_manifold_edges_count > 0:
        notes.append(f"{non_manifold_edges_count} non-manifold edges")
    if non_manifold_verts_count > 0:
        notes.append(f"{non_manifold_verts_count} non-manifold pinch vertices")
    if intersections is not None and intersections > 0:
        notes.append(f"{intersections} self-intersecting faces")

    return MeshHealthReport(
        vertex_count=v_count,
        face_count=f_count,
        euler_characteristic=euler_char,
        is_watertight=is_watertight,
        boundary_edges=boundary_edges_count,
        boundary_loops=boundary_loops_count,
        non_manifold_edges=non_manifold_edges_count,
        non_manifold_vertices=non_manifold_verts_count,
        self_intersections=intersections,
        is_export_ready=is_export_ready,
        notes=notes,
    )


def repair_mesh_for_export(
    mesh: trimesh.Trimesh,
    config: MeshRepairConfig | None = None,
) -> trimesh.Trimesh:
    """Repair pass before export (STL/STEP)."""
    cfg = config or MeshRepairConfig()
    if not cfg.auto_repair or cfg.repair_mode == "none":
        return mesh.copy()

    from graphite.repair.repair_suite import repair_trimesh
    repaired, _ = repair_trimesh(
        mesh,
        mode=cfg.repair_mode,
        escalate_to_full=cfg.escalate_to_full,
        poisson_depth=cfg.poisson_depth,
    )
    return repaired


def gmsh_stl_file_to_step(
    stl_path: str | Path,
    output_path: str | Path,
    options: StepExportOptions | None = None,
) -> list[str]:
    """
    Convert an STL file to faceted STEP using Gmsh (in-process; main thread only).

    Prefer :func:`mesh_to_step_gmsh` for meshes; this is used by the subprocess worker.
    """
    try:
        import gmsh
    except ImportError as exc:
        raise RuntimeError(
            "Gmsh is required for STEP export. Install with: pip install gmsh"
        ) from exc

    opts = options or StepExportOptions()
    stl_path = Path(stl_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    angle_rad = float(np.deg2rad(opts.feature_angle_deg))
    gmsh_initialized = False
    notes: list[str] = []
    mesh = trimesh.load(str(stl_path), force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = mesh.dump(concatenate=True)
    face_count = len(mesh.faces)

    try:
        gmsh.initialize()
        gmsh_initialized = True
        if opts.silent_gmsh:
            gmsh.option.setNumber("General.Terminal", 0)

        if opts.geometry_tolerance is not None:
            gmsh.option.setNumber("Geometry.Tolerance", float(opts.geometry_tolerance))
        else:
            lo = np.min(mesh.vertices, axis=0)
            hi = np.max(mesh.vertices, axis=0)
            diag = float(np.linalg.norm(hi - lo))
            tol = max(1e-9, min(1e-2, diag * 1e-5))
            gmsh.option.setNumber("Geometry.Tolerance", tol)

        gmsh.model.add("graphite_lattice")
        gmsh.merge(str(stl_path))
        gmsh.model.geo.synchronize()

        gmsh.model.mesh.classifySurfaces(angle_rad, True, True, angle_rad)
        try:
            gmsh.model.mesh.createGeometry()
        except Exception:
            # Fallback for noisy/organic triangulations where reparametrization fails.
            gmsh.model.mesh.createTopology(True, True)
        gmsh.model.geo.synchronize()

        surfaces = gmsh.model.getEntities(2)
        if not surfaces:
            raise RuntimeError(
                "Gmsh found no surfaces after classification. "
                "Try coarser resolution or verify the lattice mesh is closed."
            )

        existing_vols = gmsh.model.getEntities(3)
        if not existing_vols:
            surface_loop = gmsh.model.geo.addSurfaceLoop([s[1] for s in surfaces])
            gmsh.model.geo.addVolume([surface_loop])
            gmsh.model.geo.synchronize()
            existing_vols = gmsh.model.getEntities(3)
        if not existing_vols:
            raise RuntimeError(
                "Gmsh could not create any 3D volume entities for STEP export."
            )

        gmsh.model.occ.synchronize()
        gmsh.write(str(output_path))

        notes.append(
            f"Wrote faceted STEP ({len(surfaces)} classified surfaces, "
            f"{face_count:,} input faces)."
        )
    finally:
        if gmsh_initialized:
            gmsh.finalize()

    return notes


def _mesh_to_step_subprocess(
    stl_path: Path,
    output_path: Path,
    options: StepExportOptions,
) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "graphite.io.step_export_worker",
        str(stl_path),
        str(output_path),
        "--feature-angle-deg",
        str(options.feature_angle_deg),
    ]
    if options.geometry_tolerance is not None:
        cmd.extend(["--geometry-tolerance", str(options.geometry_tolerance)])

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(
            "STEP export subprocess failed"
            + (f": {detail}" if detail else ".")
        )
    stdout = (result.stdout or "").strip()
    if stdout:
        try:
            notes = json.loads(stdout)
            if isinstance(notes, list):
                return [str(n) for n in notes]
        except json.JSONDecodeError:
            pass
    return []


def mesh_to_step_gmsh(
    mesh: trimesh.Trimesh,
    output_path: str | Path,
    options: StepExportOptions | None = None,
) -> list[str]:
    """
    Convert a triangle mesh to faceted STEP using Gmsh.

    Uses a subprocess when not on the main thread (Streamlit-safe).

    Returns diagnostic notes (empty on success).
    """
    opts = options or StepExportOptions()
    notes: list[str] = []

    work_mesh = repair_mesh_for_export(mesh) if opts.repair_mesh else mesh.copy()
    if opts.max_faces is not None and len(work_mesh.faces) > int(opts.max_faces):
        raise ValueError(
            f"Mesh has {len(work_mesh.faces):,} faces, exceeding STEP export limit "
            f"({opts.max_faces:,}). Coarsen voxel resolution or enable simplification."
        )
    if not work_mesh.is_watertight:
        notes.append(
            "Mesh is not watertight after repair; Gmsh classification may fail."
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    temp_stl: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as tmp:
            temp_stl = tmp.name
        work_mesh.export(temp_stl)
        stl_path = Path(temp_stl)

        worker_opts = StepExportOptions(
            feature_angle_deg=opts.feature_angle_deg,
            max_faces=opts.max_faces,
            repair_mesh=False,
            geometry_tolerance=opts.geometry_tolerance,
            silent_gmsh=opts.silent_gmsh,
        )

        if _step_export_in_subprocess():
            gmsh_notes = _mesh_to_step_subprocess(stl_path, output_path, worker_opts)
        else:
            gmsh_notes = gmsh_stl_file_to_step(stl_path, output_path, worker_opts)

        notes.extend(gmsh_notes)
    finally:
        if temp_stl is not None and os.path.exists(temp_stl):
            os.remove(temp_stl)

    return notes


def export_mesh(
    mesh: trimesh.Trimesh,
    path: str | Path,
    *,
    formats: Sequence[str] | str | None = None,
    step_options: StepExportOptions | None = None,
    repair_config: MeshRepairConfig | None = None,
) -> ExportResult:
    """
    Export a lattice mesh to one or more file formats with automated repair and verification.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Lattice or boundary mesh to write.
    path : path-like
        Base output path. Extension may be ``.stl``, ``.3mf``, ``.step``, or ``.stp``; when
        multiple formats are requested, the stem is used and per-format extensions
        are appended.
    formats : sequence of str or str, optional
        ``("stl",)``, ``("step",)``, ``("3mf",)``, ``("stl", "3mf")``, ``("stl", "step")``,
        or a request string (``stl`` / ``step`` / ``3mf`` / ``both``). If None, inferred from ``path`` suffix.
    step_options : StepExportOptions, optional
        Gmsh conversion settings when STEP is requested.
    repair_config : MeshRepairConfig, optional
        Options controlling automated repair, escalation, and health reporting.
        Defaults to ``MeshRepairConfig()`` (gentle triage enabled).

    Returns
    -------
    ExportResult
    """
    path = Path(path)
    resolved = resolve_export_formats(path, formats)
    if not resolved:
        raise ValueError("No export formats resolved; pass formats= or a path with a suffix.")

    repair_cfg = repair_config or MeshRepairConfig()

    # Pre-export automated repair
    if repair_cfg.auto_repair and repair_cfg.repair_mode != "none":
        from graphite.repair.repair_suite import repair_trimesh_gentle, repair_trimesh_poisson

        work_mesh = repair_trimesh_gentle(mesh)
        repair_notes = ["Applied gentle triage (dedup, winding, normals, hole filling)."]

        # Health check after gentle triage
        health = compute_mesh_health(
            work_mesh,
            check_intersections=repair_cfg.check_intersections,
        )

        # Escalate to full (Poisson) if requested or if gentle triage left defects and escalate_to_full=True
        should_escalate = (
            repair_cfg.repair_mode in ("full", "implicit")
            or (repair_cfg.escalate_to_full and not health.is_export_ready)
        )
        if should_escalate:
            try:
                repair_notes.append(
                    f"Deploying Screened Poisson reconstruction (depth={repair_cfg.poisson_depth})."
                )
                work_mesh = repair_trimesh_poisson(work_mesh, depth=repair_cfg.poisson_depth)
                repair_notes.append("Completed Screened Poisson reconstruction.")
                health = compute_mesh_health(
                    work_mesh,
                    check_intersections=repair_cfg.check_intersections,
                )
            except Exception as exc:
                repair_notes.append(f"Poisson reconstruction failed ({exc}); retaining gentle triage mesh.")
    else:
        work_mesh = mesh.copy()
        repair_notes = []
        health = compute_mesh_health(
            work_mesh,
            check_intersections=repair_cfg.check_intersections,
        )

    if repair_cfg.log_health:
        status_tag = "[PASS]" if health.is_export_ready else "[WARN]"
        print(
            f"{status_tag} Mesh health verification: "
            f"V={health.vertex_count:,} F={health.face_count:,} "
            f"chi={health.euler_characteristic} "
            f"watertight={health.is_watertight} "
            f"boundary_edges={health.boundary_edges} "
            f"boundary_loops={health.boundary_loops} "
            f"nm_edges={health.non_manifold_edges} "
            f"nm_verts={health.non_manifold_vertices}"
        )
        if not health.is_export_ready and health.notes:
            for note in health.notes:
                print(f"       Health note: {note}")

    if not health.is_export_ready:
        import warnings

        warnings.warn(
            f"Exported mesh is not strictly 2-manifold or watertight: {'; '.join(health.notes)}. "
            "Slicers such as Formlabs PreForm may flag this file as broken.",
            UserWarning,
            stacklevel=2,
        )

    stem = _output_stem(path)
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)

    paths_written: list[Path] = []
    step_notes: list[str] = []

    if "stl" in resolved:
        stl_path = parent / f"{stem}.stl"
        work_mesh.export(str(stl_path))
        paths_written.append(stl_path)

    if "3mf" in resolved:
        threemf_path = parent / f"{stem}.3mf"
        work_mesh.export(str(threemf_path), file_type="3mf")
        paths_written.append(threemf_path)

    if "step" in resolved:
        step_path = parent / f"{stem}.step"
        step_notes = mesh_to_step_gmsh(work_mesh, step_path, step_options)
        paths_written.append(step_path)

    all_notes = repair_notes + step_notes
    return ExportResult(
        paths_written=paths_written,
        face_count=len(work_mesh.faces),
        vertex_count=len(work_mesh.vertices),
        watertight=bool(health.is_watertight),
        step_notes=all_notes,
        health_report=health,
    )
