"""
Mesh export utilities — STL (trimesh) and faceted STEP (Gmsh mesh-to-CAD).

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
class ExportResult:
    """Paths and metadata from ``export_mesh``."""

    paths_written: list[Path] = field(default_factory=list)
    face_count: int = 0
    vertex_count: int = 0
    watertight: bool = False
    step_notes: list[str] = field(default_factory=list)


def formats_from_request(export_format: str) -> tuple[str, ...]:
    """
    Parse UI/config export format strings.

    Accepts: ``stl``, ``step``, ``both`` (case-insensitive), or labels like
    ``STL only``, ``STEP only``, ``STL + STEP``.
    """
    key = str(export_format).strip().lower()
    if key in ("stl", "stl only"):
        return ("stl",)
    if key in ("step", "step only", "stp"):
        return ("step",)
    if key in ("both", "stl + step", "stl+step", "stl and step"):
        return ("stl", "step")
    raise ValueError(
        f"Unknown export_format {export_format!r}; use 'stl', 'step', or 'both'."
    )


def resolve_export_formats(
    output_path: str | Path | None,
    export_formats: tuple[str, ...] | str | None = None,
) -> tuple[str, ...]:
    """Resolve export formats from explicit request and/or output path suffix."""
    if export_formats is not None:
        if isinstance(export_formats, str):
            return formats_from_request(export_formats)
        normalized = tuple(str(f).lower() for f in export_formats)
        for fmt in normalized:
            if fmt not in ("stl", "step"):
                raise ValueError(f"Unsupported export format: {fmt!r}")
        return normalized
    if output_path is None:
        return ()
    suffix = Path(output_path).suffix.lower()
    if suffix in (".step", ".stp"):
        return ("step",)
    return ("stl",)


def _output_stem(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in (".stl", ".step", ".stp"):
        return path.stem
    return path.name


def _step_export_in_subprocess() -> bool:
    """True when Gmsh must not run in the current thread (e.g. Streamlit)."""
    return threading.current_thread() is not threading.main_thread()


def repair_mesh_for_export(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Light repair pass before CAD export."""
    out = mesh.copy()
    out.update_faces(out.unique_faces())
    out.update_faces(out.nondegenerate_faces())
    out.remove_unreferenced_vertices()
    trimesh.repair.fix_normals(out)
    trimesh.repair.fill_holes(out)
    return out


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
    formats: tuple[str, ...] | str | None = None,
    step_options: StepExportOptions | None = None,
) -> ExportResult:
    """
    Export a lattice mesh to one or more file formats.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Lattice or boundary mesh to write.
    path : path-like
        Base output path. Extension may be ``.stl``, ``.step``, or ``.stp``; when
        multiple formats are requested, the stem is used and per-format extensions
        are appended.
    formats : tuple of str or str, optional
        ``("stl",)``, ``("step",)``, ``("stl", "step")``, or a request string
        (``stl`` / ``step`` / ``both``). If None, inferred from ``path`` suffix.
    step_options : StepExportOptions, optional
        Gmsh conversion settings when STEP is requested.

    Returns
    -------
    ExportResult
    """
    path = Path(path)
    resolved = resolve_export_formats(path, formats)
    if not resolved:
        raise ValueError("No export formats resolved; pass formats= or a path with a suffix.")

    stem = _output_stem(path)
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)

    paths_written: list[Path] = []
    step_notes: list[str] = []

    if "stl" in resolved:
        stl_path = parent / f"{stem}.stl"
        mesh.export(str(stl_path))
        paths_written.append(stl_path)

    if "step" in resolved:
        step_path = parent / f"{stem}.step"
        step_notes = mesh_to_step_gmsh(mesh, step_path, step_options)
        paths_written.append(step_path)

    return ExportResult(
        paths_written=paths_written,
        face_count=len(mesh.faces),
        vertex_count=len(mesh.vertices),
        watertight=bool(mesh.is_watertight),
        step_notes=step_notes,
    )
