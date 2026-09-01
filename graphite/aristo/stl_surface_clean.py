"""
Dual-backend isotropic surface remesh for STL skins before TPMS/gmsh volume meshing.

Open3D (in-process or via an isolated Python 3.12 ``.venv312/`` worker) or native
trimesh fallback when Open3D is unavailable.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import trimesh
from trimesh import repair as trimesh_repair
from trimesh.remesh import subdivide_to_size

from graphite.aristo.aristo_log import aristo_log, aristo_stage

try:
    import open3d as o3d

    HAS_OPEN3D = True
except ImportError:
    o3d = None
    HAS_OPEN3D = False

_VENV312_DIRNAME = ".venv312"
_WORKER_ENV_FLAG = "ARISTO_SURFACE_CLEAN_WORKER"
_WORKER_PACKAGES = ("open3d", "trimesh", "numpy", "scipy")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_venv312_dir(config=None) -> Path:
    if config is not None:
        override = getattr(config, "fea_surface_clean_venv312_dir", None)
        if override:
            return Path(override).expanduser().resolve()
    env_override = os.environ.get("ARISTO_SURFACE_CLEAN_VENV312", "").strip()
    if env_override:
        return Path(env_override).expanduser().resolve()
    return _repo_root() / _VENV312_DIRNAME


def _venv312_python_executable(venv_dir: Path) -> Path:
    if sys.platform == "win32":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _py312_worker_enabled(config=None) -> bool:
    if os.environ.get("ARISTO_SURFACE_CLEAN_DISABLE_PY312", "").strip().lower() in (
        "1",
        "true",
        "yes",
    ):
        return False
    if config is not None and not getattr(
        config, "fea_surface_clean_py312_worker", True
    ):
        return False
    if os.environ.get(_WORKER_ENV_FLAG, "").strip() == "1":
        return False
    return True


def _python312_venv_create_commands(venv_dir: Path) -> list[list[str]]:
    commands: list[list[str]] = []
    if shutil.which("py"):
        commands.append(["py", "-3.12", "-m", "venv", str(venv_dir)])
    if shutil.which("python3.12"):
        commands.append(["python3.12", "-m", "venv", str(venv_dir)])
    if shutil.which("python3.12.exe"):
        commands.append(["python3.12.exe", "-m", "venv", str(venv_dir)])
    return commands


def _venv312_has_open3d(python_exe: Path) -> bool:
    try:
        proc = subprocess.run(
            [str(python_exe), "-c", "import open3d as o3d; print(o3d.__version__)"],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(_repo_root()),
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return proc.returncode == 0


def _create_venv312(venv_dir: Path) -> bool:
    if venv_dir.exists():
        return _venv312_python_executable(venv_dir).is_file()
    venv_dir.parent.mkdir(parents=True, exist_ok=True)
    for cmd in _python312_venv_create_commands(venv_dir):
        aristo_log(f"creating {_VENV312_DIRNAME}: {' '.join(cmd)}")
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(_repo_root()),
            )
        except OSError as exc:
            aristo_log(f"venv create failed ({exc}); trying next launcher")
            continue
        if proc.returncode == 0 and _venv312_python_executable(venv_dir).is_file():
            return True
        aristo_log(
            f"venv create failed ({proc.returncode}): "
            f"{(proc.stderr or proc.stdout or '').strip()}"
        )
    return False


def _install_venv312_packages(python_exe: Path) -> bool:
    aristo_log(f"installing {_WORKER_PACKAGES} into {_VENV312_DIRNAME}")
    try:
        bootstrap = subprocess.run(
            [str(python_exe), "-m", "pip", "install", "-U", "pip"],
            capture_output=True,
            text=True,
            cwd=str(_repo_root()),
        )
        if bootstrap.returncode != 0:
            aristo_log(f"pip bootstrap failed: {(bootstrap.stderr or '').strip()}")
        proc = subprocess.run(
            [str(python_exe), "-m", "pip", "install", *_WORKER_PACKAGES],
            capture_output=True,
            text=True,
            cwd=str(_repo_root()),
        )
    except OSError as exc:
        aristo_log(f"pip install failed: {exc}")
        return False
    if proc.returncode != 0:
        aristo_log(f"pip install failed: {(proc.stderr or proc.stdout or '').strip()}")
        return False
    return _venv312_has_open3d(python_exe)


_VENV312_PYTHON_CACHE: Path | None = None
_VENV312_CACHE_CHECKED = False


def ensure_venv312_worker(config=None, *, force_refresh: bool = False) -> Path | None:
    """
    Ensure ``.venv312`` exists with Open3D installed; return its python executable.

    Returns None when Python 3.12 is unavailable or setup fails.
    """
    global _VENV312_PYTHON_CACHE, _VENV312_CACHE_CHECKED
    if _VENV312_CACHE_CHECKED and not force_refresh:
        return _VENV312_PYTHON_CACHE

    _VENV312_CACHE_CHECKED = True
    _VENV312_PYTHON_CACHE = None

    if not _py312_worker_enabled(config):
        return None
    if HAS_OPEN3D:
        return None

    venv_dir = _resolve_venv312_dir(config)
    python_exe = _venv312_python_executable(venv_dir)
    if not python_exe.is_file():
        if not _create_venv312(venv_dir):
            aristo_log(
                "Python 3.12 not found for Open3D worker "
                "(install Python 3.12 or set ARISTO_SURFACE_CLEAN_DISABLE_PY312=1)"
            )
            return None
        python_exe = _venv312_python_executable(venv_dir)

    if not _venv312_has_open3d(python_exe):
        if not _install_venv312_packages(python_exe):
            return None
    _VENV312_PYTHON_CACHE = python_exe
    return python_exe


def _invoke_clean_stl_surface_worker(
    input_stl: Path,
    output_stl: Path,
    target_edge_mm: float,
    *,
    axis: int = 2,
    config=None,
) -> None:
    python_exe = ensure_venv312_worker(config)
    if python_exe is None:
        raise RuntimeError(
            "Open3D Python 3.12 worker is not available. "
            "Install Python 3.12, or disable ARISTO_SURFACE_CLEAN_DISABLE_PY312=0."
        )

    script = _repo_root() / "scripts" / "clean_stl_surface.py"
    output_stl.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(python_exe),
        str(script),
        str(input_stl),
        "-o",
        str(output_stl),
        "--h",
        str(float(target_edge_mm)),
        "--axis",
        str(int(axis)),
    ]
    env = os.environ.copy()
    env[_WORKER_ENV_FLAG] = "1"
    env["PYTHONPATH"] = str(_repo_root())
    env.setdefault("PYTHONUNBUFFERED", "1")

    aristo_log(f"Open3D worker subprocess: {' '.join(cmd)}")
    with aristo_stage("open3d_py312_worker_subprocess"):
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            cwd=str(_repo_root()),
        )
    if proc.stdout:
        for line in proc.stdout.splitlines():
            aristo_log(f"worker| {line}")
    if proc.returncode != 0:
        raise RuntimeError(
            "Open3D worker subprocess failed "
            f"(exit {proc.returncode}):\n{(proc.stderr or proc.stdout or '').strip()}"
        )
    if not output_stl.is_file():
        raise RuntimeError(f"Open3D worker did not write output STL: {output_stl}")


def surface_clean_backend(config=None) -> str:
    """Return the backend that ``clean_stl_surface`` will use."""
    if HAS_OPEN3D:
        return "open3d"
    if ensure_venv312_worker(config) is not None:
        return "open3d_subprocess"
    return "trimesh"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _remove_duplicate_faces(mesh: trimesh.Trimesh) -> None:
    mesh.update_faces(mesh.unique_faces())


def _remove_degenerate_faces(mesh: trimesh.Trimesh) -> None:
    if hasattr(mesh, "remove_degenerate_faces"):
        mesh.remove_degenerate_faces()
    else:
        mesh.update_faces(mesh.nondegenerate_faces())


def _max_edge_length_trimesh(mesh: trimesh.Trimesh) -> float:
    lengths = mesh.edges_unique_length
    return float(lengths.max()) if len(lengths) else 0.0


def _cap_plane_levels(
    vertices: np.ndarray,
    *,
    axis: int = 2,
) -> tuple[float, float]:
    coords = vertices[:, axis]
    return float(coords.min()), float(coords.max())


def _cap_face_vertex_indices(
    mesh: trimesh.Trimesh,
    *,
    axis: int = 2,
    cap_normal_dot_min: float = 0.95,
    target_edge_mm: float = 0.15,
    plane_tol_factor: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Vertex indices on flat bottom/top cap faces near the bounding planes."""
    face_normals = mesh.face_normals
    centroids = mesh.triangles_center
    z_min, z_max = _cap_plane_levels(np.asarray(mesh.vertices), axis=axis)
    plane_tol = float(target_edge_mm) * float(plane_tol_factor)
    bottom_faces = (face_normals[:, axis] <= -float(cap_normal_dot_min)) & (
        centroids[:, axis] <= z_min + plane_tol
    )
    top_faces = (face_normals[:, axis] >= float(cap_normal_dot_min)) & (
        centroids[:, axis] >= z_max - plane_tol
    )
    bottom_verts = (
        np.unique(mesh.faces[bottom_faces].ravel())
        if np.any(bottom_faces)
        else np.array([], dtype=np.int64)
    )
    top_verts = (
        np.unique(mesh.faces[top_faces].ravel())
        if np.any(top_faces)
        else np.array([], dtype=np.int64)
    )
    return bottom_verts, top_verts


def _project_cap_vertices_trimesh(
    mesh: trimesh.Trimesh,
    *,
    target_edge_mm: float,
    axis: int = 2,
    cap_normal_dot_min: float = 0.95,
    plane_tol_factor: float = 2.0,
    plane_min: float | None = None,
    plane_max: float | None = None,
) -> None:
    """Snap vertices on flat cap faces onto exact bottom/top planes (in-place)."""
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    if verts.size == 0:
        return

    z_min, z_max = _cap_plane_levels(verts, axis=axis)
    if plane_min is not None:
        z_min = float(plane_min)
    if plane_max is not None:
        z_max = float(plane_max)

    bottom_verts, top_verts = _cap_face_vertex_indices(
        mesh,
        axis=axis,
        cap_normal_dot_min=cap_normal_dot_min,
        target_edge_mm=target_edge_mm,
        plane_tol_factor=plane_tol_factor,
    )
    snap_tol = float(target_edge_mm) * 0.1
    if bottom_verts.size:
        near_bottom = np.abs(verts[bottom_verts, axis] - z_min) <= snap_tol
        verts[bottom_verts[near_bottom], axis] = z_min
    if top_verts.size:
        near_top = np.abs(verts[top_verts, axis] - z_max) <= snap_tol
        verts[top_verts[near_top], axis] = z_max
    mesh.vertices = verts


def _laplacian_smooth_plane_caps(
    mesh: trimesh.Trimesh,
    bottom_verts: np.ndarray,
    top_verts: np.ndarray,
    *,
    axis: int = 2,
    plane_min: float,
    plane_max: float,
    lamb: float = 0.3,
    iterations: int = 2,
) -> None:
    """
    In-plane Laplacian smooth on flat cap vertices only (Z held fixed).

    Relaxes needle/sliver corners on caps without moving the lattice skin.
    """
    if iterations <= 0 or lamb <= 0.0:
        return

    cap_verts = np.unique(
        np.concatenate(
            [
                np.asarray(bottom_verts, dtype=np.int64),
                np.asarray(top_verts, dtype=np.int64),
            ]
        )
    )
    if cap_verts.size == 0:
        return

    cap_set = set(int(v) for v in cap_verts)
    adjacency: dict[int, set[int]] = defaultdict(set)
    for tri in mesh.faces:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        adjacency[a].update((b, c))
        adjacency[b].update((a, c))
        adjacency[c].update((a, b))

    # Smooth only cap-interior vertices so rim edges to the lattice skin stay fixed.
    interior_cap = {
        vi
        for vi in cap_set
        if adjacency.get(vi) and all(n in cap_set for n in adjacency[vi])
    }
    if not interior_cap:
        return

    in_plane = [i for i in range(3) if i != axis]
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    for _ in range(int(iterations)):
        updated = verts.copy()
        for vi in interior_cap:
            neighbors = {n for n in adjacency.get(vi, ()) if n in cap_set}
            if not neighbors:
                continue
            nb = np.array(sorted(neighbors), dtype=np.int64)
            mean_nb = verts[nb].mean(axis=0)
            for comp in in_plane:
                updated[vi, comp] = (1.0 - lamb) * verts[vi, comp] + lamb * mean_nb[comp]
        verts = updated
        verts[bottom_verts, axis] = plane_min
        verts[top_verts, axis] = plane_max
    mesh.vertices = verts


def _merge_vertices_clustering(mesh: trimesh.Trimesh, target_edge_mm: float) -> None:
    """Tight vertex clustering (fraction of h) to seal microscopic boundary gaps."""
    span = float(np.max(mesh.extents))
    merge_tol = float(target_edge_mm) * 0.05
    if span <= 0.0 or merge_tol <= 0.0:
        return
    digits = int(np.clip(np.floor(-np.log10(merge_tol / max(span, 1e-9))), 1, 8))
    aristo_log(
        f"trimesh vertex clustering seal: merge_tol~{merge_tol:.5f} mm digits_vertex={digits}"
    )
    mesh.merge_vertices(merge_tex=True, merge_norm=True, digits_vertex=digits)


def _repair_trimesh_topology_after_subdivide(mesh: trimesh.Trimesh) -> None:
    """
    Strict topological repair after ``subdivide_to_size``.

    Merges redundant vertices, removes bad faces, fills holes, and fixes winding.
    """
    mesh.merge_vertices(merge_tex=True, merge_norm=True)
    _remove_duplicate_faces(mesh)
    mesh.remove_infinite_values()
    mesh.remove_unreferenced_vertices()
    _remove_degenerate_faces(mesh)
    trimesh_repair.fill_holes(mesh)
    trimesh_repair.fix_normals(mesh)
    trimesh_repair.fix_inversion(mesh)
    mesh.remove_unreferenced_vertices()


def _force_seal_trimesh_boundaries(mesh: trimesh.Trimesh, target_edge_mm: float) -> None:
    """Last-resort seal when ``is_watertight`` is still False after standard repair."""
    aristo_log("trimesh watertight=False after repair; trying broken_faces")
    trimesh_repair.broken_faces(mesh)
    _remove_duplicate_faces(mesh)
    mesh.remove_infinite_values()
    mesh.remove_unreferenced_vertices()
    _remove_degenerate_faces(mesh)
    if mesh.is_watertight:
        return

    aristo_log("trimesh still open after broken_faces; applying vertex clustering seal")
    _merge_vertices_clustering(mesh, target_edge_mm)
    _remove_duplicate_faces(mesh)
    mesh.remove_unreferenced_vertices()
    _remove_degenerate_faces(mesh)
    trimesh_repair.fill_holes(mesh)
    trimesh_repair.fix_normals(mesh)
    trimesh_repair.fix_inversion(mesh)
    mesh.remove_unreferenced_vertices()


def _diagnose_trimesh_watertight(mesh: trimesh.Trimesh) -> None:
    """Log and warn if the cleaned skin is not watertight for gmsh."""
    euler = int(mesh.euler_number) if mesh.euler_number is not None else None
    if mesh.is_watertight:
        aristo_log(f"trimesh topology check passed: watertight=True euler={euler}")
        return
    warnings.warn(
        "[Aristo] trimesh surface clean: mesh is not watertight after repair "
        f"(euler={euler}, faces={len(mesh.faces):,}, "
        f"winding_consistent={mesh.is_winding_consistent}). "
        "GMSH TPMS meshing may fail with overlapping or open boundaries.",
        stacklevel=3,
    )
    aristo_log(
        f"trimesh topology check FAILED: watertight=False euler={euler} "
        f"faces={len(mesh.faces):,}"
    )


def _prepare_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    m = mesh.copy()
    _remove_duplicate_faces(m)
    _remove_degenerate_faces(m)
    m.remove_unreferenced_vertices()
    return m


# ---------------------------------------------------------------------------
# Open3D backend
# ---------------------------------------------------------------------------


def trimesh_to_o3d(mesh: trimesh.Trimesh):
    if not HAS_OPEN3D:
        raise ImportError("Open3D backend is not available.")
    tm = _prepare_mesh(mesh)
    o3d_mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(tm.vertices, dtype=np.float64)),
        o3d.utility.Vector3iVector(np.asarray(tm.faces, dtype=np.int32)),
    )
    o3d_mesh.remove_duplicated_vertices()
    o3d_mesh.remove_duplicated_triangles()
    o3d_mesh.remove_degenerate_triangles()
    return o3d_mesh


def o3d_to_trimesh(o3d_mesh) -> trimesh.Trimesh:
    verts = np.asarray(o3d_mesh.vertices, dtype=np.float64)
    faces = np.asarray(o3d_mesh.triangles, dtype=np.int64)
    if verts.size == 0 or faces.size == 0:
        raise ValueError("Open3D surface remesh produced an empty mesh.")
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    _remove_duplicate_faces(mesh)
    _remove_degenerate_faces(mesh)
    mesh.remove_unreferenced_vertices()
    return mesh


def _max_edge_length_o3d(o3d_mesh) -> float:
    tris = np.asarray(o3d_mesh.triangles, dtype=np.int64)
    verts = np.asarray(o3d_mesh.vertices, dtype=np.float64)
    if tris.size == 0:
        return 0.0
    v0 = verts[tris[:, 0]]
    v1 = verts[tris[:, 1]]
    v2 = verts[tris[:, 2]]
    lengths = np.concatenate(
        [
            np.linalg.norm(v1 - v0, axis=1),
            np.linalg.norm(v2 - v1, axis=1),
            np.linalg.norm(v0 - v2, axis=1),
        ]
    )
    return float(lengths.max()) if lengths.size else 0.0


def _project_cap_vertices_o3d(
    o3d_mesh,
    *,
    target_edge_mm: float,
    axis: int = 2,
    cap_normal_dot_min: float = 0.95,
    plane_tol_factor: float = 2.0,
    plane_min: float | None = None,
    plane_max: float | None = None,
) -> None:
    verts = np.asarray(o3d_mesh.vertices, dtype=np.float64)
    if verts.size == 0:
        return

    z_min, z_max = _cap_plane_levels(verts, axis=axis)
    if plane_min is not None:
        z_min = float(plane_min)
    if plane_max is not None:
        z_max = float(plane_max)

    o3d_mesh.compute_triangle_normals()
    normals = np.asarray(o3d_mesh.triangle_normals, dtype=np.float64)
    tris = np.asarray(o3d_mesh.triangles, dtype=np.int64)
    if tris.size == 0:
        return

    centroids = verts[tris].mean(axis=1)
    plane_tol = float(target_edge_mm) * float(plane_tol_factor)
    bottom_faces = (normals[:, axis] <= -float(cap_normal_dot_min)) & (
        centroids[:, axis] <= z_min + plane_tol
    )
    top_faces = (normals[:, axis] >= float(cap_normal_dot_min)) & (
        centroids[:, axis] >= z_max - plane_tol
    )
    snap_tol = float(target_edge_mm) * 0.1
    if np.any(bottom_faces):
        bottom_verts = np.unique(tris[bottom_faces].ravel())
        near_bottom = np.abs(verts[bottom_verts, axis] - z_min) <= snap_tol
        verts[bottom_verts[near_bottom], axis] = z_min
    if np.any(top_faces):
        top_verts = np.unique(tris[top_faces].ravel())
        near_top = np.abs(verts[top_verts, axis] - z_max) <= snap_tol
        verts[top_verts[near_top], axis] = z_max
    o3d_mesh.vertices = o3d.utility.Vector3dVector(verts)


def _cluster_to_target_edge(o3d_mesh, target_edge_mm: float):
    voxel_size = float(target_edge_mm) / np.sqrt(2.0)
    return o3d_mesh.simplify_vertex_clustering(
        voxel_size=voxel_size,
        contraction=o3d.geometry.SimplificationContraction.Average,
    )


def _remesh_isotropic_surface_open3d(
    o3d_mesh,
    target_edge_mm: float,
    *,
    axis: int = 2,
    max_remesh_passes: int = 6,
    max_subdiv_per_pass: int = 2,
):
    h = float(target_edge_mm)
    long_edge = 1.6 * h
    mesh = o3d_mesh
    for pass_idx in range(int(max_remesh_passes)):
        max_edge = _max_edge_length_o3d(mesh)
        if max_edge <= long_edge and pass_idx > 0:
            break

        mesh.remove_duplicated_vertices()
        mesh.remove_duplicated_triangles()
        mesh.remove_degenerate_triangles()

        subdivisions = 0
        while _max_edge_length_o3d(mesh) > long_edge and subdivisions < max_subdiv_per_pass:
            mesh = mesh.subdivide_midpoint(number_of_iterations=1)
            subdivisions += 1
            _project_cap_vertices_o3d(mesh, target_edge_mm=h, axis=axis)

        mesh = _cluster_to_target_edge(mesh, h)
        _project_cap_vertices_o3d(mesh, target_edge_mm=h, axis=axis)

        aristo_log(
            f"open3d surface clean pass {pass_idx + 1}: "
            f"verts={len(mesh.vertices):,} tris={len(mesh.triangles):,} "
            f"max_edge={_max_edge_length_o3d(mesh):.4f} mm"
        )

    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    _project_cap_vertices_o3d(mesh, target_edge_mm=h, axis=axis)
    return mesh


def _clean_stl_surface_open3d(
    mesh: trimesh.Trimesh,
    target_edge_mm: float,
    *,
    axis: int = 2,
) -> trimesh.Trimesh:
    o3d_mesh = trimesh_to_o3d(mesh)
    o3d_mesh = _remesh_isotropic_surface_open3d(o3d_mesh, target_edge_mm, axis=axis)
    return o3d_to_trimesh(o3d_mesh)


# ---------------------------------------------------------------------------
# Native trimesh backend
# ---------------------------------------------------------------------------


def _clean_stl_surface_trimesh(
    mesh: trimesh.Trimesh,
    target_edge_mm: float,
    *,
    axis: int = 2,
    subdivide_max_iter: int = 10,
    laplacian_iterations: int = 1,
    laplacian_lambda: float = 0.2,
) -> trimesh.Trimesh:
    h = float(target_edge_mm)
    m = _prepare_mesh(mesh)
    plane_min = float(m.bounds[0][axis])
    plane_max = float(m.bounds[1][axis])

    aristo_log(
        f"trimesh subdivide_to_size max_edge={h:.4f} mm "
        f"(faces_in={len(m.faces):,})"
    )
    verts, faces = subdivide_to_size(
        m.vertices,
        m.faces,
        max_edge=h,
        max_iter=int(subdivide_max_iter),
    )
    m = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    with aristo_stage("trimesh_topology_repair"):
        aristo_log(
            f"trimesh post-subdiv: faces={len(m.faces):,} "
            f"watertight={m.is_watertight} euler={m.euler_number}"
        )
        _repair_trimesh_topology_after_subdivide(m)
        aristo_log(
            f"trimesh after standard repair: watertight={m.is_watertight} "
            f"euler={m.euler_number} faces={len(m.faces):,}"
        )
        if not m.is_watertight:
            _force_seal_trimesh_boundaries(m, h)
        _diagnose_trimesh_watertight(m)

    _project_cap_vertices_trimesh(
        m,
        target_edge_mm=h,
        axis=axis,
        plane_min=plane_min,
        plane_max=plane_max,
    )

    max_edge = _max_edge_length_trimesh(m)
    if max_edge > 1.2 * h:
        bottom_verts, top_verts = _cap_face_vertex_indices(m, axis=axis, target_edge_mm=h)
        aristo_log(
            f"trimesh cap in-plane laplacian iterations={laplacian_iterations} "
            f"lambda={laplacian_lambda} (max_edge={max_edge:.4f} mm > {1.2 * h:.4f})"
        )
        _laplacian_smooth_plane_caps(
            m,
            bottom_verts,
            top_verts,
            axis=axis,
            plane_min=plane_min,
            plane_max=plane_max,
            lamb=float(laplacian_lambda),
            iterations=int(laplacian_iterations),
        )
        _project_cap_vertices_trimesh(
            m,
            target_edge_mm=h,
            axis=axis,
            plane_min=plane_min,
            plane_max=plane_max,
        )

    _remove_duplicate_faces(m)
    _remove_degenerate_faces(m)
    m.remove_unreferenced_vertices()
    _project_cap_vertices_trimesh(
        m,
        target_edge_mm=h,
        axis=axis,
        plane_min=plane_min,
        plane_max=plane_max,
    )

    aristo_log(
        f"trimesh surface clean done: faces={len(m.faces):,} "
        f"max_edge={_max_edge_length_trimesh(m):.4f} mm "
        f"watertight={m.is_watertight} euler={m.euler_number}"
    )
    return m


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _load_trimesh(path: str | Path) -> trimesh.Trimesh:
    mesh = trimesh.load_mesh(str(path), process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = mesh.dump(concatenate=True)
    return mesh


def _clean_stl_surface_open3d_subprocess(
    mesh: trimesh.Trimesh,
    target_edge_mm: float,
    *,
    axis: int = 2,
    config=None,
) -> trimesh.Trimesh:
    with tempfile.TemporaryDirectory(prefix="aristo_o3d_worker_") as tmp:
        inp = Path(tmp) / "input.ply"
        out = Path(tmp) / "output.ply"
        mesh.export(str(inp))
        _invoke_clean_stl_surface_worker(
            inp, out, target_edge_mm, axis=axis, config=config
        )
        return _load_trimesh(out)


def clean_stl_surface(
    mesh: trimesh.Trimesh,
    target_edge_mm: float,
    *,
    axis: int = 2,
    config=None,
) -> trimesh.Trimesh:
    """
    Return a surface mesh with more uniform triangles at length ~h.

    Uses in-process Open3D, an isolated Python 3.12 worker, or native trimesh.
    """
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"mesh must be trimesh.Trimesh, got {type(mesh)}.")
    if float(target_edge_mm) <= 0:
        raise ValueError("target_edge_mm must be positive.")

    backend = surface_clean_backend(config)
    with aristo_stage(f"stl_isotropic_surface_remesh ({backend})"):
        aristo_log(
            f"backend={backend} input: {len(mesh.faces):,} faces, "
            f"max_edge={_max_edge_length_trimesh(mesh):.4f} mm, h={target_edge_mm} mm"
        )
        if backend == "open3d":
            cleaned = _clean_stl_surface_open3d(mesh, target_edge_mm, axis=axis)
        elif backend == "open3d_subprocess":
            cleaned = _clean_stl_surface_open3d_subprocess(
                mesh, target_edge_mm, axis=axis, config=config
            )
        else:
            cleaned = _clean_stl_surface_trimesh(mesh, target_edge_mm, axis=axis)
        aristo_log(
            f"backend={backend} output: {len(cleaned.faces):,} faces, "
            f"max_edge={_max_edge_length_trimesh(cleaned):.4f} mm, "
            f"watertight={cleaned.is_watertight}"
        )
        return cleaned


def clean_stl_surface_file(
    input_path: str | Path,
    output_path: str | Path,
    target_edge_mm: float,
    *,
    axis: int = 2,
    config=None,
) -> Path:
    """Load STL, remesh, and write ``output_path``."""
    input_path = Path(input_path).resolve()
    output_path = Path(output_path).resolve()
    backend = surface_clean_backend(config)
    if backend == "open3d_subprocess":
        _invoke_clean_stl_surface_worker(
            input_path, output_path, target_edge_mm, axis=axis, config=config
        )
        return output_path

    mesh = _load_trimesh(input_path)
    cleaned = clean_stl_surface(mesh, target_edge_mm, axis=axis, config=config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.export(str(output_path))
    return output_path


def clean_stl_surface_to_temp_stl(
    mesh: trimesh.Trimesh,
    target_edge_mm: float,
    *,
    axis: int = 2,
    config=None,
    directory: str | Path | None = None,
    prefix: str = "aristo_clean_surface_",
) -> Path:
    """Remesh and write a temporary STL; caller may delete when finished."""
    directory = Path(directory) if directory is not None else Path(tempfile.gettempdir())
    directory.mkdir(parents=True, exist_ok=True)
    fd, raw_path = tempfile.mkstemp(suffix=".stl", prefix=prefix, dir=str(directory))
    os.close(fd)
    path = Path(raw_path)

    backend = surface_clean_backend(config)
    if backend == "open3d_subprocess":
        with tempfile.NamedTemporaryFile(
            suffix=".stl", prefix="aristo_o3d_in_", delete=False, dir=str(directory)
        ) as tmp_in:
            inp = Path(tmp_in.name)
        try:
            mesh.export(str(inp))
            _invoke_clean_stl_surface_worker(
                inp, path, target_edge_mm, axis=axis, config=config
            )
        finally:
            inp.unlink(missing_ok=True)
        return path

    cleaned = clean_stl_surface(mesh, target_edge_mm, axis=axis, config=config)
    cleaned.export(str(path))
    return path


def clean_stl_surface_file_to_temp_stl(
    input_path: str | Path,
    target_edge_mm: float,
    *,
    axis: int = 2,
    config=None,
    directory: str | Path | None = None,
    prefix: str = "aristo_clean_surface_",
) -> Path:
    """File-to-temp STL clean (preferred for large STLs on Python 3.13)."""
    directory = Path(directory) if directory is not None else Path(tempfile.gettempdir())
    directory.mkdir(parents=True, exist_ok=True)
    fd, raw_path = tempfile.mkstemp(suffix=".stl", prefix=prefix, dir=str(directory))
    os.close(fd)
    path = Path(raw_path)
    clean_stl_surface_file(
        input_path,
        path,
        target_edge_mm,
        axis=axis,
        config=config,
    )
    return path


def stl_surface_clean_enabled(config=None) -> bool:
    """True when config or ``ARISTO_CLEAN_STL_SURFACE`` requests preprocessing."""
    if config is not None and getattr(config, "fea_clean_stl_surface", False):
        return True
    return os.environ.get("ARISTO_CLEAN_STL_SURFACE", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def maybe_clean_stl_surface(
    mesh: trimesh.Trimesh,
    target_edge_mm: float,
    *,
    config=None,
    axis: int = 2,
    input_stl_path: str | Path | None = None,
) -> trimesh.Trimesh:
    """
    Apply surface clean when enabled via config or environment.

    When ``input_stl_path`` is set and the Open3D 3.12 worker is used, cleans via
    subprocess file handoff without re-exporting from an in-memory mesh.
    """
    if not stl_surface_clean_enabled(config):
        return mesh

    backend = surface_clean_backend(config)
    if backend == "open3d_subprocess" and input_stl_path is not None:
        with aristo_stage("stl_surface_clean (open3d_subprocess file handoff)"):
            temp_stl = clean_stl_surface_file_to_temp_stl(
                input_stl_path,
                target_edge_mm,
                axis=axis,
                config=config,
            )
            aristo_log(f"loaded cleaned STL from worker: {temp_stl}")
            return _load_trimesh(temp_stl)

    return clean_stl_surface(mesh, target_edge_mm, axis=axis, config=config)
