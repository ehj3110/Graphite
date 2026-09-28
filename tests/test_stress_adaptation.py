"""
Unit tests for the closed-loop stress-adaptive topology optimization engine (Fully Stressed Design).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from graphite.fea.aristo_bridge import (
    apply_surface_traction,
    create_box_continuum_mesh,
    find_boundary_nodes_by_plane,
)
from graphite.fea.stress_adaptation import (
    OptimizationIterationRecord,
    StressAdaptationConfig,
    TwoScaleOptimizationResult,
    apply_volume_bisection_scaling,
    build_neighborhood_filter,
    export_optimization_result_vtk,
    optimize_lattice_density_fsd,
)
from graphite.fea.surrogate import MaterialTensorSurrogate


@pytest.fixture
def calibrated_surrogate() -> MaterialTensorSurrogate:
    """Load or provide calibrated Gyroid sheet surrogate."""
    path = Path("outputs/fea/gyroid_sheet_surrogate.json")
    if path.exists():
        return MaterialTensorSurrogate.load(str(path))

    # Fast fallback surrogate for tests if file not found
    from graphite.fea.homogenization import EngineeringConstants
    from graphite.fea.surrogate import SurrogateCalibrationPoint

    pts = []
    base_E = 2000.0
    for phi in [0.10, 0.20, 0.35, 0.50, 0.65]:
        C = np.eye(6) * (base_E * (phi**1.8))
        C[0, 1] = C[0, 2] = C[1, 2] = C[0, 0] * 0.3
        C[1, 0] = C[2, 0] = C[2, 1] = C[0, 1]
        c = EngineeringConstants(
            E_x=base_E * phi,
            E_y=base_E * phi,
            E_z=base_E * phi,
            G_xy=base_E * phi * 0.35,
            G_yz=base_E * phi * 0.35,
            G_zx=base_E * phi * 0.35,
            nu_xy=0.3,
            nu_yx=0.3,
            nu_xz=0.3,
            nu_zx=0.3,
            nu_yz=0.3,
            nu_zy=0.3,
            bulk_modulus=base_E * phi,
            zener_anisotropy=1.1,
        )
        pts.append(SurrogateCalibrationPoint(phi, C, c))

    return MaterialTensorSurrogate.fit(
        lattice_type="Gyroid",
        sample_points=pts,
        symmetry_mode="cubic",
        fitting_method="pchip",
    )


class TestStressAdaptationEngine:
    """Test mathematical correctness, convergence, and physics of stress adaptation."""

    def test_neighborhood_filter_properties(self):
        """Filter matrix must be a valid partition of unity and preserve uniform fields."""
        mesh = create_box_continuum_mesh(
            bounds=((0.0, 0.0, 0.0), (10.0, 4.0, 4.0)),
            subdivisions=(5, 2, 2),
            elem_type="tet4",
        )
        M = mesh.elements.shape[0]

        W = build_neighborhood_filter(mesh, filter_radius=3.0)

        assert W.shape == (M, M)
        # Partition of unity: each row must sum to 1.0
        row_sums = np.array(W.sum(axis=1)).flatten()
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-12)

        # Applying filter to uniform field must be invariant
        phi_uniform = np.full(M, 0.30)
        phi_filtered = W.dot(phi_uniform)
        np.testing.assert_allclose(phi_filtered, phi_uniform, atol=1e-12)

    def test_volume_bisection_scaling(self):
        """Bisection volume scaling must strictly enforce target volume fraction."""
        M = 100
        np.random.seed(42)
        densities = np.random.uniform(0.15, 0.55, M)
        element_volumes = np.random.uniform(1.0, 5.0, M)

        target_vf = 0.32
        min_d, max_d = 0.10, 0.60

        scaled = apply_volume_bisection_scaling(
            densities=densities,
            element_volumes=element_volumes,
            target_volume_fraction=target_vf,
            min_density=min_d,
            max_density=max_d,
        )

        assert np.all(scaled >= min_d - 1e-12)
        assert np.all(scaled <= max_d + 1e-12)

        resulting_vf = float(np.sum(element_volumes * scaled) / np.sum(element_volumes))
        np.testing.assert_allclose(resulting_vf, target_vf, atol=1e-5)

    def test_uniform_tensile_bar_fsd_convergence(self, calibrated_surrogate):
        """In pure uniform tension, FSD must converge quickly toward a uniform target stress."""
        # 20 x 4 x 4 mm bar
        mesh = create_box_continuum_mesh(
            bounds=((0.0, 0.0, 0.0), (20.0, 4.0, 4.0)),
            subdivisions=(10, 2, 2),
            elem_type="tet4",
        )

        # Clamped at x = 0
        left_nodes = find_boundary_nodes_by_plane(mesh, axis=0, value=0.0, tol=1e-3)
        # Tension at x = 20
        right_nodes = find_boundary_nodes_by_plane(mesh, axis=0, value=20.0, tol=1e-3)
        f_ext = apply_surface_traction(mesh, right_nodes, total_force=np.array([400.0, 0.0, 0.0]))

        # Target stress: choose a target stress of 25 MPa
        config = StressAdaptationConfig(
            target_stress=25.0,
            target_volume_fraction=None,
            relaxation_eta=0.35,
            move_limit=0.10,
            min_density=0.10,
            max_density=0.60,
            max_iterations=20,
            convergence_tol=1e-3,
            filter_radius=0.0,
        )

        result = optimize_lattice_density_fsd(
            mesh=mesh,
            surrogate=calibrated_surrogate,
            fixed_nodes=left_nodes,
            forces=f_ext,
            config=config,
            initial_densities=0.25,
        )

        # Must complete and record telemetry
        assert result.iterations_completed > 1
        assert len(result.history) == result.iterations_completed
        assert result.total_time_s > 0.0

        # Mean von Mises stress should converge to target_stress within 5%
        final_mean_vm = result.history[-1].mean_von_mises
        np.testing.assert_allclose(final_mean_vm, config.target_stress, rtol=0.05)

        # Densities in the unconstrained uniform section (x > 14 mm, away from clamped face)
        # must remain uniform according to Saint-Venant's principle
        c = mesh.element_centroids
        interior = np.where(c[:, 0] > 14.0)[0]
        densities_int = result.optimal_densities[interior]
        std_phi = float(np.std(densities_int))
        mean_phi = float(np.mean(densities_int))
        assert (std_phi / mean_phi) < 0.10

    def test_volume_fraction_conservation_in_optimization(self, calibrated_surrogate):
        """When target_volume_fraction is set, every iteration must conserve total mass."""
        mesh = create_box_continuum_mesh(
            bounds=((0.0, 0.0, 0.0), (15.0, 5.0, 5.0)),
            subdivisions=(6, 2, 2),
            elem_type="tet4",
        )

        left_nodes = find_boundary_nodes_by_plane(mesh, axis=0, value=0.0, tol=1e-3)
        right_nodes = find_boundary_nodes_by_plane(mesh, axis=0, value=15.0, tol=1e-3)
        f_ext = apply_surface_traction(mesh, right_nodes, total_force=np.array([0.0, -300.0, 0.0]))

        target_vf = 0.35
        config = StressAdaptationConfig(
            target_stress=40.0,
            target_volume_fraction=target_vf,
            relaxation_eta=0.35,
            move_limit=0.08,
            min_density=0.12,
            max_density=0.58,
            max_iterations=8,
            filter_radius=0.0,
        )

        result = optimize_lattice_density_fsd(
            mesh=mesh,
            surrogate=calibrated_surrogate,
            fixed_nodes=left_nodes,
            forces=f_ext,
            config=config,
            initial_densities=target_vf,
        )

        # Verify volume fraction in history matches target_vf
        for rec in result.history:
            np.testing.assert_allclose(rec.volume_fraction, target_vf, atol=1e-4)

    def test_cantilever_beam_bending_gradient(self, calibrated_surrogate):
        """In cantilever bending, FSD must allocate higher density to top/bottom surfaces than the core."""
        # 30 x 6 x 6 mm beam
        mesh = create_box_continuum_mesh(
            bounds=((0.0, -3.0, -3.0), (30.0, 3.0, 3.0)),
            subdivisions=(10, 4, 4),
            elem_type="tet4",
        )

        # Clamped at x = 0
        left_nodes = find_boundary_nodes_by_plane(mesh, axis=0, value=0.0, tol=1e-3)
        # Tip downward load at x = 30
        right_nodes = find_boundary_nodes_by_plane(mesh, axis=0, value=30.0, tol=1e-3)
        f_ext = apply_surface_traction(mesh, right_nodes, total_force=np.array([0.0, 0.0, -500.0]))

        config = StressAdaptationConfig(
            target_stress=30.0,
            target_volume_fraction=0.30,
            relaxation_eta=0.30,
            move_limit=0.08,
            min_density=0.10,
            max_density=0.60,
            max_iterations=12,
            filter_radius=2.5,
        )

        result = optimize_lattice_density_fsd(
            mesh=mesh,
            surrogate=calibrated_surrogate,
            fixed_nodes=left_nodes,
            forces=f_ext,
            config=config,
            initial_densities=0.30,
        )

        # Check bending gradient: elements near root (x < 10) outer flanges (|z| > 1.5)
        # should have significantly higher density than elements near neutral axis (|z| < 1.0)
        c = mesh.element_centroids
        root_outer = np.where((c[:, 0] < 12.0) & (np.abs(c[:, 2]) > 1.5))[0]
        root_core = np.where((c[:, 0] < 12.0) & (np.abs(c[:, 2]) < 0.8))[0]

        mean_outer_density = np.mean(result.optimal_densities[root_outer])
        mean_core_density = np.mean(result.optimal_densities[root_core])

        assert mean_outer_density > mean_core_density

    def test_json_and_vtu_export(self, calibrated_surrogate, tmp_path):
        """Verify summary telemetry export to JSON and ParaView VTU export."""
        mesh = create_box_continuum_mesh(
            bounds=((0.0, 0.0, 0.0), (10.0, 4.0, 4.0)),
            subdivisions=(4, 2, 2),
            elem_type="tet4",
        )
        left_nodes = find_boundary_nodes_by_plane(mesh, axis=0, value=0.0, tol=1e-3)
        right_nodes = find_boundary_nodes_by_plane(mesh, axis=0, value=10.0, tol=1e-3)
        f_ext = apply_surface_traction(mesh, right_nodes, total_force=np.array([200.0, 0.0, 0.0]))

        config = StressAdaptationConfig(max_iterations=3)
        result = optimize_lattice_density_fsd(
            mesh=mesh,
            surrogate=calibrated_surrogate,
            fixed_nodes=left_nodes,
            forces=f_ext,
            config=config,
        )

        # JSON save
        json_path = tmp_path / "opt_summary.json"
        result.save(json_path)
        assert json_path.exists()

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert "history" in data
        assert len(data["history"]) == result.iterations_completed

        # VTU export
        vtu_path = tmp_path / "opt_result.vtu"
        export_optimization_result_vtk(result, vtu_path)
        assert vtu_path.exists()
        assert vtu_path.stat().st_size > 0

    def test_png_rendering_and_tpms_realization(self, calibrated_surrogate, tmp_path):
        """Verify automated PNG rendering and physical TPMS STL realization."""
        from graphite.fea.stress_adaptation import (
            realize_optimized_tpms_lattice,
            render_optimization_summary_png,
        )

        mesh = create_box_continuum_mesh(
            bounds=((0.0, 0.0, 0.0), (10.0, 4.0, 4.0)),
            subdivisions=(4, 2, 2),
            elem_type="tet4",
        )
        left_nodes = find_boundary_nodes_by_plane(mesh, axis=0, value=0.0, tol=1e-3)
        right_nodes = find_boundary_nodes_by_plane(mesh, axis=0, value=10.0, tol=1e-3)
        f_ext = apply_surface_traction(mesh, right_nodes, total_force=np.array([200.0, 0.0, 0.0]))

        config = StressAdaptationConfig(max_iterations=2)
        result = optimize_lattice_density_fsd(
            mesh=mesh,
            surrogate=calibrated_surrogate,
            fixed_nodes=left_nodes,
            forces=f_ext,
            config=config,
        )

        # 1. Test PNG rendering
        png_3d = tmp_path / "opt_3d.png"
        png_conv = tmp_path / "opt_conv.png"
        p3d, pconv = render_optimization_summary_png(result, png_3d, png_conv)
        assert p3d.exists()
        assert p3d.stat().st_size > 0
        assert pconv is not None and pconv.exists()
        assert pconv.stat().st_size > 0

        # 2. Test physical TPMS STL generation
        stl_path = tmp_path / "opt_lattice.stl"
        lattice_mesh = realize_optimized_tpms_lattice(
            result,
            lattice_type="Gyroid",
            cell_size=5.0,
            resolution=(30, 15, 15),
            out_stl=stl_path,
            taubin_iterations=5,
        )
        assert stl_path.exists()
        assert stl_path.stat().st_size > 0
        assert len(lattice_mesh.vertices) > 0
        assert len(lattice_mesh.faces) > 0

    def test_clean_miter_strut_realization(self, calibrated_surrogate, tmp_path):
        """Verify explicit strut lattice generation with clean mitered truss joints."""
        from graphite.fea.stress_adaptation import realize_optimized_strut_lattice

        mesh = create_box_continuum_mesh(
            bounds=((0.0, 0.0, 0.0), (4.0, 4.0, 4.0)),
            subdivisions=(2, 2, 2),
            elem_type="tet4",
        )
        left_nodes = find_boundary_nodes_by_plane(mesh, axis=0, value=0.0, tol=1e-3)
        right_nodes = find_boundary_nodes_by_plane(mesh, axis=0, value=4.0, tol=1e-3)
        f_ext = apply_surface_traction(mesh, right_nodes, total_force=np.array([100.0, 0.0, 0.0]))

        config = StressAdaptationConfig(max_iterations=2)
        result = optimize_lattice_density_fsd(
            mesh=mesh,
            surrogate=calibrated_surrogate,
            fixed_nodes=left_nodes,
            forces=f_ext,
            config=config,
            initial_densities=0.20,
        )

        stl_path = tmp_path / "opt_octet.stl"
        strut_mesh = realize_optimized_strut_lattice(
            result,
            rule_name="octet",
            cell_size=2.0,
            out_stl=stl_path,
            clean_miter=True,
            circular_segments=8,
        )

        assert stl_path.exists()
        assert stl_path.stat().st_size > 0
        assert strut_mesh.is_watertight
        assert len(strut_mesh.vertices) > 0
        assert len(strut_mesh.faces) > 0

