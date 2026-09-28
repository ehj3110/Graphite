"""
Unit tests for the Aristo Continuum Bridge and Two-Scale Homogenized FEA.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from graphite.fea.aristo_bridge import (
    MacroMesh,
    TwoScaleFEAResult,
    apply_surface_traction,
    assemble_anisotropic_global_K,
    create_box_continuum_mesh,
    export_two_scale_result_vtk,
    find_boundary_nodes_by_plane,
    generate_macro_continuum_mesh,
    map_grading_field_to_centroids,
    run_two_scale_macro_fea,
)
from graphite.fea.homogenization import (
    EngineeringConstants,
    build_isotropic_material_matrix,
)
from graphite.fea.surrogate import (
    MaterialTensorSurrogate,
    SurrogateCalibrationPoint,
)


def _make_test_surrogate(E_min: float = 100.0, E_max: float = 1000.0, nu: float = 0.0) -> MaterialTensorSurrogate:
    """Create a calibrated surrogate where E scales with param."""
    vals = [0.1, 0.25, 0.5, 0.75, 1.0]
    pts = []
    for v in vals:
        E_eff = E_min + (E_max - E_min) * (v - 0.1) / 0.9
        C = build_isotropic_material_matrix(E_eff, nu)
        S = np.linalg.inv(C)
        consts = EngineeringConstants(
            E_x=float(1.0 / S[0, 0]),
            E_y=float(1.0 / S[1, 1]),
            E_z=float(1.0 / S[2, 2]),
            G_xy=float(1.0 / S[3, 3]),
            G_yz=float(1.0 / S[4, 4]),
            G_zx=float(1.0 / S[5, 5]),
            nu_xy=nu,
            nu_yx=nu,
            nu_xz=nu,
            nu_zx=nu,
            nu_yz=nu,
            nu_zy=nu,
            bulk_modulus=float(np.sum(C[:3, :3]) / 9.0),
            zener_anisotropy=1.0,
        )
        pts.append(
            SurrogateCalibrationPoint(
                param_value=v,
                solid_fraction=v,
                C_tensor=C,
                constants=consts,
            )
        )
    return MaterialTensorSurrogate(
        param_name="phi",
        param_range=(0.1, 1.0),
        sample_points=pts,
        symmetry_type="cubic",
        fitting_method="pchip",
    )


class TestAristoBridge:
    """Test suite for two-scale macro continuum FEA bridge."""

    def test_box_continuum_mesh_generation(self):
        """Structured box mesh must generate valid positive volumes summing to total volume."""
        L, W, H = 20.0, 10.0, 5.0
        subdivs = (8, 4, 2)
        mesh = create_box_continuum_mesh(
            bounds=((0.0, 0.0, 0.0), (L, W, H)),
            subdivisions=subdivs,
            elem_type="tet4",
        )

        assert mesh.nodes.shape[1] == 3
        assert mesh.elements.shape[1] == 4
        assert mesh.elem_type == "tet4"
        assert len(mesh.element_volumes) == len(mesh.elements)
        assert np.all(mesh.element_volumes > 0.0)

        expected_vol = L * W * H
        np.testing.assert_allclose(np.sum(mesh.element_volumes), expected_vol, rtol=1e-5)

        # Centroids within bounding box
        assert np.all(mesh.element_centroids[:, 0] >= 0.0)
        assert np.all(mesh.element_centroids[:, 0] <= L)

    def test_boundary_detection_and_traction(self):
        """Detect boundary faces and verify surface traction integral equals applied load."""
        mesh = create_box_continuum_mesh(
            bounds=((0.0, 0.0, 0.0), (30.0, 10.0, 10.0)),
            subdivisions=(6, 2, 2),
            elem_type="tet4",
        )
        left_nodes = find_boundary_nodes_by_plane(mesh, axis=0, value=0.0)
        right_nodes = find_boundary_nodes_by_plane(mesh, axis=0, value=30.0)

        assert len(left_nodes) > 0
        assert len(right_nodes) > 0
        assert len(np.intersect1d(left_nodes, right_nodes)) == 0

        # Apply 500 N force in X direction
        F_total = (500.0, 0.0, 0.0)
        F_vec = apply_surface_traction(mesh, right_nodes, F_total)

        assert F_vec.shape == (3 * len(mesh.nodes),)
        # Sum of X components must equal 500.0
        assert pytest.approx(np.sum(F_vec[::3])) == 500.0
        assert pytest.approx(np.sum(F_vec[1::3])) == 0.0
        assert pytest.approx(np.sum(F_vec[2::3])) == 0.0

    def test_uniform_tensile_bar_analytical_limit(self):
        """Uniform bar under uniaxial tension must match analytical delta = F*L / (A*E)."""
        L = 40.0
        W, H = 4.0, 4.0
        A = W * H
        E_val = 2000.0
        F_val = 800.0

        surrogate = _make_test_surrogate(E_min=E_val, E_max=E_val, nu=0.0)
        mesh = create_box_continuum_mesh(
            bounds=((0.0, 0.0, 0.0), (L, W, H)),
            subdivisions=(20, 2, 2),
            elem_type="tet4",
        )

        left_nodes = find_boundary_nodes_by_plane(mesh, axis=0, value=0.0)
        right_nodes = find_boundary_nodes_by_plane(mesh, axis=0, value=L)
        F_vec = apply_surface_traction(mesh, right_nodes, (F_val, 0.0, 0.0))

        result = run_two_scale_macro_fea(
            mesh=mesh,
            surrogate=surrogate,
            grading_source=0.5,
            fixed_nodes=left_nodes,
            forces=F_vec,
            fixed_components=(0, 1, 2),
        )

        u_tip = np.mean(result.displacements[right_nodes, 0])
        u_analytical = (F_val * L) / (A * E_val)

        # 3D continuum with tet4 elements matches 1D bar within 0.5%
        assert pytest.approx(u_analytical, rel=0.005) == u_tip
        # Strain energy U = 0.5 * F * u
        assert pytest.approx(0.5 * F_val * u_tip, rel=0.005) == result.compliance_energy

    def test_graded_tensile_bar_analytical_solution(self):
        """Functionally graded bar must match closed-form logarithmic tip deflection."""
        L = 50.0
        W, H = 5.0, 5.0
        A = W * H
        E0 = 200.0
        EL = 1000.0
        F_val = 1000.0

        # Linear E(x) through linear phi(x) from 0.1 to 1.0
        surrogate = _make_test_surrogate(E_min=E0, E_max=EL, nu=0.0)
        mesh = create_box_continuum_mesh(
            bounds=((0.0, 0.0, 0.0), (L, W, H)),
            subdivisions=(25, 2, 2),
            elem_type="tet4",
        )

        def grading_fn(coords: np.ndarray) -> np.ndarray:
            x = coords[:, 0]
            return 0.1 + (1.0 - 0.1) * (x / L)

        left_nodes = find_boundary_nodes_by_plane(mesh, axis=0, value=0.0)
        right_nodes = find_boundary_nodes_by_plane(mesh, axis=0, value=L)
        F_vec = apply_surface_traction(mesh, right_nodes, (F_val, 0.0, 0.0))

        result = run_two_scale_macro_fea(
            mesh=mesh,
            surrogate=surrogate,
            grading_source=grading_fn,
            fixed_nodes=left_nodes,
            forces=F_vec,
            fixed_components=(0, 1, 2),
        )

        u_tip_fem = float(np.mean(result.displacements[right_nodes, 0]))
        # Analytical 1D solution: u(L) = (F * L) / (A * (EL - E0)) * ln(EL / E0)
        u_tip_analytical = (F_val * L) / (A * (EL - E0)) * np.log(EL / E0)

        # Must agree within 1.5% for discretized 3D tets
        assert pytest.approx(u_tip_analytical, rel=0.015) == u_tip_fem
        assert result.max_displacement > 0.0
        assert result.max_von_mises > 0.0

    def test_vtk_export_roundtrip(self, tmp_path: Path):
        """Export TwoScaleFEAResult to VTU and verify attributes via PyVista."""
        import pyvista as pv

        surrogate = _make_test_surrogate(E_min=500.0, E_max=1500.0, nu=0.3)
        mesh = create_box_continuum_mesh(
            bounds=((0.0, 0.0, 0.0), (10.0, 10.0, 10.0)),
            subdivisions=(4, 4, 4),
            elem_type="tet4",
        )
        fixed = find_boundary_nodes_by_plane(mesh, axis=2, value=0.0)
        top = find_boundary_nodes_by_plane(mesh, axis=2, value=10.0)
        F_vec = apply_surface_traction(mesh, top, (0.0, 0.0, -200.0))

        result = run_two_scale_macro_fea(
            mesh=mesh,
            surrogate=surrogate,
            grading_source=0.5,
            fixed_nodes=fixed,
            forces=F_vec,
        )

        vtu_path = tmp_path / "test_macro_result.vtu"
        export_two_scale_result_vtk(result, vtu_path)
        assert vtu_path.is_file()

        # Load back with PyVista
        grid = pv.read(str(vtu_path))
        assert grid.n_points == len(mesh.nodes)
        assert grid.n_cells == len(mesh.elements)
        assert "displacement_mm" in grid.point_data
        assert "von_mises_nodal_MPa" in grid.point_data
        assert "von_mises_elem_MPa" in grid.cell_data
        assert "grading_field" in grid.cell_data

    def test_gmsh_cad_envelope_meshing(self):
        """Generate macro continuum mesh from CAD Trimesh envelope via Gmsh."""
        import trimesh

        box = trimesh.creation.box((12.0, 12.0, 12.0))
        macro_mesh = generate_macro_continuum_mesh(box, target_element_size=4.0)

        assert macro_mesh.nodes.shape[0] > 10
        assert macro_mesh.elements.shape[0] > 10
        assert macro_mesh.elem_type == "tet4"
        assert np.all(macro_mesh.element_volumes > 0.0)
        # Volume within 5% of CAD box volume (12^3 = 1728)
        assert pytest.approx(1728.0, rel=0.05) == np.sum(macro_mesh.element_volumes)
