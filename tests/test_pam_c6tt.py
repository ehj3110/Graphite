"""
Unit tests for C-6-TT (Cubic Truncated Tetrahedral) 3D bulk PAM (Zhou et al., Science 2025).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from graphite.explicit.geometry_module import _trimesh_to_manifold
from graphite.explicit.interlinked import (
    generate_truncated_tetrahedron_particle,
    generate_c6tt_cubic_tiling,
    generate_pam_lattice,
    export_pam_multibody_stl,
    recalibrate_pam_lattice,
)


ROOT = Path(__file__).resolve().parents[1]
OUT_C6TT = ROOT / "outputs" / "c6tt_3d_bulk.stl"


class TestC6TTParticle:
    def test_truncated_tetrahedron_topology(self):
        p = generate_truncated_tetrahedron_particle(size=10.0, center=(1.0, 2.0, 3.0), particle_id=5)
        assert p.geometry_type == "TT"
        assert p.nodes.shape == (12, 3)
        assert p.struts.shape == (18, 2)
        assert p.particle_id == 5
        np.testing.assert_allclose(p.center, (1.0, 2.0, 3.0))

        # Check tuple unpacking support: nodes, struts = p
        nodes, struts = p
        assert nodes.shape == (12, 3)
        assert struts.shape == (18, 2)

        # Edge lengths should all be identical: 2/3 * size
        expected_edge = (2.0 / 3.0) * 10.0
        for i, j in p.struts:
            edge_dist = float(np.linalg.norm(p.nodes[i] - p.nodes[j]))
            assert np.isclose(edge_dist, expected_edge, atol=1e-4)


class TestC6TTBulkLattice:
    @pytest.fixture(scope="class")
    def tiling_result(self):
        return generate_c6tt_cubic_tiling(
            repeats=(2, 2, 2),
            size=10.0,
            strut_radius=0.50,
            min_clearance=0.30,
            build_meshes=True,
        )

    def test_tiling_particle_count_and_code(self, tiling_result):
        assert tiling_result.tripartite_code == "C-6-TT"
        assert len(tiling_result.particles) == 8
        assert len(tiling_result.meshes) == 8
        assert tiling_result.clearance_valid

    def test_surface_clearance_dfam_criterion(self, tiling_result):
        # Minimum surface clearance must exceed 0.30 mm
        assert tiling_result.min_clearance_mm >= 0.30
        assert tiling_result.min_clearance_mm > 0.60

    def test_zero_solid_manifold_collision(self, tiling_result):
        # Convert meshes to Manifold3D and compute all pairwise Boolean intersections
        manifolds = [_trimesh_to_manifold(m) for m in tiling_result.meshes]
        for m in tiling_result.meshes:
            assert m.is_watertight

        for i in range(len(manifolds)):
            for j in range(i + 1, len(manifolds)):
                intersection = manifolds[i] ^ manifolds[j]
                inter_vol = float(intersection.volume())
                assert inter_vol <= 1e-4, f"Pair ({i}, {j}) collided with volume {inter_vol}"

    def test_multibody_stl_export(self, tiling_result):
        out_path = export_pam_multibody_stl(tiling_result, OUT_C6TT)
        assert out_path.is_file()
        assert out_path.stat().st_size > 1000

    def test_generate_pam_lattice_router(self):
        res = generate_pam_lattice("C-6-TT", unit_cell_size=10.0, repeats=(2, 2, 2), strut_radius=0.50, min_clearance=0.30)
        assert res.tripartite_code == "C-6-TT"
        assert res.clearance_valid
        assert len(res.particles) == 8

    def test_recalibrate_pam_lattice_c6tt(self):
        best_a0 = recalibrate_pam_lattice("C-6-TT", target_d=10.0, r=0.50, repeats=(2, 2, 2))
        assert best_a0 > 10.0
        assert 11.5 <= best_a0 <= 14.0

    def test_generate_c6tt_lattice_alias(self):
        from graphite.explicit.interlinked import generate_c6tt_lattice
        res = generate_c6tt_lattice(repeats=(1, 1, 1), size=10.0, strut_radius=0.50, build_meshes=False)
        assert res.tripartite_code == "C-6-TT"
        assert len(res.particles) == 1


class TestContactMechanicsAndJamming:
    def test_classify_particle_contact(self):
        from graphite.explicit.interlinked import (
            generate_truncated_tetrahedron_particle,
            classify_particle_contact,
            analyze_interparticle_contact_manifold,
        )
        p1 = generate_truncated_tetrahedron_particle(size=10.0, center=(0.0, 0.0, 0.0), particle_id=0)
        p2 = generate_truncated_tetrahedron_particle(size=10.0, center=(12.5, 0.0, 0.0), particle_id=1)

        # Outer vertex contact (near edge of p1 and p2) -> tensile
        pt_tensile = np.array([6.25, 3.0, 3.0], dtype=np.float64)
        c_type = classify_particle_contact(p1, p2, contact_point=pt_tensile, outer_fraction=0.60)
        assert c_type == "tensile"

        # Inner cavity contact (near center of particle) -> compressive
        pt_comp = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        c_type_comp = classify_particle_contact(p1, p2, contact_point=pt_comp, outer_fraction=0.60)
        assert c_type_comp == "compressive"

        # Detailed analysis
        analysis = analyze_interparticle_contact_manifold(p1, p2, strut_radius=0.50)
        assert "contact_type" in analysis
        assert "clearance_mm" in analysis
        assert "contact_point" in analysis
        assert analysis["contact_type"] in ("tensile", "compressive")

    def test_jamming_power_laws(self):
        from graphite.explicit.interlinked import (
            compute_jammed_bending_modulus,
            compute_jamming_compressive_modulus,
        )
        import json

        # Unjammed state: Z < Z0
        assert compute_jammed_bending_modulus(Z_avg=4.0, grid_rotation_deg=0.0) == 0.0
        assert compute_jammed_bending_modulus(Z_avg=5.0, grid_rotation_deg=45.0) == 0.0  # Z0 is 5.29

        # Jammed state: Z >= Z0
        E_bend_0 = compute_jammed_bending_modulus(Z_avg=6.0, grid_rotation_deg=0.0)
        assert E_bend_0 > 0.0
        # Formula: 0.159 * (6.0 - 4.89)^2.348 ≈ 0.159 * 1.11^2.348 ≈ 0.203
        assert np.isclose(E_bend_0, 0.159 * ((6.0 - 4.89) ** 2.348), atol=1e-3)

        # 45-degree rotation increases threshold Z0 to 5.29, lowering modulus for same Z
        E_bend_45 = compute_jammed_bending_modulus(Z_avg=6.0, grid_rotation_deg=45.0)
        assert 0.0 < E_bend_45 < E_bend_0

        # Zhou et al. compressive jamming: E* = prefactor * (Z - Z0)^n
        assert compute_jamming_compressive_modulus(Z=4.5, Z0=5.0) == 0.0
        E_comp = compute_jamming_compressive_modulus(Z=7.0, Z0=5.0, n=1.0, prefactor=2.5)
        assert np.isclose(E_comp, 5.0)

        # Export reviewable contact & jamming report
        out_report = ROOT / "outputs" / "phase1_c6tt_contact_analysis.json"
        report_data = {
            "tripartite_code": "C-6-TT",
            "wang_nature_2021": {
                "Z_critical_0deg": 4.89,
                "Z_critical_45deg": 5.29,
                "E_bend_at_Z6_0deg_MPa": float(E_bend_0),
                "E_bend_at_Z6_45deg_MPa": float(E_bend_45),
            },
            "zhou_science_2025": {
                "Z0": 5.0,
                "n": 1.0,
                "E_comp_at_Z7": float(E_comp),
            },
        }
        with open(out_report, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)
        assert out_report.is_file()
