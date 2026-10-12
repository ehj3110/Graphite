"""
tests/test_pam_turnkey_cli.py

Unit tests for the beginner-friendly turnkey PAM interface:
  - quick_lattice / quick_fabric
  - InterlinkedLatticeResult (.summary, .export, .render)
  - CLI runner (python -m graphite.explicit.interlinked / scripts/generate_pam.py)
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest
import numpy as np

from graphite.explicit.interlinked import (
    quick_lattice,
    quick_fabric,
    chainmail,
    space_fabric,
    shock_absorber,
    impact_cushion,
    PRESETS,
    InterlinkedRegistry,
    InterlinkedLatticeResult,
)
from graphite.explicit.interlinked.cli import main as cli_main


def test_quick_lattice_preset_shock_absorber(tmp_path: Path):
    """Test generating a small 3D shock absorber lattice with preset and 3MF export."""
    out_3mf = tmp_path / "shock.3mf"
    result = quick_lattice(
        preset="shock_absorber",
        grid_size=(2, 2, 2),
        strut_diameter=0.75,
        clearance=0.20,
        export_path=out_3mf,
    )
    assert isinstance(result, InterlinkedLatticeResult)
    assert result.num_rings == 64  # 2x2x2 Diamond cubic supercell has 64 basis sites
    assert result.clearance_valid is True
    assert result.min_clearance >= 0.19  # ~0.20 mm within numerical tolerance
    assert out_3mf.exists()
    assert out_3mf.stat().st_size > 1000  # valid instanced 3MF package


def test_quick_lattice_bounding_box_fit():
    """Test automatic grid repetition calculation from a target bounding box in mm."""
    result = quick_lattice(
        cell="c6tt",
        bounding_box=(25.0, 25.0, 25.0),
        strut_diameter=0.75,
        clearance=0.20,
    )
    # c6tt pitch is ~7.23 mm for d=0.75, Delta=0.20 -> 25 / 7.23 ~= 3.46 -> 3 or 4 repetitions
    assert result.num_rings > 0
    assert result.clearance_valid is True
    summary_text = result.summary()
    assert "C-6-TT" in summary_text or "c6tt" in summary_text.lower()
    assert "Minimum Clearance:" in summary_text


def test_quick_fabric_space_fabric(tmp_path: Path):
    """Test generating a 2D NASA space fabric sheet with quick_fabric."""
    out_stl = tmp_path / "fabric.stl"
    result = quick_fabric(
        pattern="space_fabric",
        width=30.0,
        length=30.0,
        strut_diameter=0.80,
        clearance=0.25,
        export_path=out_stl,
    )
    assert result.num_rings > 0
    assert out_stl.exists()
    assert out_stl.stat().st_size > 0


def test_result_render_preview(tmp_path: Path):
    """Test .render() method producing a matte scientific render image."""
    result = quick_lattice(
        cell="c6tt",
        grid_size=(1, 1, 1),
        strut_diameter=0.75,
        clearance=0.20,
    )
    out_png = tmp_path / "test_render.png"
    result.render(out_png, title="Test Specimen", cutaway=False)
    assert out_png.exists()
    assert out_png.stat().st_size > 1000


def test_cli_dry_run_json(capsys):
    """Test CLI execution in dry-run mode with JSON output."""
    rc = cli_main(["--preset", "shock_absorber", "--box", "40", "40", "40", "--dry-run", "--json"])
    assert rc == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["cell"] == "d4tet"
    assert payload["preset"] == "shock_absorber"
    assert payload["unit_cell_pitch_mm"] > 9.0
    assert payload["cage_diameter_mm"] > 6.0
    assert payload["is_feasible"] is True


def test_cli_list_presets(capsys):
    """Test CLI --list-presets flag."""
    rc = cli_main(["--list-presets"])
    assert rc == 0
    captured = capsys.readouterr()
    assert "shock_absorber" in captured.out
    assert "space_fabric" in captured.out
    assert "chainmail" in captured.out


def test_cli_list_cells(capsys):
    """Test CLI --list-cells flag."""
    rc = cli_main(["--list-cells"])
    assert rc == 0
    captured = capsys.readouterr()
    assert "d4tet" in captured.out
    assert "c6tt" in captured.out
    assert "j4ring" in captured.out


def test_risk_profiles_low_and_high():
    """Verify that low_risk and high_risk profiles calibrate correct strut and clearance values."""
    res_low = quick_lattice("d4tet", grid_size=(1, 1, 1), profile="low_risk")
    assert pytest.approx(res_low.wire_radius, 1e-3) == 0.500  # 1.0 mm diameter
    assert pytest.approx(res_low.min_clearance, 1e-2) == 0.200

    res_high = quick_lattice("d4tet", grid_size=(1, 1, 1), profile="high_risk")
    assert pytest.approx(res_high.wire_radius, 1e-3) == 0.375  # 0.75 mm diameter
    assert pytest.approx(res_high.min_clearance, 1e-2) == 0.150


def test_convenience_one_liners(tmp_path: Path):
    """Test dedicated 1-liner functions chainmail, space_fabric, shock_absorber, impact_cushion."""
    f_maille = tmp_path / "maille.3mf"
    res_m = chainmail(width=30, length=30, export_path=f_maille)
    assert res_m.num_rings > 0
    assert res_m.clearance_valid is True
    assert f_maille.exists()

    res_shock = shock_absorber(size=25, profile="high_risk")
    assert res_shock.num_rings > 0
    assert res_shock.clearance_valid is True
    assert res_shock.min_clearance >= 0.14

    res_cushion = impact_cushion(size=(25, 25, 25), profile="low_risk")
    assert res_cushion.num_rings > 0
    assert res_cushion.clearance_valid is True


def test_colloquial_alias_resolution():
    """Test forgiving cell argument resolution with common aliases."""
    res_tet = quick_lattice("tet", grid_size=(1, 1, 1))
    assert "d-4-tet" in res_tet.metadata.get("cell_name", "").lower() or "d4tet" in res_tet.metadata.get("cell_name", "").lower()

    res_cushion = quick_lattice("cushion", grid_size=(1, 1, 1))
    assert "c-6-tt" in res_cushion.metadata.get("cell_name", "").lower() or "c6tt" in res_cushion.metadata.get("cell_name", "").lower()


def test_natural_sizing_scalar_and_2d():
    """Test natural size argument handling (scalar cube and 2D sheet)."""
    res_scalar = quick_lattice("d4tet", size=25.0)
    assert res_scalar.num_rings > 0

    res_2d = quick_lattice("chainmail", size=(30.0, 30.0))
    assert res_2d.num_rings > 0


def test_discovery_helpers():
    """Test list_presets, list_cells, and describe_cell helpers."""
    from graphite.explicit.interlinked import list_presets, list_cells, describe_cell

    presets = list_presets()
    assert len(presets) >= 4
    preset_names = [p["preset"] for p in presets]
    assert "chainmail" in preset_names
    assert "shock_absorber" in preset_names

    cells = list_cells()
    assert len(cells) >= 6

    info = describe_cell("shock")
    assert info["tag"] == "d4tet"
    assert info["coordination_number"] == 4
    assert info["kappa"] > 0.0


def test_cli_positional_and_profile(capsys):
    """Test CLI with positional target, --profile high_risk, and --size."""
    rc = cli_main(["shock_absorber", "--profile", "high_risk", "--size", "30", "30", "30", "--dry-run", "--json"])
    assert rc == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["cell"] == "d4tet"
    assert payload["risk_profile"] == "high_risk"
    assert pytest.approx(payload["wire_diameter_mm"], 1e-3) == 0.75
    assert pytest.approx(payload["clearance_mm"], 1e-3) == 0.15

