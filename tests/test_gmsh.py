from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.meshing.generate_conformal_lattices import generate_base_scaffold


def test_gmsh_volume_generation():
    heavy_stl = ROOT / "top_part_new_BMeshRepaired_Repaired.stl"
    if not heavy_stl.exists():
        pytest.skip(
            "Heavy integration STL is missing: "
            "top_part_new_BMeshRepaired_Repaired.stl. "
            "Restore it to run this integration test."
        )

    nodes, tets, surface_faces = generate_base_scaffold(str(heavy_stl), target_size=25.0)

    assert len(nodes) > 0
    assert len(tets) > 0
    assert len(surface_faces) > 0
