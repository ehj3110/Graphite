from pathlib import Path
import sys

import trimesh


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.meshing.generate_conformal_lattices import extract_tet_struts
from src.meshing.generate_conformal_lattices import generate_base_scaffold
from src.meshing.generate_conformal_lattices import sweep_to_manifold
from src.repair.repair_suite import repair_stl


def test_pipeline_smoke(tmp_path):
    box = trimesh.creation.box(extents=[10, 10, 10])
    raw_stl = tmp_path / "smoke_box.stl"
    box.export(raw_stl)

    repair_stl(str(raw_stl))
    repaired_stl = tmp_path / "smoke_box_Repaired.stl"
    assert repaired_stl.exists(), "Repair suite did not emit repaired STL."

    nodes, tets, _surface_faces = generate_base_scaffold(str(repaired_stl), target_size=5.0)
    assert len(nodes) > 0
    assert len(tets) > 0

    struts = extract_tet_struts(tets)
    assert len(struts) > 0

    lattice = sweep_to_manifold(nodes, struts, radius=0.4)
    lattice_stl = tmp_path / "smoke_box_lattice.stl"
    lattice.export(lattice_stl)

    assert lattice_stl.exists(), "Lattice export file was not created."
    lattice_mesh = trimesh.load_mesh(lattice_stl)
    assert len(lattice_mesh.vertices) > 0
