import numpy as np
import trimesh
import sys
from pathlib import Path

# Add repo root to sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from graphite.explicit.supercell_module import (
    generate_cartesian_nodes,
    apply_face_centric_kagome,
    cull_kagome_lattice
)
from graphite.explicit.geometry_module import generate_geometry

def test_visualize_snapping_distortion():
    print("="*60)
    print(" VISUALIZING CONFORMAL SNAPPING DISTORTION ".center(60, '='))
    print("="*60)
    
    # 1. Create a Pyramid mesh
    pyramid = trimesh.creation.cone(radius=20, height=40, sections=4)
    pyramid.apply_translation([0, 0, 20])
    
    Path("outputs").mkdir(exist_ok=True)
    pyramid.export("outputs/Distortion_Test_C_Pyramid_Reference.stl")
    
    L = 10.0 # Unit cell size
    cell_type = "Truncated Octa-Tetra Honeycomb Kagome"
    element_size = L / (1.4142 * 3.0)
    
    bounds = pyramid.bounds
    
    # 2. Generate Base Mathematical Grid
    fcc_nodes = generate_cartesian_nodes(bounds, element_size, cell_type, padding_blocks=1)
    kag_nodes, kag_struts, tet_elements, tet_coords = apply_face_centric_kagome(
        fcc_nodes, element_size, cell_type
    )
    
    print(f"[+] Full Mathematical Grid: {len(kag_nodes)} Nodes, {len(kag_struts)} Struts")
    
    # 3. Formally Cull the Lattice (50% Rule)
    print("[+] Formally culling lattice (Deleting completely/partially exterior elements)...")
    cull_nodes, cull_struts, kept_elements, kept_tet_coords = cull_kagome_lattice(
        kag_nodes, kag_struts, tet_elements, tet_coords, pyramid
    )
    
    print(f"    Culled Matrix: {len(cull_nodes)} Nodes, {len(cull_struts)} Struts.")
    
    # 4. Aggressively morph the surviving mesh to perfectly wrap the surface via Parent Super-Cells
    print("[+] Executing Phase 7: Super-Cell Deformation Morphing...")
    
    from graphite.explicit.supercell_module import apply_supercell_stretching
    
    final_nodes = apply_supercell_stretching(
        kag_nodes, kept_elements, kept_tet_coords, pyramid
    )
    
    # Compress nodes so we only export the culled matrix mapped nodes
    used_nodes = np.unique(cull_struts.ravel())
    cull_nodes_unmorphed = kag_nodes[used_nodes]
    cull_nodes_morphed = final_nodes[used_nodes]
    
    # Remap strut indices
    mapping = np.full(len(kag_nodes), -1, dtype=np.int32)
    mapping[used_nodes] = np.arange(len(used_nodes))
    cull_struts_mapped = mapping[cull_struts]
    
    # 5. Export Final True Geometry
    radius = 0.5
    Path("outputs").mkdir(exist_ok=True)
    
    # Culled un-morphed reference
    mesh_raw = generate_geometry(cull_nodes_unmorphed, cull_struts_mapped, radius)
    mesh_raw.export("outputs/Final_Phase7_A_Culled_Rigid.stl")
    
    # Aggressively Morphed version
    mesh_snapped = generate_geometry(cull_nodes_morphed, cull_struts_mapped, radius)
    out_path = "outputs/Final_Phase7_B_SuperCell_Deformed.stl"
    mesh_snapped.export(out_path)
    
    print(f"    SUCCESS: Saved mathematically perfect conformal wrap to {out_path}")


def test_generate_conformal_lattice_exact_cad_trim():
    from graphite.implicit.conformal import generate_conformal_lattice

    # Create a simple box STL
    box = trimesh.creation.box(extents=(6.0, 6.0, 6.0))
    tmp_path = Path("scratch/test_box_exact_trim.stl")
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    box.export(str(tmp_path))

    # Generate conformal gyroid with exact_cad_trim=True
    mesh_out = generate_conformal_lattice(
        stl_path=tmp_path,
        lattice_type="Gyroid",
        resolution=0.5,
        unit_cell_size=3.0,
        solid_fraction=0.33,
        exact_cad_trim=True,
    )

    assert isinstance(mesh_out, trimesh.Trimesh)
    assert len(mesh_out.faces) > 0
    assert mesh_out.is_watertight
    # Bounding box should stay bounded by original box
    assert np.all(mesh_out.extents <= box.extents + 0.1)


if __name__ == "__main__":
    test_visualize_snapping_distortion()
