# -*- coding: utf-8 -*-
"""
Generate two 5mm cube gyroids for comparison:
  - F_less_than_t.stl  : solid = F < t  (25% SF)
  - F_greater_than_t.stl: solid = F > t  (75% SF)

1 unit cell = L = 5 mm.  Saves to the same directory as this script.
"""

import os
import numpy as np
import trimesh

# Import from main generator
from General_Lattice_Generator import (
    gyroid_field,
    gyroid_gradient_field,
    gyroid_isovalue,
    dual_contour,
)

CUBE_MM = 5.0
SOLID_FRAC = 0.25
VOXELS_PER_PERIOD = 32

def main():
    L_mm = CUBE_MM
    res = L_mm / VOXELS_PER_PERIOD
    t = gyroid_isovalue(SOLID_FRAC)  # t = -0.75 for 25%

    # Grid: 0 to CUBE_MM in each dimension, with boundary padding
    n = int(np.ceil(CUBE_MM / res)) + 2
    origin = np.array([-res, -res, -res])
    xi = origin[0] + np.arange(n) * res
    yi = origin[1] + np.arange(n) * res
    zi = origin[2] + np.arange(n) * res
    X, Y, Z = np.meshgrid(xi, yi, zi, indexing='ij')

    F = gyroid_field(X, Y, Z, L_mm).astype(np.float64)
    del X, Y, Z

    # Create 5mm cube for boolean
    cube = trimesh.creation.box(extents=[CUBE_MM, CUBE_MM, CUBE_MM])
    cube.vertices += np.array([CUBE_MM/2, CUBE_MM/2, CUBE_MM/2])  # center at 2.5,2.5,2.5

    out_dir = os.path.dirname(os.path.abspath(__file__))

    t_75 = gyroid_isovalue(0.75)  # t = 0.75 for 75% solid with F > t

    for name, iso, use_negated in [
        ("F_less_than_t", t, False),           # solid = F < t (25%)
        ("F_greater_than_t", -t_75, True),     # solid = F > t_75 (75%) via -F < -t_75
    ]:
        if use_negated:
            F_use = -F.copy()
            grad_fn_use = lambda X,Y,Z,L: tuple(-g for g in gyroid_gradient_field(X,Y,Z,L))
        else:
            F_use = F.copy()
            grad_fn_use = gyroid_gradient_field

        G = F_use - iso
        G[0,:,:] = 1.0; G[-1,:,:] = 1.0
        G[:,0,:] = 1.0; G[:,-1,:] = 1.0
        G[:,:,0] = 1.0; G[:,:,-1] = 1.0

        print(f"\n--- {name} (iso={iso:.4f}) ---")
        verts, faces = dual_contour(G, 0.0, res, origin, L_mm, gradient_fn=grad_fn_use)
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=True)

        # Boolean intersect with cube
        try:
            import manifold3d as mf
            def _to_mf(tm):
                v = np.ascontiguousarray(tm.vertices, dtype=np.float32)
                f = np.ascontiguousarray(tm.faces, dtype=np.uint32)
                return mf.Manifold(mf.Mesh(vert_properties=v, tri_verts=f))
            m_result = _to_mf(mesh) ^ _to_mf(cube)
            raw = m_result.to_mesh()
            vb = np.array(raw.vert_properties, dtype=np.float64)
            fb = np.array(raw.tri_verts, dtype=np.int64).reshape(-1, 3)
            mesh = trimesh.Trimesh(vertices=vb, faces=fb, process=True)
        except Exception as e:
            print(f"  manifold3d failed ({e}), using PyVista fallback")
            import pyvista as pv
            def _pv(tm):
                return pv.PolyData(tm.vertices, np.c_[np.full(len(tm.faces),3,int), tm.faces])
            clipped = _pv(mesh).clip_surface(_pv(cube), invert=True).triangulate().clean()
            pts = np.array(clipped.points)
            fc = np.array(clipped.faces).reshape(-1, 4)[:, 1:]
            mesh = trimesh.Trimesh(vertices=pts, faces=fc, process=True)

        sf_label = "25pct" if "less" in name else "75pct"
        path = os.path.join(out_dir, f"GyroidCube5mm_{sf_label}_{name}.stl")
        mesh.export(path)
        print(f"  Saved: {path}  ({len(mesh.faces):,} faces)")

if __name__ == "__main__":
    main()
