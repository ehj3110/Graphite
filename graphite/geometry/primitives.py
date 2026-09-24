"""
Graphite Geometry - Primitive Generation

Utility functions for generating basic 3D mesh primitives.
"""
import trimesh


def generate_primitive(shape, size):
    """
    Generate a simple 3D mesh primitive.

    Parameters
    ----------
    shape : str
        The type of primitive to generate ('Cube', 'Sphere', 'Cylinder').
    size : float
        The characteristic size dimension of the primitive.

    Returns
    -------
    trimesh.Trimesh
        The generated 3D mesh primitive.
    """
    if shape == "Cube":
        return trimesh.creation.box(extents=[size, size, size])
    elif shape == "Sphere":
        return trimesh.creation.icosphere(radius=size / 2.0)
    elif shape == "Cylinder":
        return trimesh.creation.cylinder(radius=size / 2.0, height=size)
    elif shape in ("Torus", "Toros"):
        # Total outer diameter equals size
        major_r = float(size) * 0.35
        minor_r = float(size) * 0.15
        return trimesh.creation.torus(major_radius=major_r, minor_radius=minor_r)
    elif shape in ("Tube", "Sleeve"):
        import manifold3d as m3d
        from graphite.explicit.geometry_module import _manifold_to_trimesh
        r_out = float(size) / 2.0
        r_in = float(size) * 0.35
        h = float(size)
        c_out = m3d.Manifold.cylinder(h, r_out, r_out, circular_segments=32, center=True)
        c_in = m3d.Manifold.cylinder(h * 1.05, r_in, r_in, circular_segments=32, center=True)
        tube_m = c_out - c_in
        return _manifold_to_trimesh(tube_m)
    else:
        raise ValueError(f"Unknown primitive shape: {shape}")

