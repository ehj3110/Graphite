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
    else:
        raise ValueError(f"Unknown primitive shape: {shape}")

