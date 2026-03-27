import trimesh


def generate_primitive(shape, size):
    if shape == "Cube":
        return trimesh.creation.box(extents=[size, size, size])
    elif shape == "Sphere":
        return trimesh.creation.icosphere(radius=size / 2.0)
    elif shape == "Cylinder":
        return trimesh.creation.cylinder(radius=size / 2.0, height=size)
    else:
        raise ValueError(f"Unknown primitive shape: {shape}")

