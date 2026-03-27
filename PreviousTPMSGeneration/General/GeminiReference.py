import numpy as np
from skimage import measure
import stl
from meshlib import mrmeshpy as mm

def generate_gyroid_lattice(pore_size_mm, solid_fraction, cube_dim_mm, resolution=64):
    """
    Generates a Gyroid lattice cube based on an analytical model.
    
    Parameters:
    - pore_size_mm: The desired diameter of the inscribed sphere (pore).
    - solid_fraction: The volume fraction (0.0 to 1.0).
    - cube_dim_mm: The total size of the cube side in mm.
    - resolution: Voxel resolution per side (higher = smoother, slower).
    """
    
    # 1. CALCULATE UNIT CELL SIZE (L)
    # For a Gyroid, the relationship between pore diameter (Dp), 
    # Unit cell length (L), and volume fraction (phi) is approx:
    # Dp = L * (1 - C * phi), where C is a geometric constant.
    # A common empirical fit for Gyroids is L = Dp / (1 - 1.15 * solid_fraction)
    # Note: This is valid for solid_fractions between ~0.1 and ~0.5
    
    if solid_fraction >= 0.8:
        print("Warning: Solid fraction too high for standard Gyroid models.")
        
    unit_cell_size = pore_size_mm / (1.0 - 1.15 * solid_fraction)
    
    # 2. CALCULATE ISO-VALUE (t)
    # The level set eq: sin(kx)cos(ky) + sin(ky)cos(kz) + sin(kz)cos(kx) = t
    # t controls the volume fraction. 
    # For a Gyroid, a linear approximation for t based on volume fraction phi is:
    # t is approx 1.5 * (2 * phi - 1)
    # At t=0, phi=0.5. As t increases, solid volume increases.
    iso_value = 1.5 * (2 * solid_fraction - 1)
    
    print(f"Calculated Unit Cell Size (L): {unit_cell_size:.4f} mm")
    print(f"Calculated ISO Threshold (t): {iso_value:.4f}")

    # 3. GENERATE COORDINATE GRID
    num_cells = cube_dim_mm / unit_cell_size
    k = (2 * np.pi) / unit_cell_size
    
    # Create coordinate arrays
    x = np.linspace(0, cube_dim_mm, resolution)
    y = np.linspace(0, cube_dim_mm, resolution)
    z = np.linspace(0, cube_dim_mm, resolution)
    X, Y, Z = np.meshgrid(x, y, z)

    # 4. GYROID LEVEL SET EQUATION
    # G = sin(kx)cos(ky) + sin(ky)cos(kz) + sin(kz)cos(kx)
    gyroid = (np.sin(k * X) * np.cos(k * Y) + 
              np.sin(k * Y) * np.cos(k * Z) + 
              np.sin(k * Z) * np.cos(k * X))

    # 5. MARCHING CUBES TO GENERATE MESH
    # The surface is where gyroid - iso_value = 0
    verts, faces, normals, values = measure.marching_cubes(gyroid, level=iso_value)

    # Scale vertices to real mm dimensions
    verts = verts * (cube_dim_mm / (resolution - 1))

    # 6. EXPORT TO STL
    cube_mesh = stl.mesh.Mesh(np.zeros(faces.shape[0], dtype=stl.mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            cube_mesh.vectors[i][j] = verts[f[j],:]

    filename = f"gyroid_P{pore_size_mm}_S{solid_fraction}.stl"
    cube_mesh.save(filename)
    print(f"Success! Mesh saved as {filename}")

if __name__ == "__main__":
    # USER INPUTS
    DESIRED_PORE_SIZE = 0.8   # mm
    SOLID_FRACTION = 0.3      # 30% solid
    CUBE_SIDE_LENGTH = 5.0    # 5mm cube
    QUALITY = 80              # Grid resolution (100+ for 3D printing)

    generate_gyroid_lattice(
        DESIRED_PORE_SIZE, 
        SOLID_FRACTION, 
        CUBE_SIDE_LENGTH, 
        resolution=QUALITY
    )