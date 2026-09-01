import numpy as np

"""
D3Q19 Lattice Boltzmann Constants.
Defines the 19 discrete velocity vectors, their equilibrium weights, 
and their exact opposites (required for bounce-back boundary conditions).
"""

# Number of discrete velocities
Q = 19

# Discrete velocity vectors (c_i)
# Order: Rest (0), Faces (1-6), Edges (7-18)
c = np.array([
    [ 0,  0,  0], # 0: Rest
    [ 1,  0,  0], [-1,  0,  0], # 1, 2: X-axis
    [ 0,  1,  0], [ 0, -1,  0], # 3, 4: Y-axis
    [ 0,  0,  1], [ 0,  0, -1], # 5, 6: Z-axis
    [ 1,  1,  0], [-1, -1,  0], # 7, 8: XY plane
    [ 1, -1,  0], [-1,  1,  0], # 9, 10: XY plane
    [ 1,  0,  1], [-1,  0, -1], # 11, 12: XZ plane
    [ 1,  0, -1], [-1,  0,  1], # 13, 14: XZ plane
    [ 0,  1,  1], [ 0, -1, -1], # 15, 16: YZ plane
    [ 0,  1, -1], [ 0, -1,  1]  # 17, 18: YZ plane
], dtype=np.int32)

# Lattice weights (w_i) corresponding to each velocity vector
w = np.array([
    1./3.,  # Rest
    1./18., 1./18., 1./18., 1./18., 1./18., 1./18., # Faces
    1./36., 1./36., 1./36., 1./36., 1./36., 1./36., # Edges
    1./36., 1./36., 1./36., 1./36., 1./36., 1./36.  # Edges
], dtype=np.float32)

# Opposite indices array
# If a particle travels in direction i, its exact reverse is opposite[i]
# Crucial for the no-slip "Bounce-Back" condition on TPMS walls.
opposite = np.array([
    0, 
    2, 1, 
    4, 3, 
    6, 5, 
    8, 7, 
    10, 9, 
    12, 11, 
    14, 13, 
    16, 15, 
    18, 17
], dtype=np.int32)

# Lattice speed of sound squared
cs2 = 1.0 / 3.0
