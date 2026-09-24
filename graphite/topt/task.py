"""
Graphite Topology Optimization - Task Abstraction

This module provides a unified interface for defining and managing topology 
optimization domains, boundary conditions, and loads. It wraps the `sktopt` 
finite element definitions for compatibility with Graphite's solvers.
"""
import numpy as np
import skfem
import sktopt
from sktopt import core, mesh
import gmsh
import logging

logger = logging.getLogger(__name__)

class ToptTask:
    """
    Abstraction layer for Topology Optimization tasks.
    
    Wraps `sktopt.mesh.LinearElasticity` to provide a simplified interface 
    for initializing meshes, calculating element volumes, and applying 
    boundary conditions and loads.

    Parameters
    ----------
    nodes : ndarray
        Mesh nodal coordinates (N x 3).
    elements : ndarray
        Mesh element connectivity (M x 4 for tetrahedra).
    basis : skfem.Basis
        The finite element basis for the mesh.
    task : sktopt.mesh.LinearElasticity
        The underlying linear elasticity problem definition.
    """
    def __init__(self, nodes, elements, basis, task):
        self.nodes = nodes
        self.elements = elements
        self.basis = basis
        self.task = task
        
        # Shortcuts for MMA solver
        self.all_elements = np.arange(len(elements))
        self.design_elements = np.arange(len(elements))
        self.fixed_elements = np.array([], dtype=int)
        
        # Calculate element volumes
        t = basis.mesh.t
        p = basis.mesh.p
        v1, v2, v3, v4 = p[:, t[0]], p[:, t[1]], p[:, t[2]], p[:, t[3]]
        mats = np.empty((t.shape[1], 3, 3))
        mats[:, 0, :] = (v1 - v4).T
        mats[:, 1, :] = (v2 - v4).T
        mats[:, 2, :] = (v3 - v4).T
        self.elements_volume = np.abs(np.linalg.det(mats)) / 6.0

    @classmethod
    def from_msh(cls, msh_path):
        """
        Initialize a ToptTask from a Gmsh (.msh) file.

        Parameters
        ----------
        msh_path : str
            Path to the Gmsh file to load.

        Returns
        -------
        ToptTask
            A new ToptTask instance built from the mesh data.
        """
        gmsh.initialize()
        gmsh.open(msh_path)
        nodes = gmsh.model.mesh.getNodes()[1].reshape(-1, 3)
        elements = gmsh.model.mesh.getElementsByType(4)[1].reshape(-1, 4) - 1
        gmsh.finalize()
        
        m = skfem.MeshTet(nodes.T, elements.T)
        basis = skfem.Basis(m, skfem.ElementVector(skfem.ElementTetP1()))
        
        # Default empty task, needs update_problem
        # We use a dummy LinearElasticity that we will update later
        dummy_task = mesh.LinearElasticity(basis)
        
        return cls(nodes, elements, basis, dummy_task)

    def update_problem(self, fixed_elements=None, neumann_linear=None, dirichlet_nodes=None, dirichlet_dofs=None):
        """
        Update the physical problem definition (boundary conditions and loads).

        Parameters
        ----------
        fixed_elements : array_like, optional
            Indices of elements locked to maximum density (rho=1.0).
        neumann_linear : Callable, optional
            Function defining the surface tractions/loads.
        dirichlet_nodes : array_like, optional
            Indices of nodes to lock in space.
        dirichlet_dofs : dict, optional
            Specific degrees of freedom to lock for Dirichlet nodes.
        """
        if fixed_elements is not None:
            self.fixed_elements = np.array(fixed_elements)
            self.design_elements = np.delete(self.all_elements, self.fixed_elements)
            
        self.task = mesh.LinearElasticity(
            basis=self.basis,
            fixed_elements=self.fixed_elements,
            neumann_linear=neumann_linear,
            dirichlet_nodes=dirichlet_nodes,
            dirichlet_dofs=dirichlet_dofs
        )

    def exlude_dirichlet_from_design(self):
        """Required by sktopt. Just pass as we handle it in update_problem."""
        pass

    # Proxy properties to sktopt task
    @property
    def n_tasks(self):
        """
        Get the number of load cases (tasks) defined.

        Returns
        -------
        int
            The number of independent load cases.
        """
        return self.task.n_tasks
