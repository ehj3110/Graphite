# Topology Optimization Module (`graphite.topt`)

The `topt` module provides a flexible, unstructured-mesh topology optimization engine for Graphite. It is designed to handle complex CAD geometries by working directly on tetrahedral meshes generated from STL files.

## Architecture

The module is built on two primary mathematical backends:
1.  **`scikit-fem`**: Used for assembly of finite element matrices (Stiffness, Mass) and solving the linear elasticity system.
2.  **`nlopt`**: Used for the optimization loops, specifically the **Method of Moving Asymptotes (MMA)** for multi-constraint problems.

### Core Components
- **`task.py`**: Bridges the gap between GMSH/skfem meshes and the `sktopt` objective functions.
- **`mma_solver.py`**: Contains the industrial-grade MMA wrapper and specialized optimizers:
    - `Compliance_Heaviside_Optimizer`: Minimizes compliance with a volume constraint.
    - `Stress_P_Norm_Optimizer`: Minimizes aggregated P-Norm stress with compliance and volume constraints.

## Advanced Features

### 1. Stress Aggregation (P-Norm)
To optimize for stress on an unstructured mesh, we utilize the P-Norm aggregation function to collapse element-wise stresses into a single differentiable scalar objective:
$$J = \left( \sum_{e} v_e \sigma_e^p \right)^{1/p}$$
where $p$ is usually set between 6 and 10 to approximate the maximum stress.

### 2. Stress Relaxation
To avoid the "singularity problem" in topology optimization (where stress is undefined for zero-density elements), we use the $q$-relaxation technique:
$$\hat{\sigma}_e = \rho_e^q \sigma_e$$

### 3. Heaviside Projection & Filtering
To ensure a manufacturable, binary (solid/void) result, the module uses a three-stage density pipeline:
1.  **Helmholtz Filter**: Smoothes the design variables to prevent checkerboarding.
2.  **Heaviside Projection**: Sharpen the filtered densities using a $\beta$-ramp scheduling (ramping $\beta$ from 1 to 32).
3.  **Passive Density Override**: Foolproof hard-clamping of functional interfaces (load pads, clamping zones) to ensure they are never deleted by the filter math.

## J-Hook Structural Baseline Case Study

In April 2026, a series of diagnostic runs were performed on a J-Hook geometry to validate the load path connectivity.

### Findings:
- **Baseline Compliance**: Established at **211,130** for a uniform 2mm mesh.
- **Interface Locking**: Failure to lock the 8 Cartesian load faces (indices 0-7) at the tip leads to "cheating" where the optimizer shortens the lever arm, resulting in an artificially low compliance (~4,600).
- **Architecture Inversion**: Minimizing stress while constraining compliance to $1.1 \times C_{base}$ proved to be the most stable multi-constraint configuration.

## Usage

Example of running a compliance-driven optimization:
```python
from graphite.topt.mma_solver import Compliance_Heaviside_Optimizer
from types import SimpleNamespace

config = SimpleNamespace(target_volume_fraction=0.5, max_iterations=150, rho_min=1e-4)
opt = Compliance_Heaviside_Optimizer(task, config, beta_init=2.0, beta_max=32.0)
final_rho = opt.optimize()
```

## See also

- [Topology optimization checkpoint (sandbox layout & handoff notes)](../optimization/checkpoints/topology-optimization-2026-05/CHECKPOINT.md)
