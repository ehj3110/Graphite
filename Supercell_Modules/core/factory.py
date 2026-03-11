"""
LatticeFactory: topology + micro-rule composition pipeline.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.spatial import Delaunay

from .lattice_rules import (
    apply_icosahedral_rule,
    apply_kagome_rule,
    apply_rhombic_rule,
    apply_voronoi_rule,
)
from .topologies import (
    generate_a15_seeds,
    generate_bitruncated_cubic_seeds,
    generate_rhombicuboct_seeds,
    generate_truncated_oct_tet_seeds,
)


def _tetra_volumes(points: np.ndarray, simplices: np.ndarray) -> np.ndarray:
    """Compute tetrahedron volumes."""
    t = points[simplices]
    mat = np.stack(
        (
            t[:, 1, :] - t[:, 0, :],
            t[:, 2, :] - t[:, 0, :],
            t[:, 3, :] - t[:, 0, :],
        ),
        axis=1,
    )
    return np.abs(np.linalg.det(mat)) / 6.0


class LatticeFactory:
    """Build merged lattice graphs from selected topology and micro-rule."""

    def __init__(self) -> None:
        self._topology_map: dict[str, Callable[..., np.ndarray]] = {
            "bitruncated": generate_bitruncated_cubic_seeds,
            "truncated_oct_tet": generate_truncated_oct_tet_seeds,
            "rhombicuboct": generate_rhombicuboct_seeds,
            "a15": generate_a15_seeds,
        }
        self._rule_map: dict[str, Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]] = {
            "kagome": apply_kagome_rule,
            "voronoi": apply_voronoi_rule,
            "icosahedral": apply_icosahedral_rule,
            "rhombic": apply_rhombic_rule,
        }

    def generate_lattice(
        self,
        topology_name: str,
        rule_name: str,
        cell_size: float,
        nx: int = 1,
        ny: int = 1,
        nz: int = 1,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate merged (nodes, struts) lattice.

        Steps:
          1) topology seed generation
          2) Delaunay tetrahedralization
          3) degenerate tet purge
          4) per-tet micro-rule application
          5) global node/strut merge
        """
        topo_key = topology_name.strip().lower()
        rule_key = rule_name.strip().lower()
        if topo_key not in self._topology_map:
            raise ValueError(f"Unknown topology '{topology_name}'.")
        if rule_key not in self._rule_map:
            raise ValueError(f"Unknown rule '{rule_name}'.")
        if cell_size <= 0:
            raise ValueError("cell_size must be > 0.")

        seed_fn = self._topology_map[topo_key]
        rule_fn = self._rule_map[rule_key]
        points = seed_fn(nx=nx, ny=ny, nz=nz, cell_size=cell_size)
        if points.shape[0] < 4:
            raise RuntimeError("Not enough seed points for tetrahedralization.")

        simplices = np.asarray(Delaunay(points).simplices, dtype=np.int64)
        volumes = _tetra_volumes(points, simplices)
        clean_simplices = simplices[volumes >= 1e-8]
        if clean_simplices.shape[0] == 0:
            raise RuntimeError("All Delaunay tetrahedra were degenerate.")

        node_map: dict[tuple[float, float, float], int] = {}
        nodes_list: list[np.ndarray] = []
        strut_set: set[tuple[int, int]] = set()

        def _key(xyz: np.ndarray) -> tuple[float, float, float]:
            return (
                round(float(xyz[0]), 8),
                round(float(xyz[1]), 8),
                round(float(xyz[2]), 8),
            )

        def add_node(xyz: np.ndarray) -> int:
            k = _key(xyz)
            idx = node_map.get(k)
            if idx is None:
                idx = len(nodes_list)
                nodes_list.append(np.asarray(xyz, dtype=np.float64))
                node_map[k] = idx
            return idx

        for tet in clean_simplices:
            tet_coords = points[tet]
            local_nodes, local_struts = rule_fn(tet_coords)
            local_to_global = [add_node(local_nodes[i]) for i in range(local_nodes.shape[0])]
            for u, v in local_struts:
                a = local_to_global[int(u)]
                b = local_to_global[int(v)]
                if a == b:
                    continue
                strut_set.add((min(a, b), max(a, b)))

        nodes = np.vstack(nodes_list) if nodes_list else np.empty((0, 3), dtype=np.float64)
        struts = np.array(sorted(strut_set), dtype=np.int64) if strut_set else np.empty((0, 2), dtype=np.int64)
        return nodes, struts
