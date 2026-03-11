"""
Supercell Unit Cell Visualizer — Interactive 3D plots of lattice topologies.

Plots individual unit cells with struts colored by sub-cell / local basis
to reveal how each cell is constructed.

Usage:
    python visualize_supercells.py --cell BCC
    python visualize_supercells.py --cell Diamond
    python visualize_supercells.py --cell Kelvin
    python visualize_supercells.py --cell A15
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import numpy as np


@dataclass
class UnitCell:
    """Nodes and struts with optional group labels for coloring."""

    nodes: np.ndarray  # (N, 3)
    struts: np.ndarray  # (S, 2) indices
    strut_groups: list[tuple[str, list[int]]]  # (group_name, [strut indices])


def get_bcc_unit_cell() -> UnitCell:
    """
    BCC: 9 nodes (8 corners + 1 body center) in 1x1x1 box.
    Struts: 8 body-to-corner (one per octant).
    """
    nodes = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
            [0.5, 0.5, 0.5],  # body center
        ],
        dtype=np.float64,
    )
    body_idx = 8
    corner_indices = list(range(8))
    struts = np.array([[body_idx, c] for c in corner_indices], dtype=np.int64)
    # Group by octant: each corner-body pair is one sub-cell
    strut_groups = [(f"Octant_{i}", [i]) for i in range(8)]
    return UnitCell(nodes=nodes, struts=struts, strut_groups=strut_groups)


def get_diamond_unit_cell() -> UnitCell:
    """
    Diamond: FCC with 2-atom basis. 8 nodes in 1x1x1.
    Nodes: [0,0,0], [0.5,0.5,0], [0.5,0,0.5], [0,0.5,0.5] (FCC1)
           [0.25,0.25,0.25], [0.75,0.75,0.25], [0.75,0.25,0.75], [0.25,0.75,0.75] (FCC2)
    Struts: 4 bonds per atom (tetrahedral).
    """
    nodes = np.array(
        [
            [0, 0, 0],
            [0.5, 0.5, 0],
            [0.5, 0, 0.5],
            [0, 0.5, 0.5],
            [0.25, 0.25, 0.25],
            [0.75, 0.75, 0.25],
            [0.75, 0.25, 0.75],
            [0.25, 0.75, 0.75],
        ],
        dtype=np.float64,
    )
    # Tetrahedral bonds: each atom connects to 4 neighbors
    # (0,0,0) -> (0.25,0.25,0.25), (0.75,0.25,0.25), (0.25,0.75,0.25), (0.25,0.25,0.75)
    # Using periodic images: (0.25,0.25,0.25), (0.75,0.25,0.25), (0.25,0.75,0.25), (0.25,0.25,0.75)
    # Node 0 (0,0,0) -> 4, 5, 6, 7 (with wraparound: 5=(0.75,0.75,0.25) - no, 5 is (0.75,0.75,0.25)
    # From (0,0,0), neighbors at (0.25,0.25,0.25), (0.75,0.25,0.25), (0.25,0.75,0.25), (0.25,0.25,0.75)
    # (0.25,0.25,0.25)=4, (0.75,0.25,0.25) - not in list. (0.25,0.75,0.25) - not. (0.25,0.25,0.75) - not.
    # Diamond: each C has 4 neighbors. From (0,0,0): (+0.25,+0.25,+0.25), (+0.25,-0.25,-0.25), (-0.25,+0.25,-0.25), (-0.25,-0.25,+0.25)
    # In fractional: (0.25,0.25,0.25), (0.25,0.75,0.75), (0.75,0.25,0.75), (0.75,0.75,0.25)
    # So nodes 4, 7, 6, 5
    bonds = [
        (0, 4),
        (0, 7),
        (0, 6),
        (0, 5),
        (1, 5),
        (1, 6),
        (1, 4),
        (1, 7),
        (2, 6),
        (2, 5),
        (2, 7),
        (2, 4),
        (3, 7),
        (3, 4),
        (3, 5),
        (3, 6),
    ]
    struts = np.array(sorted([(min(a, b), max(a, b)) for a, b in bonds]), dtype=np.int64)
    struts = np.unique(struts, axis=0)
    # Group by bond direction (4 tetrahedral directions)
    def dir_key(v: np.ndarray) -> str:
        s = np.sign(v + 1e-9)
        return f"Dir_{int(s[0])}{int(s[1])}{int(s[2])}"

    groups: dict[str, list[int]] = {}
    for i, (a, b) in enumerate(struts):
        vec = nodes[b] - nodes[a]
        key = dir_key(vec)
        if key not in groups:
            groups[key] = []
        groups[key].append(i)
    strut_groups = list(groups.items())
    return UnitCell(nodes=nodes, struts=struts, strut_groups=strut_groups)


def get_kelvin_unit_cell() -> UnitCell:
    """
    Kelvin (Truncated Octahedron): nodes are permutations of [0, 1/2, 1/4].
    12 nodes in 1x1x1. Edges from permutohedron adjacency (transposition).
    """
    from itertools import permutations

    nodes_set: set[tuple[float, float, float]] = set()
    for vals in ([0.0, 0.5, 0.25], [0.0, 0.5, 0.75]):
        for p in permutations(vals):
            nodes_set.add(p)
    nodes = np.array(sorted(nodes_set), dtype=np.float64)

    def transposition_adjacent(t1: tuple[float, ...], t2: tuple[float, ...]) -> bool:
        diffs = [(a, b) for a, b in zip(t1, t2) if abs(a - b) > 1e-6]
        if len(diffs) != 2:
            return False
        return abs(diffs[0][0] - diffs[0][1]) == abs(diffs[1][0] - diffs[1][1])

    struts_list: list[tuple[int, int]] = []
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            if transposition_adjacent(tuple(nodes[i]), tuple(nodes[j])):
                struts_list.append((i, j))
    struts = np.array(struts_list, dtype=np.int64)

    # Group by edge length: square edges (shorter) vs hex edges (longer)
    lengths = np.linalg.norm(nodes[struts[:, 1]] - nodes[struts[:, 0]], axis=1)
    median_len = np.median(lengths)
    strut_groups = [
        ("Square_face", [i for i in range(len(struts)) if lengths[i] < median_len]),
        ("Hex_face", [i for i in range(len(struts)) if lengths[i] >= median_len]),
    ]
    strut_groups = [(n, idx) for n, idx in strut_groups if idx]

    return UnitCell(nodes=nodes, struts=struts, strut_groups=strut_groups)


def get_a15_unit_cell() -> UnitCell:
    """
    A15 (Cr3Si): 8 nodes. 2 BCC at (0,0,0), (0.5,0.5,0.5), 6 face at
    (0.25,0.5,0), (0.75,0.5,0), (0,0.25,0.5), (0,0.75,0.5), (0.5,0,0.25), (0.5,0,0.75).
    """
    nodes = np.array(
        [
            [0, 0, 0],
            [0.5, 0.5, 0.5],
            [0.25, 0.5, 0],
            [0.75, 0.5, 0],
            [0, 0.25, 0.5],
            [0, 0.75, 0.5],
            [0.5, 0, 0.25],
            [0.5, 0, 0.75],
        ],
        dtype=np.float64,
    )
    bcc_idx = [0, 1]
    face_idx = [2, 3, 4, 5, 6, 7]
    struts_list: list[tuple[int, int, str]] = []
    for b in bcc_idx:
        for f in face_idx:
            struts_list.append((min(b, f), max(b, f), "BCC_Face"))
    struts_list.append((2, 3, "Chain_1"))
    struts_list.append((4, 5, "Chain_2"))
    struts_list.append((6, 7, "Chain_3"))
    struts = np.array([(a, b) for a, b, _ in struts_list], dtype=np.int64)
    strut_groups: list[tuple[str, list[int]]] = []
    for i, (_, _, g) in enumerate(struts_list):
        found = False
        for j, (name, indices) in enumerate(strut_groups):
            if name == g:
                strut_groups[j][1].append(i)
                found = True
                break
        if not found:
            strut_groups.append((g, [i]))
    return UnitCell(nodes=nodes, struts=struts, strut_groups=strut_groups)


def _plot_matplotlib(cell: UnitCell, title: str, save_path: str | None = None) -> None:
    """Plot using matplotlib 3D."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(cell.strut_groups), 1)))
    for gi, (group_name, indices) in enumerate(cell.strut_groups):
        segs = [np.array([cell.nodes[cell.struts[i, 0]], cell.nodes[cell.struts[i, 1]]]) for i in indices]
        ax.add_collection3d(
            Line3DCollection(segs, colors=[colors[gi % len(colors)]], linewidths=2.5, label=group_name)
        )

    ax.scatter(
        cell.nodes[:, 0],
        cell.nodes[:, 1],
        cell.nodes[:, 2],
        c="black",
        s=50,
        depthshade=True,
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    ax.set_box_aspect([1, 1, 1])
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()


def _plot_plotly(cell: UnitCell, title: str, save_path: str | None = None) -> None:
    """Plot using plotly (interactive)."""
    import plotly.graph_objects as go

    fig = go.Figure()
    colors = [
        "rgb(255,0,0)",
        "rgb(0,128,0)",
        "rgb(0,0,255)",
        "rgb(255,165,0)",
        "rgb(128,0,128)",
        "rgb(0,128,128)",
        "rgb(255,0,255)",
        "rgb(128,128,0)",
    ]
    for gi, (group_name, indices) in enumerate(cell.strut_groups):
        for i in indices:
            a, b = cell.struts[i, 0], cell.struts[i, 1]
            x = [cell.nodes[a, 0], cell.nodes[b, 0], None]
            y = [cell.nodes[a, 1], cell.nodes[b, 1], None]
            z = [cell.nodes[a, 2], cell.nodes[b, 2], None]
            fig.add_trace(
                go.Scatter3d(
                    x=x,
                    y=y,
                    z=z,
                    mode="lines",
                    line=dict(color=colors[gi % len(colors)], width=4),
                    name=group_name,
                    showlegend=(i == indices[0]),
                )
            )
    fig.add_trace(
        go.Scatter3d(
            x=cell.nodes[:, 0],
            y=cell.nodes[:, 1],
            z=cell.nodes[:, 2],
            mode="markers",
            marker=dict(size=6, color="black"),
            name="Nodes",
        )
    )
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(range=[0, 1], title="X"),
            yaxis=dict(range=[0, 1], title="Y"),
            zaxis=dict(range=[0, 1], title="Z"),
            aspectmode="cube",
        ),
        showlegend=True,
    )
    if save_path:
        fig.write_html(save_path)
    else:
        fig.show()


def plot_unit_cell(
    cell: UnitCell,
    title: str,
    backend: str = "matplotlib",
    save_path: str | None = None,
) -> None:
    """Plot the unit cell with struts colored by group."""
    if backend == "plotly":
        try:
            _plot_plotly(cell, title, save_path)
        except ImportError:
            _plot_matplotlib(cell, title, save_path.replace(".html", ".png") if save_path else None)
    else:
        _plot_matplotlib(cell, title, save_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize supercell unit cell topologies")
    parser.add_argument(
        "--cell",
        choices=["BCC", "Diamond", "Kelvin", "A15"],
        default="BCC",
        help="Unit cell topology to visualize",
    )
    parser.add_argument(
        "--backend",
        choices=["matplotlib", "plotly"],
        default="matplotlib",
        help="Plotting backend (plotly is more interactive)",
    )
    parser.add_argument(
        "--save",
        metavar="PATH",
        help="Save to file instead of showing (e.g. output/bcc.png)",
    )
    args = parser.parse_args()

    cells = {
        "BCC": get_bcc_unit_cell,
        "Diamond": get_diamond_unit_cell,
        "Kelvin": get_kelvin_unit_cell,
        "A15": get_a15_unit_cell,
    }
    cell = cells[args.cell]()
    title = f"{args.cell} Unit Cell (1×1×1)"
    plot_unit_cell(cell, title, backend=args.backend, save_path=args.save)


if __name__ == "__main__":
    main()
