import numpy as np


def evaluate_tpms(lattice_type, k, X, Y, Z):
    """
    Evaluate a TPMS field for the given lattice type.
    """
    l_type = lattice_type.lower()
    if l_type == "gyroid":
        return (
            np.sin(k * X) * np.cos(k * Y)
            + np.sin(k * Y) * np.cos(k * Z)
            + np.sin(k * Z) * np.cos(k * X)
        )
    elif l_type in ["schwarz-p", "schwarz primitive", "schwarz"]:
        return np.cos(k * X) + np.cos(k * Y) + np.cos(k * Z)
    elif l_type == "diamond":
        return (
            np.sin(k * X) * np.sin(k * Y) * np.sin(k * Z)
            + np.sin(k * X) * np.cos(k * Y) * np.cos(k * Z)
            + np.cos(k * X) * np.sin(k * Y) * np.cos(k * Z)
            + np.cos(k * X) * np.cos(k * Y) * np.sin(k * Z)
        )
    elif l_type == "neovius":
        return 3.0 * (np.cos(k * X) + np.cos(k * Y) + np.cos(k * Z)) + 4.0 * (
            np.cos(k * X) * np.cos(k * Y) * np.cos(k * Z)
        )
    elif l_type == "split-p":
        t1 = (
            np.sin(2 * k * X) * np.sin(k * Z) * np.cos(k * Y)
            + np.sin(2 * k * Y) * np.sin(k * X) * np.cos(k * Z)
            + np.sin(2 * k * Z) * np.sin(k * Y) * np.cos(k * X)
        )
        t2 = (
            np.cos(2 * k * X) * np.cos(2 * k * Y)
            + np.cos(2 * k * Y) * np.cos(2 * k * Z)
            + np.cos(2 * k * Z) * np.cos(2 * k * X)
        )
        t3 = np.cos(2 * k * X) + np.cos(2 * k * Y) + np.cos(2 * k * Z)
        return 1.1 * t1 - 0.2 * t2 - 0.4 * t3
    else:
        raise ValueError(f"Unknown lattice type: {lattice_type}")

