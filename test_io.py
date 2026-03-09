"""
Graphite I/O Module — Test Suite

Tests load_and_verify_mesh() with:
  1. In-memory baseline: trimesh box exported to temp STL (no external files)
  2. Real parts: all .stl files in test_parts/ (if directory exists)

Run from project root:
    python test_io.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import trimesh

from io_module import load_and_verify_mesh, MeshVerificationResult


# -----------------------------------------------------------------------------
# Test Configuration
# -----------------------------------------------------------------------------

MOCK_BOX_EXTENTS = [20.0, 20.0, 20.0]
EXPECTED_VOLUME = 20.0 * 20.0 * 20.0  # 8000.0

TEST_PARTS_DIR = Path("test_parts")


# -----------------------------------------------------------------------------
# Baseline Test (In-Memory Mock Box)
# -----------------------------------------------------------------------------


def _run_baseline_test() -> None:
    """
    Run the mock box test as a regression baseline.
    Uses tempfile to avoid polluting the working directory.
    """
    print("--- Testing: In-Memory Baseline Box ---")
    mock_mesh = trimesh.creation.box(extents=MOCK_BOX_EXTENTS)

    with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as tmp:
        tmp_path = Path(tmp.name)
        mock_mesh.export(str(tmp_path))

    try:
        result = load_and_verify_mesh(tmp_path)
        print(f"Success! Volume: {result.volume:.2f}, BBox: {result.bounding_box}")

        assert result.mesh.is_watertight, "Mesh must be watertight after load."
        assert abs(result.volume - EXPECTED_VOLUME) < 0.01, (
            f"Volume mismatch: got {result.volume}, expected ~{EXPECTED_VOLUME}"
        )
        assert result.bounding_box == (20.0, 20.0, 20.0), (
            f"Bounding box mismatch: got {result.bounding_box}"
        )
        print("[OK] Baseline assertions passed.")
    finally:
        tmp_path.unlink(missing_ok=True)


# -----------------------------------------------------------------------------
# Real Parts Test (test_parts/*.stl)
# -----------------------------------------------------------------------------


def _run_real_parts_test() -> None:
    """
    Run load_and_verify_mesh on all .stl files in test_parts/.
    Skips gracefully if the directory does not exist.
    """
    if not TEST_PARTS_DIR.exists():
        print(f"\nDirectory '{TEST_PARTS_DIR}' not found. Skipping real-parts test.")
        return

    stl_files = sorted(
        f for f in TEST_PARTS_DIR.iterdir()
        if f.is_file() and f.suffix.lower() == ".stl"
    )

    if not stl_files:
        print(f"\nNo .stl files in '{TEST_PARTS_DIR}'. Skipping.")
        return

    for filepath in stl_files:
        print(f"\n--- Testing: {filepath.name} ---")
        try:
            result = load_and_verify_mesh(filepath)
            print("Status: Watertight and Ready.")
            print(f"Volume: {result.volume:.2f}")
            print(f"Bounding Box: {result.bounding_box}")
        except Exception as e:
            print(f"REJECTED: {e}")


# -----------------------------------------------------------------------------
# Entry Point
# -----------------------------------------------------------------------------


def run_io_test() -> None:
    """Execute baseline test, then real-parts test if available."""
    print("=" * 60)
    print("Graphite I/O Module — Test")
    print("=" * 60)

    _run_baseline_test()
    _run_real_parts_test()

    print("\n" + "=" * 60)


if __name__ == "__main__":
    run_io_test()
