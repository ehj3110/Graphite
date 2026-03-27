"""
Graphite Production Benchmark — High-performance telemetry for Adapter and MariaTubeRack.

Runs:
  1. Part2_Adapter.STL | Kagome | 6.0mm | 15% VF | fast_solve=True
  2. MariaTubeRack_Full.STL | Kagome | 6.0mm | 10% VF | fast_solve=True

Tracks: Scaffolding, Topology, Math, Geometry (cylinders, union, intersect, convert), Export.
Reports: Predicted vs Achieved Vf, Overlap Error %, Bottleneck Analysis.
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np
import trimesh

from graphite.explicit.geometry_module import generate_geometry
from graphite.explicit.scaffold_module import generate_conformal_scaffold
from solver import _strut_lengths, calculate_k_analytical
from graphite.explicit.topology_module import generate_topology

# Suppress geometry Performance Audit during benchmark (we print our own)
_ORIG_STDOUT = sys.stdout


def _quiet_geometry():
    """Temporarily suppress geometry_module print during benchmark."""
    import io
    sys.stdout = io.StringIO()


def _restore_stdout():
    sys.stdout = _ORIG_STDOUT


def run_benchmark(
    stl_path: Path,
    target_vf: float,
    element_size: float,
    output_name: str,
) -> dict:
    """Run full pipeline with telemetry. Returns timing and accuracy dict."""
    timings: dict = {}
    total_start = time.perf_counter()

    # Load mesh (process=True for watertight)
    t0 = time.perf_counter()
    mesh = trimesh.load(str(stl_path), force="mesh", process=True)
    timings["mesh_load"] = time.perf_counter() - t0
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    total_volume = float(mesh.volume)

    # [Scaffolding]
    t0 = time.perf_counter()
    scaffold = generate_conformal_scaffold(
        mesh=mesh,
        target_element_size=element_size,
    )
    timings["scaffolding"] = time.perf_counter() - t0

    # [Topology]
    t0 = time.perf_counter()
    nodes, struts = generate_topology(
        nodes=scaffold.nodes,
        elements=scaffold.elements,
        surface_faces=scaffold.surface_faces,
        topology_type="kagome",
        include_surface_cage=True,
        target_element_size=element_size,
    )
    nodes = np.asarray(nodes, dtype=np.float64)
    timings["topology"] = time.perf_counter() - t0

    # [Math]
    t0 = time.perf_counter()
    k = calculate_k_analytical(
        target_vf=target_vf,
        nodes=nodes,
        struts=struts,
        total_volume=total_volume,
        topology_type="kagome",
    )
    timings["math"] = time.perf_counter() - t0

    lengths = _strut_lengths(nodes, struts)
    radii = lengths * k
    radii = np.clip(radii, 0.1, 2.0)

    # Predicted Vf: analytical formula targets this (V_cyl * overlap_factor = target_vol)
    from solver import OVERLAP_FACTOR_KAGOME
    sum_L3 = float(np.sum(lengths**3))
    V_cyl = math.pi * (k**2) * sum_L3
    predicted_vf = (V_cyl * OVERLAP_FACTOR_KAGOME) / total_volume  # Should ≈ target_vf

    # [Geometry] — with timings dict
    geo_timings: dict = {}
    _quiet_geometry()
    t0_geo_total = time.perf_counter()
    try:
        result = generate_geometry(
            nodes=nodes,
            struts=struts,
            strut_radius=radii,
            boundary_mesh=mesh,
            add_spheres=False,
            crop_to_boundary=True,
            return_manifold=False,
            union_batch_size=10,
            use_native_cylinders=True,
            timings=geo_timings,
        )
    finally:
        _restore_stdout()

    timings["geometry_total"] = time.perf_counter() - t0_geo_total
    final_mesh, achieved_volume = result
    achieved_vf = achieved_volume / total_volume

    timings["geometry_cylinders"] = geo_timings.get("geometry_cylinders", 0)
    timings["geometry_cylinders_count"] = geo_timings.get("geometry_cylinders_count", 0)
    timings["geometry_union"] = geo_timings.get("geometry_union", 0)
    timings["geometry_union_batch_size"] = geo_timings.get("geometry_union_batch_size", 10)
    timings["geometry_intersect"] = geo_timings.get("geometry_intersect", 0)
    timings["geometry_volume"] = geo_timings.get("geometry_volume", 0)
    timings["geometry_convert"] = geo_timings.get("geometry_convert", 0)

    # [Export]
    t0 = time.perf_counter()
    out_path = Path("results") / "benchmark" / output_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    final_mesh.export(str(out_path))
    timings["export"] = time.perf_counter() - t0

    timings["total"] = time.perf_counter() - total_start

    # Accuracy
    overlap_error_pct = (achieved_vf - target_vf) / target_vf * 100 if target_vf > 0 else 0

    return {
        "timings": timings,
        "predicted_vf": predicted_vf,
        "achieved_vf": achieved_vf,
        "target_vf": target_vf,
        "overlap_error_pct": overlap_error_pct,
        "n_struts": len(struts),
        "n_cylinders": timings["geometry_cylinders_count"],
    }


def print_telemetry(name: str, data: dict) -> None:
    """Print enhanced telemetry for a run."""
    t = data["timings"]
    print("\n" + "=" * 70)
    print(f"  {name}")
    print("=" * 70)
    print("\n  [Telemetry]")
    print(f"    [Mesh Load]    process=True:          {t.get('mesh_load', 0):.3f} s")
    print(f"    [Scaffolding]  GMSH nodes/struts:     {t['scaffolding']:.3f} s")
    print(f"    [Topology]     Internal vs surface:   {t['topology']:.3f} s")
    print(f"    [Math]         calculate_k_analytical: {t['math']:.6f} s")
    print(f"    [Geometry]     Cylinders ({t['geometry_cylinders_count']}): {t['geometry_cylinders']:.3f} s")
    print(f"    [Boolean]      recursive_batch_union (batch={t['geometry_union_batch_size']}): {t['geometry_union']:.3f} s")
    print(f"    [Boolean]      Intersect (clip):      {t['geometry_intersect']:.3f} s")
    print(f"    [Boolean]      volume() (eval):       {t.get('geometry_volume', 0):.3f} s  <- actual CSG eval")
    print(f"    [Export]       manifold_to_trimesh + STL: {t['geometry_convert']:.3f} s + {t['export']:.3f} s")
    print(f"\n  [Accuracy]")
    print(f"    Predicted Vf (analytical): {data['predicted_vf']:.4%}")
    print(f"    Achieved Vf (manifold):    {data['achieved_vf']:.4%}")
    print(f"    Target Vf:                 {data['target_vf']:.4%}")
    print(f"    Overlap Error %:           {data['overlap_error_pct']:+.2f}%")
    accounted = (
        t.get("mesh_load", 0) + t["scaffolding"] + t["topology"] + t["math"]
        + t["geometry_cylinders"] + t["geometry_union"] + t["geometry_intersect"]
        + t.get("geometry_volume", 0) + t["geometry_convert"] + t["export"]
    )
    if t.get("geometry_total"):
        print(f"    [Geometry]    Total (wall):          {t['geometry_total']:.3f} s")
    print(f"\n  [Total] {t['total']:.2f} s (accounted: {accounted:.2f} s)")


def bottleneck_analysis(runs: list[tuple[str, dict]]) -> None:
    """Print bottleneck analysis: which step took >40% of total time."""
    print("\n" + "=" * 70)
    print("  BOTTLENECK ANALYSIS")
    print("=" * 70)

    for name, data in runs:
        t = data["timings"]
        total = t["total"]
        steps = [
        ("Mesh Load", t.get("mesh_load", 0)),
        ("Scaffolding", t["scaffolding"]),
        ("Topology", t["topology"]),
        ("Math", t["math"]),
        ("Cylinder Creation", t["geometry_cylinders"]),
        ("recursive_batch_union", t["geometry_union"]),
        ("Intersect (clip)", t["geometry_intersect"]),
        ("volume() (Boolean eval)", t.get("geometry_volume", 0)),
        ("manifold_to_trimesh", t["geometry_convert"]),
        ("STL Export", t["export"]),
    ]
        bottlenecks = [s for s, v in steps if total > 0 and (v / total) > 0.40]
        pcts = {s: (v / total * 100) for s, v in steps}
        print(f"\n  {name}:")
        for step, pct in sorted(pcts.items(), key=lambda x: -x[1]):
            marker = " <<< BOTTLENECK (>40%)" if pct > 40 else ""
            print(f"    {step}: {pct:.1f}%{marker}")
        if bottlenecks:
            print(f"  -> Primary bottleneck(s): {', '.join(bottlenecks)}")


def union_scaling_analysis(runs: list[tuple[str, dict]]) -> None:
    """Analyze recursive_batch_union scaling: linear vs exponential."""
    print("\n" + "=" * 70)
    print("  RECURSIVE_BATCH_UNION SCALING")
    print("=" * 70)

    for i, (name, data) in enumerate(runs):
        t = data["timings"]
        n = data["n_cylinders"]
        t_union = t["geometry_union"]
        print(f"\n  {name}:")
        print(f"    Cylinders: {n}")
        print(f"    Union time: {t_union:.3f} s")
        if n > 0:
            print(f"    Time per cylinder: {t_union / n * 1000:.3f} ms")

    if len(runs) >= 2:
        n1, t1 = runs[0][1]["n_cylinders"], runs[0][1]["timings"]["geometry_union"]
        n2, t2 = runs[1][1]["n_cylinders"], runs[1][1]["timings"]["geometry_union"]
        if n1 > 0 and t1 > 0:
            ratio_n = n2 / n1
            ratio_t = t2 / t1
            print(f"\n  Adapter -> MariaTubeRack:")
            print(f"    Cylinder count ratio: {ratio_n:.2f}x")
            print(f"    Union time ratio:     {ratio_t:.2f}x")
            if ratio_t < ratio_n * 1.2:
                print(f"    Scaling: ~LINEAR (union time scales with cylinder count)")
            else:
                print(f"    Scaling: SUPER-LINEAR (possible exponential component)")


def main() -> None:
    print("=" * 70)
    print("  GRAPHITE PRODUCTION BENCHMARK")
    print("=" * 70)

    base = Path("test_parts")
    if not base.exists():
        base = Path(".")

    runs: list[tuple[str, dict]] = []

    # Run 1: Part2_Adapter
    adapter_path = base / "Part2_Adapter.STL"
    if not adapter_path.exists():
        adapter_path = Path("Part2_Adapter.STL")
    if adapter_path.exists():
        print("\n  Run 1: Part2_Adapter | Kagome | 6mm | 15% VF | fast_solve=True")
        data = run_benchmark(
            stl_path=adapter_path,
            target_vf=0.15,
            element_size=6.0,
            output_name="benchmark_Adapter_15pct.stl",
        )
        runs.append(("Part2_Adapter", data))
        print_telemetry("Part2_Adapter (15% VF)", data)
    else:
        print(f"\n  Skipping Adapter: not found at {adapter_path}")

    # Run 2: MariaTubeRack
    rack_path = base / "MariaTubeRack_Full.STL"
    if not rack_path.exists():
        rack_path = Path("MariaTubeRack_Full.STL")
    if rack_path.exists():
        print("\n  Run 2: MariaTubeRack_Full | Kagome | 6mm | 10% VF | fast_solve=True")
        data = run_benchmark(
            stl_path=rack_path,
            target_vf=0.10,
            element_size=6.0,
            output_name="benchmark_MariaTubeRack_10pct.stl",
        )
        runs.append(("MariaTubeRack_Full", data))
        print_telemetry("MariaTubeRack_Full (10% VF)", data)
    else:
        print(f"\n  Skipping MariaTubeRack: not found at {rack_path}")

    if runs:
        bottleneck_analysis(runs)
        union_scaling_analysis(runs)

    print("\n  Optimization notes:")
    print("  - If volume() (Boolean eval) is the bottleneck: manifold3d defers CSG until")
    print("    volume()/to_mesh(); consider simplifying the boundary or reducing cylinders.")
    print("  - If Cylinder Creation is the bottleneck: investigate batch Manifold creation")
    print("    (single C++ call for many cylinders).")
    print("=" * 70)


if __name__ == "__main__":
    main()
