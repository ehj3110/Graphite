"""Launcher & batch recipe CLI for Graphite Implicit Studio.

Usage:
    .venv\\Scripts\\python.exe scripts\\launch_implicit_gui.py
    .venv\\Scripts\\python.exe scripts\\launch_implicit_gui.py --check-env
    .venv\\Scripts\\python.exe scripts\\launch_implicit_gui.py --recipe recipe.json
    .venv\\Scripts\\python.exe scripts\\launch_implicit_gui.py --batch r1.json r2.json --output-dir outputs/batch
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Graphite Implicit Studio Desktop & Headless Recipe Launcher")
    parser.add_argument("--check-env", action="store_true", help="Verify workstation dependencies and exit")
    parser.add_argument("--recipe", type=Path, default=None, help="Run single saved JSON recipe headlessly")
    parser.add_argument("--batch", type=Path, nargs="+", default=None, help="Run multiple saved JSON recipes sequentially")
    parser.add_argument("--output-dir", type=Path, default=None, help="Override output root directory for headless runs")
    args, qt_args = parser.parse_known_args(argv)

    if args.check_env:
        from scripts.setup_implicit_studio_env import check_environment

        ok, lines = check_environment()
        print("\n".join(lines))
        return 0 if ok else 1

    recipe_queue: list[Path] = []
    if args.recipe is not None:
        recipe_queue.append(args.recipe)
    if args.batch:
        recipe_queue.extend(args.batch)

    if recipe_queue:
        from graphite.ui.implicit_state import run_recipe_batch

        results = run_recipe_batch(
            recipe_queue,
            output_root=args.output_dir,
            progress_cb=lambda idx, total, rpath, msg: print(
                f"[{min(idx + 1, max(total, 1))}/{max(total, 1)} {rpath.name}] {msg}"
            ),
        )
        for rpath, res in results:
            files = ", ".join(str(p) for p in res.output_paths)
            print(f"Completed {rpath.name} in {res.elapsed_seconds:.1f}s -> {files}")
        return 0

    from graphite.ui.implicit_studio import launch_implicit_studio

    return launch_implicit_studio([sys.argv[0], *qt_args])


if __name__ == "__main__":
    raise SystemExit(main())
