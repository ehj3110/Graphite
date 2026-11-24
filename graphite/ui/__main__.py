"""
Graphite UI entrypoint.

Usage:
  python -m graphite.ui                 # Launch Graphite Implicit Studio (PySide6 + VTK)
  python -m graphite.ui --legacy-trame  # Launch legacy Trame + PyVista web app
  python -m graphite.ui.cli ...         # Run headless CLI recipe runner
"""
from __future__ import annotations

import sys


def main():
    if "--legacy-trame" in sys.argv[1:]:
        from graphite.ui.trame_app import create_app

        server = create_app()
        port = 8080
        host = "localhost"
        print("\n========================================================")
        print("  Graphite Conformal Lattice Studio (Legacy Trame)")
        print(f"  Serving at: http://{host}:{port}/")
        print("========================================================\n")
        server.start(port=port, host=host)
        return

    from graphite.ui.implicit_studio import launch_implicit_studio

    raise SystemExit(launch_implicit_studio(sys.argv))


if __name__ == "__main__":
    main()
