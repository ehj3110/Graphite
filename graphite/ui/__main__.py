"""
Graphite UI entrypoint.

Usage:
  python -m graphite.ui          # Launch interactive Trame + PyVista web app
  python -m graphite.ui.cli ...  # Run headless CLI recipe runner
"""
from __future__ import annotations

import sys
from graphite.ui.trame_app import create_app


def main():
    server = create_app()
    port = 8080
    host = "localhost"
    print(f"\n========================================================")
    print(f"  Graphite Conformal Lattice Studio (Trame + PyVista)")
    print(f"  Serving at: http://{host}:{port}/")
    print(f"========================================================\n")
    server.start(port=port, host=host)


if __name__ == "__main__":
    main()
