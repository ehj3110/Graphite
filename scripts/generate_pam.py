#!/usr/bin/env python3
"""
scripts/generate_pam.py

Turnkey Generator for Print-Ready Interlinked Metamaterials & Chainmail Fabrics.

Usage:
  # 3D Shock Absorber (40x40x40 mm box, 750 um strut, 200 um clearance):
  python scripts/generate_pam.py --preset shock_absorber --box 40 40 40 --out outputs/shock.3mf

  # NASA JPL Space Fabric (80x80 mm sheet):
  python scripts/generate_pam.py --preset space_fabric --width 80 --length 80 --out outputs/fabric.3mf

  # DfAM Preflight Dry Run (Checks cage diameter and clearance without generating mesh):
  python scripts/generate_pam.py --cell d4tet --box 50 50 50 --strut 0.75 --clearance 0.20 --dry-run

  # Tensile Specimen with Solid Perimeter Handling Frame and Scientific Preview:
  python scripts/generate_pam.py --cell d4tet --box 40 40 20 --frame --preview outputs/specimen.png
"""

from __future__ import annotations

import sys
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from graphite.explicit.interlinked.cli import main

if __name__ == "__main__":
    sys.exit(main())
