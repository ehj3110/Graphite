"""
graphite/explicit/interlinked/__main__.py

Package entrypoint allowing: python -m graphite.explicit.interlinked [options]
"""

from __future__ import annotations

import sys
from .cli import main

if __name__ == "__main__":
    sys.exit(main())
