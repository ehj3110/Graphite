"""
conftest.py — Project-level pytest configuration for Graphite.

Adds the workspace root to sys.path so that 'graphite' is importable
without a formal package install.
"""
import sys
import os

# Ensure the project root is on sys.path so `graphite` is importable
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
