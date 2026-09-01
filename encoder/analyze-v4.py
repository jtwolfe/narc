#!/usr/bin/env python3
"""Launcher — v4 encoder lives in encoder/v4/ so v0–v4r scripts stay untouched."""
from __future__ import annotations

import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parent / "v4" / "analyze.py"), run_name="__main__")
