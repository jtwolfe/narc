#!/usr/bin/env python3
"""Launcher — real v4r encoder lives in encoder/v4r/ so v0–v3 scripts stay untouched."""
from __future__ import annotations

import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parent / "v4r" / "analyze.py"), run_name="__main__")
