#!/usr/bin/env python3
"""Launcher — v4.1 K′ bake lives in encoder/v4.1 so v4 stays frozen."""
from __future__ import annotations

import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parent / "v4.1" / "bake.py"), run_name="__main__")
