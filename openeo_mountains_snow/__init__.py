"""Workspace package shim.

This makes the repository root behave like an installed package by
resolving submodules from src/openeo_mountains_snow.
"""

from pathlib import Path

_pkg_dir = Path(__file__).resolve().parent
_src_pkg_dir = _pkg_dir.parent / "src" / "openeo_mountains_snow"

if _src_pkg_dir.is_dir():
    __path__ = [str(_src_pkg_dir)]
else:
    __path__ = [str(_pkg_dir)]
