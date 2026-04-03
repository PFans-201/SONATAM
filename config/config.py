"""
config.py
=========
Load ``config.yaml`` and expose a single ``CFG`` dict.

Usage
-----
>>> from harmonic_kg_project.config import CFG
>>> midi_root = CFG["data"]["midi_root"]
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

# ── Optional PyYAML; fall back to a hard-coded empty dict ────────────────────
try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False

__all__ = ["CFG", "load"]

_DEFAULT_CONFIG_PATH = Path(__file__).parent / "config.yaml"


def load(path: str | Path | None = None) -> Dict[str, Any]:
    """
    Load the YAML config file and return it as a nested dict.

    Parameters
    ----------
    path : str or Path, optional
        Override the default location (``config/config.yaml``).

    Returns
    -------
    dict
    """
    p = Path(path) if path else _DEFAULT_CONFIG_PATH
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    if not _YAML_AVAILABLE:
        raise ImportError("PyYAML is required: pip install pyyaml")
    with open(p, "r") as f:
        return yaml.safe_load(f)


# ── Module-level singleton ─────────────────────────────────────────────────
try:
    CFG: Dict[str, Any] = load()
except Exception:
    CFG = {}   # silently empty if yaml not installed yet
