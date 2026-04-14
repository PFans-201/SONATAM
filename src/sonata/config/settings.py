"""
settings.py
===========
Load ``config.yaml`` and expose a single ``CFG`` dict.

Usage
-----
>>> from sonata.config.settings import CFG, resolve_path
>>> midi_root = resolve_path(CFG["data"]["midi_root"])

# Or reload from a custom path:
>>> from sonata.config.settings import load
>>> cfg = load("/path/to/my_config.yaml")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Union

# ── Optional PyYAML; fall back to an empty dict ───────────────────────────────
try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False

__all__ = ["CFG", "PROJECT_ROOT", "load", "resolve_path"]

# Config lives at the project root: <repo_root>/config/config.yaml
# __file__ = src/sonata/config/settings.py → parents[3] = project root
PROJECT_ROOT: Path = Path(__file__).parents[3].resolve()
_DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


def resolve_path(relative: Union[str, Path]) -> Path:
    """
    Resolve a config-relative path to an absolute path under ``PROJECT_ROOT``.

    All paths in ``config.yaml`` are written relative to the project root
    (e.g. ``"data/raw/lmd_matched"``).  This function prepends the project
    root so the result is valid regardless of the current working directory.

    If the path is already absolute it is returned unchanged.

    Parameters
    ----------
    relative : str or Path
        A path string from ``CFG``, e.g. ``CFG["data"]["midi_root"]``.

    Returns
    -------
    Path
        Absolute path.

    Examples
    --------
    >>> resolve_path("data/raw/lmd_matched")
    PosixPath('/home/.../SONATAM/data/raw/lmd_matched')
    >>> resolve_path("/absolute/path")   # already absolute — unchanged
    PosixPath('/absolute/path')
    """
    p = Path(relative)
    if p.is_absolute():
        return p
    return (PROJECT_ROOT / p).resolve()


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
    CFG = {}   # silently empty if yaml not installed yet or config not present
