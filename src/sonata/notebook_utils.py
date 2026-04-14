"""
notebook_utils.py
=================
Small display helpers for use inside Jupyter notebooks.

These are notebook-only utilities and are intentionally kept separate from
the core library so that the library itself has no Jupyter dependency.

Usage
-----
>>> from sonata.notebook_utils import rp, pp, show_path, show_paths
>>> print(f"Saved to {rp(some_path)}")
>>> midi_root = pp(CFG["data"]["midi_root"])   # absolute path from config string
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Union

# Re-export for convenience — notebooks can do a single import line
from sonata.config.settings import resolve_path  # noqa: F401

__all__ = ["project_root", "rp", "pp", "resolve_path", "show_path", "show_paths"]

# ── Project root resolution ───────────────────────────────────────────────────
# __file__ = src/sonata/notebook_utils.py
#  parents[0] = src/sonata/
#  parents[1] = src/
#  parents[2] = <project root>
project_root: Path = Path(__file__).parents[2].resolve()


def pp(cfg_value: Union[str, Path]) -> Path:
    """
    Convert a config-relative path string to an **absolute Path**.

    Shorthand for :func:`sonata.config.settings.resolve_path` — designed
    for concise use in notebook cells.

    Parameters
    ----------
    cfg_value : str or Path
        A raw path value from ``CFG``, e.g. ``CFG["data"]["midi_root"]``.

    Returns
    -------
    Path
        Absolute path under the project root.

    Examples
    --------
    >>> pp("data/raw/lmd_matched")
    PosixPath('/home/.../SONATAM/data/raw/lmd_matched')
    """
    return resolve_path(cfg_value)


def rp(path: Union[str, Path]) -> str:
    """
    Return *path* as a string **relative to the project root**.

    If the path is not inside the project tree (e.g. an absolute system path
    given by the user), the full path is returned unchanged so nothing is lost.

    Parameters
    ----------
    path : str or Path

    Returns
    -------
    str
        Relative path string, e.g. ``"data/processed/curated_dataset.parquet"``.

    Examples
    --------
    >>> rp("/home/user/SONATAM/data/processed/curated_dataset.parquet")
    'data/processed/curated_dataset.parquet'
    >>> rp("data/raw/lmd_matched")   # already relative — returned as-is
    'data/raw/lmd_matched'
    """
    p = Path(path)
    # Try both the raw path and the fully-resolved path so that paths
    # constructed from sys.prefix (not resolved) and paths that have been
    # through Path.resolve() both work correctly.
    for candidate in (p, p.resolve()):
        try:
            return str(candidate.relative_to(project_root))
        except ValueError:
            continue
    # Neither form is inside the project root — return as-is
    return str(path)


def show_path(label: str, path: Union[str, Path]) -> None:
    """Print ``<label>: <relative path>``."""
    print(f"{label}: {rp(path)}")


def show_paths(paths: Iterable[tuple[str, Union[str, Path]]]) -> None:
    """
    Print a labelled list of relative paths.

    Parameters
    ----------
    paths : iterable of (label, path) tuples

    Examples
    --------
    >>> show_paths([("MIDI", midi_path), ("MusicXML", xml_path)])
    MIDI    : data/generated/demo.mid
    MusicXML: data/generated/demo.musicxml
    """
    pairs = list(paths)
    width = max(len(label) for label, _ in pairs)
    for label, path in pairs:
        print(f"  {label:<{width}} → {rp(path)}")
