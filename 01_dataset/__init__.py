"""
01_dataset — Dataset curation and manipulation.

Modules
-------
harmonic_analyzer : MIDIHarmonicAnalyzer class
msd_reader        : read_msd_metadata() for MSD HDF5 files
linker            : LakhMSDLinker — builds unified MIDI + metadata DataFrames
"""

from .harmonic_analyzer import MIDIHarmonicAnalyzer
from .msd_reader import read_msd_metadata, KEY_NAMES
from .linker import LakhMSDLinker

__all__ = ["MIDIHarmonicAnalyzer", "read_msd_metadata", "KEY_NAMES", "LakhMSDLinker"]
