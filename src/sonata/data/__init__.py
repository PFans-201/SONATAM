"""sonata.data — dataset ingestion, dual-branch feature extraction."""

from sonata.data.linker import LakhMSDLinker
from sonata.data.msd_reader import read_msd_metadata
from sonata.data.semantic_analyzer import SemanticAnalyzer

__all__ = [
    "LakhMSDLinker",
    "SemanticAnalyzer",
    "read_msd_metadata",
]

# JSymbolicExtractor is optional (requires Java + JAR)
try:
    from sonata.data.jsymbolic_wrapper import JSymbolicExtractor
    __all__.append("JSymbolicExtractor")
except Exception:
    pass
