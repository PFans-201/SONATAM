"""
schema.py
=========
RDF ontology definition for the SONATAM harmonic knowledge graph.

Defines node types, edge types, and data properties for the hybrid
recommendation / link-prediction knowledge graph.

Node types
----------
MusicalPiece, Artist, Genre, MusicalKey, ChordProgression, Chord, Era, User, Interaction

Edge types
----------
hasArtist, hasGenre, hasGlobalKey, hasProgression, hasChord, hasEra,
transitionsTo, listenedTo, rated

Main class
----------
HarmonicKGSchema
    Provides namespace URIs, node/edge type enums, and URI factory methods.
"""

from __future__ import annotations

from typing import Tuple

try:
    from rdflib import Namespace, URIRef, Literal, RDF, RDFS, XSD
    _RDF_AVAILABLE = True
except ImportError:
    _RDF_AVAILABLE = False

__all__ = ["HarmonicKGSchema"]

# -- Namespace ---------------------------------------------------------------
HKG = Namespace("http://harmonic-kg.org/") if _RDF_AVAILABLE else None


class HarmonicKGSchema:
    """
    Ontology / schema for the SONATAM knowledge graph.

    Provides:

    * **Node type** constants and their RDF classes.
    * **Edge type** constants and their RDF properties.
    * **URI factory** methods for minting unique resource identifiers.
    * **Data properties** list for each node type.

    All URIs live under ``http://harmonic-kg.org/``.
    """

    NAMESPACE = "http://harmonic-kg.org/"

    # -- Node types -----------------------------------------------------------
    NODE_TYPES = [
        "MusicalPiece",
        "Artist",
        "Genre",
        "MusicalKey",
        "ChordProgression",
        "Chord",
        "Era",
        "User",
        "Interaction",
    ]

    # -- Edge types (predicate, src_type, dst_type) ---------------------------
    EDGE_TYPES = [
        ("hasArtist",      "MusicalPiece", "Artist"),
        ("hasGenre",       "MusicalPiece", "Genre"),
        ("hasGlobalKey",   "MusicalPiece", "MusicalKey"),
        ("hasProgression", "MusicalPiece", "ChordProgression"),
        ("hasChord",       "ChordProgression", "Chord"),
        ("hasEra",         "MusicalPiece", "Era"),
        ("transitionsTo",  "Chord", "Chord"),
        ("listenedTo",     "User", "MusicalPiece"),
        ("rated",          "User", "MusicalPiece"),
    ]

    # -- Data properties per node type ----------------------------------------
    DATA_PROPERTIES = {
        "MusicalPiece": [
            "title", "release", "year", "duration", "tempo",
            "time_signature", "loudness", "danceability", "energy",
        ],
        "Artist":           ["artist_name"],
        "Genre":            ["genre_name"],
        "MusicalKey":       ["key_name", "mode"],
        "ChordProgression": ["progression_string"],
        "Chord":            ["chord_label", "roman_numeral"],
        "Era":              ["era_label", "decade_start"],
        "User":             ["user_id"],
        "Interaction":      ["play_count", "rating", "timestamp"],
    }

    # ------------------------------------------------------------------
    #  URI factory methods
    # ------------------------------------------------------------------

    @staticmethod
    def song_uri(track_id: str) -> str:
        """Mint a URI for a MusicalPiece node."""
        return f"{HarmonicKGSchema.NAMESPACE}piece/{track_id}"

    @staticmethod
    def artist_uri(artist_id: str) -> str:
        """Mint a URI for an Artist node."""
        safe = artist_id.replace(" ", "_")
        return f"{HarmonicKGSchema.NAMESPACE}artist/{safe}"

    @staticmethod
    def genre_uri(genre: str) -> str:
        """Mint a URI for a Genre node."""
        safe = genre.lower().replace(" ", "_").replace("/", "_")
        return f"{HarmonicKGSchema.NAMESPACE}genre/{safe}"

    @staticmethod
    def key_uri(key_name: str, mode: str = "") -> str:
        """Mint a URI for a MusicalKey node."""
        label = f"{key_name}_{mode}" if mode else key_name
        return f"{HarmonicKGSchema.NAMESPACE}key/{label}"

    @staticmethod
    def chord_uri(chord_label: str) -> str:
        """Mint a URI for a Chord node."""
        safe = chord_label.replace("#", "sharp").replace(" ", "_")
        return f"{HarmonicKGSchema.NAMESPACE}chord/{safe}"

    @staticmethod
    def progression_uri(track_id: str, index: int = 0) -> str:
        """Mint a URI for a ChordProgression node."""
        return f"{HarmonicKGSchema.NAMESPACE}progression/{track_id}_{index}"

    @staticmethod
    def era_uri(era_label: str) -> str:
        """Mint a URI for an Era node."""
        return f"{HarmonicKGSchema.NAMESPACE}era/{era_label}"

    @staticmethod
    def user_uri(user_id: str) -> str:
        """Mint a URI for a User node."""
        return f"{HarmonicKGSchema.NAMESPACE}user/{user_id}"

    @staticmethod
    def interaction_uri(user_id: str, track_id: str) -> str:
        """Mint a URI for a User-Piece interaction."""
        return f"{HarmonicKGSchema.NAMESPACE}interaction/{user_id}_{track_id}"

    # ------------------------------------------------------------------
    #  RDF helpers
    # ------------------------------------------------------------------

    @staticmethod
    def rdf_class(node_type: str):
        """Return the rdflib URIRef for a node-type class."""
        if not _RDF_AVAILABLE:
            raise ImportError("rdflib is required: pip install rdflib")
        return HKG[node_type]

    @staticmethod
    def rdf_property(edge_type: str):
        """Return the rdflib URIRef for an edge-type property."""
        if not _RDF_AVAILABLE:
            raise ImportError("rdflib is required: pip install rdflib")
        return HKG[edge_type]
