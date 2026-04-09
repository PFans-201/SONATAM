"""
schema.py
=========
Knowledge-graph schema constants for the SONATA project.

Defines
-------
* RDF namespace objects (``rdflib.Namespace``)
* Node-type URIs — Song, Artist, Genre, Chord, Key, ChordProgression
* Edge-type URIs — hasArtist, hasGenre, hasKey, hasChord, transitionsTo, …
* ``HarmonicKGSchema`` — static container class for all namespaces & URIs
"""

from __future__ import annotations

from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import OWL, RDF, RDFS, XSD

__all__ = ["HarmonicKGSchema", "NS"]


# ── Project-level RDF namespace ────────────────────────────────────────
NS = Namespace("http://harmonic-kg.org/")


class HarmonicKGSchema:
    """
    Static container for all RDF namespaces and URI factory methods used
    by the SONATA project.

    Usage
    -----
    >>> from sonata.kg.schema import HarmonicKGSchema as S
    >>> g = rdflib.Graph()
    >>> g.bind("hkg", S.NS)
    >>> song_uri = S.song_uri("TRAAAGR128F425B14B")
    """

    # Namespaces
    NS     = NS
    OWL    = OWL
    RDF    = RDF
    RDFS   = RDFS
    XSD    = XSD

    # ── Class URIs ─────────────────────────────────────────────────────
    Song              = NS["Song"]
    Artist            = NS["Artist"]
    Genre             = NS["Genre"]
    MusicalKey        = NS["MusicalKey"]
    Chord             = NS["Chord"]
    ChordProgression  = NS["ChordProgression"]
    ChordTransition   = NS["ChordTransition"]

    # ── Property URIs ──────────────────────────────────────────────────
    # Song-level
    hasTrackId        = NS["hasTrackId"]
    hasTitle          = NS["hasTitle"]
    hasArtist         = NS["hasArtist"]
    hasArtistName     = NS["hasArtistName"]
    hasYear           = NS["hasYear"]
    hasTempo          = NS["hasTempo"]
    hasDuration       = NS["hasDuration"]
    hasLoudness       = NS["hasLoudness"]
    hasDanceability   = NS["hasDanceability"]
    hasEnergy         = NS["hasEnergy"]
    hasTimeSignature  = NS["hasTimeSignature"]
    hasMatchScore     = NS["hasMatchScore"]

    # Genre / tags
    hasGenre          = NS["hasGenre"]
    hasPrimaryGenre   = NS["hasPrimaryGenre"]
    genreLabel        = NS["genreLabel"]

    # Key / harmony
    hasGlobalKey      = NS["hasGlobalKey"]
    keyName           = NS["keyName"]
    keyMode           = NS["keyMode"]
    msdKeyName        = NS["msdKeyName"]
    msdKeyConfidence  = NS["msdKeyConfidence"]

    # Chord & progression
    hasProgression    = NS["hasProgression"]
    hasChord          = NS["hasChord"]
    chordHarte        = NS["chordHarte"]
    chordRoman        = NS["chordRoman"]
    chordFunction     = NS["chordFunction"]
    chordPosition     = NS["chordPosition"]
    chordDuration     = NS["chordDuration"]

    # Transition
    transitionsTo     = NS["transitionsTo"]
    transitionProb    = NS["transitionProb"]
    fromChord         = NS["fromChord"]
    toChord           = NS["toChord"]

    # Harmonic features (numeric)
    numModulations         = NS["numModulations"]
    chordVocabRoman        = NS["chordVocabRoman"]
    uniqueChordRatio       = NS["uniqueChordRatio"]
    transitionEntropy      = NS["transitionEntropy"]
    harmRhythmMean         = NS["harmRhythmMean"]
    avgChordCardinality    = NS["avgChordCardinality"]
    intervalClassVector    = NS["intervalClassVector"]
    funcRatioT             = NS["funcRatioT"]
    funcRatioD             = NS["funcRatioD"]
    funcRatioS             = NS["funcRatioS"]
    funcRatioPD            = NS["funcRatioPD"]

    # ── URI factory methods ────────────────────────────────────────────

    @staticmethod
    def song_uri(track_id: str) -> URIRef:
        """URIRef for a Song node, e.g. ``hkg:song/TRAAAGR128F425B14B``."""
        return NS[f"song/{track_id}"]

    @staticmethod
    def artist_uri(artist_id: str) -> URIRef:
        """URIRef for an Artist node."""
        return NS[f"artist/{artist_id}"]

    @staticmethod
    def genre_uri(genre_label: str) -> URIRef:
        """URIRef for a Genre node (label is slugified)."""
        slug = genre_label.replace(" ", "_").replace("/", "-").lower()
        return NS[f"genre/{slug}"]

    @staticmethod
    def key_uri(key_name: str) -> URIRef:
        """URIRef for a MusicalKey node, e.g. ``hkg:key/C_major``."""
        slug = key_name.replace(" ", "_").replace("#", "sharp").replace("-", "flat")
        return NS[f"key/{slug}"]

    @staticmethod
    def chord_uri(harte_label: str) -> URIRef:
        """URIRef for a Chord node, e.g. ``hkg:chord/C:maj``."""
        slug = harte_label.replace(":", "_").replace("#", "sharp").replace("/", "-")
        return NS[f"chord/{slug}"]

    @staticmethod
    def progression_uri(track_id: str, midi_file: str) -> URIRef:
        """URIRef for a ChordProgression node tied to a specific MIDI file."""
        return NS[f"progression/{track_id}/{midi_file.replace('.', '_')}"]

    @staticmethod
    def transition_uri(from_harte: str, to_harte: str) -> URIRef:
        """URIRef for a ChordTransition node between two Harte-labelled chords."""
        a = from_harte.replace(":", "_").replace("#", "sharp")
        b = to_harte.replace(":", "_").replace("#", "sharp")
        return NS[f"transition/{a}__{b}"]

    # ── Graph initialiser ──────────────────────────────────────────────

    @classmethod
    def new_graph(cls) -> Graph:
        """Return a new rdflib Graph with all project namespaces pre-bound."""
        g = Graph()
        g.bind("hkg",  cls.NS)
        g.bind("owl",  OWL)
        g.bind("rdf",  RDF)
        g.bind("rdfs", RDFS)
        g.bind("xsd",  XSD)
        return g
