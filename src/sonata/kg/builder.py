"""
builder.py
==========
Convert the curated pandas DataFrame into an RDF knowledge graph (rdflib)
and/or a NetworkX directed graph for graph-ML pipelines.

Main class
----------
KGBuilder
    from_dataframe(df)         → rdflib.Graph  (RDF triples)
    to_networkx(df)            → nx.DiGraph     (for GNN / graph analytics)
    save(graph, path, fmt)     → writes Turtle / JSON-LD / N-Triples
    load(path)                 → rdflib.Graph
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Optional

import pandas as pd
from rdflib import Graph, Literal, URIRef
from rdflib.namespace import RDF, RDFS, XSD

from sonata.kg.schema import HarmonicKGSchema as S

__all__ = ["KGBuilder"]

# Supported rdflib serialisation formats
_FORMAT_MAP = {
    ".ttl":      "turtle",
    ".nt":       "ntriples",
    ".jsonld":   "json-ld",
    ".json":     "json-ld",
    ".n3":       "n3",
    ".xml":      "xml",
}


class KGBuilder:
    """
    Build an RDF knowledge graph from the unified curated dataset.

    The graph represents the following entity types and relationships::

        Song  ─── hasArtist    ──► Artist
              ─── hasGenre     ──► Genre
              ─── hasGlobalKey ──► MusicalKey
              ─── hasProgression──► ChordProgression
                                        └── hasChord ──► Chord
        Chord ─── transitionsTo ──► Chord  (with probability literal)

    Usage
    -----
    >>> from sonata.kg.builder import KGBuilder
    >>> builder = KGBuilder()
    >>> g = builder.from_dataframe(df)
    >>> builder.save(g, "data/processed/harmonic_kg.ttl")
    """

    def __init__(self) -> None:
        pass

    # ─────────────────────────────────────────────────────────────────────
    #  Main entry: DataFrame → RDF Graph
    # ─────────────────────────────────────────────────────────────────────

    def from_dataframe(
        self,
        df: pd.DataFrame,
        include_progressions: bool = True,
        verbose: bool = True,
    ) -> Graph:
        """
        Convert a curated dataset DataFrame into an rdflib RDF graph.

        Parameters
        ----------
        df : pd.DataFrame
            Output of ``LakhMSDLinker.build_dataset()``.
        include_progressions : bool
            If True (and chord-level columns are present in df), add
            per-song ChordProgression nodes with individual Chord nodes.
            Set to False for a lightweight metadata-only graph.
        verbose : bool
            Print progress.

        Returns
        -------
        rdflib.Graph
        """
        g = S.new_graph()

        for i, row in df.iterrows():
            track_id = str(row.get("track_id", f"unknown_{i}"))
            song_uri = S.song_uri(track_id)

            # ── Song node ─────────────────────────────────────────────
            g.add((song_uri, RDF.type, S.Song))

            self._add_literal(g, song_uri, S.hasTrackId, track_id, XSD.string)

            if pd.notna(row.get("title")):
                self._add_literal(g, song_uri, S.hasTitle, str(row["title"]), XSD.string)
            if pd.notna(row.get("artist_name")):
                self._add_literal(g, song_uri, S.hasArtistName, str(row["artist_name"]), XSD.string)
            if pd.notna(row.get("year")) and int(row["year"]) > 0:
                self._add_literal(g, song_uri, S.hasYear, int(row["year"]), XSD.integer)
            if pd.notna(row.get("msd_tempo")):
                self._add_literal(g, song_uri, S.hasTempo, float(row["msd_tempo"]), XSD.float)
            if pd.notna(row.get("msd_duration")):
                self._add_literal(g, song_uri, S.hasDuration, float(row["msd_duration"]), XSD.float)
            if pd.notna(row.get("msd_loudness")):
                self._add_literal(g, song_uri, S.hasLoudness, float(row["msd_loudness"]), XSD.float)
            if pd.notna(row.get("msd_danceability")):
                self._add_literal(g, song_uri, S.hasDanceability, float(row["msd_danceability"]), XSD.float)
            if pd.notna(row.get("msd_energy")):
                self._add_literal(g, song_uri, S.hasEnergy, float(row["msd_energy"]), XSD.float)
            if pd.notna(row.get("msd_time_sig")):
                self._add_literal(g, song_uri, S.hasTimeSignature, int(row["msd_time_sig"]), XSD.integer)
            if pd.notna(row.get("match_score")):
                self._add_literal(g, song_uri, S.hasMatchScore, float(row["match_score"]), XSD.float)

            # ── Artist node ───────────────────────────────────────────
            if pd.notna(row.get("artist_id")):
                artist_uri = S.artist_uri(str(row["artist_id"]))
                g.add((artist_uri, RDF.type, S.Artist))
                g.add((song_uri, S.hasArtist, artist_uri))
                if pd.notna(row.get("artist_name")):
                    self._add_literal(g, artist_uri, RDFS.label, str(row["artist_name"]), XSD.string)

            # ── Genre nodes ───────────────────────────────────────────
            if pd.notna(row.get("top3_genres")):
                for genre_label in str(row["top3_genres"]).split(";"):
                    genre_label = genre_label.strip()
                    if not genre_label:
                        continue
                    genre_uri = S.genre_uri(genre_label)
                    g.add((genre_uri, RDF.type, S.Genre))
                    self._add_literal(g, genre_uri, S.genreLabel, genre_label, XSD.string)
                    g.add((song_uri, S.hasGenre, genre_uri))

            # ── Key node ──────────────────────────────────────────────
            global_key = row.get("global_key") or row.get("msd_key_name")
            global_mode = row.get("global_mode") or row.get("msd_mode_name", "")
            if pd.notna(global_key):
                key_str = f"{global_key} {global_mode}".strip()
                key_uri = S.key_uri(key_str)
                g.add((key_uri, RDF.type, S.MusicalKey))
                self._add_literal(g, key_uri, S.keyName, str(global_key), XSD.string)
                self._add_literal(g, key_uri, S.keyMode, str(global_mode), XSD.string)
                g.add((song_uri, S.hasGlobalKey, key_uri))

            # ── Harmonic feature literals on Song ─────────────────────
            harm_map = {
                "num_modulations":       (S.numModulations,      XSD.integer),
                "chord_vocab_roman":     (S.chordVocabRoman,     XSD.integer),
                "unique_chord_ratio":    (S.uniqueChordRatio,    XSD.float),
                "transition_entropy":    (S.transitionEntropy,   XSD.float),
                "harm_rhythm_mean":      (S.harmRhythmMean,      XSD.float),
                "avg_chord_cardinality": (S.avgChordCardinality, XSD.float),
                "interval_class_vector": (S.intervalClassVector, XSD.string),
                "func_ratio_T":          (S.funcRatioT,          XSD.float),
                "func_ratio_D":          (S.funcRatioD,          XSD.float),
                "func_ratio_S":          (S.funcRatioS,          XSD.float),
                "func_ratio_PD":         (S.funcRatioPD,         XSD.float),
            }
            for col, (prop, dtype) in harm_map.items():
                if col in row and pd.notna(row[col]):
                    try:
                        val = int(row[col]) if dtype == XSD.integer else (
                            float(row[col]) if dtype == XSD.float else str(row[col])
                        )
                        self._add_literal(g, song_uri, prop, val, dtype)
                    except (ValueError, TypeError):
                        pass

        if verbose:
            n = len(g)
            print(f"  ✓ RDF graph built: {n:,} triples  ({len(df)} songs)")

        return g

    # ─────────────────────────────────────────────────────────────────────
    #  NetworkX export (for GNN / graph analytics)
    # ─────────────────────────────────────────────────────────────────────

    def to_networkx(self, df: pd.DataFrame):
        """
        Build a lightweight NetworkX DiGraph where:
          * Song, Artist, Genre, Key are nodes with attribute dicts
          * Edges carry the relationship type as ``rel`` attribute

        Requires: ``pip install networkx``
        """
        try:
            import networkx as nx
        except ImportError as exc:
            raise ImportError("networkx is required: pip install networkx") from exc

        G = nx.DiGraph()

        for _, row in df.iterrows():
            track_id = str(row.get("track_id", "?"))
            song_node = f"song:{track_id}"

            G.add_node(song_node, type="Song",
                       title=str(row.get("title", "")),
                       artist=str(row.get("artist_name", "")),
                       year=int(row["year"]) if pd.notna(row.get("year")) else 0,
                       tempo=float(row["msd_tempo"]) if pd.notna(row.get("msd_tempo")) else 0.0,
                       match_score=float(row["match_score"]) if pd.notna(row.get("match_score")) else 0.0,
                       global_key=str(row.get("global_key", "")),
                       primary_genre=str(row.get("primary_genre", "")))

            if pd.notna(row.get("artist_id")):
                artist_node = f"artist:{row['artist_id']}"
                G.add_node(artist_node, type="Artist", name=str(row.get("artist_name", "")))
                G.add_edge(song_node, artist_node, rel="hasArtist")

            if pd.notna(row.get("top3_genres")):
                for g_label in str(row["top3_genres"]).split(";"):
                    g_label = g_label.strip()
                    if g_label:
                        genre_node = f"genre:{g_label}"
                        G.add_node(genre_node, type="Genre", label=g_label)
                        G.add_edge(song_node, genre_node, rel="hasGenre")

            global_key = row.get("global_key") or row.get("msd_key_name")
            if pd.notna(global_key):
                key_node = f"key:{global_key}"
                G.add_node(key_node, type="MusicalKey", name=str(global_key))
                G.add_edge(song_node, key_node, rel="hasGlobalKey")

        return G

    # ─────────────────────────────────────────────────────────────────────
    #  Persistence helpers
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def save(graph: Graph, path: str, fmt: Optional[str] = None) -> None:
        """
        Serialise the graph to disk.

        Parameters
        ----------
        graph : rdflib.Graph
        path : str
            Output file path. Extension determines format if ``fmt`` is None:
            ``.ttl`` → Turtle, ``.nt`` → N-Triples, ``.jsonld`` → JSON-LD.
        fmt : str, optional
            Override rdflib serialisation format string.
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        rdflib_fmt = fmt or _FORMAT_MAP.get(p.suffix.lower(), "turtle")
        graph.serialize(destination=str(p), format=rdflib_fmt)
        print(f"  ✓ Saved {len(graph):,} triples → {p}  [{rdflib_fmt}]")

    @staticmethod
    def load(path: str) -> Graph:
        """Load an RDF graph from a serialised file."""
        g = S.new_graph()
        g.parse(path)
        print(f"  ✓ Loaded {len(g):,} triples ← {path}")
        return g

    # ─────────────────────────────────────────────────────────────────────
    #  Internal helpers
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def _add_literal(g: Graph, subject: URIRef, predicate: URIRef, value, dtype) -> None:
        g.add((subject, predicate, Literal(value, datatype=dtype)))
