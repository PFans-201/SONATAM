"""
builder.py
==========
Build the SONATAM RDF knowledge graph and / or a NetworkX graph from the
curated feature DataFrame.

The **KGBuilder** uses :class:`sonata.kg.schema.HarmonicKGSchema` to mint
URIs and then populates an ``rdflib.Graph`` with:

* MusicalPiece nodes (with jSymbolic2 + semantic feature literals)
* Artist, Genre, MusicalKey, Era entity nodes
* Edges: hasArtist, hasGenre, hasGlobalKey, hasEra
* (Optional) User interaction edges: listenedTo, rated

Main class
----------
KGBuilder
    from_dataframe(df, ...)          -> rdflib.Graph
    to_networkx(df, ...)             -> nx.DiGraph
    save(graph, path, fmt)           -> None
    load(path, fmt)                  -> rdflib.Graph
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from sonata.config.settings import CFG
from sonata.kg.schema import HarmonicKGSchema

__all__ = ["KGBuilder"]

log = logging.getLogger(__name__)


class KGBuilder:
    """
    Construct an RDF knowledge graph from a SONATAM feature DataFrame.

    Parameters
    ----------
    include_progressions : bool
        If ``True``, chord-level and progression-level nodes are added
        (creates a much larger graph).
    """

    def __init__(self, include_progressions: bool = False) -> None:
        cfg_kg = CFG.get("knowledge_graph", {})
        self.include_progressions = include_progressions or cfg_kg.get(
            "include_progressions", False
        )

    # ------------------------------------------------------------------
    #  DataFrame -> rdflib.Graph
    # ------------------------------------------------------------------

    def from_dataframe(
        self,
        df: pd.DataFrame,
        user_interactions_df: Optional[pd.DataFrame] = None,
        verbose: bool = True,
    ):
        """
        Convert the feature DataFrame into an RDF graph.

        Parameters
        ----------
        df : pd.DataFrame
            Output of ``LakhMSDLinker.build_dataset()``.
        user_interactions_df : pd.DataFrame, optional
            Columns: ``user_id``, ``track_id``, ``play_count``, ``rating``.
        verbose : bool
            Print progress information.

        Returns
        -------
        rdflib.Graph
        """
        from rdflib import Graph, URIRef, Literal, Namespace, RDF, RDFS, XSD

        HKG = Namespace(HarmonicKGSchema.NAMESPACE)
        g = Graph()
        g.bind("hkg", HKG)

        schema = HarmonicKGSchema
        stats = {
            "pieces": 0,
            "artists": set(),
            "genres": set(),
            "keys": set(),
            "eras": set(),
        }

        for _, row in df.iterrows():
            track_id = str(row.get("track_id", ""))
            if not track_id:
                continue

            piece = URIRef(schema.song_uri(track_id))
            g.add((piece, RDF.type, HKG.MusicalPiece))

            # -- Scalar data properties --
            if pd.notna(row.get("title")):
                g.add((piece, HKG.title, Literal(str(row["title"]))))
            if pd.notna(row.get("year")) and int(row["year"]) > 0:
                g.add((piece, HKG.year, Literal(int(row["year"]), datatype=XSD.integer)))
            if pd.notna(row.get("msd_duration")):
                g.add((piece, HKG.duration, Literal(float(row["msd_duration"]), datatype=XSD.double)))
            if pd.notna(row.get("msd_tempo")):
                g.add((piece, HKG.tempo, Literal(float(row["msd_tempo"]), datatype=XSD.double)))

            # -- Feature literals (jsym_* and sem_*) --
            for col in df.columns:
                if (col.startswith("jsym_") or col.startswith("sem_")) and pd.notna(row.get(col)):
                    try:
                        val = float(row[col])
                        g.add((piece, HKG[col], Literal(val, datatype=XSD.double)))
                    except (ValueError, TypeError):
                        g.add((piece, HKG[col], Literal(str(row[col]))))

            # -- Artist --
            artist_id = str(row.get("artist_id", ""))
            if artist_id:
                artist = URIRef(schema.artist_uri(artist_id))
                g.add((artist, RDF.type, HKG.Artist))
                if pd.notna(row.get("artist_name")):
                    g.add((artist, HKG.artist_name, Literal(str(row["artist_name"]))))
                g.add((piece, HKG.hasArtist, artist))
                stats["artists"].add(artist_id)

            # -- Genre(s) --
            for col in ("top3_genres", "primary_genre"):
                if col in row and pd.notna(row.get(col)):
                    for gl in str(row[col]).split(";"):
                        gl = gl.strip()
                        if gl:
                            genre = URIRef(schema.genre_uri(gl))
                            g.add((genre, RDF.type, HKG.Genre))
                            g.add((genre, HKG.genre_name, Literal(gl)))
                            g.add((piece, HKG.hasGenre, genre))
                            stats["genres"].add(gl)
                    break  # only use first available genre column

            # -- Musical key --
            for col in ("sem_global_key", "msd_key_name"):
                if col in row and pd.notna(row.get(col)):
                    key_name = str(row[col])
                    mode = str(row.get("sem_global_mode", row.get("msd_mode_name", "")))
                    key = URIRef(schema.key_uri(key_name, mode))
                    g.add((key, RDF.type, HKG.MusicalKey))
                    g.add((key, HKG.key_name, Literal(key_name)))
                    if mode:
                        g.add((key, HKG.mode, Literal(mode)))
                    g.add((piece, HKG.hasGlobalKey, key))
                    stats["keys"].add(f"{key_name}_{mode}")
                    break

            # -- Era (from year) --
            if pd.notna(row.get("year")) and int(row["year"]) > 0:
                yr = int(row["year"])
                decade = (yr // 10) * 10
                era_label = f"{decade}s"
                era = URIRef(schema.era_uri(era_label))
                g.add((era, RDF.type, HKG.Era))
                g.add((era, HKG.era_label, Literal(era_label)))
                g.add((era, HKG.decade_start, Literal(decade, datatype=XSD.integer)))
                g.add((piece, HKG.hasEra, era))
                stats["eras"].add(era_label)

            stats["pieces"] += 1

        # -- User interactions --
        n_interactions = 0
        if user_interactions_df is not None and not user_interactions_df.empty:
            for _, irow in user_interactions_df.iterrows():
                uid = str(irow["user_id"])
                tid = str(irow["track_id"])
                user = URIRef(schema.user_uri(uid))
                piece = URIRef(schema.song_uri(tid))
                g.add((user, RDF.type, HKG.User))
                g.add((user, HKG.user_id, Literal(uid)))

                play_count = irow.get("play_count", 0)
                if pd.notna(play_count) and int(play_count) > 0:
                    g.add((user, HKG.listenedTo, piece))
                    inter = URIRef(schema.interaction_uri(uid, tid))
                    g.add((inter, RDF.type, HKG.Interaction))
                    g.add((inter, HKG.play_count, Literal(int(play_count), datatype=XSD.integer)))

                rating = irow.get("rating")
                if pd.notna(rating):
                    g.add((user, HKG.rated, piece))

                n_interactions += 1

        if verbose:
            print(f"  RDF graph: {len(g)} triples")
            print(f"      {stats['pieces']} pieces, "
                  f"{len(stats['artists'])} artists, "
                  f"{len(stats['genres'])} genres, "
                  f"{len(stats['keys'])} keys, "
                  f"{len(stats['eras'])} eras, "
                  f"{n_interactions} user interactions")

        return g

    # ------------------------------------------------------------------
    #  DataFrame -> NetworkX
    # ------------------------------------------------------------------

    def to_networkx(
        self,
        df: pd.DataFrame,
        user_interactions_df: Optional[pd.DataFrame] = None,
        verbose: bool = True,
    ):
        """
        Build a NetworkX directed graph from the feature DataFrame.

        Each node gets a ``type`` attribute; edges are typed via the
        ``relation`` attribute.

        Returns
        -------
        networkx.DiGraph
        """
        import networkx as nx

        G = nx.DiGraph()

        for _, row in df.iterrows():
            track_id = str(row.get("track_id", ""))
            if not track_id:
                continue

            # MusicalPiece node
            G.add_node(
                f"piece:{track_id}",
                type="MusicalPiece",
                title=str(row.get("title", "")),
                year=int(row.get("year", 0)),
            )

            # Artist
            artist_id = str(row.get("artist_id", ""))
            if artist_id:
                G.add_node(
                    f"artist:{artist_id}",
                    type="Artist",
                    name=str(row.get("artist_name", "")),
                )
                G.add_edge(f"piece:{track_id}", f"artist:{artist_id}", relation="hasArtist")

            # Genre
            for col in ("top3_genres", "primary_genre"):
                if col in row and pd.notna(row.get(col)):
                    for gl in str(row[col]).split(";"):
                        gl = gl.strip()
                        if gl:
                            G.add_node(f"genre:{gl}", type="Genre", name=gl)
                            G.add_edge(f"piece:{track_id}", f"genre:{gl}", relation="hasGenre")
                    break

            # Key
            for col in ("sem_global_key", "msd_key_name"):
                if col in row and pd.notna(row.get(col)):
                    key = str(row[col])
                    G.add_node(f"key:{key}", type="MusicalKey", name=key)
                    G.add_edge(f"piece:{track_id}", f"key:{key}", relation="hasGlobalKey")
                    break

            # Era
            if pd.notna(row.get("year")) and int(row["year"]) > 0:
                era = f"{(int(row['year']) // 10) * 10}s"
                G.add_node(f"era:{era}", type="Era", label=era)
                G.add_edge(f"piece:{track_id}", f"era:{era}", relation="hasEra")

        # User interactions
        if user_interactions_df is not None and not user_interactions_df.empty:
            for _, irow in user_interactions_df.iterrows():
                uid = str(irow["user_id"])
                tid = str(irow["track_id"])
                G.add_node(f"user:{uid}", type="User")
                G.add_edge(f"user:{uid}", f"piece:{tid}", relation="listenedTo")

        if verbose:
            print(f"  NetworkX graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

        return G

    # ------------------------------------------------------------------
    #  Serialisation
    # ------------------------------------------------------------------

    @staticmethod
    def save(graph, path: str, fmt: str = "turtle") -> None:
        """
        Serialise an rdflib.Graph to disk.

        Parameters
        ----------
        graph : rdflib.Graph
        path : str
            Destination file path.
        fmt : str
            Serialisation format: ``"turtle"``, ``"nt"``, ``"xml"``, ``"json-ld"``.
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        graph.serialize(destination=str(p), format=fmt)
        log.info("Saved RDF graph -> %s (%s)", p, fmt)

    @staticmethod
    def load(path: str, fmt: str = "turtle"):
        """
        Load an rdflib.Graph from disk.

        Returns
        -------
        rdflib.Graph
        """
        from rdflib import Graph

        g = Graph()
        g.parse(path, format=fmt)
        log.info("Loaded RDF graph <- %s (%d triples)", path, len(g))
        return g
