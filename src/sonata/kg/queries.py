"""
queries.py
==========
SPARQL query helpers and graph traversal utilities for the SONATA KG.

Class
-----
KGQueries
    Convenience wrappers around rdflib SPARQL queries and pandas conversion.

Usage
-----
>>> from sonata.kg.queries import KGQueries
>>> q = KGQueries(graph)
>>> df = q.songs_by_genre("rock")
>>> transitions = q.chord_transitions()
"""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd
from rdflib import Graph, URIRef
from rdflib.namespace import RDF

from sonata.kg.schema import HarmonicKGSchema as S

__all__ = ["KGQueries"]


class KGQueries:
    """
    SPARQL query helpers for the SONATA rdflib graph.

    Parameters
    ----------
    graph : rdflib.Graph
        A fully populated graph returned by ``KGBuilder.from_dataframe()``.
    """

    def __init__(self, graph: Graph) -> None:
        self.g = graph

    # ─────────────────────────────────────────────────────────────────────
    #  Generic SPARQL runner
    # ─────────────────────────────────────────────────────────────────────

    def sparql(self, query: str) -> pd.DataFrame:
        """
        Run a SPARQL SELECT query and return results as a DataFrame.

        Parameters
        ----------
        query : str
            SPARQL query string. The ``hkg:`` prefix is pre-defined.

        Returns
        -------
        pd.DataFrame with column names matching the SELECT variables.
        """
        prefix = f"PREFIX hkg: <{S.NS}>\n"
        result = self.g.query(prefix + query)
        rows   = [dict(zip(result.vars, row)) for row in result]
        return pd.DataFrame(rows)

    # ─────────────────────────────────────────────────────────────────────
    #  Pre-built queries
    # ─────────────────────────────────────────────────────────────────────

    def all_songs(self) -> pd.DataFrame:
        """Return all Song nodes with title, artist, year, key, and primary genre."""
        return self.sparql("""
            SELECT ?song ?trackId ?title ?artist ?year ?key ?genre
            WHERE {
              ?song a hkg:Song .
              OPTIONAL { ?song hkg:hasTrackId ?trackId }
              OPTIONAL { ?song hkg:hasTitle ?title }
              OPTIONAL { ?song hkg:hasArtistName ?artist }
              OPTIONAL { ?song hkg:hasYear ?year }
              OPTIONAL { ?song hkg:hasGlobalKey ?keyNode .
                         ?keyNode hkg:keyName ?key }
              OPTIONAL { ?song hkg:hasPrimaryGenre ?genre }
            }
        """)

    def songs_by_genre(self, genre_label: str) -> pd.DataFrame:
        """Return all songs tagged with a given genre label (case-insensitive substring)."""
        return self.sparql(f"""
            SELECT ?song ?trackId ?title ?artist ?year
            WHERE {{
              ?song a hkg:Song .
              ?song hkg:hasGenre ?genreNode .
              ?genreNode hkg:genreLabel ?genreLabel .
              FILTER(CONTAINS(LCASE(STR(?genreLabel)), LCASE("{genre_label}")))
              OPTIONAL {{ ?song hkg:hasTrackId ?trackId }}
              OPTIONAL {{ ?song hkg:hasTitle ?title }}
              OPTIONAL {{ ?song hkg:hasArtistName ?artist }}
              OPTIONAL {{ ?song hkg:hasYear ?year }}
            }}
        """)

    def songs_by_key(self, key_name: str, mode: Optional[str] = None) -> pd.DataFrame:
        """Return all songs whose global key matches the given key name (e.g., 'C', 'G#')."""
        mode_filter = f'FILTER(?mode = "{mode}")' if mode else ""
        return self.sparql(f"""
            SELECT ?song ?trackId ?title ?key ?mode
            WHERE {{
              ?song a hkg:Song .
              ?song hkg:hasGlobalKey ?keyNode .
              ?keyNode hkg:keyName ?key .
              OPTIONAL {{ ?keyNode hkg:keyMode ?mode }}
              FILTER(?key = "{key_name}")
              {mode_filter}
              OPTIONAL {{ ?song hkg:hasTrackId ?trackId }}
              OPTIONAL {{ ?song hkg:hasTitle ?title }}
            }}
        """)

    def genre_distribution(self) -> pd.DataFrame:
        """Return genre labels sorted by number of songs."""
        return self.sparql("""
            SELECT ?genreLabel (COUNT(?song) AS ?count)
            WHERE {
              ?song a hkg:Song .
              ?song hkg:hasGenre ?genreNode .
              ?genreNode hkg:genreLabel ?genreLabel .
            }
            GROUP BY ?genreLabel
            ORDER BY DESC(?count)
        """)

    def key_distribution(self) -> pd.DataFrame:
        """Return global key names sorted by frequency across all songs."""
        return self.sparql("""
            SELECT ?key ?mode (COUNT(?song) AS ?count)
            WHERE {
              ?song a hkg:Song .
              ?song hkg:hasGlobalKey ?keyNode .
              ?keyNode hkg:keyName ?key .
              OPTIONAL { ?keyNode hkg:keyMode ?mode }
            }
            GROUP BY ?key ?mode
            ORDER BY DESC(?count)
        """)

    def high_entropy_songs(self, min_entropy: float = 3.0) -> pd.DataFrame:
        """Return songs with harmonic transition entropy above a threshold."""
        return self.sparql(f"""
            SELECT ?song ?trackId ?title ?entropy
            WHERE {{
              ?song a hkg:Song .
              ?song hkg:transitionEntropy ?entropy .
              FILTER(?entropy >= {min_entropy})
              OPTIONAL {{ ?song hkg:hasTrackId ?trackId }}
              OPTIONAL {{ ?song hkg:hasTitle ?title }}
            }}
            ORDER BY DESC(?entropy)
        """)

    def songs_with_modulations(self, min_modulations: int = 2) -> pd.DataFrame:
        """Return songs that modulate at least N times."""
        return self.sparql(f"""
            SELECT ?song ?trackId ?title ?numMod
            WHERE {{
              ?song a hkg:Song .
              ?song hkg:numModulations ?numMod .
              FILTER(?numMod >= {min_modulations})
              OPTIONAL {{ ?song hkg:hasTrackId ?trackId }}
              OPTIONAL {{ ?song hkg:hasTitle ?title }}
            }}
            ORDER BY DESC(?numMod)
        """)

    # ─────────────────────────────────────────────────────────────────────
    #  Lightweight graph traversal (no SPARQL)
    # ─────────────────────────────────────────────────────────────────────

    def songs_for_artist(self, artist_id: str) -> List[URIRef]:
        """Return all Song URIRefs linked to a given artist_id."""
        artist_uri = S.artist_uri(artist_id)
        return list(self.g.subjects(S.hasArtist, artist_uri))

    def genres_for_song(self, track_id: str) -> List[str]:
        """Return genre label strings for a given track_id."""
        song_uri = S.song_uri(track_id)
        genres   = []
        for genre_node in self.g.objects(song_uri, S.hasGenre):
            for label in self.g.objects(genre_node, S.genreLabel):
                genres.append(str(label))
        return genres

    def summary(self) -> Dict:
        """
        Return a summary dict with counts of each node type and total triples.
        """
        node_types = [
            ("Song",             S.Song),
            ("Artist",           S.Artist),
            ("Genre",            S.Genre),
            ("MusicalKey",       S.MusicalKey),
            ("Chord",            S.Chord),
            ("ChordProgression", S.ChordProgression),
        ]
        counts = {name: sum(1 for _ in self.g.subjects(RDF.type, uri))
                  for name, uri in node_types}
        counts["total_triples"] = len(self.g)
        return counts
