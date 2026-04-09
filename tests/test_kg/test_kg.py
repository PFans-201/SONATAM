"""
test_kg.py
==========
Unit tests for sonata.kg (schema, builder, queries).
"""

import pytest


def test_schema_import():
    from sonata.kg.schema import HarmonicKGSchema, NS
    assert HarmonicKGSchema is not None
    assert NS is not None


def test_song_uri():
    from sonata.kg.schema import HarmonicKGSchema as S
    uri = S.song_uri("TRAAAGR128F425B14B")
    assert "TRAAAGR128F425B14B" in str(uri)


def test_genre_uri_slugify():
    from sonata.kg.schema import HarmonicKGSchema as S
    uri = S.genre_uri("Heavy Metal")
    assert "heavy_metal" in str(uri)


def test_new_graph():
    from sonata.kg.schema import HarmonicKGSchema as S
    g = S.new_graph()
    assert g is not None
    assert len(g) == 0


def test_builder_import():
    from sonata.kg.builder import KGBuilder
    builder = KGBuilder()
    assert builder is not None
