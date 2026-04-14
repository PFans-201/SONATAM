""""""

test_kg.pytest_kg.py

====================

Unit tests for sonata.kg (schema, builder, converter).Unit tests for sonata.kg (schema, builder, queries).

""""""



import pytestimport pytest





# ── Schema ────────────────────────────────────────────────────────────def test_schema_import():

    from sonata.kg.schema import HarmonicKGSchema, NS

def test_schema_import():    assert HarmonicKGSchema is not None

    from sonata.kg.schema import HarmonicKGSchema, NS    assert NS is not None

    assert HarmonicKGSchema is not None

    assert NS is not None

def test_song_uri():

    from sonata.kg.schema import HarmonicKGSchema as S

def test_song_uri():    uri = S.song_uri("TRAAAGR128F425B14B")

    from sonata.kg.schema import HarmonicKGSchema as S    assert "TRAAAGR128F425B14B" in str(uri)

    uri = S.song_uri("TRAAAGR128F425B14B")

    assert "TRAAAGR128F425B14B" in str(uri)

def test_genre_uri_slugify():

    from sonata.kg.schema import HarmonicKGSchema as S

def test_genre_uri_slugify():    uri = S.genre_uri("Heavy Metal")

    from sonata.kg.schema import HarmonicKGSchema as S    assert "heavy_metal" in str(uri)

    uri = S.genre_uri("Heavy Metal")

    assert "heavy_metal" in str(uri)

def test_new_graph():

    from sonata.kg.schema import HarmonicKGSchema as S

def test_era_uri():    g = S.new_graph()

    from sonata.kg.schema import HarmonicKGSchema as S    assert g is not None

    uri = S.era_uri(1990)    assert len(g) == 0

    assert "1990s" in str(uri)



def test_builder_import():

def test_user_uri():    from sonata.kg.builder import KGBuilder

    from sonata.kg.schema import HarmonicKGSchema as S    builder = KGBuilder()

    uri = S.user_uri("user123")    assert builder is not None

    assert "user123" in str(uri)


def test_interaction_uri():
    from sonata.kg.schema import HarmonicKGSchema as S
    uri = S.interaction_uri("user123", "TRABC")
    assert "user123_TRABC" in str(uri)


def test_new_graph():
    from sonata.kg.schema import HarmonicKGSchema as S
    g = S.new_graph()
    assert g is not None
    assert len(g) == 0


# ── Builder ───────────────────────────────────────────────────────────

def test_builder_import():
    from sonata.kg.builder import KGBuilder
    builder = KGBuilder()
    assert builder is not None


def test_builder_from_dataframe():
    import pandas as pd
    from sonata.kg.builder import KGBuilder

    df = pd.DataFrame([
        {
            "track_id": "TRABC123",
            "title": "Test Song",
            "artist_name": "Test Artist",
            "primary_genre": "Rock",
            "global_key": "C major",
            "year": 1995,
            "duration": 200.0,
            "jsym_feature1": 0.5,
            "sem_feature1": 0.8,
        },
    ])

    builder = KGBuilder()
    g = builder.from_dataframe(df)

    assert len(g) > 0
    # Should have at least the piece type triple + several property triples
    assert len(g) >= 5


def test_builder_to_networkx():
    import pandas as pd
    from sonata.kg.builder import KGBuilder

    df = pd.DataFrame([
        {"track_id": "TRABC123", "artist_name": "Artist A", "primary_genre": "Rock"},
        {"track_id": "TRDEF456", "artist_name": "Artist B", "primary_genre": "Jazz"},
    ])

    builder = KGBuilder()
    G = builder.to_networkx(df)

    assert G.number_of_nodes() > 0
    assert G.number_of_edges() > 0


# ── Converter ─────────────────────────────────────────────────────────

def test_converter_import():
    from sonata.kg.converter import HeteroGraphConverter
    converter = HeteroGraphConverter()
    assert converter is not None
