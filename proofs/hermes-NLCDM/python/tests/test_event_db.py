"""Tests for mabench.event_db — structured event database."""

import pickle
from datetime import date, datetime

from mabench.event_db import Event, EventDB


def test_add_single_event():
    db = EventDB()
    db.add("Alice", "flew", "American Airlines", chunk_id=0)
    assert db.n_events() == 1
    assert db.n_entities() == 2  # "alice" + "american airlines"


def test_add_batch():
    db = EventDB()
    triples = [
        ("Alice", "flew", "American Airlines"),
        ("Alice", "received", "crystal chandelier"),
    ]
    db.add_batch(triples, chunk_id=0)
    assert db.n_events() == 2
    assert "alice" in db._entity_index
    assert "american airlines" in db._entity_index
    assert "crystal chandelier" in db._entity_index


def test_lookup_by_entity():
    db = EventDB()
    db.add("Alice", "flew", "American Airlines", chunk_id=0)
    db.add("Bob", "visited", "Rome", chunk_id=1)
    db.add("Alice", "stayed at", "Hilton", chunk_id=2)

    results = db.lookup(["alice"])
    chunk_ids = [cid for cid, _ in results]
    assert 0 in chunk_ids
    assert 2 in chunk_ids
    assert 1 not in chunk_ids


def test_lookup_scores_normalized():
    db = EventDB()
    # chunk 0 has 2 entity matches, chunk 1 has 1
    db.add("Alice", "flew", "Delta", chunk_id=0)
    db.add("Alice", "booked", "Hilton", chunk_id=0)
    db.add("Alice", "visited", "Rome", chunk_id=1)

    results = db.lookup(["alice"])
    scores = {cid: score for cid, score in results}
    assert scores[0] == 1.0  # max hits
    assert scores[1] < 1.0


def test_lookup_with_date_filter():
    db = EventDB()
    db.add("Alice", "flew", "Delta", chunk_id=0, event_date=date(2023, 3, 1))
    db.add("Alice", "flew", "United", chunk_id=1, event_date=date(2023, 9, 1))

    # Window around March — should only get chunk 0
    results = db.lookup(
        ["alice"], date_center=date(2023, 3, 15), date_window_days=30,
    )
    chunk_ids = [cid for cid, _ in results]
    assert 0 in chunk_ids
    assert 1 not in chunk_ids


def test_lookup_date_none_events_pass_through():
    """Events without dates should always pass through date filter."""
    db = EventDB()
    db.add("Alice", "flew", "Delta", chunk_id=0)  # no date
    db.add("Alice", "flew", "United", chunk_id=1, event_date=date(2023, 9, 1))

    results = db.lookup(
        ["alice"], date_center=date(2023, 3, 15), date_window_days=30,
    )
    chunk_ids = [cid for cid, _ in results]
    assert 0 in chunk_ids  # no date = passes through
    assert 1 not in chunk_ids  # too far from March


def test_lookup_empty():
    db = EventDB()
    db.add("Alice", "flew", "Delta", chunk_id=0)
    results = db.lookup(["nonexistent"])
    assert results == []


def test_lookup_top_k():
    db = EventDB()
    for i in range(20):
        db.add(f"entity{i}", "pred", "obj", chunk_id=i)
    # Each entity maps to exactly 1 chunk, so lookup all 20
    all_ents = [f"entity{i}" for i in range(20)]
    results = db.lookup(all_ents, top_k=5)
    assert len(results) == 5


def test_add_batch_with_event_times():
    db = EventDB()
    triples = [("Alice", "flew", "Delta"), ("Bob", "visited", "Rome")]
    event_times = [datetime(2023, 3, 1, 10, 0), None]
    db.add_batch(triples, chunk_id=0, event_times=event_times)

    # Check date index populated for first triple
    assert date(2023, 3, 1) in db._date_index
    assert len(db._date_index[date(2023, 3, 1)]) >= 1


def test_short_entities_ignored():
    db = EventDB()
    db.add("I", "am", "X", chunk_id=0)  # "i" and "x" are < 2 chars
    assert db.n_entities() == 0
    assert db.n_events() == 0


def test_pickle_roundtrip():
    db = EventDB()
    db.add("Alice", "flew", "Delta", chunk_id=0, event_date=date(2023, 3, 1))
    db.add("Bob", "visited", "Rome", chunk_id=1)

    data = pickle.dumps(db)
    db2 = pickle.loads(data)

    assert db2.n_events() == db.n_events()
    assert db2.n_entities() == db.n_entities()
    results = db2.lookup(["alice"])
    assert len(results) == 1
    assert results[0][0] == 0


def test_case_insensitive_lookup():
    db = EventDB()
    db.add("American Airlines", "hub", "Dallas", chunk_id=0)
    results = db.lookup(["american airlines"])
    assert len(results) == 1
    results2 = db.lookup(["AMERICAN AIRLINES"])
    assert len(results2) == 1


def test_dedup_chunk_ids():
    """Multiple events in same chunk should count as hits, not duplicate entries."""
    db = EventDB()
    db.add("Alice", "flew", "Delta", chunk_id=0)
    db.add("Alice", "booked", "Hilton", chunk_id=0)
    db.add("Alice", "ate at", "Olive Garden", chunk_id=0)

    results = db.lookup(["alice"])
    assert len(results) == 1  # single chunk
    assert results[0][0] == 0
    assert results[0][1] == 1.0  # all hits in one chunk


# ---------------------------------------------------------------------------
# Type index
# ---------------------------------------------------------------------------

def test_type_index_populated():
    db = EventDB()
    db.add("MoMA", "type", "museum", chunk_id=0)
    db.add("Rijksmuseum", "type", "museum", chunk_id=1)
    db.add("Python", "type", "programming language", chunk_id=2)
    assert "museum" in db._type_index
    assert "moma" in db._type_index["museum"]
    assert "rijksmuseum" in db._type_index["museum"]
    assert "programming language" in db._type_index
    assert "python" in db._type_index["programming language"]


def test_type_index_from_batch():
    db = EventDB()
    triples = [("MoMA", "type", "museum"), ("user", "visited", "MoMA")]
    db.add_batch(triples, chunk_id=0)
    assert "museum" in db._type_index
    assert "moma" in db._type_index["museum"]


def test_lookup_by_type_returns_chunk_ids():
    """Type resolution: 'museum' -> {MoMA, Rijksmuseum} -> lookup entities."""
    db = EventDB()
    db.add("MoMA", "type", "museum", chunk_id=0)
    db.add("user", "visited", "MoMA", chunk_id=1)
    db.add("Rijksmuseum", "type", "museum", chunk_id=2)
    db.add("user", "visited", "Rijksmuseum", chunk_id=3)

    results = db.lookup_by_type("museum")
    chunk_ids = [cid for cid, _ in results]
    assert 0 in chunk_ids
    assert 1 in chunk_ids
    assert 2 in chunk_ids
    assert 3 in chunk_ids


def test_lookup_by_type_case_insensitive():
    db = EventDB()
    db.add("MoMA", "type", "Museum", chunk_id=0)
    results1 = db.lookup_by_type("museum")
    assert len(results1) >= 1
    results2 = db.lookup_by_type("MUSEUM")
    assert len(results2) >= 1


def test_lookup_by_type_nonexistent():
    db = EventDB()
    db.add("MoMA", "type", "museum", chunk_id=0)
    results = db.lookup_by_type("restaurant")
    assert results == []


def test_lookup_by_type_top_k():
    db = EventDB()
    for i in range(20):
        db.add(f"entity{i}", "type", "widget", chunk_id=i)
    results = db.lookup_by_type("widget", top_k=5)
    assert len(results) <= 5


# ---------------------------------------------------------------------------
# Predicate index
# ---------------------------------------------------------------------------

def test_predicate_index_populated():
    db = EventDB()
    db.add("user", "visited", "MoMA", chunk_id=0)
    db.add("user", "flew", "Delta", chunk_id=1)
    db.add("user", "visited", "Rome", chunk_id=2)
    assert "visited" in db._predicate_index
    assert len(db._predicate_index["visited"]) == 2
    assert "flew" in db._predicate_index


def test_lookup_by_predicate():
    db = EventDB()
    db.add("user", "visited", "MoMA", chunk_id=0)
    db.add("user", "flew", "Delta", chunk_id=1)
    db.add("user", "visited", "Rome", chunk_id=2)

    results = db.lookup_by_predicate("visited")
    chunk_ids = [cid for cid, _ in results]
    assert 0 in chunk_ids
    assert 2 in chunk_ids
    assert 1 not in chunk_ids


def test_lookup_by_predicate_empty():
    db = EventDB()
    db.add("user", "visited", "MoMA", chunk_id=0)
    results = db.lookup_by_predicate("nonexistent")
    assert results == []


# ---------------------------------------------------------------------------
# Chunk index + get_facts_for_chunks
# ---------------------------------------------------------------------------

def test_get_facts_for_chunks():
    db = EventDB()
    db.add("user", "visited", "MoMA", chunk_id=0)
    db.add("user", "flew", "Delta", chunk_id=0)
    db.add("Bob", "likes", "cats", chunk_id=1)

    facts = db.get_facts_for_chunks([0, 1])
    assert 0 in facts
    assert len(facts[0]) == 2
    assert any("MoMA" in f for f in facts[0])
    assert 1 in facts
    assert len(facts[1]) == 1


def test_get_facts_for_chunks_empty():
    db = EventDB()
    db.add("user", "visited", "MoMA", chunk_id=0)
    facts = db.get_facts_for_chunks([99])
    assert facts == {}


# ---------------------------------------------------------------------------
# Tests for llm_extract_query_terms
# ---------------------------------------------------------------------------

def test_llm_extract_query_terms_empty():
    """Empty/blank input returns empty list without calling LLM."""
    from mabench.retrieval import llm_extract_query_terms
    assert llm_extract_query_terms("") == []
    assert llm_extract_query_terms("   ") == []


def test_llm_extract_query_terms_parse(monkeypatch):
    """Verify JSON parsing and lowercasing of LLM response."""
    from mabench import retrieval as ret

    class FakeChoice:
        class message:
            content = '{"search_terms": ["Cat", "pet", "Feline"]}'

    class FakeResp:
        choices = [FakeChoice()]

    class FakeClient:
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    return FakeResp()

    monkeypatch.setattr(ret, "_query_llm_client", FakeClient())
    terms = ret.llm_extract_query_terms("What is the name of my cat?")
    assert terms == ["cat", "pet", "feline"]


def test_llm_extract_query_terms_markdown_fence(monkeypatch):
    """LLM sometimes wraps JSON in markdown fences — verify stripping."""
    from mabench import retrieval as ret

    class FakeChoice:
        class message:
            content = '```json\n{"search_terms": ["bedroom", "walls", "color"]}\n```'

    class FakeResp:
        choices = [FakeChoice()]

    class FakeClient:
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    return FakeResp()

    monkeypatch.setattr(ret, "_query_llm_client", FakeClient())
    terms = ret.llm_extract_query_terms("What color did I repaint my bedroom walls?")
    assert "bedroom" in terms
    assert "walls" in terms
    assert "color" in terms


def test_llm_extract_query_terms_failure_returns_empty(monkeypatch):
    """On LLM failure, returns empty list (falls through to BM25)."""
    from mabench import retrieval as ret

    class FakeClient:
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    raise RuntimeError("API down")

    monkeypatch.setattr(ret, "_query_llm_client", FakeClient())
    terms = ret.llm_extract_query_terms("What is my cat's name?")
    assert terms == []


def test_llm_extract_query_terms_filters_short(monkeypatch):
    """Terms shorter than 2 chars are filtered out."""
    from mabench import retrieval as ret

    class FakeChoice:
        class message:
            content = '{"search_terms": ["a", "cat", "I", "pet"]}'

    class FakeResp:
        choices = [FakeChoice()]

    class FakeClient:
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    return FakeResp()

    monkeypatch.setattr(ret, "_query_llm_client", FakeClient())
    terms = ret.llm_extract_query_terms("test")
    assert terms == ["cat", "pet"]  # "a" and "I" filtered
