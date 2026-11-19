from utils.logger import pack_result


def test_basic_shape():
    out = pack_result("hi", ["a.md"], 0.1234)
    assert out == {"answer": "hi", "sources": ["a.md"], "latency": 0.123}


def test_sources_deduped_and_sorted():
    out = pack_result("x", ["b.md", "a.md", "b.md"], 0.0)
    assert out["sources"] == ["a.md", "b.md"]


def test_refusal_carries_empty_sources():
    # The score-threshold gate refuses with sources=[] -- pack_result must
    # pass that through as an empty list, not drop or fill it.
    out = pack_result("Not enough info in the docs.", [], 0.05)
    assert out["sources"] == []


def test_none_sources_become_empty_list():
    out = pack_result("x", None, 0.0)
    assert out["sources"] == []


def test_extra_fields_merged():
    out = pack_result("x", [], 0.0, extra={"model": "test"})
    assert out["model"] == "test"


def test_latency_rounded_to_3():
    out = pack_result("x", [], 1.23456789)
    assert out["latency"] == 1.235
