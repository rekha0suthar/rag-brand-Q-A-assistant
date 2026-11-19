from chain import filter_by_score


class FakeDoc:
    """Stand-in for a LangChain Document -- filter_by_score never inspects it."""

    def __init__(self, name):
        self.metadata = {"source": name}
        self.page_content = f"content of {name}"


def _results(*pairs):
    return [(FakeDoc(name), score) for name, score in pairs]


def test_keeps_docs_at_or_below_threshold():
    results = _results(("a.md", 0.8), ("b.md", 1.1), ("c.md", 1.4))
    kept = filter_by_score(results, 1.1)
    assert [d.metadata["source"] for d, _ in kept] == ["a.md", "b.md"]


def test_all_pass_when_scores_low():
    results = _results(("a.md", 0.2), ("b.md", 0.3))
    assert len(filter_by_score(results, 1.1)) == 2


def test_refusal_case_returns_empty_list():
    # Off-topic query: every chunk is far away (high L2 distance).
    results = _results(("a.md", 1.5), ("b.md", 1.7))
    assert filter_by_score(results, 1.1) == []


def test_empty_input():
    assert filter_by_score([], 1.1) == []


def test_lower_is_better_semantics():
    # FAISS here returns L2 distance: a SMALLER score must survive a
    # threshold that rejects a LARGER one.
    close = _results(("close.md", 0.5))
    far = _results(("far.md", 2.0))
    assert filter_by_score(close, 1.0) and not filter_by_score(far, 1.0)
