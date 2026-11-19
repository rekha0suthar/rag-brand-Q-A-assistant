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


# ---- relative-margin rule (session-upload gate) ----

def test_margin_keeps_near_best_chunks_above_threshold():
    # Best (1.54) passes the 1.6 gate; 1.55 and 1.70 are within +0.2 of the
    # best, so they are kept even though 1.70 is above the threshold.
    results = _results(("a.md", 1.54), ("b.md", 1.55), ("c.md", 1.70), ("d.md", 1.80))
    kept = filter_by_score(results, 1.6, margin=0.2)
    assert [d.metadata["source"] for d, _ in kept] == ["a.md", "b.md", "c.md"]


def test_margin_refuses_when_best_above_threshold():
    # Even with a margin, nothing is kept if the BEST chunk fails the gate.
    results = _results(("a.md", 1.66), ("b.md", 1.70))
    assert filter_by_score(results, 1.6, margin=0.2) == []


def test_margin_none_matches_plain_threshold():
    results = _results(("a.md", 0.9), ("b.md", 1.3), ("c.md", 1.5))
    plain = filter_by_score(results, 1.25)
    assert [d.metadata["source"] for d, _ in plain] == ["a.md"]
    assert plain == filter_by_score(results, 1.25, margin=None)


def test_margin_does_not_shrink_threshold():
    # A chunk below the threshold is always kept, even when it is further
    # than best+margin: the margin only ever widens the cutoff.
    results = _results(("a.md", 0.5), ("b.md", 1.2))
    kept = filter_by_score(results, 1.25, margin=0.2)
    assert [d.metadata["source"] for d, _ in kept] == ["a.md", "b.md"]


def test_margin_empty_input():
    assert filter_by_score([], 1.6, margin=0.2) == []
