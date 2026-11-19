import pytest

from chain import make_asker
from utils.errors import QuotaExceededError


class FakeDoc:
    def __init__(self, source, content):
        self.metadata = {"source": source}
        self.page_content = content


class StubVectorstore:
    """similarity_search_with_score stub returning canned (doc, score) pairs."""

    def __init__(self, results):
        self.results = results
        self.last_query = None
        self.last_k = None

    def similarity_search_with_score(self, query, k):
        self.last_query = query
        self.last_k = k
        return self.results


class StubLLM:
    def __init__(self, reply="Grounded answer."):
        self.reply = reply
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        return type("Out", (), {"content": self.reply})()


CFG = {"top_k": 4, "score_threshold": 1.25, "refusal_text": "Not enough info in the docs."}


def test_pass_gate_answers_with_sources():
    vs = StubVectorstore([
        (FakeDoc("a.md", "alpha facts"), 0.8),
        (FakeDoc("b.md", "beta facts"), 1.1),
    ])
    llm = StubLLM("The discount was 15%.")
    ask = make_asker(vs, CFG, llm)
    out = ask("What discount ran?")
    assert out["answer"] == "The discount was 15%."
    assert out["sources"] == ["a.md", "b.md"]
    assert len(llm.calls) == 1
    assert vs.last_k == 4


def test_refusal_when_nothing_passes_gate():
    vs = StubVectorstore([
        (FakeDoc("a.md", "alpha facts"), 1.5),
        (FakeDoc("b.md", "beta facts"), 1.9),
    ])
    llm = StubLLM()
    ask = make_asker(vs, CFG, llm)
    out = ask("Off-topic question")
    assert out["answer"] == "Not enough info in the docs."
    assert out["sources"] == []
    # The refusal gate must short-circuit BEFORE the LLM.
    assert llm.calls == []


def test_refusal_on_empty_retrieval():
    ask = make_asker(StubVectorstore([]), CFG, StubLLM())
    out = ask("Anything")
    assert out["sources"] == []
    assert "not enough info" in out["answer"].lower()


def test_context_contains_only_passing_chunks():
    vs = StubVectorstore([
        (FakeDoc("keep.md", "KEEP-THIS-CHUNK"), 0.5),
        (FakeDoc("drop.md", "DROP-THIS-CHUNK"), 2.0),
    ])
    llm = StubLLM()
    ask = make_asker(vs, CFG, llm)
    out = ask("q")
    assert out["sources"] == ["keep.md"]
    prompt_text = llm.calls[0][-1].content
    assert "KEEP-THIS-CHUNK" in prompt_text
    assert "DROP-THIS-CHUNK" not in prompt_text


def test_result_shape():
    vs = StubVectorstore([(FakeDoc("a.md", "x"), 0.1)])
    out = make_asker(vs, CFG, StubLLM())("q")
    assert set(out) == {"answer", "sources", "latency"}
    assert isinstance(out["latency"], float)


# ---- provider error mapping ----

class RaisingLLM:
    def __init__(self, exc):
        self.exc = exc

    def invoke(self, messages):
        raise self.exc


def test_quota_error_mapped_to_user_facing_exception():
    vs = StubVectorstore([(FakeDoc("a.md", "alpha facts"), 0.8)])
    exc = Exception("Error code: 429 - {'error': {'message': 'Rate limit reached'}}")
    ask = make_asker(vs, CFG, RaisingLLM(exc))
    with pytest.raises(QuotaExceededError) as excinfo:
        ask("q")
    assert "quota" in str(excinfo.value).lower()


def test_non_quota_llm_error_propagates_unchanged():
    vs = StubVectorstore([(FakeDoc("a.md", "alpha facts"), 0.8)])
    ask = make_asker(vs, CFG, RaisingLLM(ConnectionError("connection refused")))
    with pytest.raises(ConnectionError):
        ask("q")


# ---- session gate params (threshold override + relative margin) ----

def test_session_threshold_override_answers_above_default_gate():
    # 1.45 refuses under the demo threshold (1.25) but answers under the
    # session threshold (1.6) -- the exact bug: natural questions over
    # uploaded docs score higher L2 than the demo corpus.
    vs = StubVectorstore([(FakeDoc("resume.txt", "worked at Flipkart"), 1.45)])
    llm = StubLLM("Flipkart.")
    ask = make_asker(vs, CFG, llm, score_threshold=1.6, score_margin=0.2)
    out = ask("What companies has this person worked at?")
    assert out["answer"] == "Flipkart."
    assert out["sources"] == ["resume.txt"]


def test_session_margin_keeps_sibling_chunk():
    # Best chunk passes at 1.54; sibling at 1.70 is above the 1.6 threshold
    # but within +0.2 of the best, so it must reach the LLM context too.
    vs = StubVectorstore([
        (FakeDoc("doc.md", "BEST-CHUNK"), 1.54),
        (FakeDoc("doc.md", "SIBLING-CHUNK"), 1.70),
        (FakeDoc("doc.md", "FAR-CHUNK"), 1.90),
    ])
    llm = StubLLM()
    ask = make_asker(vs, CFG, llm, score_threshold=1.6, score_margin=0.2)
    ask("q")
    prompt_text = llm.calls[0][-1].content
    assert "BEST-CHUNK" in prompt_text
    assert "SIBLING-CHUNK" in prompt_text
    assert "FAR-CHUNK" not in prompt_text


def test_session_refusal_when_best_above_session_threshold():
    vs = StubVectorstore([(FakeDoc("doc.md", "x"), 1.82)])
    llm = StubLLM()
    ask = make_asker(vs, CFG, llm, score_threshold=1.6, score_margin=0.2)
    out = ask("What is the capital of France?")
    assert out["answer"] == "Not enough info in the docs."
    assert out["sources"] == []
    assert llm.calls == []


def test_refusal_reports_best_score_and_threshold():
    # Refusal transparency: the UI captions "closest match X vs cutoff Y".
    vs = StubVectorstore([
        (FakeDoc("doc.md", "x"), 1.82),
        (FakeDoc("doc.md", "y"), 1.95),
    ])
    ask = make_asker(vs, CFG, StubLLM(), score_threshold=1.6, score_margin=0.2)
    out = ask("off-topic")
    assert out["best_score"] == 1.82
    assert out["threshold"] == 1.6


def test_refusal_on_empty_retrieval_has_no_best_score():
    ask = make_asker(StubVectorstore([]), CFG, StubLLM(), score_threshold=1.6)
    out = ask("anything")
    assert out["best_score"] is None
    assert out["threshold"] == 1.6


def test_default_gate_unchanged_without_params():
    # Demo mode calls make_asker without gate params: cfg score_threshold
    # (1.25) must still apply exactly.
    vs = StubVectorstore([(FakeDoc("a.md", "x"), 1.3)])
    llm = StubLLM()
    out = make_asker(vs, CFG, llm)("q")
    assert out["sources"] == []
    assert llm.calls == []
    assert out["threshold"] == 1.25
