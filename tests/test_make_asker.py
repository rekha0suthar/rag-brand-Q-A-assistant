from chain import make_asker


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
