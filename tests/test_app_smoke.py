"""Headless Streamlit smoke test via AppTest.

Boots the real app (real embeddings, real FAISS index, real config) and asks
an off-corpus question in demo mode. The score-threshold gate refuses BEFORE
any LLM call, so this exercises the full retrieve-then-gate path end-to-end
with no provider, no key, and no network beyond the one-time model download.

Slower than the unit tests (loads MiniLM + the index once); kept green in CI
by caching the HF model directory.
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from utils.config import load_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def at():
    # from_file resolves relative paths against THIS file's directory, so the
    # app path must be absolute.
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=300)
    app.run()
    return app


def test_app_boots_without_exception(at):
    assert not at.exception


def test_demo_mode_lists_full_corpus(at):
    # The sidebar caption reports the committed corpus size; it must agree
    # with what's actually on disk.
    from pathlib import Path

    n_docs = len(list(Path("docs").glob("*.md")))
    assert any(f"{n_docs} brand documents" in str(c.value) for c in at.caption)


def test_off_corpus_question_refuses_without_llm(at):
    # "2023 revenue" is a deliberately-absent fact: its best chunk scores
    # above the calibrated threshold, so the gate refuses pre-LLM. If this
    # ever tries to call the LLM, no provider is running here and the app
    # would render its generic error instead of the refusal info box.
    at.chat_input[0].set_value("What was our 2023 revenue?").run()
    assert not at.exception
    history = at.session_state["history_demo"]
    assert history, "no messages recorded"
    last = history[-1]
    cfg = load_config()
    assert last["refused"] is True
    assert last["sources"] == []
    assert cfg["refusal_text"].lower() in last["answer"].lower()


def test_on_corpus_question_passes_the_gate(at):
    # An on-corpus question must get PAST the gate (chunks kept, LLM invoked).
    # Environment-agnostic check: with no provider running, the invoke fails
    # and the app shows its generic error (history unchanged); with a live
    # local provider it answers normally. Either way, the one outcome that
    # must NOT happen is a gate refusal.
    before = len(at.session_state["history_demo"])
    at.chat_input[0].set_value("What discount ran in the Summer 2025 campaign?").run()
    assert not at.exception
    history = at.session_state["history_demo"]
    if len(history) == before:
        # LLM unavailable: the generic error proves the gate let it through.
        assert any("Something went wrong" in str(e.value) for e in at.error)
    else:
        assert history[-1]["refused"] is False
        assert history[-1]["sources"]
