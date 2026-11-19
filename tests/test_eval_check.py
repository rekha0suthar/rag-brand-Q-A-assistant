import importlib.util
from pathlib import Path

import pytest

# eval_samples is a flat script directory (no __init__.py); load by path.
_RUN_EVAL = Path(__file__).resolve().parents[1] / "eval_samples" / "run_eval.py"
_spec = importlib.util.spec_from_file_location("run_eval", _RUN_EVAL)
run_eval = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_eval)
check = run_eval.check


def test_refuse_passes_on_refusal_with_empty_sources():
    item = {"type": "refuse"}
    assert check(item, "Not enough info in the docs.", []) is True


def test_refuse_fails_when_sources_cited():
    # A refusal that cites sources is the old bug -- must count as a failure.
    item = {"type": "refuse"}
    assert check(item, "Not enough info in the docs.", ["brand_guidelines.md"]) is False


def test_refuse_fails_on_non_refusal_answer():
    item = {"type": "refuse"}
    assert check(item, "Our 2023 revenue was 10 crore.", []) is False


def test_contains_case_insensitive():
    item = {"type": "contains", "text": "limited-time offer"}
    assert check(item, "Include: Limited-Time Offer; terms apply.", ["a.md"]) is True
    assert check(item, "No disclaimer here.", ["a.md"]) is False


def test_keywords_any_of():
    item = {"type": "keywords", "any_of": ["best", "guaranteed"]}
    assert check(item, "We avoid the word GUARANTEED.", ["a.md"]) is True
    assert check(item, "We avoid superlatives.", ["a.md"]) is False


def test_unknown_type_raises():
    with pytest.raises(ValueError):
        check({"type": "nope"}, "x", [])
