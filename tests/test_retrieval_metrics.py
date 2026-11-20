"""Unit tests for the retriever-only eval: metric math + labeled-set hygiene.

The metric functions are imported from run_retrieval_eval.py (heavy deps are
inside run(), so importing them pulls in no torch/FAISS). The hygiene tests
keep retrieval.jsonl honest as the corpus grows: every expected file must
exist, and every corpus doc must be covered by at least one question.
"""

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location(
    "run_retrieval_eval", ROOT / "eval_samples" / "run_retrieval_eval.py"
)
rre = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rre)


# ---------- metric math ----------


def test_hit_at_3_counts_any_expected_source_in_top_3():
    ranked = ["a.md", "b.md", "c.md", "d.md"]
    assert rre.hit_at_k(ranked, ["c.md"], k=3) == 1
    assert rre.hit_at_k(ranked, ["d.md"], k=3) == 0
    assert rre.hit_at_k(ranked, ["x.md", "b.md"], k=3) == 1


def test_hit_at_k_empty_ranking():
    assert rre.hit_at_k([], ["a.md"], k=3) == 0


def test_reciprocal_rank_first_match_wins():
    ranked = ["a.md", "b.md", "b.md", "c.md"]
    assert rre.reciprocal_rank(ranked, ["b.md"]) == 0.5
    assert rre.reciprocal_rank(ranked, ["c.md"]) == 0.25
    assert rre.reciprocal_rank(ranked, ["a.md", "c.md"]) == 1.0
    assert rre.reciprocal_rank(ranked, ["z.md"]) == 0.0


# ---------- labeled-set hygiene ----------


def _items():
    return rre.load_items()


def test_labeled_set_is_large_enough():
    assert len(_items()) >= 20


def test_expected_sources_exist_in_docs():
    docs = {p.name for p in (ROOT / "docs").glob("*.md")}
    for item in _items():
        assert item["expected"], f"empty expected list: {item['q']}"
        for src in item["expected"]:
            assert src in docs, f"{src} (for {item['q']!r}) is not a file in docs/"


def test_every_doc_is_covered_by_at_least_one_question():
    docs = {p.name for p in (ROOT / "docs").glob("*.md")}
    covered = {src for item in _items() for src in item["expected"]}
    assert docs <= covered, f"docs with no retrieval question: {sorted(docs - covered)}"


# ---------- end-to-end golden set hygiene ----------


def test_qa_golden_set_size_and_refusals():
    lines = (ROOT / "eval_samples" / "qa.jsonl").read_text().splitlines()
    items = [json.loads(ln) for ln in lines if ln.strip()]
    assert len(items) >= 15
    refusals = [i for i in items if i["type"] == "refuse"]
    assert len(refusals) >= 3
    for item in items:
        assert item["type"] in {"refuse", "contains", "keywords"}
