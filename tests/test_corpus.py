"""Corpus and calibration invariants.

These tests pin the facts the refusal-gate calibration depends on: the corpus
stays at demo scale, the deliberately-absent facts STAY absent (the refusal
demo depends on them), and the configured threshold stays inside the gap
measured by scripts/calibrate.py.
"""

import re
from pathlib import Path

from utils.config import load_config

ROOT = Path(__file__).resolve().parents[1]
DOCS = sorted((ROOT / "docs").glob("*.md"))


def test_corpus_size():
    # The committed demo corpus: big enough to make retrieval non-trivial,
    # small enough to re-embed in seconds. Growing it is fine -- recalibrate
    # (scripts/calibrate.py) and re-run `python ingest.py` when you do.
    assert len(DOCS) >= 15


def test_docs_are_realistically_sized():
    # New docs target 300-800 words. The two original seed stubs predate the
    # rule and are grandfathered; nothing else may be stub-sized.
    grandfathered = {"brand_guidelines.md", "campaign_diwali_2024.md"}
    for p in DOCS:
        words = len(p.read_text(encoding="utf-8").split())
        if p.name in grandfathered:
            continue
        assert 250 <= words <= 900, f"{p.name}: {words} words"


def test_refusal_demo_facts_stay_absent():
    # The refusal demo only works because these facts are nowhere in the
    # corpus. If one of these patterns starts matching, a doc leaked a fact
    # the eval expects the assistant to refuse on.
    forbidden = [
        r"\brevenue\b",
        r"\bheadcount\b",
        r"\bemployees\b",
        r"\bCEO\b",
        r"\bfounder\b",
        r"\bfunding\b",
        r"\bvaluation\b",
    ]
    for p in DOCS:
        text = p.read_text(encoding="utf-8")
        for pat in forbidden:
            assert not re.search(pat, text, re.IGNORECASE), f"{p.name} matches {pat}"


def test_threshold_sits_inside_measured_gap():
    # scripts/calibrate.py (17-doc corpus, 2026-11): worst on-topic best-chunk
    # L2 was 1.206, best clearly-off-topic was 1.339. The configured threshold
    # must sit strictly inside that gap or the gate mislabels measured queries.
    cfg = load_config()
    assert 1.206 < cfg["score_threshold"] < 1.339


def test_session_gate_unchanged():
    # The session-upload gate is calibrated independently of the demo corpus;
    # corpus changes must not drag it along.
    cfg = load_config()
    assert cfg["session_score_threshold"] == 1.6
    assert cfg["session_score_margin"] == 0.2
