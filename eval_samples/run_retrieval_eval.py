"""Retriever-only eval: hit@3 and MRR over the labeled set in retrieval.jsonl.

Each item maps a question to the doc(s) that genuinely contain its answer
(several docs legitimately share some facts -- any of them counts as a hit).
No LLM is involved: this isolates retrieval quality from generation quality,
runs offline, and costs nothing.

Exits non-zero when hit@3 drops below HIT3_FLOOR so CI fails on retrieval
regressions. Run after `python ingest.py`:

    python eval_samples/run_retrieval_eval.py
"""

import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

EVAL_FILE = Path(__file__).resolve().parent / "retrieval.jsonl"

HIT3_FLOOR = 0.9  # fail the eval below this
RETRIEVE_K = 10  # depth for MRR; hit@3 only looks at the first 3


def hit_at_k(ranked_sources, expected, k=3):
    """1 if any expected source appears in the first k ranked sources, else 0."""
    return int(any(src in expected for src in ranked_sources[:k]))


def reciprocal_rank(ranked_sources, expected):
    """1/rank of the FIRST expected source in the ranking (0 if absent)."""
    for i, src in enumerate(ranked_sources, start=1):
        if src in expected:
            return 1.0 / i
    return 0.0


def load_items():
    return [json.loads(line) for line in EVAL_FILE.read_text().splitlines() if line.strip()]


def run():
    # Heavy imports inside run() so the pure metric functions above are
    # importable by tests without pulling in torch/FAISS.
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings

    from utils.config import embedding_model_name, load_config

    cfg = load_config()
    emb = HuggingFaceEmbeddings(model_name=embedding_model_name(cfg))
    vs = FAISS.load_local(
        cfg.get("index_dir", "faiss_index"),
        embeddings=emb,
        allow_dangerous_deserialization=True,
    )

    items = load_items()
    hits, rrs = [], []
    for item in items:
        results = vs.similarity_search_with_score(item["q"], k=RETRIEVE_K)
        ranked = [d.metadata.get("source", "?") for d, _ in results]
        h = hit_at_k(ranked, item["expected"], k=3)
        rr = reciprocal_rank(ranked, item["expected"])
        hits.append(h)
        rrs.append(rr)
        status = "HIT " if h else "MISS"
        print(f"[{status}] rr={rr:.2f}  {item['q']}")
        if not h:
            print(f"        expected {item['expected']}, top-3 was {ranked[:3]}")

    hit3 = sum(hits) / len(hits)
    mrr = sum(rrs) / len(rrs)
    print(f"\nRetrieval: n={len(hits)}  hit@3={hit3:.3f}  MRR={mrr:.3f}")
    if hit3 < HIT3_FLOOR:
        print(f"FAIL: hit@3 {hit3:.3f} is below the {HIT3_FLOOR} floor.")
        return 1
    print(f"PASS: hit@3 >= {HIT3_FLOOR}")
    return 0


if __name__ == "__main__":
    sys.exit(run())
