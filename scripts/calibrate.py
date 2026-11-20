"""Measure the on-topic vs off-topic L2 distance gap for the demo refusal gate.

Loads the committed FAISS index, embeds a labeled set of questions, and prints
the best-chunk L2 distance for each. The demo-mode `score_threshold` in
config.yaml is set from the measured gap: above every on-topic best score,
below every off-topic best score, with headroom on both sides.

Run after any corpus change (`python ingest.py` first):

    python scripts/calibrate.py

No LLM calls -- embeddings are local MiniLM, so this is free and offline.
"""

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from utils.config import embedding_model_name, load_config

# Natural questions a user would actually ask, covering every doc in docs/.
ON_TOPIC = [
    "What words does our brand voice avoid?",
    "What disclaimer must discount messaging include?",
    "What offer ran during Diwali 2024?",
    "What discount ran in the Summer 2025 campaign?",
    "Are health claims allowed in our marketing?",
    "How many emoji can a social caption have?",
    "What is our one-line brand positioning?",
    "How much does the Glow Diya Set cost?",
    "What material are the Breeze cushion covers made of?",
    "What scents come in the Hearth Candle Trio?",
    "How long do the candles burn?",
    "What are the three price bands?",
    "What is the free shipping threshold?",
    "What is the maximum discount we can offer?",
    "What is the theme of the Monsoon 2025 campaign?",
    "What offer runs for Diwali 2025?",
    "What is the return window for products?",
    "How do customers report a damaged item?",
    "How long do refunds take to process?",
    "Is the paint on the diyas lead-free?",
    "Can we call our candles eco-friendly?",
    "Can copy claim the cushion covers are organic cotton?",
    "How long does compliance review take?",
    "When do asset approvals expire?",
    "Can we mention competitors in our copy?",
    "What is the rule for reposting user-generated content?",
    "When was the gift note feature added to checkout?",
    "What is the Your Corners gallery?",
    "Give an example of rewriting copy that uses banned superlatives.",
    "What label must paid partnership posts carry?",
]

# Facts deliberately absent from the corpus + unrelated world questions.
OFF_TOPIC = [
    "What was our 2023 revenue?",
    "How many employees does the company have?",
    "Who is the CEO of the company?",
    "Where is the company headquarters located?",
    "How much funding has the company raised?",
    "What is the marketing budget for this year?",
    "Who won the 1998 FIFA World Cup?",
    "What is the boiling point of nitrogen?",
    "How do I file income tax returns in Germany?",
    "What's a good recipe for chicken biryani?",
    "Explain how a transformer neural network works.",
    "What is the capital of Australia?",
    "How do I fix a Python ImportError?",
    "What are the side effects of ibuprofen?",
    "Best exercises for lower back pain?",
]


def best_scores(vs, questions, k):
    out = []
    for q in questions:
        results = vs.similarity_search_with_score(q, k=k)
        best_doc, best = min(results, key=lambda r: r[1])
        out.append((q, float(best), best_doc.metadata.get("source", "?")))
    return out


def summarize(label, rows):
    scores = sorted(s for _, s, _ in rows)
    print(
        f"\n{label}: n={len(scores)}  min={scores[0]:.3f}  "
        f"max={scores[-1]:.3f}  mean={sum(scores) / len(scores):.3f}"
    )
    for q, s, src in sorted(rows, key=lambda r: r[1]):
        print(f"  {s:.3f}  {q}  [{src}]")
    return scores


def main():
    cfg = load_config()
    emb = HuggingFaceEmbeddings(model_name=embedding_model_name(cfg))
    vs = FAISS.load_local(
        cfg.get("index_dir", "faiss_index"),
        embeddings=emb,
        allow_dangerous_deserialization=True,
    )
    k = cfg.get("top_k", 4)

    on = summarize("ON-TOPIC (best-chunk L2, lower = closer)", best_scores(vs, ON_TOPIC, k))
    off = summarize("OFF-TOPIC (best-chunk L2)", best_scores(vs, OFF_TOPIC, k))

    worst_on, best_off = on[-1], off[0]
    print(f"\nGap: worst on-topic {worst_on:.3f}  vs  best off-topic {best_off:.3f}")
    if worst_on < best_off:
        mid = (worst_on + best_off) / 2
        print(f"Clean separation. Midpoint threshold suggestion: {mid:.2f}")
    else:
        n_overlap = sum(1 for s in on if s >= best_off)
        print(
            f"OVERLAP -- {n_overlap} on-topic question(s) score past the "
            "best off-topic one. Inspect them before picking a threshold."
        )
    print(f"Current config score_threshold: {cfg.get('score_threshold')}")


if __name__ == "__main__":
    main()
