# Architecture

One shared pipeline (`make_asker()` in `chain.py`) behind three entry points
(Streamlit, FastAPI, CLI), with a deterministic refusal gate in front of the LLM.

## Data flow

```
INGEST (offline, python ingest.py)
  docs/*.md|.txt|.pdf
        |  utils/io.load_docs: extract + clean text
        v
  chunk_texts (RecursiveCharacterTextSplitter, 800 chars / 150 overlap)
        |
        v
  MiniLM embeddings (local, all-MiniLM-L6-v2)
        |
        v
  faiss_index/  (committed to the repo)

QUERY (every entry point: Streamlit app.py, FastAPI api/main.py, cli.py, eval)
  question
        |
        v
  embed(question) --> FAISS top-k search --> [(chunk, L2 distance), ...]
        |
        v
  +-- refusal gate (filter_by_score) ----------------------------+
  | demo corpus:   keep chunk if distance <= 1.27                |
  | session docs:  keep if distance <= 1.6, OR within +0.2 of    |
  |                the best chunk when the best passes (margin)  |
  +---------------------------+-----------------------------------+
        |                     |
   nothing kept          chunks kept
        |                     |
        v                     v
  refuse WITHOUT         prompt = instructions + kept chunks only
  calling the LLM             |
  (empty sources,             v
  best_score reported)   LLM (Ollama local / Groq hosted)
                              |
                              v
                 answer + [source: filename] citations + latency
```

## Key decisions and why

- **Local MiniLM embeddings, never an API.** Embedding is needed at ingest, at
  every query, and for session uploads; a hosted embedding API would add a key
  requirement, cost, and a second failure mode. MiniLM (~90MB) fits the 1GB
  Streamlit Community Cloud host — loaded once and shared by both app modes.
- **Committed FAISS index.** The demo corpus is 17 small markdown docs
  (~6,800 words) for one fictional home-decor brand; committing the
  index makes deploys reproducible and start fast (no re-embedding at boot),
  and CI/tests never need to build one. Tradeoff: it must be rebuilt by hand
  (`python ingest.py`) when `docs/` or `embedding_model` changes.
- **Hand-rolled chain instead of LCEL/RetrievalQA.** The pipeline is four
  explicit steps (search, gate, prompt, invoke). Writing them out keeps the
  refusal gate inspectable and unit-testable with plain stubs (`tests/` run
  with no models or keys) — an LCEL graph would hide the exact behavior this
  project exists to demonstrate.
- **Deterministic refusal gate before the LLM.** FAISS returns L2 distance
  (lower = closer). If no chunk clears the threshold, the chain refuses
  without invoking the model and cites nothing — refusal does not depend on
  the LLM following instructions, and a refusal can never cite documents it
  didn't use.
- **Two thresholds + a margin rule.** Thresholds are per-corpus calibrations,
  not constants, measured by `scripts/calibrate.py` (local embeddings only —
  free, offline, re-run after any corpus change). On the 17-doc corpus
  (2026-11): 30 on-topic questions' best chunks measured **0.594–1.206**
  (mean 0.870); 15 off-topic/absent-fact questions measured **1.339–1.853**
  (mean 1.63) → gate at **1.27**, the midpoint of the gap. One probe sits
  below any usable threshold ("marketing budget" at 1.066 — campaign docs
  talk about budget-adjacent things), which is exactly why the context-only
  prompt exists as the second layer: the gate catches *clearly* off-corpus
  questions cheaply, the prompt handles near-domain questions whose answer
  isn't in the retrieved text. Arbitrary uploads score systematically higher
  (synthetic resume + warranty corpus: answerable 1.08–1.59, off-topic
  1.66–2.11) → session gate at **1.6**, plus a relative margin: when the best
  chunk passes, chunks within **+0.2** of it are kept too, because neighbors
  of the best match are usually part of the same answer even when they sit
  above the absolute cutoff.
- **Eval harness as a golden set, not an LLM judge.** Two independent layers:
  - *Retriever-only* (`eval_samples/run_retrieval_eval.py`, runs in CI, no
    LLM): 34 labeled question → expected-source pairs covering every doc,
    scored as **hit@3 0.941 / MRR 0.953**; CI fails below a 0.9 hit@3 floor.
  - *End-to-end* (`eval_samples/run_eval.py`, needs a live provider):
    `eval_samples/qa.jsonl` holds 15 cases of three mechanical types —
    `refuse` (must refuse with no sources; 3 cases over deliberately absent
    facts), `contains` (exact required phrase, e.g. a disclaimer), `keywords`
    (any accepted phrasing). Checks are string-level and reproducible; no
    second model grades the first.

## Known tradeoffs

- **Per-corpus calibration.** The distance thresholds were measured on
  specific corpora. An unusual uploaded corpus may need
  `session_score_threshold`/`session_score_margin` re-tuned in `config.yaml`;
  refusal on arbitrary uploads is best-effort (the UI says so, and captions
  the closest-match distance on every session-mode refusal).
- **No auth or rate limiting on the API variant.** `api/main.py` is a demo
  surface: anyone who can reach it can spend LLM quota. Put it behind a
  gateway (or add an API key middleware) before exposing it publicly.
- **Prompt-injection surface via retrieved content.** Retrieved chunks are
  pasted into the prompt. A malicious uploaded document could carry
  instructions to the model; the system prompt constrains tone and grounding
  but is not a security boundary. Acceptable for a demo whose uploads are
  session-local and user-supplied — not for multi-tenant ingestion.
