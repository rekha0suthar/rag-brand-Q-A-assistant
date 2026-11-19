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
  | demo corpus:   keep chunk if distance <= 1.25                |
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
- **Committed FAISS index.** The demo corpus is 6 small files; committing the
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
  not constants. Demo corpus: answerable questions' best chunks measured at
  0.85–1.12, off-topic at 1.33–1.98 → gate at **1.25**. Arbitrary uploads score
  systematically higher (synthetic resume + warranty corpus: answerable
  1.08–1.59, off-topic 1.66–2.11) → session gate at **1.6**, plus a relative
  margin: when the best chunk passes, chunks within **+0.2** of it are kept
  too, because neighbors of the best match are usually part of the same
  answer even when they sit above the absolute cutoff.
- **Eval harness as a golden set, not an LLM judge.** `eval_samples/qa.jsonl`
  holds 10 cases of three mechanical types — `refuse` (must refuse with no
  sources), `contains` (exact required phrase, e.g. a disclaimer), `keywords`
  (any accepted phrasing). Checks are string-level and reproducible; no
  second model grades the first. Current result: 10/10.

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
