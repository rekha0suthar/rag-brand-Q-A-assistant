# RAG for Marketing Briefs

[![CI](https://github.com/rekha0suthar/rag-brand-Q-A-assistant/actions/workflows/ci.yml/badge.svg)](https://github.com/rekha0suthar/rag-brand-Q-A-assistant/actions/workflows/ci.yml)

**Live demo:** https://rag-brand-q-a-assistant-kewjcwl4ezwdteptjda5ny.streamlit.app/

A **brand-safe Retrieval-Augmented Generation (RAG)** assistant that answers questions **only** from your own documents (brand guidelines, past campaigns, compliance). It provides **source citations** and **refuses** when the answer isn’t in the docs.

* **Retrieval:** FAISS + MiniLM embeddings, with a calibrated **score-threshold refusal gate**
* **Generation:** Local **Ollama** (e.g., `llama3`) or hosted **Groq** via LangChain
* **UX / API:** Two-mode Streamlit app (`app.py`) — brand demo + upload-your-own-documents, FastAPI service (`api/main.py`), CLI (`cli.py`)
* **Safety:** Deterministic refusal below the retrieval threshold + “use only context” prompt
* **Ops:** Config-driven (`config.yaml`), Dockerfile, CI (ruff lint + 85 tests + retrieval-metric gate, gated deploys), two-layer eval harness (retriever hit@3/MRR + end-to-end golden set)

See [ARCHITECTURE.md](ARCHITECTURE.md) for the data-flow diagram, the design
decisions behind the refusal gate, and the known tradeoffs.

---
## Demo Video
![screen-capture (1)](https://github.com/user-attachments/assets/e71e267f-084c-40a6-bd41-fbe71ee69b15)



## ✨ Features

* **Grounded answers** with **[source: filename]** citations
* **Strict refusal**: returns `"Not enough info in the docs."` with an **empty sources list** when no chunk clears the retrieval score threshold — the LLM is never called
* **Local-first inference** with **Ollama** (no API key), **cloud-ready** with **Groq** (`LLM_PROVIDER=groq`)
* **Two-mode Streamlit app**:
  * **Brand knowledge demo** — chat over the committed 17-doc index, with example-question chips (including an off-limits one that shows the refusal) and a knowledge-base overview
  * **Your documents** — upload `.pdf/.md/.txt` (≤50 files, ≤10 MB total); they are chunked with the same splitter, embedded with the same shared model, and indexed **in memory for the session only** (never written to disk). Uploads use a session-calibrated refusal gate (see below); files with little readable text (scanned/image PDFs) are skipped with a visible warning instead of being indexed as garbage.
* **FastAPI endpoint** (`POST /ask`) for programmatic use (and Swagger at `/docs`)
* **Clean failure modes**: Groq rate/quota errors surface as a friendly "free AI
  quota is temporarily exhausted" message (HTTP 429 on the API); unexpected
  errors show a short message and log server-side — never a traceback in the UI
* **Config-only tuning**: chunk size, `k`, models, thresholds, refusal text
* **Lightweight eval harness** (optional) to sanity-check accuracy/refusal/latency

---

## 🧠 Architecture

Full version with design rationale: [ARCHITECTURE.md](ARCHITECTURE.md).

```
                 ┌───────────────────────────────────────────┐
                 │                 Ingest                    │
                 │  docs/ → split → embed (MiniLM) → FAISS  │
                 └───────────────────────────────────────────┘
                                    │
                                    ▼
User question ──► embed(q) ──► FAISS retrieve top-k chunks ──► Prompt:
                                                     [instructions + CONTEXT]
                                                              │
                                                              ▼
                                                 LLM (Ollama llama3 / Groq)
                                                              │
                                                              ▼
                                        Answer + [source: filename] + latency
                                     (or “Not enough info in the docs.”)
```

---

## 📁 Project Structure

```
rag-brand-Q-A-assistant/
├─ app.py                 # Two-mode Streamlit UI (brand demo + uploads)
├─ cli.py                 # Interactive command-line Q&A
├─ api/
│  └─ main.py             # FastAPI service (POST /ask, /docs, /healthz)
├─ chain.py               # Retrieval + threshold gate + prompt + generation
├─ ingest.py              # Build FAISS index from docs/
├─ utils/
│  ├─ config.py           # Shared config loader + embedding-model default
│  ├─ errors.py           # Quota/rate-limit error classification
│  ├─ io.py               # Load/clean .md/.txt/.pdf
│  └─ logger.py           # ask() result packaging
├─ config.yaml            # Chunking/k/models/safety knobs
├─ docs/                  # 17 marketing docs for one fictional home-decor
│  │                      # brand: guidelines, tone FAQ + voice examples,
│  │                      # 4 campaign docs (Diwali 24/25, Summer/Monsoon 25),
│  │                      # 3 product spec sheets, pricing/discount policy,
│  │                      # returns & support, materials/safety & compliance,
│  │                      # internal FAQ, website release notes
│  └─ ...                 # (deliberately ABSENT: revenue, headcount, CEO --
│                         #  the refusal demo depends on these staying absent)
├─ faiss_index/           # Committed vector index (rebuild with ingest.py)
├─ eval_samples/
│  ├─ qa.jsonl            # End-to-end golden set (answers, keywords, refusals)
│  ├─ run_eval.py         # End-to-end runner -- writes eval_samples/report.md
│  ├─ retrieval.jsonl     # 34 labeled question -> expected-source pairs
│  ├─ run_retrieval_eval.py  # Retriever-only hit@3 / MRR gate (no LLM, in CI)
│  └─ report.md           # Last committed end-to-end eval report
├─ scripts/
│  └─ calibrate.py        # Measures on/off-topic L2 distributions for the gate
├─ tests/                 # Unit tests (pytest) -- no models or API keys needed
├─ .github/workflows/ci.yml   # lint + tests, then main -> release promotion
├─ ARCHITECTURE.md        # Data flow, design decisions, tradeoffs
├─ Dockerfile             # Streamlit image with baked MiniLM weights
├─ requirements.txt       # Pinned runtime deps
├─ requirements-dev.txt   # Dev deps (pytest, ruff, httpx)
├─ ruff.toml              # Lint/format config
├─ render.yaml            # Render Blueprint (Streamlit demo)
├─ .env.example
├─ LICENSE
└─ README.md
```

---

## 🔧 Prerequisites

* **Python 3.10+**
* An LLM provider — either of:

### Run with Ollama (local, default)

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3
ollama serve   # if not already running
```

### Run with Groq (hosted)

No local LLM server needed — set the provider and an API key
(from [console.groq.com](https://console.groq.com)):

```bash
# .env (see .env.example) or exported in the shell
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_...
```

`LLM_PROVIDER` overrides `provider` in `config.yaml`, so deploy hosts
(e.g. Render) only set env vars. The model is `groq_model` in `config.yaml`.

> Embeddings are always **HF MiniLM run locally** (no API key required) in
> both modes — only generation switches between Ollama and Groq.
> If you use scanned PDFs, add OCR later; for now prefer `.md/.txt` or text-based PDFs.

---

## ⚙️ Configuration (`config.yaml`)

```yaml
# retrieval
chunk_size: 800
chunk_overlap: 150
top_k: 4

# index
index_dir: faiss_index
embedding_model: sentence-transformers/all-MiniLM-L6-v2  # rebuild index if changed

# safety
# L2 distance cutoff for retrieved chunks (lower = closer). Chunks above it
# are dropped; if none remain, the chain refuses without calling the LLM.
score_threshold: 1.27
refusal_text: "Not enough info in the docs."

# "Your documents" (session upload) gate -- see the calibration note below
session_score_threshold: 1.6
session_score_margin: 0.2

# generation
provider: ollama            # or "groq"; LLM_PROVIDER env var overrides
ollama_model: llama3
groq_model: openai/gpt-oss-20b
temperature: 0.0
```

Tune here—no code changes needed.

---

## 🚀 Setup & Run (Local)

1. **Create environment & install deps**

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

(The extra index pulls CPU-only torch wheels — much smaller than the default
CUDA build. Drop it if you want GPU torch.)

2. **Add/verify docs** in `./docs` (use `.md/.txt` to start)

3. **Build the index** (optional — a prebuilt `faiss_index/` is committed;
   only needed after changing `docs/`)

```bash
python ingest.py
# Expected: [ingest] OK - indexed N chunks into faiss_index
# After changing docs/, re-measure the refusal gate: python scripts/calibrate.py
```

4a) **Run the demo (Streamlit)**

```bash
streamlit run app.py
# open http://localhost:8501
```

The app opens in **Brand knowledge demo** mode (the committed index). Switch
the sidebar to **Your documents** to upload your own files and chat over a
session-only in-memory index — same retrieval, same prompt, via the shared
`make_asker()` code path in `chain.py`.

An honest note on the refusal gate: an L2-distance threshold is a **per-corpus
calibration**, not a universal constant. The demo threshold (1.27) was measured
on the 17 committed brand docs with `scripts/calibrate.py` (30 on-topic
questions: best chunks 0.594–1.206; 15 off-topic/absent-fact questions:
1.339–1.853, one near-domain outlier at 1.066 handled by the context-only
prompt). On arbitrary uploads, natural questions score
systematically higher distances (measured on a synthetic resume + warranty
policy: best chunks for answerable questions landed at 1.08–1.59, off-topic
questions at 1.66–2.11), so the demo gate wrongly refused most legitimate
questions. Upload mode therefore uses a looser `session_score_threshold` (1.6)
plus a **relative-margin rule**: if the best chunk passes the gate, chunks
within `session_score_margin` (0.2) of it are kept too; if even the best chunk
fails, the app refuses and captions the closest-match distance vs the cutoff so
the behavior is explainable. On an unusual corpus you may still need to re-tune
these two values in `config.yaml` — that is the inherent trade-off of a
distance-threshold gate.

4b) **Run the API (FastAPI)**

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
# open http://localhost:8000/docs (Swagger)
```

4c) **Run with Docker**

```bash
docker build -t rag-brand-qa .
docker run -p 8501:8501 -e LLM_PROVIDER=groq -e GROQ_API_KEY=gsk_... rag-brand-qa
# open http://localhost:8501
```

The image bakes the MiniLM weights and the committed `faiss_index/` at build
time, so it starts without downloading models or re-ingesting.

**Call the API**

```bash
curl -s -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"Give the mandatory discount disclaimer."}'
```

---

## ✅ Tests & Lint

```bash
pip install -r requirements-dev.txt
pytest -q            # 69 unit tests -- no models, index or API keys needed
ruff check .         # lint
ruff format --check .
```

CI runs both on every push and PR.

---

## 🚢 Deployment

The live demo runs on **Streamlit Community Cloud**, tracking the **`release`**
branch — not `main`. CI promotes `main` → `release` automatically (a
force-push in the `promote` job of
[`.github/workflows/ci.yml`](.github/workflows/ci.yml)) **only after the test
job passes**, so a commit that breaks the suite can land on `main` but never
reaches the deployed app. The first CI run on `main` creates the remote
`release` branch; point the Streamlit Cloud app at it under
*Settings → Branch*.

---

## 🧪 Quick Test Prompts

* “List 3 words our brand voice avoids.”
* “Give the mandatory discount disclaimer.”
* “What was our 2023 revenue?” → should respond **“Not enough info in the docs.”**

---

## 🧭 Document Guidelines

* Keep documents short and focused (1–3 pages each).
* Suggested files:

  * `brand_guidelines.md` — tone, prohibited words, voice examples
  * `campaign_*.md` — what worked, offers, copy snippets
  * `compliance.md` — do-nots, mandatory disclaimers
* After changing docs, re-run:

  ```bash
  python ingest.py
  ```

---

## 🧪 Evaluation Harness

Two independent layers, split so retrieval quality is measured separately from
generation quality:

| Metric | Value | How it's measured |
|---|---|---|
| Retrieval hit@3 | **0.941** | 34 labeled question → expected-source pairs (every doc covered); hit = an expected doc in the top 3 chunks |
| Retrieval MRR | **0.953** | reciprocal rank of the first expected doc (depth 10) |
| End-to-end golden set | **15/15** | live chain (Groq, `openai/gpt-oss-20b`), string-level checks |

**Retriever-only** (no LLM, local embeddings, runs in CI and fails the build
when hit@3 drops below 0.9):

```bash
python eval_samples/run_retrieval_eval.py
```

**End-to-end** -- `eval_samples/qa.jsonl` holds 15 cases: `refuse` cases (3 of
them -- facts deliberately absent from the corpus, like 2023 revenue and
headcount, where the gate must refuse without citing anything), `contains`
cases (the answer must include an exact phrase from the docs, like a
disclaimer), and `keywords` cases (the answer must include at least one of a
few valid phrasings). Run it against the live chain:

```bash
python eval_samples/run_eval.py
```

It prints a pass/fail + latency per question and writes
[`eval_samples/report.md`](eval_samples/report.md).

The demo-gate threshold these evals exercise is re-measured with
`scripts/calibrate.py` whenever the corpus changes -- measured distributions
are recorded in `config.yaml` and ARCHITECTURE.md.

---


## 🧰 Troubleshooting

* **Empty index / FAISS “list index out of range”:** docs had no extractable text. Use `.md/.txt` or text-based PDFs; then `python ingest.py`.
* **Ollama connection refused:** ensure `ollama serve` is running and `ollama list` shows `llama3`.
* **Answers seem off:** lower `chunk_size` to `600`, `chunk_overlap` to `120`; try `top_k: 5`.
* **No citations shown:** ask queries that are actually covered by your docs; check `docs/` contents.

---

## 🔒 Safety & Guardrails

* **Score-threshold refusal gate (deterministic).** Retrieval uses
  `similarity_search_with_score`; FAISS returns **L2 distance** (lower =
  closer). Chunks scoring above `score_threshold` are dropped **before** the
  prompt is built. If nothing passes, the chain returns the refusal text with
  an **empty sources list** and **never calls the LLM** — refusal does not
  depend on the model behaving, and a refusal can't cite documents it didn't
  use. The threshold was calibrated against the committed index: on-corpus
  queries score well below it, off-topic queries well above.
* **Context-only prompt.** The system prompt enforces “use ONLY context” for
  questions that pass retrieval but aren't actually answered by the docs.
* **Sources = evidence.** Citations come only from chunks that cleared the
  threshold and were included in the prompt.
* You can add simple PII redaction in `chain.py` before prompting if needed.

---

## 🗺️ Roadmap

* Hybrid retrieval (BM25 + dense) or reranking
* Better evaluations (semantic similarity, judge-LLM)
* Telemetry: token/latency logs, caching
* Semantic Kernel orchestration variant
* OCR for scanned PDFs

---

## 📄 License

MIT — see [LICENSE](LICENSE).

