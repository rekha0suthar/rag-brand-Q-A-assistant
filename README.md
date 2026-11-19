# RAG for Marketing Briefs

**Live demo:** _coming soon_ <!-- TODO: live URL after deploy -->

A **brand-safe Retrieval-Augmented Generation (RAG)** assistant that answers questions **only** from your own documents (brand guidelines, past campaigns, compliance). It provides **source citations** and **refuses** when the answer isn’t in the docs.

* **Retrieval:** FAISS + MiniLM embeddings, with a calibrated **score-threshold refusal gate**
* **Generation:** Local **Ollama** (e.g., `llama3`) or hosted **Groq** via LangChain
* **UX / API:** Streamlit demo (`app.py`), FastAPI service (`api/main.py`), CLI (`cli.py`)
* **Safety:** Deterministic refusal below the retrieval threshold + “use only context” prompt
* **Ops:** Config-driven (`config.yaml`), Dockerfile, CI, simple eval harness

---
## Demo Video
![screen-capture (1)](https://github.com/user-attachments/assets/e71e267f-084c-40a6-bd41-fbe71ee69b15)



## ✨ Features

* **Grounded answers** with **[source: filename]** citations
* **Strict refusal**: returns `"Not enough info in the docs."` with an **empty sources list** when no chunk clears the retrieval score threshold — the LLM is never called
* **Local-first inference** with **Ollama** (no API key), **cloud-ready** with **Groq** (`LLM_PROVIDER=groq`)
* **Streamlit demo** for quick showcase and screenshots
* **FastAPI endpoint** (`POST /ask`) for programmatic use (and Swagger at `/docs`)
* **Config-only tuning**: chunk size, `k`, model, refusal text, etc.
* **Lightweight eval harness** (optional) to sanity-check accuracy/refusal/latency

---

## 🧠 Architecture

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
├─ app.py                 # Streamlit demo UI
├─ cli.py                 # Interactive command-line Q&A
├─ api/
│  └─ main.py             # FastAPI service (POST /ask, /docs, /healthz)
├─ chain.py               # Retrieval + threshold gate + prompt + generation
├─ ingest.py              # Build FAISS index from docs/
├─ utils/
│  ├─ io.py               # Load/clean .md/.txt/.pdf
│  └─ logger.py           # Packaging + simple timing/log helpers
├─ config.yaml            # Chunking/k/models/safety knobs
├─ docs/                  # Marketing docs (brand, campaigns, compliance)
│  ├─ brand_guidelines.md
│  ├─ campaign_diwali_2024.md
│  ├─ campaign_summer_2025.md
│  ├─ compliance.md
│  ├─ product_messaging.md
│  └─ tone_of_voice_faq.md
├─ faiss_index/           # Committed vector index (rebuild with ingest.py)
├─ eval_samples/
│  ├─ qa.jsonl            # Golden set (answers, keywords, refusals)
│  ├─ run_eval.py         # Evaluation runner -- writes eval_samples/report.md
│  └─ report.md           # Last committed eval report
├─ tests/                 # Unit tests (pytest) -- no models or API keys needed
├─ .github/workflows/ci.yml
├─ Dockerfile             # Streamlit image with baked MiniLM weights
├─ requirements.txt       # Pinned runtime deps
├─ requirements-dev.txt   # Dev deps (pytest)
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

# safety
# L2 distance cutoff for retrieved chunks (lower = closer). Chunks above it
# are dropped; if none remain, the chain refuses without calling the LLM.
score_threshold: 1.25
refusal_text: "Not enough info in the docs."

# generation
provider: ollama            # or "groq"; LLM_PROVIDER env var overrides
ollama_model: llama3
groq_model: llama-3.3-70b-versatile
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
# Expected: [ingest] ✅ Indexed N chunks into faiss_index
```

4a) **Run the demo (Streamlit)**

```bash
streamlit run app.py
# open http://localhost:8501
```

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

`eval_samples/qa.jsonl` holds a small golden set -- each item is either a
`refuse` case (the doc genuinely doesn't have the answer), a `contains` case
(the answer must include an exact phrase from the docs, like a disclaimer),
or a `keywords` case (the answer must include at least one of a few valid
phrasings). Run it against the live chain:

```bash
python eval_samples/run_eval.py
```

It prints a pass/fail + latency per question and writes
`eval_samples/report.md`. The committed report predates the score-threshold
gate and the corpus expansion (re-run pending). Last committed run — 5 of the
now 10 cases (Groq, `llama-3.3-70b-versatile`):

**Score:** 5/5 &nbsp;|&nbsp; **Avg latency:** 0.36s &nbsp;|&nbsp; **Max latency:** 0.54s

| # | Question | Result |
|---|----------|--------|
| 1 | List 3 words our brand voice avoids. | ✅ PASS |
| 2 | Give a compliant disclaimer we must include for discounts. | ✅ PASS |
| 3 | What was our 2023 revenue? | ✅ PASS (correctly refused) |
| 4 | Write one playful headline for Diwali 2024. | ✅ PASS |
| 5 | Are health claims allowed in our marketing? | ✅ PASS (correctly refused -- not in docs) |

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

