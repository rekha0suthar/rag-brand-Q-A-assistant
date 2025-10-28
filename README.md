# RAG for Marketing Briefs

A **brand-safe Retrieval-Augmented Generation (RAG)** assistant that answers questions **only** from your own documents (brand guidelines, past campaigns, compliance). It provides **source citations** and **refuses** when the answer isn’t in the docs.

* **Retrieval:** FAISS + MiniLM embeddings
* **Generation:** Local **Ollama** (e.g., `llama3`) via LangChain
* **UX / API:** Streamlit demo (`app.py`) + FastAPI service (`api/main.py`)
* **Safety:** “Use only context” prompt + explicit **refusal** message
* **Ops:** Config-driven (`config.yaml`), Docker Compose, simple eval harness

---

## ✨ Features

* **Grounded answers** with **[source: filename]** citations
* **Strict refusal**: returns `"Not enough info in the docs."` when evidence is missing
* **Fast local inference** with **Ollama** (no external LLM API required)
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
                                                 LLM (Ollama: llama3)
                                                              │
                                                              ▼
                                        Answer + [source: filename] + latency
                                     (or “Not enough info in the docs.”)
```

---

## 📁 Project Structure

```
rag-ad-briefs/
├─ app.py                 # Streamlit demo UI
├─ api/
│  └─ main.py             # FastAPI service (POST /ask, /docs, /healthz)
├─ chain.py               # Retrieval + prompt + generation (Ollama)
├─ ingest.py              # Build FAISS index from docs/
├─ utils/
│  ├─ io.py               # Load/clean .md/.txt/.pdf
│  └─ logger.py           # Packaging + simple timing/log helpers
├─ config.yaml            # Chunking/k/models/safety knobs
├─ docs/                  # Your marketing docs (brand, campaigns, compliance)
│  ├─ brand_guidelines.md
│  ├─ campaign_diwali_2024.md
├─ faiss_index/           # (generated) vector index
├─ eval/
│  ├─ qa.jsonl            # Tiny golden set (optional)
│  └─ run_eval.py         # Simple evaluation runner (optional)
├─ requirements.txt (if you keep one)
└─ README.md
```

---

## 🔧 Prerequisites

* **Python 3.10+**
* **Ollama** installed and running locally

  ```bash
  curl -fsSL https://ollama.com/install.sh | sh
  ollama pull llama3
  ollama serve   # if not already running
  ```

> The project uses **HF MiniLM embeddings** locally (no API key required).
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
min_context_chars: 200
refusal_text: "Not enough info in the docs."

# generation (Ollama)
ollama_model: llama3
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
pip install -U \
  langchain langchain-core langchain-community langchain-text-splitters \
  langchain-ollama langchain-huggingface sentence-transformers \
  faiss-cpu chromadb pypdf python-dotenv pyyaml tiktoken tenacity streamlit fastapi uvicorn
```

2. **Add/verify docs** in `./docs` (use `.md/.txt` to start)

3. **Build the index**

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

## 🧪 (Optional) Evaluation Harness

`eval/qa.jsonl`

```json
{"q":"List 3 words our brand voice avoids.","ref":"best|No.1|guaranteed"}
{"q":"Give the mandatory discount disclaimer.","ref":"Limited-time offer; terms apply"}
{"q":"What was our 2023 revenue?","ref":"REFUSE"}
```

Run:

```bash
python eval/run_eval.py
```

Outputs pass/fail per item + latency, and a small overall score.

---


## 🧰 Troubleshooting

* **Empty index / FAISS “list index out of range”:** docs had no extractable text. Use `.md/.txt` or text-based PDFs; then `python ingest.py`.
* **Ollama connection refused:** ensure `ollama serve` is running and `ollama list` shows `llama3`.
* **Answers seem off:** lower `chunk_size` to `600`, `chunk_overlap` to `120`; try `top_k: 5`.
* **No citations shown:** ask queries that are actually covered by your docs; check `docs/` contents.

---

## 🔒 Safety & Guardrails

* System prompt enforces **“use ONLY context”**.
* If retrieved context is too small (`min_context_chars`), returns **refusal** text.
* You can add simple PII redaction in `chain.py` before prompting if needed.

---

## 🗺️ Roadmap

* Hybrid retrieval (BM25 + dense) or reranking
* Better evaluations (semantic similarity, judge-LLM)
* Telemetry: token/latency logs, caching
* Semantic Kernel orchestration variant
* OCR for scanned PDFs

