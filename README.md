# RAG for Marketing Briefs

**What this is:**  
A Retrieval-Augmented Generation app that answers marketing/brand questions **only** from your provided documents (brand guidelines, past campaigns, compliance). If the answer isn’t in the docs, it says **“Not enough info in the docs.”** Sources are cited.

**Why it matters:**  
Agencies need brand-safe assistants that respect compliance and provenance. This shows practical GenAI: Python, LangChain, embeddings, vector search, prompt guardrails, and basic evals.

---

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill OPENAI_API_KEY
python ingest.py       # build FAISS index from /docs
python cli.py          # CLI test
# or
streamlit run app.py   # UI
