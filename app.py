# app.py -- two-mode Streamlit UI over the shared make_asker() pipeline.
#
# Mode 1 "Brand knowledge demo": committed faiss_index + the 6 docs/ files.
# Mode 2 "Your documents": uploads -> in-memory FAISS in session_state
#                          (never written to disk).
#
# Memory notes for Streamlit Community Cloud (~1GB RAM): the MiniLM embedding
# model, the LLM client, and the demo index are each loaded ONCE via
# st.cache_resource and shared by both modes. Uploads are capped.

import logging
from pathlib import Path

import streamlit as st
import yaml

st.set_page_config(page_title="RAG Brand Q&A", page_icon="🛡️", layout="centered")

from chain import build_llm, make_asker
from ingest import chunk_texts
from utils.errors import QuotaExceededError
from utils.io import has_enough_text, load_bytes

log = logging.getLogger(__name__)

MAX_FILES = 50
MAX_TOTAL_BYTES = 10 * 1024 * 1024  # 10 MB across all uploads

EXAMPLES = [
    ("What discount ran in the Summer 2025 campaign?", False),
    ("What words does our brand voice avoid?", False),
    ("Are health claims allowed in our marketing?", False),
    ("What was our 2023 revenue?", True),  # off-limits -> demonstrates refusal
]

st.markdown(
    """
    <style>
    .src-badge {
        display: inline-block;
        padding: 0.1rem 0.5rem;
        margin-right: 0.3rem;
        border-radius: 0.6rem;
        border: 1px solid rgba(128, 128, 128, 0.35);
        background: rgba(128, 128, 128, 0.12);
        font-size: 0.78rem;
        font-family: monospace;
    }
    .src-line { margin: 0.25rem 0 0.1rem 0; font-size: 0.8rem; opacity: 0.85; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------- cached resources (loaded once, shared by both modes) ----------


@st.cache_resource(show_spinner=False)
def get_cfg():
    return yaml.safe_load(Path("config.yaml").read_text())


@st.cache_resource(show_spinner=False)
def get_embeddings():
    # The single embedding model instance for BOTH the demo index and
    # session indexes -- MiniLM is ~90MB; loading it twice would be fatal
    # on a 1GB host.
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


@st.cache_resource(show_spinner=False)
def get_llm():
    return build_llm(get_cfg())


@st.cache_resource(show_spinner=False)
def get_demo_asker():
    from langchain_community.vectorstores import FAISS

    cfg = get_cfg()
    vs = FAISS.load_local(
        cfg.get("index_dir", "faiss_index"),
        embeddings=get_embeddings(),
        allow_dangerous_deserialization=True,
    )
    return make_asker(vs, cfg, get_llm())


@st.cache_data(show_spinner=False)
def kb_summaries():
    """(filename, title, one-line description) for each doc in docs/."""
    items = []
    for p in sorted(Path("docs").glob("*.md")):
        raw_text = p.read_text(encoding="utf-8", errors="ignore")
        lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]
        fallback_title = p.stem.replace("_", " ").title()
        title = next(
            (ln.lstrip("# ").strip() for ln in lines if ln.startswith("#")), fallback_title
        )
        desc = next((ln.lstrip("-* ").strip() for ln in lines if not ln.startswith("#")), "")
        if len(desc) > 110:
            desc = desc[:107] + "…"
        items.append((p.name, title, desc))
    return items


# ---------- answer rendering ----------


def is_refusal(out, cfg):
    refusal = cfg.get("refusal_text", "Not enough info in the docs.").lower()
    return refusal in out["answer"].lower() and not out["sources"]


def render_assistant(msg):
    if msg["refused"]:
        st.info(msg["answer"], icon="🛡️")
        st.caption(
            "This answer isn't in the loaded documents. "
            "This assistant only answers from them — that's the safety guarantee."
        )
        # Session mode: make the gate explainable — show how far the closest
        # chunk was from the refusal cutoff.
        if msg.get("gate_note"):
            st.caption(msg["gate_note"])
    else:
        st.markdown(msg["answer"])
        if msg["sources"]:
            badges = " ".join(f'<span class="src-badge">{s}</span>' for s in msg["sources"])
            st.markdown(f'<div class="src-line">Sources: {badges}</div>', unsafe_allow_html=True)
    st.caption(f"{msg['latency']:.1f}s")


def run_question(question, asker, cfg, history, session_mode=False):
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Retrieving and generating…"):
                out = asker(question)
        except QuotaExceededError as e:
            st.warning(str(e), icon="⏳")
            return
        except Exception:
            # Never dump a traceback into the UI; details go to the server log.
            log.exception("ask() failed for question: %r", question)
            st.error("Something went wrong while generating the answer. Please try again.")
            return
        refused = is_refusal(out, cfg)
        gate_note = None
        if refused and session_mode and out.get("best_score") is not None:
            gate_note = (
                f"Closest match scored {out['best_score']:.2f} — "
                f"above the {out['threshold']:.2f} cutoff."
            )
        msg = {
            "role": "assistant",
            "answer": out["answer"],
            "sources": out["sources"],
            "latency": out["latency"],
            "refused": refused,
            "gate_note": gate_note,
        }
        render_assistant(msg)
    history.append({"role": "user", "content": question})
    history.append(msg)


# ---------- header ----------

st.title("🛡️ RAG Brand Q&A")
st.markdown("**Answers only from the loaded documents — with citations, or an honest refusal.**")

# ---------- sidebar ----------

with st.sidebar:
    mode = st.radio("Mode", ["Brand knowledge demo", "Your documents"])
    demo_mode = mode == "Brand knowledge demo"
    st.divider()

    if demo_mode:
        st.subheader("Knowledge base")
        kb = kb_summaries()
        st.caption(f"{len(kb)} brand documents (committed demo corpus):")
        for fname, _, _ in kb:
            st.markdown(f"- `{fname}`")
    else:
        st.subheader("Your documents")
        uploads = st.file_uploader(
            "Upload .pdf, .md or .txt files",
            type=["pdf", "md", "txt"],
            accept_multiple_files=True,
            help=f"Up to {MAX_FILES} files, {MAX_TOTAL_BYTES // (1024 * 1024)} MB total. "
            "Indexed in memory for this session only — nothing is stored.",
        )
        if st.button("Index documents", disabled=not uploads, use_container_width=True):
            if len(uploads) > MAX_FILES:
                st.error(f"Too many files: {len(uploads)} (max {MAX_FILES}).")
            elif sum(f.size for f in uploads) > MAX_TOTAL_BYTES:
                st.error(f"Uploads exceed {MAX_TOTAL_BYTES // (1024 * 1024)} MB total.")
            else:
                cfg = get_cfg()
                parsed, skipped = [], []
                for f in uploads:
                    try:
                        text = load_bytes(f.name, f.getvalue())
                    except Exception:
                        text = ""
                    # Guard: a scanned/image PDF gives pypdf (almost) no text.
                    # Indexing it would silently produce garbage matches, so
                    # skip it with a visible warning instead.
                    if has_enough_text(text):
                        parsed.append((f.name, text))
                    else:
                        skipped.append((f.name, text))
                        st.warning(
                            f"**{f.name}** has little readable text — "
                            "scanned/image PDFs can't be read. Skipped."
                        )
                chunks = chunk_texts(
                    parsed,
                    chunk_size=cfg.get("chunk_size", 800),
                    chunk_overlap=cfg.get("chunk_overlap", 150),
                )
                if not chunks:
                    st.error("No readable text found in the uploaded files — nothing was indexed.")
                else:
                    from langchain_community.vectorstores import FAISS

                    with st.spinner("Embedding and indexing…"):
                        vs = FAISS.from_texts(
                            [c["page_content"] for c in chunks],
                            get_embeddings(),
                            metadatas=[c["metadata"] for c in chunks],
                        )
                    # In-memory only: lives in session_state, never on disk.
                    st.session_state["session_vs"] = vs
                    st.session_state["session_stats"] = (len(parsed), len(chunks))
                    st.session_state["history_docs"] = []
        if "session_stats" in st.session_state:
            n_docs, n_chunks = st.session_state["session_stats"]
            st.success(f"Indexed {n_docs} documents ({n_chunks} chunks).")

    st.divider()
    with st.expander("How this works"):
        st.markdown(
            "1. Your question is embedded (MiniLM) and matched against the FAISS index.\n"
            "2. A **calibrated L2-distance threshold** gates retrieval: chunks that "
            "score too far from the question are dropped.\n"
            "3. If *no* chunk passes, the app refuses **without calling the LLM** "
            "and cites nothing — citations only ever point at evidence actually used.\n"
            "4. Otherwise the LLM answers **only from the retrieved context**, "
            "with per-source citations.\n\n"
            "[Source on GitHub](https://github.com/rekha0suthar/rag-brand-Q-A-assistant)"
        )

# ---------- shared model warm-up (first load only) ----------

cfg = get_cfg()
with st.spinner("Loading embedding model — first load takes ~20s on this free host"):
    get_embeddings()

# ---------- main area ----------

chip_question = None

if demo_mode:
    with st.expander("What's in the knowledge base"):
        for fname, title, desc in kb_summaries():
            st.markdown(f"**{title}** · `{fname}`  \n{desc}")

    st.caption("Try an example:")
    cols = st.columns(2)
    for i, (q, off_limits) in enumerate(EXAMPLES):
        label = f"🚫 Off-limits: {q}" if off_limits else q
        if cols[i % 2].button(label, key=f"chip_{i}", use_container_width=True):
            chip_question = q

    asker = get_demo_asker()
    history = st.session_state.setdefault("history_demo", [])
    input_disabled = False
    placeholder = "Ask about the brand, campaigns or compliance…"
else:
    st.caption(
        "Note: your documents use a looser, session-calibrated refusal "
        "threshold — refusal is best-effort on arbitrary corpora."
    )
    if "session_vs" in st.session_state:
        # Session gate: same code path as demo mode, but with the looser
        # session threshold + relative-margin rule (see config.yaml).
        asker = make_asker(
            st.session_state["session_vs"],
            cfg,
            get_llm(),
            score_threshold=cfg.get("session_score_threshold", 1.6),
            score_margin=cfg.get("session_score_margin", 0.2),
        )
        input_disabled = False
        placeholder = "Ask about your uploaded documents…"
    else:
        st.info("Upload documents in the sidebar and click **Index documents** to start.")
        asker = None
        input_disabled = True
        placeholder = "Index some documents first…"
    history = st.session_state.setdefault("history_docs", [])

# history (per session, per mode)
for msg in history:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
    else:
        with st.chat_message("assistant"):
            render_assistant(msg)

question = st.chat_input(placeholder, disabled=input_disabled)
if chip_question and asker is not None:
    question = chip_question

if question and asker is not None:
    run_question(question, asker, cfg, history, session_mode=not demo_mode)
