# Streamlit demo image. Build:  docker build -t rag-brand-qa .
# Run:    docker run -p 8501:8501 -e LLM_PROVIDER=groq -e GROQ_API_KEY=... rag-brand-qa
#
# Runs as a non-root user (uid 1000) so it works on hosts that refuse root
# containers (e.g. Hugging Face Spaces) — and because root containers are
# bad hygiene anyway.
FROM python:3.11-slim

RUN useradd -m -u 1000 appuser
WORKDIR /app

# Install pinned deps first so the layer caches across code changes.
# CPU-only torch via the PyTorch wheel index keeps the image GPU-free.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Bake the MiniLM embedding weights into the image AS THE RUNTIME USER so
# cold starts don't download ~90MB (or fail on hosts without egress) and the
# cache is readable at runtime (root-owned caches are invisible to uid 1000).
USER appuser
ENV HOME=/home/appuser
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# Copy the app, including the committed faiss_index/ so no ingest is needed.
COPY --chown=appuser:appuser . .

ENV PORT=8501 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
EXPOSE 8501

# Shell form so $PORT expands (Render and friends inject their own PORT).
# CORS/XSRF relaxed so the app works behind iframe proxies (HF Spaces).
CMD streamlit run app.py --server.port ${PORT} --server.address 0.0.0.0 \
    --server.enableCORS false --server.enableXsrfProtection false
