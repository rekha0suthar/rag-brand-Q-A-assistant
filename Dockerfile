# Streamlit demo image. Build:  docker build -t rag-brand-qa .
# Run:    docker run -p 8501:8501 -e LLM_PROVIDER=groq -e GROQ_API_KEY=... rag-brand-qa
FROM python:3.11-slim

WORKDIR /app

# Install pinned deps first so the layer caches across code changes.
# CPU-only torch via the PyTorch wheel index keeps the image GPU-free.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Bake the MiniLM embedding weights into the image so cold starts don't
# download ~90MB from Hugging Face (or fail on hosts without egress).
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# Copy the app, including the committed faiss_index/ so no ingest is needed.
COPY . .

ENV PORT=8501
EXPOSE 8501

# Shell form so $PORT expands (Render and friends inject their own PORT).
CMD streamlit run app.py --server.port ${PORT} --server.address 0.0.0.0
