# Single-container image for Cloud Run / Hugging Face Spaces / any Docker host.
# Serves the FastAPI backend AND the static frontend on one port, and calls
# Claude (no GPU needed). Set ANTHROPIC_API_KEY at runtime, not here.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.hf_cache

WORKDIR /app

# Install deps first for better layer caching.
COPY requirements.txt .
RUN pip install -r requirements.txt

# Deployment identity, stamped at build time so the running container can report
# exactly which commit it came from:
#   docker build --build-arg BUILD_SHA=$(git rev-parse --short HEAD) \
#                --build-arg BUILD_DATE=$(date -u +%Y-%m-%dT%H:%M:%SZ) .
ARG BUILD_SHA=""
ARG BUILD_DATE=""
ENV BUILD_SHA=$BUILD_SHA \
    BUILD_DATE=$BUILD_DATE

# Copy the rest of the repo (corpus under data/, backend, frontend).
COPY . .

# Build the TF-IDF index now (CPU is available during the build) and bake the
# pickle into the image. At runtime the app just loads it — Cloud Run throttles
# CPU between requests, so a runtime build would stall and never finish.
RUN python scripts/build_index.py

# Embed the corpus AND bake the ONNX model into the image (HF_HOME above), so the
# container never reaches out to Hugging Face on a cold start.
RUN python scripts/build_dense_index.py

# Claude (Sonnet) is the canonical default everywhere — hosted and local dev.
# LLM re-ranker ON (Haiku only for the rerank call; other flows use Sonnet).
# RERANK_COMBINED_PUBMED (#2): re-rank protocols.io + PubMed jointly so Haiku vets
# PubMed (+0.05-0.07 nDCG@10 on held-out eval vs the old lexical blend).
ENV LLM_PROVIDER=claude \
    CLAUDE_MODEL=claude-sonnet-4-6 \
    EXECUTION_STRATEGY=agentic \
    ENABLE_RERANKER=true \
    RERANKER_MODEL=claude-haiku-4-5 \
    RERANKER_SHORTLIST_PROFILE=30 \
    RERANKER_SHORTLIST_LEXICAL=30 \
    RERANK_COMBINED_PUBMED=true \
    COMBINED_PUBMED_CANDIDATES=5 \
    PUBMED_QUERY_PROVIDER=claude \
    PUBMED_QUERY_MODEL=claude-haiku-4-5 \
    DENSE_FUSION=1 \
    DENSE_TOP_K=60 \
    HF_HUB_OFFLINE=1 \
    QUERY_LOGGING_ENABLED=1 \
    QUERY_LOG_S3_BUCKET=protocolsnerd-query-logs-921135845455 \
    QUERY_LOG_S3_PREFIX=query_logs/ \
    QUERY_LOG_VIEWER_OPEN=1

# Run from the backend dir so PROTOCOLS_DATA_DIR ("../data/protocols") resolves,
# matching the Render start command. Cloud Run/Spaces inject $PORT (default 8080).
WORKDIR /app/protocolnerd-backend
ENV PORT=8080
EXPOSE 8080
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}"]
