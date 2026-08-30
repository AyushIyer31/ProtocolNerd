# Installation Guide

ProtocolNerd runs as a FastAPI backend plus a static frontend, launched together by `run.py`.

## Prerequisites

- Python 3.11+
- An Anthropic API key (all default model calls go to Claude)

## Install

```bash
git clone https://github.com/AyushIyer31/ProtocolNerd.git
cd ProtocolNerd

python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

cp protocolnerd-backend/variables.env.example protocolnerd-backend/variables.env
# then edit variables.env and set ANTHROPIC_API_KEY
```

## Run

```bash
python3 run.py
```

This starts the backend on `http://localhost:8001`, the frontend on `http://localhost:5555`,
and opens `http://localhost:5555/chat.html`.

The corpus and both search indexes ship prebuilt under `data/`, so the first request works
immediately.

## Configuration

All settings live in `protocolnerd-backend/variables.env` (gitignored; the example file is the
template). The only required key is `ANTHROPIC_API_KEY`. See the
[Configuration](README.md#configuration) section of the README for the full variable table,
and [Corpus](README.md#corpus) for rebuilding the corpus and indexes from scratch.

## Tests and evaluation

```bash
venv/bin/python -m unittest discover tests           # unit tests
```

The evaluation harness behind the paper's experiment lives in
[protocolnerd-backend/benchmarking/](protocolnerd-backend/benchmarking/README.md).

## Deployment

- **Render**: `render.yaml` defines the backend web service and the static frontend.
- **Netlify**: `netlify.toml` publishes `protocolnerd-website/`.
- **Docker**: single-container image via the root `Dockerfile` (backend + frontend on one port).
- Older EC2 notes live in `deploy/`.
