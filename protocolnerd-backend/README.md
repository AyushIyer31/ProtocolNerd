# Backend

The FastAPI service behind `/chat`. A request moves through five stages —
domain routing, profile building, retrieval, re-ranking, explanation — each
using a language model suited to the task, or none where none is needed.

## Layout, by stage

```
main.py                  the orchestrator: /chat, SSE progress, session state
domains/                 the plugin layer (see below)
llm_providers.py         one interface for every model call: Claude / OpenAI / Gemini / Ollama
claude_client.py         the LLM calls themselves (profile, queries, explanations)

retrievers.py            Retriever contract + registry; per-domain source gating
protocolsio_client.py    protocols.io API
pubmed_client.py         NCBI E-utilities (biology's literature lane)
europepmc_client.py      Europe PMC METHODS: search (chemistry's literature lane)

protocol_rag.py          TF-IDF index and keyword search over the corpus
dense_index.py           MiniLM embedding search
blend_ranking.py         reciprocal-rank fusion of the retrieval lanes
reranker.py              LLM re-rank passes (protocol pass + joint protocols-and-papers pass)
field_ranking.py         deterministic profile-field scoring

query_logger.py          query logging to S3
storage/                 session persistence
benchmarking/            the paper's evaluation — see benchmarking/README.md
tests/                   backend unit tests (repo-level tests/ holds the ranking suite)
```

## The domain plugin layer

`domains/` is where discipline-specific behavior lives, behind the `Domain`
interface in `base.py`:

- a one-line `description` that becomes the domain's entry in the router's menu
- its profile fields and clarification logic
- its prompts, one file per domain (`biology_prompts.py`, `chemistry_prompts.py`)
- `paper_sources` — which literature lane runs alongside protocols.io
  (biology: PubMed; chemistry: Europe PMC)

Registration is two lines in `registry.py`. Retrieval, fusion, re-ranking, and
explanation are reused unchanged because they operate on the structured profile
rather than on domain vocabulary; the paper's Section 6 walks through the steps
and evaluates the chemistry extension built this way.

## Running locally

```bash
# from the repo root, venv active, variables.env holding ANTHROPIC_API_KEY
cd protocolnerd-backend
python -m uvicorn main:app --host 0.0.0.0 --port 8001
```

The server loads the prebuilt index from `data/protocol_index.pkl` at startup
(build it first via `scripts/build_index.py` if missing). `GET /health` reports
status plus the running build's commit and date.
