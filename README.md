# ProtocolNerd

[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green.svg)](https://fastapi.tiangolo.com/)
[![Claude](https://img.shields.io/badge/Claude-Sonnet%204.6-orange.svg)](https://www.anthropic.com/)

An interactive AI research assistant for **scientific protocol discovery**, built for the
NYU research project with Prof. Dennis Shasha. Biology is the primary, evaluated domain;
chemistry ships as a second domain built on the same extension mechanism.

**Live:** <https://protocolnerd.wirelessnerd.org>

A bench scientist describes an experiment in plain English. ProtocolNerd turns that into a
structured experiment profile, asks a clarifying question when a required detail is missing,
searches a curated corpus of **22,724 Protocols.io protocols** plus the domain's literature source (PubMed for biology, Europe PMC for chemistry), and returns a
single ranked list explaining why each result fits and what it does not cover.

## Why it exists

Searching Protocols.io and PubMed means visiting both, guessing the right keywords for each,
and then opening every result to find out whether it actually applies to your organism,
sample, and readout. Keyword search misses protocols that use different vocabulary for the
same technique. ProtocolNerd adds a profiling, retrieval, and ranking layer on top of both
sources so the fit judgement happens before you start reading.

## How a request flows

1. **Domain routing** — a Claude call decides which scientific domain the request belongs to,
   from a menu each registered domain declares about itself. Biology and chemistry are
   registered; the rest of the request runs with the routed domain's fields and prompts.
2. **Profile building** — Claude extracts a structured experiment profile (22 fields in biology, 13 in chemistry: organism,
   technique, sample type, readout, conditions, …). Fields it cannot infer are marked
   "not specified" rather than guessed.
3. **Clarification** — if a required field is missing, the system asks one targeted question
   with clickable options. It can also explain *why* it is asking.
4. **Candidate queries** — the profile becomes a checklist of differently-phrased search
   queries the user can edit or deselect before anything runs.
5. **Retrieval** — TF-IDF and semantic (all-MiniLM-L6-v2) search over the local corpus, merged
   with Reciprocal Rank Fusion, plus the domain's literature lane: a live PubMed search for
   biology, Europe PMC's Methods-section search for chemistry.
6. **Re-ranking** — a rule-based pass, then two Claude Haiku 4.5 passes: protocols alone, then
   protocols and papers jointly. Protocols and papers compete in one ranking, no source quotas.
7. **Explanation** — every result carries why it matches, what may not fit, and which profile
   fields remain unresolved.

A request with no clarification uses seven model calls before results appear: the confirmation request repeats domain routing, so four Sonnet calls join three Haiku calls.

## Quick start

**Prerequisites:** Python 3.11+, and an Anthropic API key.

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

cp protocolnerd-backend/variables.env.example protocolnerd-backend/variables.env
# then edit variables.env and set ANTHROPIC_API_KEY

python3 scripts/build_index.py          # TF-IDF index, ~30s
python3 scripts/build_dense_index.py    # semantic index, several minutes

python3 run.py
```

This starts the backend on `http://localhost:8001`, the frontend on `http://localhost:5555`,
and opens `http://localhost:5555/chat.html`.

The protocol corpus is not included in the repository: the protocols belong to their
authors on Protocols.io, so each developer builds their own copy. Before the index
commands above, crawl the corpus once (set `PROTOCOLS_IO_TOKEN` in `variables.env`;
a full crawl takes a few hours and is resumable):

```bash
python3 scripts/fetch_protocols.py
```

The two search indexes are build artifacts on top of that corpus — the TF-IDF one is
386 MB, past GitHub's file-size limit — so both are always built locally by the
commands above; no API key is needed for the index builds themselves.

The semantic index has no such fallback. Without it, retrieval is TF-IDF only, which is *not*
the configuration the paper evaluates. To match that configuration, build it and set
`DENSE_FUSION=1` in `variables.env`.

## Configuration

Set in `protocolnerd-backend/variables.env` (gitignored; use `variables.env.example` as the
template). The only required key is `ANTHROPIC_API_KEY`.

| Variable | Purpose |
|---|---|
| `ANTHROPIC_API_KEY` | Required. All default model calls go to Claude. |
| `CLAUDE_MODEL` | Default `claude-sonnet-4-6`. |
| `RERANKER_MODEL` | Default `claude-haiku-4-5`, used for the two re-ranking passes and the PubMed query. |
| `LLM_PROVIDER` | `claude` (default), `openai`, `gemini`, or `ollama`. |
| `NCBI_API_KEY` | Optional. Raises PubMed rate limits. |
| `PROTOCOLS_IO_TOKEN` | Needed only to rebuild the corpus. |
| `ENABLE_EUROPEPMC` | `1` by default. `0` is a kill switch that removes chemistry's Europe PMC lane everywhere. |

Every model call goes through one provider interface, so switching providers is a config
change rather than a code change.

## Guides

| Part | Guide |
|---|---|
| Offline pipeline: corpus crawl + index builds | [scripts/README.md](scripts/README.md) |
| Frontend | [protocolnerd-website/README.md](protocolnerd-website/README.md) |
| Backend serving | [protocolnerd-backend/README.md](protocolnerd-backend/README.md) |
| Evaluation (the paper's numbers) | [protocolnerd-backend/benchmarking/README.md](protocolnerd-backend/benchmarking/README.md) |
| AWS deployment | [deploy/AWS_DEPLOYMENT.md](deploy/AWS_DEPLOYMENT.md) |

## Project structure

```
protocolnerd-backend/
  main.py                  FastAPI app: /chat orchestration, SSE, domain gating
  claude_client.py         all LLM calls (profile, queries, routing, explanations)
  llm_providers.py         provider abstraction: Claude / OpenAI / Gemini / Ollama
  domains/                 domain plugins
    base.py                the Domain interface a new domain implements
    registry.py            LLM router + keyword fallback
    biology.py             the primary domain (paper Section 5); pairs with PubMed
    chemistry.py           second domain (paper Section 6); pairs with Europe PMC
  retrievers.py            retrieval sources, gated per domain
  europepmc_client.py      Europe PMC METHODS: search (chemistry's literature lane)
  protocol_rag.py          TF-IDF index and search
  dense_index.py           MiniLM embedding index
  reranker.py              Claude Haiku re-ranking passes
  pubmed_client.py         NCBI E-utilities client
  protocolsio_client.py    Protocols.io API client
  benchmarking/            evaluation harness — see benchmarking/README.md

protocolnerd-website/
  chat.html                the protocol search UI
  index.html               root redirect to chat.html

data/                      your crawled corpus + index build outputs (not in git)
tests/                     unit tests
deploy/                    deployment notes
scripts/                   corpus and index build scripts
```

## Extending to a new domain

Domain-specific behavior lives behind the `Domain` interface in
`protocolnerd-backend/domains/base.py`. A new domain supplies its own profile fields,
clarification questions, query shaping, ranking, and prompts, then registers itself. The
retrieval, fusion, re-ranking, and explanation stages are reused unchanged, because they
operate on the profile rather than on domain vocabulary.

The router needs no edit either: it builds its menu from the one-line `description` each
registered domain declares about itself.

`chemistry.py` is the worked example, live in the deployed system: it declares 13 chemistry
profile fields and its own prompts, pairs protocols.io with Europe PMC instead of PubMed, and
was evaluated with the same citation-grounded method as biology (paper Section 6).

## Evaluation

The paper reports one experiment: a citation-grounded benchmark of 100 papers, each confirmed
from its own Methods section to have actually used a specific Protocols.io protocol.

| Method | Find rate (top 10) |
|---|---|
| **ProtocolNerd** | **46%** |
| LLM + web search, given the full abstract | 34% |
| LLM + web search, given the same short query | 19% |
| DP (Wang et al. Detailed Prompt) keyword baseline | 22% |

The chemistry extension, evaluated on its own 100 confirmed pairs with the same method and no
baseline comparison, finds the confirmed protocol 32% of the time against a corpus built for
biology.

See [protocolnerd-backend/benchmarking/README.md](protocolnerd-backend/benchmarking/README.md)
for how to reproduce these.

## Corpus

Protocols.io has no bulk-export API, so the corpus is built by searching its public API with a
curated list of 761 terms covering biology techniques, organisms, sample
types, and assays. Protocols are deduplicated by ID.

```bash
python3 scripts/fetch_protocols.py      # fetch/refresh protocols (resumable)
python3 scripts/build_index.py          # TF-IDF index
python3 scripts/build_dense_index.py    # semantic index
```

Updates fetch newest-first and stop at the first already-saved protocol, so refreshes are
incremental.

## Deployment

Runs as a single container on AWS ECS Fargate (us-east-1), serving the backend and the chat
frontend from one port, behind an application load balancer and Cloudflare. Secrets come from
AWS Secrets Manager, never from the image.

```bash
SHA=$(git rev-parse --short=8 HEAD)
REPO=<account>.dkr.ecr.us-east-1.amazonaws.com/protocolsnerd

aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin "$REPO"
docker build --platform linux/amd64 --build-arg BUILD_SHA="$SHA" -t "$REPO:$SHA" .
docker push "$REPO:$SHA"
# then register a new task definition revision with that image and update the service
```

The full runbook, including verification, rollback, configuration, and corpus refresh, is
[deploy/AWS_DEPLOYMENT.md](deploy/AWS_DEPLOYMENT.md).

`--platform linux/amd64` is required: the task definition is pinned to X86_64, so an Apple
Silicon build without it produces an image that will not start. `GET /health` reports the
build SHA, so you can confirm what is actually live.

## API

| Method | Path | Description |
|---|---|---|
| `POST` | `/chat` | Main endpoint: profiling, clarification, search, ranking, explanation |
| `GET` | `/health` | Health check, reports the running build SHA |
| `GET` | `/sse?session_id=...` | Progress stream for a session |
| `GET` | `/fetch_backend_mode` | Backend mode and available strategies |
| `GET` | `/ollama_status` | Ollama reachability and local models |
