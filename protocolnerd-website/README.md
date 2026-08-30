# Frontend

Static HTML/JS with no build step. In production the same container serves these
pages alongside the API; locally, any static server works.

```
chat.html         the protocol search UI (the product)
index.html        root redirect to chat.html
debug.html        backend-mode and model inspection helpers
query_logs.html   viewer for the S3 query logs
```

## Running locally

```bash
cd protocolnerd-website
python3 -m http.server 8080
# open http://localhost:8080/chat.html — it talks to the backend on :8001
```

The backend URL resolves from `window.PROTOCOLSNERD_API_URL` (written by
`scripts/write_frontend_env.py` for deployments) and falls back to
`http://localhost:8001` for local development.

## Where per-domain behavior lives

`chat.html` is domain-aware through small lookup tables rather than branches:

- `DOMAIN_PROFILE_VIEW` — how each domain's experiment profile renders: its
  field rows, labels, and which fields count as required per sub-intent. Adding
  a domain to the UI means adding one entry here; without it the domain still
  works but renders with the default (biology) field labels.
- `SOURCE_BADGE` — the label and colors for each retrieval source
  (protocols.io, PubMed, Europe PMC). A new source needs one line here or its
  results are badged with the fallback.

Debug mode (`?debug=1`) additionally shows the models in use, the deployment
commit and build time from `/health`, and the exact queries sent to each source.
