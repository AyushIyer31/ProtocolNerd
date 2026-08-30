# Offline pipeline: corpus and indexes

Everything here runs before the server ever starts. The output is the corpus in
`data/` and the two search indexes the backend loads at boot. The corpus is not
distributed with the public repository, since the protocols belong to their
authors on Protocols.io: each developer crawls their own copy with the scripts
below. The Docker build re-runs the two index builds inside the image.

```
scripts/
  fetch_protocols.py         crawl protocols.io into data/protocols/ (one JSON per protocol)
  retrieve_all_protocols.sh  full crawl across the curated search-term list
  finalize_corpus.sh         dedupe and finalize a crawl
  backfill_steps.py          backfill protocol step text missing from early crawls
  build_index.py             TF-IDF index      -> data/protocol_index.pkl   (~30s)
  build_dense_index.py       MiniLM embeddings -> data/protocol_dense.npy   (minutes)
  write_frontend_env.py      generate the frontend's env.js for a deployment
  benchmark_expansion.py     one-off analysis of query-expansion settings
  count_unique.py            corpus statistics helper
```

## The corpus

Protocols.io has no bulk-export API, so the corpus is assembled by searching its
public API with a curated list of 761 biology terms (the list is Appendix A of
the paper). Each protocol lands as `data/protocols/<id>.json`; a full crawl
yields roughly 22,700 protocols after deduplication by ID. The crawl is resumable and
fetches newest-first, stopping at the first already-saved protocol, so a re-run
is an incremental refresh rather than a restart.

```bash
python3 scripts/fetch_protocols.py       # refresh the corpus (needs PROTOCOLS_IO_TOKEN)
python3 scripts/build_index.py           # rebuild TF-IDF
python3 scripts/build_dense_index.py     # rebuild embeddings
```

## Refresh reality

Nothing schedules this. The deployed index is frozen at image build time, so a
corpus refresh is: crawl, then rebuild and redeploy the image
(see [../deploy/AWS_DEPLOYMENT.md](../deploy/AWS_DEPLOYMENT.md)).
