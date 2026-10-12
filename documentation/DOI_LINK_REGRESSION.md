# DOI result-link regression

Current main 28145ae normalized a bare DOI `10.17504/protocols.io.abc123` as `https://10.17504/protocols.io.abc123`, treating the DOI prefix as a hostname. The result is not a DOI resolver link.

The patch maps bare identifiers to `https://doi.org/<identifier>`. Existing HTTP(S) URLs and scheme-less doi.org/dx.doi.org links retain their existing destinations. The no-DOI URL/slug/empty fallbacks are unchanged.

Offline command:

```
python -m pytest -q tests/test_protocolsio_links.py
```

Before: 2 failed, 7 passed. After: 9 passed. No network requests or model calls. Synthetic DOI fixtures test normalization, not whether a particular publication exists.

Original repository suite before patch: 29 passed, 5 failed. Four query-logger tests depend on collection/import order (query logger imported by main before their environment setup); those five logger tests pass when run alone. The remaining existing ranking test expects "mouse system" while current code returns "animal system". These are not hidden or fixed in this narrow link patch. Original ranking tests plus the new link tests: 37 passed, 1 failed. Production deployment and end-to-end retrieval were not performed.
