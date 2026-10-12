# DOI normalization: 20 deterministic paired runs

Each row invokes the actual baseline and patched normalizer in separate fresh Python processes. No network/model call. Synthetic DOI, not a publication-existence claim.

| Run | Case | Before (verbatim URL) | After (verbatim URL) | Result |
|---|---|---|---|---|
| 1 | bare | `https://10.17504/protocols.io.abc123` | `https://doi.org/10.17504/protocols.io.abc123` | PASS |
| 2 | padded | `https://10.17504/protocols.io.abc123` | `https://doi.org/10.17504/protocols.io.abc123` | PASS |
| 3 | resolver | `https://doi.org/10.17504/protocols.io.abc123` | `https://doi.org/10.17504/protocols.io.abc123` | PASS |
| 4 | legacy | `https://dx.doi.org/10.17504/protocols.io.abc123` | `https://dx.doi.org/10.17504/protocols.io.abc123` | PASS |
| 5 | https | `https://doi.org/10.17504/protocols.io.abc123` | `https://doi.org/10.17504/protocols.io.abc123` | PASS |
| 6 | http | `http://dx.doi.org/10.17504/protocols.io.abc123` | `http://dx.doi.org/10.17504/protocols.io.abc123` | PASS |
| 7 | fallback-url | `https://example.com/protocol` | `https://example.com/protocol` | PASS |
| 8 | fallback-slug | `https://www.protocols.io/view/published-slug` | `https://www.protocols.io/view/published-slug` | PASS |
| 9 | empty | `` | `` | PASS |
| 10 | bare | `https://10.17504/protocols.io.abc123` | `https://doi.org/10.17504/protocols.io.abc123` | PASS |
| 11 | padded | `https://10.17504/protocols.io.abc123` | `https://doi.org/10.17504/protocols.io.abc123` | PASS |
| 12 | resolver | `https://doi.org/10.17504/protocols.io.abc123` | `https://doi.org/10.17504/protocols.io.abc123` | PASS |
| 13 | legacy | `https://dx.doi.org/10.17504/protocols.io.abc123` | `https://dx.doi.org/10.17504/protocols.io.abc123` | PASS |
| 14 | https | `https://doi.org/10.17504/protocols.io.abc123` | `https://doi.org/10.17504/protocols.io.abc123` | PASS |
| 15 | http | `http://dx.doi.org/10.17504/protocols.io.abc123` | `http://dx.doi.org/10.17504/protocols.io.abc123` | PASS |
| 16 | fallback-url | `https://example.com/protocol` | `https://example.com/protocol` | PASS |
| 17 | fallback-slug | `https://www.protocols.io/view/published-slug` | `https://www.protocols.io/view/published-slug` | PASS |
| 18 | empty | `` | `` | PASS |
| 19 | bare | `https://10.17504/protocols.io.abc123` | `https://doi.org/10.17504/protocols.io.abc123` | PASS |
| 20 | padded | `https://10.17504/protocols.io.abc123` | `https://doi.org/10.17504/protocols.io.abc123` | PASS |

20/20 expected post-fix outputs. Bare DOI regression reproduced in 6/20 baseline runs; 14 unchanged cases retained.
