"""Dense (embedding) retrieval over the protocol corpus, fused with TF-IDF via RRF.

WHY: TF-IDF only matches literal word overlap, so a request phrased differently from the
protocol ("how do plants cope without water" vs "Drought stress tolerance assay") scores ~0
and never enters the candidate pool -- and the re-ranker cannot rescue what retrieval never
fetched. A sentence-embedding index matches on MEANING and fills exactly that gap.

We keep BOTH: dense fails where lexical wins (exact identifiers, rare jargon -- "Q5
polymerase", "pCAMBIA1300"), so the two are complementary and fused, never swapped.

Runtime is ONNX Runtime + tokenizers (NO torch): the CPU torch wheel would add ~1GB to the
image and a large resident footprint, and the Fargate task is 1 vCPU / 2GB. At query time we
only ever embed one short string; the corpus is embedded once at build time.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

log = logging.getLogger(__name__)

MODEL_REPO = os.getenv("DENSE_MODEL_REPO", "Xenova/all-MiniLM-L6-v2")
MODEL_FILE = os.getenv("DENSE_MODEL_FILE", "onnx/model.onnx")
MAX_TOKENS = int(os.getenv("DENSE_MAX_TOKENS", "256") or 256)   # MiniLM truncates here anyway
EMBED_DIM = 384

_session = None
_tokenizer = None


def _artifacts():
    """Lazily load the ONNX session + tokenizer (cached process-wide)."""
    global _session, _tokenizer
    if _session is None or _tokenizer is None:
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer
        model_path = os.getenv("DENSE_MODEL_PATH") or hf_hub_download(MODEL_REPO, MODEL_FILE)
        tok_path = os.getenv("DENSE_TOKENIZER_PATH") or hf_hub_download(MODEL_REPO, "tokenizer.json")
        tok = Tokenizer.from_file(tok_path)
        tok.enable_truncation(max_length=MAX_TOKENS)
        tok.enable_padding(length=None)          # pad to longest in batch
        _session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        _tokenizer = tok
    return _session, _tokenizer


def embed(texts: Sequence[str], batch_size: int = 64) -> np.ndarray:
    """Embed texts -> L2-normalized float32 matrix (n, 384).

    all-MiniLM-L6-v2 is a MEAN-POOLING model: the sentence vector is the average of the
    token vectors weighted by the attention mask (padding must not contribute), then L2
    normalized so cosine similarity is a plain dot product.
    """
    sess, tok = _artifacts()
    out = np.zeros((len(texts), EMBED_DIM), dtype=np.float32)
    for start in range(0, len(texts), batch_size):
        chunk = [t if (t and t.strip()) else " " for t in texts[start:start + batch_size]]
        encs = tok.encode_batch(chunk)
        ids = np.array([e.ids for e in encs], dtype=np.int64)
        mask = np.array([e.attention_mask for e in encs], dtype=np.int64)
        hidden = sess.run(["last_hidden_state"], {
            "input_ids": ids,
            "attention_mask": mask,
            "token_type_ids": np.zeros_like(ids),
        })[0]                                        # (b, seq, 384)
        m = mask[..., None].astype(np.float32)       # (b, seq, 1) — zero out padding
        summed = (hidden * m).sum(axis=1)
        counts = np.clip(m.sum(axis=1), 1e-9, None)
        vecs = summed / counts
        norms = np.clip(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-9, None)
        out[start:start + len(chunk)] = (vecs / norms).astype(np.float32)
    return out


def protocol_embed_text(p: Dict[str, Any], desc_chars: int = 600) -> str:
    """Focused text for the DENSE index -- deliberately NOT the TF-IDF blob.

    TF-IDF concatenates title x3 + steps + materials + guidelines, which is fine for a bag
    of words but wrong here: the model truncates at ~256 tokens, so a giant blob buries the
    identity of the protocol in boilerplate. Title + keywords + a description snippet is
    what actually says "this is what this protocol does".
    """
    from protocol_rag import _draftjs_to_text  # local import; avoids a cycle at module load
    title = (p.get("title") or "").strip()
    kws = " ".join(p.get("keywords") or [])
    desc = _draftjs_to_text(p.get("description"))[:desc_chars]
    return ". ".join(x for x in (title, kws, desc) if x).strip()


def build_dense_index(protocols: List[Dict[str, Any]], out_path: Path,
                      batch_size: int = 64) -> np.ndarray:
    """Embed every protocol and save as float16 (halves the artifact; retrieval is unaffected)."""
    texts = [protocol_embed_text(p) for p in protocols]
    log.info(f"Embedding {len(texts)} protocols with {MODEL_REPO}...")
    mat = embed(texts, batch_size=batch_size)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, mat.astype(np.float16))
    log.info(f"Saved dense index {mat.shape} -> {out_path}")
    return mat


def load_dense_index(path: Path) -> Optional[np.ndarray]:
    """Load the dense matrix as float32 (stored float16). None if absent -> caller falls back."""
    p = Path(path)
    if not p.exists():
        return None
    return np.load(p).astype(np.float32)


# Feature flag + slice size. K=60 is the swept optimum: bigger slices buy no extra reach
# (>=1-best plateaus) and cost ordering quality, pushing ambiguous queries below the
# TF-IDF-only baseline. See the DENSE_TOP_K sweep in the eval reports.
FUSION_ENABLED = os.getenv("DENSE_FUSION", "0") == "1"
DENSE_TOP_K = int(os.getenv("DENSE_TOP_K", "60") or 60)

_matrix_cache: Dict[str, Optional[np.ndarray]] = {}


def get_dense_matrix(path: Path) -> Optional[np.ndarray]:
    """Process-wide cached matrix load (17MB) -- never reload it per request."""
    key = str(path)
    if key not in _matrix_cache:
        mat = load_dense_index(path)
        if mat is None:
            log.warning(f"Dense index missing at {path}; retrieval stays TF-IDF only.")
        else:
            log.info(f"Loaded dense index {mat.shape} from {path}")
        _matrix_cache[key] = mat
    return _matrix_cache[key]


def is_enabled() -> bool:
    return FUSION_ENABLED


def dense_search(matrix: np.ndarray, queries: Sequence[str],
                 top_k: int) -> List[tuple]:
    """(row_index, cosine) for the top_k most similar protocols across ALL queries.

    Vectors are L2-normalized, so cosine similarity is a dot product. With several candidate
    queries we take each protocol's BEST similarity over them (max), not the mean — a protocol
    that nails one facet of the request shouldn't be diluted by the facets it doesn't cover.
    """
    if matrix is None or not queries:
        return []
    qv = embed(list(queries))                 # (q, 384)
    sims = matrix @ qv.T                      # (n_protocols, q)
    best = sims.max(axis=1)
    k = min(top_k, best.shape[0])
    idx = np.argpartition(-best, k - 1)[:k]   # cheap top-k, then order those
    ordered = idx[np.argsort(-best[idx])]
    return [(int(i), float(best[i])) for i in ordered]


def dense_candidates(matrix: np.ndarray, protocols: List[Dict[str, Any]], query: str,
                     cqs: Optional[Sequence[str]], top_k: int) -> List[Dict[str, Any]]:
    """Embedding-retrieved candidates as search-result dicts, best->worst.

    The RAW query leads: it carries the natural phrasing the embedding model is good at
    ("how do plants cope without water"), which the keyword-ish candidate queries strip out.
    Shared by the app and the eval so the two can't drift apart.
    """
    if matrix is None:
        return []
    from protocol_rag import _draftjs_to_text
    queries = [query] + [c for c in (cqs or [])[:3] if c and c.strip()]
    queries = [q for q in queries if q and q.strip()]
    out: List[Dict[str, Any]] = []
    for row, sim in dense_search(matrix, queries, top_k):
        p = protocols[row]
        out.append({
            "id": p.get("id"), "title": p.get("title") or "",
            "uri": p.get("uri") or "", "doi": p.get("doi") or "",
            "url": p.get("url") or (f"https://{p.get('doi')}" if p.get("doi") else ""),
            "created_on": p.get("created_on"), "published_on": p.get("published_on"),
            "description": _draftjs_to_text(p.get("description"))[:400],
            "materials_text": _draftjs_to_text(p.get("materials_text"))[:300],
            "steps_preview": [s for s in (p.get("steps") or []) if s][:3],
            "step_count": len(p.get("steps") or []),
            "stats": p.get("stats") or {},
            "authors": p.get("authors") or [], "keywords": p.get("keywords") or [],
            # TF-IDF `score` here is itself a cosine (plus 0.5*title cosine), so the dense
            # cosine sits on a comparable 0-1 scale. Carrying it (rather than 0) lets a
            # dense-only hit compete for a lexical-shortlist slot instead of being frozen out.
            "score": round(sim, 4), "_dense": True,
        })
    return out


def fuse_pool(tfidf_results: List[Dict[str, Any]],
              dense_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """RRF-fuse a TF-IDF pool with a dense pool, de-duped by protocol id.

    Where both retrievers found a protocol we keep the TF-IDF dict (it carries the lexical
    score the downstream shortlist sorts on).
    """
    if not dense_results:
        return tfidf_results
    by_id = {r["id"]: r for r in tfidf_results}
    for d in dense_results:
        by_id.setdefault(d["id"], d)
    order = rrf_fuse([r["id"] for r in tfidf_results], [d["id"] for d in dense_results])
    return [by_id[i] for i in order if i in by_id]


def rrf_fuse(*ranked_lists: Sequence[Any], k: int = 60, limit: Optional[int] = None) -> List[Any]:
    """Reciprocal Rank Fusion: score(d) = sum over lists of 1/(k + rank(d)), rank 1-based.

    Fuses on RANK, never score, because TF-IDF weights and cosine similarities live on
    different, per-query-varying scales that can't be meaningfully averaged. A document
    ranked well by EITHER retriever still makes the pool; one ranked well by BOTH rises
    to the top.
    """
    scores: Dict[Any, float] = {}
    for lst in ranked_lists:
        for rank, item in enumerate(lst, 1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    fused = sorted(scores, key=lambda d: -scores[d])
    return fused[:limit] if limit else fused
