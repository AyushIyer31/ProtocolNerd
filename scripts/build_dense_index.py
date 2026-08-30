#!/usr/bin/env python3
"""
Build the DENSE (embedding) protocol index once and save it, alongside the TF-IDF pickle.
Same rationale as build_index.py: bake it into the image so the container never embeds the
corpus at runtime -- at query time we only ever embed one short query string.

Run from the repo root (the Dockerfile does this at build time):
    python scripts/build_dense_index.py
"""

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "protocolnerd-backend"))

from protocol_rag import load_protocol_index  # noqa: E402
from dense_index import build_dense_index  # noqa: E402

INDEX_PATH = ROOT / "data" / "protocol_index.pkl"
OUT_PATH = ROOT / "data" / "protocol_dense.npy"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    # Reuse the TF-IDF index's protocol list so row i of the dense matrix is exactly
    # protocols[i] -- the fusion step relies on that alignment.
    protocols = load_protocol_index(INDEX_PATH)["protocols"]
    mat = build_dense_index(protocols, OUT_PATH)
    print(f"Built dense index {mat.shape} for {len(protocols)} protocols -> {OUT_PATH}")


if __name__ == "__main__":
    main()
