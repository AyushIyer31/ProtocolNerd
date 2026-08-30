#!/usr/bin/env python3
"""
Build the TF-IDF protocol index once and pickle it, so deploy images can ship a
prebuilt index instead of building it at runtime. Cloud Run throttles CPU
between requests, which stalls a background build — baking it avoids that.

Run from the repo root (the Dockerfile does this at build time):
    python scripts/build_index.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "protocolnerd-backend"))

from protocol_rag import build_protocol_index, save_protocol_index  # noqa: E402

DATA_DIR = ROOT / "data" / "protocols"
OUT_PATH = ROOT / "data" / "protocol_index.pkl"


def main() -> None:
    index = build_protocol_index(DATA_DIR)
    save_protocol_index(index, OUT_PATH)
    print(f"Built index for {len(index['protocols'])} protocols -> {OUT_PATH}")


if __name__ == "__main__":
    main()
