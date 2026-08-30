#!/usr/bin/env python3
"""How much chemistry does the Protocols.io corpus actually contain?

Section 6 of the paper attributes part of the chemistry find rate to coverage:
Protocols.io is dominated by biology, so a chemistry query competes for a much
shallower pool of genuinely comparable candidates. This measures that directly
against the same prebuilt index the deployed system searches, counting protocols
whose title or description mentions each of ten core techniques per domain.

The technique lists are deliberately central to each field and deliberately
symmetric in size, so the comparison is not built by choosing obscure chemistry
terms and common biology ones.

Usage:
    python corpus_coverage_by_domain.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _bootstrap  # noqa: F401
from protocol_rag import load_protocol_index  # type: ignore

INDEX = Path(__file__).resolve().parents[3] / "data" / "protocol_index.pkl"

CHEMISTRY = ["suzuki", "grignard", "raft polymer", "schlenk", "cross-coupling",
             "recrystalli", "distillation", "titration", "electrochem", "catalyst"]
BIOLOGY = ["pcr", "western blot", "cell culture", "crispr", "dna extraction",
           "transfection", "microscopy", "flow cytometry", "sequencing", "cloning"]


def count(protocols, terms):
    out = {}
    for t in terms:
        out[t] = sum(1 for p in protocols
                     if t in f"{p.get('title','')} {p.get('description','')}".lower())
    return out


def main() -> int:
    if not INDEX.exists():
        print(f"index not found at {INDEX}")
        return 1
    protocols = load_protocol_index(INDEX)["protocols"]
    print(f"corpus: {len(protocols)} protocols\n")

    for label, terms in (("CHEMISTRY", CHEMISTRY), ("BIOLOGY", BIOLOGY)):
        hits = count(protocols, terms)
        print(f"{label}, {len(terms)} core techniques: {sum(hits.values())} protocols")
        for k, v in sorted(hits.items(), key=lambda x: -x[1]):
            print(f"     {v:>5}  {k}")
        zero = [k for k, v in hits.items() if v == 0]
        if zero:
            print(f"     no protocols at all for: {', '.join(zero)}")
        print()

    c = sum(count(protocols, CHEMISTRY).values())
    b = sum(count(protocols, BIOLOGY).values())
    print(f"biology : chemistry = {b/max(c,1):.0f} to 1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
