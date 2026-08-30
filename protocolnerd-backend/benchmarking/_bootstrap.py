"""
Shared setup for the eval harness.

Import this FIRST in every eval script:

    import _bootstrap  # noqa: F401

It (1) puts the backend package on sys.path, (2) loads the API keys from
protocolnerd-backend/.env into os.environ, and (3) silences the noisy sklearn
pickle-version warning so eval output stays readable.
"""
from __future__ import annotations

import os
import shutil
import sys
import warnings
from pathlib import Path

# --- paths ------------------------------------------------------------------
EVAL_DIR = Path(__file__).resolve().parent
REPO_ROOT = EVAL_DIR.parent.parent
BACKEND_DIR = EVAL_DIR.parent
INDEX_PATH = REPO_ROOT / "data" / "protocol_index.pkl"
RESULTS_DIR = EVAL_DIR / "results"
CACHE_DIR = EVAL_DIR / "cache"
REPLAY_DIR = EVAL_DIR / "replay_cache"

RESULTS_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)


def _seed_cache_from_replay() -> None:
    """Give a fresh clone a working cache.

    `cache/` is gitignored working state, so a clone starts empty and every eval
    script would otherwise re-call the API -- the web-search baselines use Sonnet
    with a search tool, which is slow and costly. `replay_cache/` is committed and
    holds the same entries pruned to the 100 benchmark papers, so copying anything
    missing makes all four experiments replay offline for free.

    Existing files are never overwritten: a local cache always wins, so this cannot
    clobber a fresh run's results.
    """
    if not REPLAY_DIR.is_dir():
        return
    for src in REPLAY_DIR.glob("*.json"):
        dst = CACHE_DIR / src.name
        if not dst.exists():
            shutil.copy2(src, dst)


_seed_cache_from_replay()

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


# --- load env files (no hard dependency on python-dotenv) -------------------
def _load_env_file(env_file: Path) -> None:
    if not env_file.exists():
        return
    for raw in env_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        # Set if unset OR currently empty (some keys are split across files:
        # e.g. .env has an empty PROTOCOLS_IO_TOKEN, variables.env has the real one).
        if key and value and not os.environ.get(key):
            os.environ[key] = value


def _load_env() -> None:
    # variables.env first (holds PROTOCOLS_IO_TOKEN), then .env (LLM keys).
    _load_env_file(BACKEND_DIR / "variables.env")
    _load_env_file(BACKEND_DIR / ".env")


_load_env()

# The baked index was pickled with a slightly newer sklearn; the warning is
# harmless for read-only TF-IDF transforms.
warnings.filterwarnings("ignore", message=".*InconsistentVersionWarning.*")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
