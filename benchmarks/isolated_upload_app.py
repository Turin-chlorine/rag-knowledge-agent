"""Serve the real project API against a dedicated benchmark knowledge base."""

from pathlib import Path
import sys
import os

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

import config

RUN = Path(os.environ.get("RAG_BENCH_RUN_DIR", str(ROOT / "data" / "benchmarks" / "upload200" / "api_run")))
config.STORE_PATH = RUN / "vector_store.json"
config.UPLOAD_DIR = RUN / "uploads"

from main import app  # noqa: E402  - after patching the paths imported by the project
