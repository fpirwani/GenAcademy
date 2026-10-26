"""Local-only settings. Model files are managed separately by Ollama."""

import os
from pathlib import Path

# Disable optional tracing even if enabled in the launching shell.
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["OTEL_SDK_DISABLED"] = "true"

ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("DOCUQA_V2_DATA_DIR", str(ROOT / "data"))).resolve()
MODEL = os.environ.get("DOCUQA_MODEL", "qwen3:4b")
EMBEDDING_MODEL = "nomic-embed-text"
OLLAMA_URL = "http://127.0.0.1:11434"
MAX_BYTES = 25 * 1024 * 1024
MAX_PAGES = 200
MAX_CHARS = 600_000
