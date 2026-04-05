"""Cortex configuration — paths, model version, constants."""

import os
import platform
from pathlib import Path

# Paths
CORTEX_DIR = Path.home() / ".cortex"
DB_PATH = CORTEX_DIR / "cortex.db"
LOG_PATH = CORTEX_DIR / "cortex.log"
INGEST_LOG_PATH = CORTEX_DIR / "ingest.log"
STATE_PATH = CORTEX_DIR / "state.json"

# Embedding model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# Claude Code history
CLAUDE_DIR = Path.home() / ".claude"
HISTORY_PATH = CLAUDE_DIR / "history.jsonl"
PROJECTS_DIR = CLAUDE_DIR / "projects"

# Goose
if platform.system() == "Windows":
    _appdata = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    GOOSE_DB_PATH = _appdata / "Block" / "goose" / "data" / "sessions" / "sessions.db"
else:
    GOOSE_DB_PATH = Path.home() / ".local" / "share" / "goose" / "sessions" / "sessions.db"

# Distillation
DEFAULT_CLUSTER_MIN = 2  # minimum entries to form a cluster
DEFAULT_BATCH_SIZE = 10  # entries per distillation batch
DEFAULT_TOP_K = 10  # default results for query

# Entry types
ENTRY_TYPES = {"raw", "observation", "recommendation", "correction", "pattern"}

# Ranking weights
SIMILARITY_WEIGHT = 0.6
CONFIDENCE_WEIGHT = 0.25
RECENCY_WEIGHT = 0.15
