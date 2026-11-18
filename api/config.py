"""Central configuration for the PATHSNAP API."""

import os
from pathlib import Path

from dotenv import load_dotenv

# Project root: parent directory of api/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load local environment variables, if a .env file exists.
load_dotenv(PROJECT_ROOT / ".env")


def resolve_path(value: str) -> Path:
    """Resolve a path relative to the project root unless already absolute."""
    path = Path(value).expanduser()

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path.resolve()


# API server settings
API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("API_PORT", "8000"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# Dataset paths
DATA_ROOT = resolve_path(os.getenv("DATA_ROOT", "data/raw"))
TRAIN_DATA_DIR = resolve_path(
    os.getenv("TRAIN_DATA_DIR", "data/raw/train")
)
STUDENT_TEST_DIR = resolve_path(
    os.getenv("EVAL_DATA_DIR", os.getenv("STUDENT_TEST_DIR", "data/raw/student_test"))
)

# Model and configuration paths
MODEL_DIR = resolve_path(os.getenv("MODEL_DIR", "models"))
CONFIG_DIR = resolve_path(os.getenv("CONFIG_DIR", "configs"))

# Output paths
MAP_OUTPUT_DIR = resolve_path(os.getenv("MAP_OUTPUT_DIR", "maps"))
RESULTS_DIR = resolve_path(os.getenv("RESULTS_DIR", "results"))
