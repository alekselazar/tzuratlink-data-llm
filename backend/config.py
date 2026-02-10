import json
import os
from pathlib import Path
from typing import List

# Load .env: project root (local) and /app/.env (Docker mount)
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB = os.getenv("MONGO_DB", "tagger")
SEFARIA_BASE = os.getenv("SEFARIA_BASE", "https://www.sefaria.org")

# OpenAI Vision (line OCR)
OPENAI_API_KEY = (os.getenv("OPENAI_API_KEY") or "").strip()
OPENAI_MODEL = (os.getenv("OPENAI_MODEL") or "gpt-4o-mini").strip()
OPENAI_TIMEOUT_S = int(os.getenv("OPENAI_TIMEOUT_S", "60"))
OPENAI_MAX_RETRIES = int(os.getenv("OPENAI_MAX_RETRIES", "3"))
VLM_BATCH_SIZE = int(os.getenv("VLM_BATCH_SIZE", "12"))

# OpenAI Embeddings (for commentary and main-text alignment)
OPENAI_EMBEDDING_MODEL = (os.getenv("OPENAI_EMBEDDING_MODEL") or "text-embedding-3-small").strip()
OPENAI_EMBEDDING_BATCH_SIZE = int(os.getenv("OPENAI_EMBEDDING_BATCH_SIZE", "100"))
# Use embeddings for main-text segment alignment (else fuzzy text match)
USE_EMBEDDINGS_FOR_MAIN_ALIGN = os.getenv("USE_EMBEDDINGS_FOR_MAIN_ALIGN", "").strip().lower() in ("1", "true", "yes")

# Rashi Tesseract (tessdata dir containing rashi.tessdata; e.g. /data in Docker)
RASHI_TESSDATA_DIR = (os.getenv("RASHI_TESSDATA_DIR") or "data").strip()

# Commentary filter: only include Sefaria commentary whose title starts with one of these.
# Load from commentary_config.json (gitignored); copy from commentary_config.example.json.
DEFAULT_COMMENTARY_TITLE_PREFIXES: List[str] = ["Rashi on", "Tosafot on"]


def _commentary_config_path() -> Path:
    env_path = os.getenv("COMMENTARY_CONFIG_PATH", "").strip()
    if env_path and os.path.isfile(env_path):
        return Path(env_path)
    for base in (Path.cwd(), Path(__file__).resolve().parent.parent):
        p = base / "commentary_config.json"
        if p.is_file():
            return p
    return Path()


def get_commentary_title_prefixes() -> List[str]:
    path = _commentary_config_path()
    if not path or not path.is_file():
        return list(DEFAULT_COMMENTARY_TITLE_PREFIXES)
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        prefixes = data.get("commentary_title_prefixes")
        if isinstance(prefixes, list) and all(isinstance(x, str) for x in prefixes):
            return [x.strip() for x in prefixes if x.strip()]
    except (OSError, json.JSONDecodeError, TypeError):
        pass
    return list(DEFAULT_COMMENTARY_TITLE_PREFIXES)
