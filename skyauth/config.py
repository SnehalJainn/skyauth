"""Runtime configuration: API keys, paths, logger, and shared in-memory state."""

import logging
import os
from pathlib import Path


OPENWEATHER_API_KEY = os.environ.get("OPENWEATHER_API_KEY", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_DIR = PROJECT_ROOT / "templates"

SESSION_TTL_SECONDS = 120
TIMESTAMP_FRESH_SECONDS = 120
TIMESTAMP_STALE_SECONDS = 300
MAX_GPS_DRIFT_KM = 1.0
MIN_SUN_ELEVATION_DEG = 3.0

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger("skyauth")

# Single-process, in-memory stores. Replace with Redis or a DB if you scale out.
sessions: dict = {}
transactions: list = []
