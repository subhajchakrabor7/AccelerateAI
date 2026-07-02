import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
LANDING_DIR  = DATA_DIR / "landing"
PROFILES_DIR = DATA_DIR / "profiles"
STTM_DIR     = DATA_DIR / "sttm"
BRONZE_DIR   = DATA_DIR / "bronze_layer"
SILVER_DIR   = DATA_DIR / "silver_layer"
GOLD_DIR     = DATA_DIR / "gold_layer"
REPORTS_DIR  = BASE_DIR / "reports"
AUDIT_DIR    = BASE_DIR / "audit_logs"

for _d in [LANDING_DIR, PROFILES_DIR, STTM_DIR, BRONZE_DIR, SILVER_DIR, GOLD_DIR, REPORTS_DIR, AUDIT_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL   = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GEMINI_MODEL = "gemini-2.5-flash"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").strip().lower()
