import logging
from pathlib import Path

_LOG_FILE = Path(__file__).resolve().parent.parent / "audit_logs" / "pipeline.log"
_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

_logger = logging.getLogger("idamp")
if not _logger.handlers:
    _handler = logging.FileHandler(_LOG_FILE, encoding="utf-8")
    _handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    _logger.addHandler(_handler)
    _logger.setLevel(logging.INFO)
    _logger.propagate = False


def log(msg: str) -> None:
    """Write to a log file — immune to Streamlit's stdout/stderr capture and closure."""
    try:
        _logger.info(msg)
    except Exception:
        pass
