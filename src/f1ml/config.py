"""Settings read from environment variables (or a local .env file).

    SQL_SERVER      e.g.  localhost\\SQLEXPRESS   or   SAHIL\\SQLEXPRESS
    SQL_DATABASE    default F1_DB2
    SQL_DRIVER      default ODBC Driver 17 for SQL Server
    SQL_TRUSTED     yes (Windows login, default) | no
    SQL_USER / SQL_PASSWORD   only when SQL_TRUSTED=no
    DATA_DIR        folder for CSV feature exports (default ./data)
    MODEL_DIR       folder for trained model files (default ./models)
    REPORT_DIR      folder for metrics + model cards (default ./reports)
    OLLAMA_HOST     default http://localhost:11434
    OLLAMA_MODEL    default qwen2.5-coder:3b
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _load_dotenv(path: Path) -> None:
    """Minimal .env reader: KEY=VALUE per line, # comments. Real env vars win."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


PROJECT_ROOT = Path(__file__).resolve().parents[2]
_load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    sql_server: str = os.getenv("SQL_SERVER", r"localhost\SQLEXPRESS")
    sql_database: str = os.getenv("SQL_DATABASE", "F1_DB2")
    sql_driver: str = os.getenv("SQL_DRIVER", "ODBC Driver 17 for SQL Server")
    sql_trusted: bool = os.getenv("SQL_TRUSTED", "yes").lower() in ("yes", "true", "1")
    sql_user: str = os.getenv("SQL_USER", "")
    sql_password: str = os.getenv("SQL_PASSWORD", "")
    data_dir: Path = Path(os.getenv("DATA_DIR", PROJECT_ROOT / "data"))
    model_dir: Path = Path(os.getenv("MODEL_DIR", PROJECT_ROOT / "models"))
    report_dir: Path = Path(os.getenv("REPORT_DIR", PROJECT_ROOT / "reports"))
    ollama_host: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:3b")


settings = Settings()
