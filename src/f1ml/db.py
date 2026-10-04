"""SQL Server access. Imported lazily so CSV-only runs need no ODBC driver."""
from __future__ import annotations

from urllib.parse import quote_plus

import pandas as pd

from .config import settings


def get_engine():
    from sqlalchemy import create_engine  # lazy: only needed for SQL runs

    parts = [
        f"DRIVER={{{settings.sql_driver}}}",
        f"SERVER={settings.sql_server}",
        f"DATABASE={settings.sql_database}",
        "TrustServerCertificate=yes",
    ]
    if settings.sql_trusted:
        parts.append("Trusted_Connection=yes")
    else:
        parts += [f"UID={settings.sql_user}", f"PWD={settings.sql_password}"]
    odbc = ";".join(parts)
    return create_engine(
        f"mssql+pyodbc:///?odbc_connect={quote_plus(odbc)}",
        fast_executemany=True,
    )


def read_sql(query: str, params: dict | None = None) -> pd.DataFrame:
    from sqlalchemy import text

    with get_engine().connect() as conn:
        return pd.read_sql(text(query), conn, params=params)
