import os
from typing import List, Tuple, Dict, Any
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.exc import SQLAlchemyError
from rapidfuzz import process, fuzz

load_dotenv()

# ---------------- DB URL (config + fallback) ----------------
def _default_sqlite_url() -> str:
    # app/data/thy_ops.db beklenen konum
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, "..", ".."))
    candidate = os.path.join(root, "data", "thy_ops.db")
    return f"sqlite:///{candidate}"

DB_URL = os.getenv("DB_URL", _default_sqlite_url())
engine = create_engine(DB_URL, future=True)

ALLOWED_TABLES = {"flights", "complaints", "refunds", "weather"}

# ---------------- helpers ----------------
def _quote_ident(name: str) -> str:
    if engine.dialect.name == "mysql":
        return f"`{name}`"
    return f"\"{name}\""

def _validate_table_column(table_name: str, column_name: str) -> None:
    insp = inspect(engine)
    tables = set(insp.get_table_names())
    if table_name not in tables:
        raise ValueError(f"Table not found: {table_name}")
    cols = {c["name"] for c in insp.get_columns(table_name)}
    if column_name not in cols:
        raise ValueError(f"Column not found: {table_name}.{column_name}")

# ---------------- fuzzy core ----------------
def get_best_fuzzy_match(input_value, choices) -> Tuple[str, int]:
    if not choices:
        return str(input_value), 0
    match, score, _ = process.extractOne(
        str(input_value),
        [str(c) for c in choices],
        scorer=fuzz.token_set_ratio
    )
    return match, int(score)

def get_values(table_name: str, column_name: str) -> List[str]:
    _validate_table_column(table_name, column_name)
    t = _quote_ident(table_name)
    c = _quote_ident(column_name)
    q = text(f"SELECT DISTINCT {c} AS val FROM {t} WHERE {c} IS NOT NULL")
    with engine.connect() as conn:
        df = pd.read_sql(q, con=conn)
    return df["val"].dropna().astype(str).tolist()


def call_match(val: List[Any]) -> List[List[str]]:

    """
    Girdi (filter agent):
      ["yes", ["flights","aircraft_type","A320"], ["complaints","status","open, pending"], ...]
      veya başlıksız liste
    Çıktı (triples):
      [["table name:flights","column_name:aircraft_type","filter_value:A320"], ...]
    """
    if not isinstance(val, list) or len(val) == 0:
        return []
    start_idx = 1 if isinstance(val[0], str) and val[0].lower() == "yes" else 0
    items = val[start_idx:]

    final: List[List[str]] = []
    for lst in items:
        if not lst or len(lst) < 3:
            continue
        table = str(lst[0]).strip()
        column = str(lst[1]).strip()
        if table not in ALLOWED_TABLES:
            continue  # güvenlik: sadece beklenen tablolar
        raw_value = str(lst[2])
        str_vals = [i.strip() for i in raw_value.split(",") if i.strip()]
        try:
            distinct_vals = get_values(table, column)
        except Exception:
            continue
        for sub in str_vals:
            best, _ = get_best_fuzzy_match(sub, distinct_vals)
            final.append([
                f"table name:{table}",
                f"column_name:{column}",
                f"filter_value:{best}"
            ])
    return final


def normalize_filters(val: List[Any]) -> List[Any]:
    """
    Girilen filter listesine dayanarak (DB distinct’e fuzzy map),
    normalize edilmiş ["yes", ["table","column","v1, v2"], ...] döndür.
    """
    triples = call_match(val)
    if not triples:
        return ["no"]

    agg: Dict[Tuple[str, str], List[str]] = {}
    for t, c, v in triples:
        table = t.split(":", 1)[1]
        column = c.split(":", 1)[1]
        value = v.split(":", 1)[1]
        key = (table, column)
        agg.setdefault(key, [])
        if value not in agg[key]:
            agg[key].append(value)

    out: List[Any] = ["yes"]
    for (table, column), values in agg.items():
        out.append([table, column, ", ".join(values)])
    return out
