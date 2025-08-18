import os, json, time
from typing import Dict, Any, List, Tuple
from sqlalchemy import create_engine, text, inspect

DB_URL = os.getenv("DB_URL", "sqlite:///app/data/thy_ops.db")
OUT_PATH = os.getenv("VOCAB_JSON_PATH", "app/data/vocab.json")

# Prompt'u şişirmemek için sınırlar
MAX_VALUES_PER_COL = int(os.getenv("VOCAB_MAX_VALUES_PER_COL", "30"))   # categoricals için
ID_PREVIEW_TOPK    = int(os.getenv("VOCAB_ID_PREVIEW_TOPK", "20"))      # kimlikler için örnek sayısı
LOW_CARD_MAX       = int(os.getenv("VOCAB_LOW_CARD_MAX", "50"))         # uniq <= 50 ise kategorik say
SAMPLE_LIMIT       = int(os.getenv("VOCAB_SAMPLE_LIMIT", "3"))

# Zorunlu dahil edilecek kimlik sütunları (yüksek kardinalite olsa bile)
ALWAYS_INCLUDE_IDS: Dict[str, List[str]] = {
    "flights":    ["flight_number"],
    "complaints": ["flight_number", "customer_id", "complaint_id"],
    "refunds":    ["flight_number", "customer_id", "ticket_id"],
    "weather":    ["flight_number"],
}

# Çok büyük uniq setleri kategorik olarak KEŞFETMEK yerine "id_preview"e taşıyalım
ID_LIKE_NAME_HINTS = {"id", "number", "code"}

engine = create_engine(DB_URL, future=True)

def _looks_like_id(colname: str) -> bool:
    n = colname.lower()
    return any(h in n for h in ID_LIKE_NAME_HINTS)

def _is_date_col(name: str) -> bool:
    n = name.lower()
    return any(k in n for k in ["date", "time", "created", "updated", "departure"])

def _count_distinct(conn, table: str, col: str) -> int:
    try:
        return int(conn.execute(text(f"SELECT COUNT(DISTINCT {col}) FROM {table} WHERE {col} IS NOT NULL")).scalar() or 0)
    except Exception:
        return 0

def _distinct_values(conn, table: str, col: str, limit: int) -> List[Any]:
    try:
        rows = conn.execute(
            text(f"SELECT DISTINCT {col} AS v FROM {table} WHERE {col} IS NOT NULL LIMIT :n"),
            {"n": limit}
        ).fetchall()
        return [r[0] for r in rows if r[0] is not None]
    except Exception:
        return []

def _topk_by_freq(conn, table: str, col: str, k: int) -> Tuple[List[Any], int]:
    """Kimlik gibi sütunlarda: en sık geçen K örnek + toplam uniq sayısı."""
    total_uniq = _count_distinct(conn, table, col)
    try:
        rows = conn.execute(
            text(f"""
                SELECT {col} AS v, COUNT(*) AS c
                FROM {table}
                WHERE {col} IS NOT NULL
                GROUP BY {col}
                ORDER BY c DESC
                LIMIT :k
            """),
            {"k": k}
        ).fetchall()
        topk = [r[0] for r in rows if r[0] is not None]
    except Exception:
        topk = []
    return topk, total_uniq

def build_vocab() -> Dict[str, Any]:
    data: Dict[str, Any] = {"generated_at": int(time.time()), "tables": {}}
    with engine.connect() as conn:
        insp = inspect(conn)
        for t in insp.get_table_names():
            cols = insp.get_columns(t)
            info = {
                "categoricals": {},          # {col: {"uniq": U, "values":[...]}}
                "id_preview": {},            # {col: {"uniq": U, "examples":[...]}}
                "date_cols": {},             # {col: {"min":..., "max":...}}
                "sample": [],                # örnek satır (debug/gözle kontrol)
                "total_rows": 0,
            }

            # toplam satır
            try:
                info["total_rows"] = int(conn.execute(text(f"SELECT COUNT(*) FROM {t}")).scalar() or 0)
            except Exception:
                info["total_rows"] = 0

            # örnek satır
            try:
                info["sample"] = [dict(r._mapping) for r in conn.execute(text(f"SELECT * FROM {t} LIMIT :n"), {"n": SAMPLE_LIMIT})]
            except Exception:
                pass

            for c in cols:
                cname = c["name"]
                ctype = (str(c.get("type")) or "").upper()

                # tarih aralığı
                if _is_date_col(cname):
                    try:
                        mm = conn.execute(text(f"""
                            SELECT MIN({cname}) AS mi, MAX({cname}) AS ma
                            FROM {t}
                            WHERE {cname} IS NOT NULL
                        """)).mappings().fetchone()
                        if mm:
                            info["date_cols"][cname] = {"min": mm["mi"], "max": mm["ma"]}
                    except Exception:
                        pass

                uniq = _count_distinct(conn, t, cname)
                if uniq == 0:
                    continue

                # Kimlik sayılabilecek sütunlar ve "whitelist" id kolonları → id_preview’e
                if cname in ALWAYS_INCLUDE_IDS.get(t, []) or _looks_like_id(cname):
                    examples, total = _topk_by_freq(conn, t, cname, ID_PREVIEW_TOPK)
                    info["id_preview"][cname] = {"uniq": total, "examples": examples}
                    continue

                # Kategorik (metin/flag) ve düşük kardinalite ise direkt values
                looks_text = ("CHAR" in ctype) or ("TEXT" in ctype) or ctype == ""
                looks_flag = ("INT" in ctype and uniq <= 3)
                is_low_card = (uniq <= LOW_CARD_MAX) or (info["total_rows"] and uniq/float(info["total_rows"]) <= 0.2)

                if (looks_text or looks_flag) and is_low_card:
                    vals = _distinct_values(conn, t, cname, MAX_VALUES_PER_COL)
                    if vals:
                        info["categoricals"][cname] = {"uniq": uniq, "values": vals}
                else:
                    # yüksek kardinalite metin kolon — küçük bir örnek göster, uniq sayısını yaz
                    examples, total = _topk_by_freq(conn, t, cname, min(ID_PREVIEW_TOPK, MAX_VALUES_PER_COL))
                    if examples:
                        info["id_preview"][cname] = {"uniq": total, "examples": examples}

            data["tables"][t] = info

    return data

if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    vocab = build_vocab()
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    print(f"[OK] wrote {OUT_PATH}")
