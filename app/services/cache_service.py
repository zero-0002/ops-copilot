# app/services/cache_service.py
import os, json, time, hashlib, re, ast
from typing import Any, Dict, Optional, Tuple, List

from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from langchain_openai import OpenAIEmbeddings
import numpy as np

from app.tools.retriever.search import search_exact, is_fresh, canonicalize_prompt
from app.tools.retriever.schema import get_redis, PREFIX
from app.tools.retriever.fingerprint import dataset_fingerprint
from app.agents.synthesis.synthesizer import synthesize_sql  # LLM varsa kullanacağız

DB_URL = os.getenv("DB_URL", "sqlite:///app/data/thy_ops.db")
engine = create_engine(DB_URL, future=True)

CACHE_EXECUTE_SQL_ON_HIT = os.getenv("CACHE_EXECUTE_SQL_ON_HIT", "true").lower() == "true"
CACHE_REQUIRE_APPROVAL   = os.getenv("CACHE_REQUIRE_APPROVAL", "true").lower() == "true"
CACHE_ALLOWED_TAGS       = {t.strip().lower() for t in os.getenv("CACHE_ALLOWED_TAGS", "approved,seed").split(",") if t.strip()}
CACHE_TTL_SEC            = int(os.getenv("CACHE_TTL_SEC", "2592000"))
EMBED_MODEL              = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")

# --- LLM’i cache-hit’te kullanıp kullanmayacağımızı belirle ---
def _is_ascii(s: str) -> bool:
    try:
        s.encode("ascii"); return True
    except Exception:
        return False

USE_LLM_SYNTH = os.getenv("CACHE_LLM_SYNTH", "true").lower() == "true" and bool(os.getenv("OPENAI_API_KEY"))
if USE_LLM_SYNTH and not _is_ascii(os.getenv("OPENAI_API_KEY","")):
    # Türkçe karakterli sahte key verilmişse otomatik kapat.
    USE_LLM_SYNTH = False

emb = OpenAIEmbeddings(model=EMBED_MODEL)

# ---------------- Helpers ----------------
def _is_approved(doc: Dict[str, Any]) -> bool:
    raw = (doc.get("tags") or "").strip()
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode(errors="ignore")

    tags = set()
    if isinstance(raw, str):
        if raw.startswith("["):
            try:
                tags = {str(t).strip().lower() for t in (json.loads(raw) or [])}
            except Exception:
                tags = set()
        else:
            tags = {t.strip().lower() for t in raw.split("|") if t.strip()}
    elif isinstance(raw, list):
        tags = {str(t).strip().lower() for t in raw}
    return (not CACHE_REQUIRE_APPROVAL) or bool(tags & CACHE_ALLOWED_TAGS)

_SELECT_RE = re.compile(r"^\s*(?:--.*?\n|\s)*\s*(select|with)\b", re.IGNORECASE | re.DOTALL)
def _is_select_like(sql_text: str) -> bool:
    return bool(_SELECT_RE.search(sql_text or ""))

def _has_limit(sql_text: str) -> bool:
    return " limit " in f" {(sql_text or '').lower()} "

def _append_limit(sql_text: str, limit_preview: int) -> str:
    s = (sql_text or "").strip().rstrip(";")
    if not _is_select_like(s):
        return ""
    low = s.lower()
    banned = ("insert", "update", "delete", "drop", "alter", "create", "attach", "reindex", "vacuum", "pragma")
    if any(b in low for b in banned):
        return ""
    if not _has_limit(s):
        s = f"{s} LIMIT {int(limit_preview)}"
    return s

def _wrap_with_limit(sql_text: str, limit_preview: int) -> str:
    s = (sql_text or "").strip().rstrip(";")
    if not _is_select_like(s):
        return ""
    return f"SELECT * FROM ({s}) AS subq LIMIT {int(limit_preview)}"

def _exec_sql_try(sql_text: str, limit_preview: int) -> Tuple[List[Dict[str, Any]], int]:
    try:
        with engine.connect() as conn:
            res = conn.execute(text(sql_text))
            rows = [dict(r._mapping) for r in res.fetchmany(limit_preview)]
            cnt  = res.rowcount if res.rowcount is not None else len(rows)
            return rows, cnt
    except SQLAlchemyError:
        return [], 0

def _exec_sql_best_effort(sql_text: str, limit_preview: int) -> Tuple[List[Dict[str, Any]], int]:
    limited = _append_limit(sql_text, limit_preview)
    if limited:
        rows, cnt = _exec_sql_try(limited, limit_preview)
        if rows: return rows, cnt
    wrapped = _wrap_with_limit(sql_text, limit_preview)
    if wrapped:
        rows, cnt = _exec_sql_try(wrapped, limit_preview)
        if rows: return rows, cnt
    return _exec_sql_try((sql_text or "").strip().rstrip(";"), limit_preview)

def _load_sample(sample: Any, preview_rows: int) -> List[Dict[str, Any]]:
    if isinstance(sample, list):
        return sample[:preview_rows]
    if isinstance(sample, (str, bytes)):
        s = sample.decode() if isinstance(sample, (bytes, bytearray)) else sample
        try:
            data = json.loads(s)
            if isinstance(data, list): return data[:preview_rows]
        except Exception:
            pass
        try:
            import ast
            data = ast.literal_eval(s)
            if isinstance(data, list): return data[:preview_rows]
        except Exception:
            return []
    return []

def _quick_summary_tr(rows: List[Dict[str,Any]]) -> str:
    if not rows:
        return "Veri bulunamadı; bu nedenle kıyas ve toplamlar hesaplanamadı."
    keys = set(rows[0].keys())
    if {"category","complaints_cnt"} <= keys:
        top = max(rows, key=lambda r: r.get("complaints_cnt", 0))
        return f"En çok şikayet edilen kategori: {top['category']} (n={top['complaints_cnt']})."
    return "Önbellekten satır önizlemesi gösterildi."

# ---------------- Public API ----------------
async def try_cache(
    q: str,
    k: int,
    threshold: float,
    preview_rows: int,
    return_rows: bool,
    return_chart: bool,
    session_id: str | None = None,
) -> Optional[Dict[str, Any]]:
    t0 = time.time()

    doc = search_exact(q)
    if not doc:
        return None
    if not is_fresh(doc) or not _is_approved(doc):
        return None

    sql_text = (doc.get("sql_text") or "").strip()
    if not sql_text:
        return None

    rows: List[Dict[str, Any]] = []
    if (return_rows or return_chart):
        rows = _load_sample(doc.get("sample_preview"), preview_rows)
        if not rows and CACHE_EXECUTE_SQL_ON_HIT:
            rows, _ = _exec_sql_best_effort(sql_text, min(preview_rows, 150))

    analysis_text = ""
    headline_metrics: List[Dict[str, Any]] = []
    vega = None

    if USE_LLM_SYNTH:
        try:
            syn = synthesize_sql(
                question=q, sql=sql_text, rows=rows,
                want_chart=bool(return_chart), max_rows=preview_rows,
            )
            analysis_text = (syn.get("analysis_text") or "").strip()
            headline_metrics = syn.get("headline_metrics") or []
            vega = syn.get("vega_lite_spec") if return_chart else None
        except Exception:
            analysis_text = _quick_summary_tr(rows)
            headline_metrics = []
            vega = None
    else:
        # LLM yok → basit özet
        analysis_text = _quick_summary_tr(rows)
        headline_metrics = []
        vega = None

    took = int((time.time() - t0) * 1000)
    return {
        "use_web": False,
        "want_sql": True,
        "final_answer": analysis_text,
        "analysis_text": analysis_text,
        "headline_metrics": headline_metrics,
        "vega_lite_spec": vega,
        "sql": sql_text,
        "rows": (rows if return_rows else []),
        "columns": (list(rows[0].keys()) if (return_rows and rows) else []),
        "citations": [],
        "used_cache": True,
        "cache_score": 1.10,
        "cache_took_ms": took,
    }

# write-through aynı (değişmedi)
def _sig_for(prompt_norm: str, sql_text: str) -> str:
    m = hashlib.sha1(); m.update(prompt_norm.encode("utf-8")); m.update(b"||"); m.update(sql_text.encode("utf-8"))
    return m.hexdigest()

def _to_bytes_float32(vec):
    return np.array(vec, dtype=np.float32).tobytes(order="C")

async def write_through_if_needed(question: str, graph_out: Dict[str, Any], session_id: str | None = None) -> None:
    if os.getenv("CACHE_WRITE_THROUGH", "false").lower() != "true":
        return
    sql = (graph_out.get("sql") or "").strip()
    if not sql:
        return
    prompt_norm = canonicalize_prompt(question)
    try:
        r = get_redis()
        key = f"{PREFIX}{_sig_for(prompt_norm, sql)}"
        vec = emb.embed_query(prompt_norm)
        payload = {
            "prompt": question, "prompt_norm": prompt_norm, "intent_signature": "", "slots": "{}",
            "sql_text": sql, "sql_params_schema": "{}", "dialect": graph_out.get("dialect","sqlite"),
            "rowcount": str(len(graph_out.get("rows") or [])),
            "sample_preview": json.dumps((graph_out.get("rows") or [])[:50], ensure_ascii=False),
            "dataset_fingerprint": dataset_fingerprint(), "created_at": str(int(time.time())),
            "ttl_sec": str(CACHE_TTL_SEC), "tags": "", "embedding": _to_bytes_float32(vec),
            "session_id": session_id or "",
        }
        r.hset(key, mapping=payload); r.expire(key, CACHE_TTL_SEC)
    except Exception:
        pass
