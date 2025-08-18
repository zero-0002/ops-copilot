# app/orchestrators/arbiter_utils.py

import os
import re
from typing import Dict, Any, Iterable, Tuple, List
from sqlalchemy.engine.url import make_url

# retriever tarafındaki normalize ile aynı davranış
try:
    from app.tools.retriever.search import canonicalize_prompt as _canonicalize_prompt
except Exception:
    def _canonicalize_prompt(q: str) -> str:
        return " ".join((q or "").strip().lower().split())

# ---------- public: canonicalize ----------
def canonicalize(q: str) -> str:
    """User prompt normalize (lower/trim/multi-space collapse)"""
    return _canonicalize_prompt(q or "")

# ---------- public: detect_dialect ----------
def detect_dialect() -> str:
    """
    DB_URL'den backend çıkarır (sqlite/mysql). Default sqlite.
    Örn: mysql+mysqlconnector://...  -> 'mysql'
         sqlite:///file.db           -> 'sqlite'
    """
    url = os.getenv("DB_URL", "sqlite:///app/data/thy_ops.db")
    try:
        backend = make_url(url).get_backend_name()
    except Exception:
        backend = "sqlite"

    if "mysql" in backend:
        return "mysql"
    if "sqlite" in backend:
        return "sqlite"
    # başka engine'lere de açık olsun
    return backend or "sqlite"

# ---------- helpers ----------
def _jaccard(a: Iterable, b: Iterable) -> float:
    sa, sb = set(a or []), set(b or [])
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / max(1, len(sa | sb))

_WIN_RX = re.compile(r"(?i)\b(?:son|last)\s+(\d+)\s*(g[uü]n|day|hafta|week|ay|month)s?\b")

def _canon_window(s: Any) -> str:
    """
    'son 30 gün' / 'last 2 weeks' -> ISO benzeri P30D / P2W
    Zaten P* formatındaysa dokunma. Yoksa None -> ''
    """
    if not s:
        return ""
    s = str(s).strip()
    if s.upper().startswith("P"):
        return s.upper()
    m = _WIN_RX.search(s)
    if not m:
        return s  # tanıyamazsak orijinali döndür
    n = int(m.group(1))
    unit = m.group(2).lower()
    if unit in ("gün", "gun", "day"):
        return f"P{n}D"
    if unit in ("hafta", "week"):
        return f"P{n}W"
    if unit in ("ay", "month"):
        return f"P{n}M"
    return s

def _flatten_filter_dict(d: Dict[str, Any]) -> List[Tuple[str, str]]:
    flat = []
    for k, v in (d or {}).items():
        if isinstance(v, (list, tuple, set)):
            for it in v:
                flat.append((str(k), str(it).lower()))
        elif isinstance(v, dict):
            # nested dict -> key.sub=value
            for sk, sv in v.items():
                flat.append((f"{k}.{sk}", str(sv).lower()))
        else:
            flat.append((str(k), str(v).lower()))
    return flat

# ---------- public: slot_coverage ----------
def slot_coverage(cached: Dict[str, Any], live: Dict[str, Any]) -> float:
    """
    Cache'teki slot'lar ile canlı isteğin slot'ları ne kadar uyuşuyor?
    Basit ama sağlam bir skor (0..1):
      - time_window: eşitlik (P* formatına normalize)
      - filters: jaccard(set of key=value)
      - dims/metrics: jaccard(list)
      - diğer scalar alanlar: eşitlik -> 1, değilse 0
    Ağırlıklar: window 0.4, filters 0.3, dims 0.15, metrics 0.15
    """
    cached = cached or {}
    live   = live or {}

    # time window
    w_cache = _canon_window(cached.get("time_window"))
    w_live  = _canon_window(live.get("time_window"))
    win_score = 1.0 if (w_cache and w_live and w_cache == w_live) else (1.0 if not (w_cache or w_live) else 0.0)

    # filters
    f_cache = set(_flatten_filter_dict(cached.get("filters", {})))
    f_live  = set(_flatten_filter_dict(live.get("filters", {})))
    filt_score = _jaccard(f_cache, f_live)

    # dims / metrics
    dims_score    = _jaccard(cached.get("dims", []),    live.get("dims", []))
    metrics_score = _jaccard(cached.get("metrics", []), live.get("metrics", []))

    # combine
    score = 0.4*win_score + 0.3*filt_score + 0.15*dims_score + 0.15*metrics_score
    return max(0.0, min(1.0, score))

# ---------- public: derive_intent_signature ----------
def derive_intent_signature(q: str, slots: Dict[str, Any]) -> str:
    """
    Stabil bir “niyet imzası” üret: window, dims, metrics ve filtrelerden oluşan deterministic string.
    Örn: 'type:kpi|window:P30D|dims:category|metrics:complaints_cnt,refund_rate|filters:weather_impact=1'
    """
    s = slots or {}
    window  = _canon_window(s.get("time_window"))
    dims    = sorted([str(x) for x in (s.get("dims") or [])])
    metrics = sorted([str(x) for x in (s.get("metrics") or [])])

    filt_pairs = _flatten_filter_dict(s.get("filters", {}))
    filt_pairs = sorted(f"{k}={v}" for k, v in filt_pairs)

    type_hint = "kpi" if metrics else "sql"

    return (
        f"type:{type_hint}"
        f"|window:{window or ''}"
        f"|dims:{','.join(dims)}"
        f"|metrics:{','.join(metrics)}"
        f"|filters:{','.join(filt_pairs)}"
    )
