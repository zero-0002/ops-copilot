# app/agents/sql/sql_langraph.py
from __future__ import annotations

import ast
import os
import re
from typing import TypedDict, List, Dict, Any

from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

from langgraph.graph import StateGraph, START, END
# NOT: Send kullanmıyoruz (dict bekleniyor), o yüzden import etmiyoruz
from langgraph.constants import Send
from langgraph.checkpoint.memory import MemorySaver 

# i18n normalize (tek giriş noktası)
from app.i18n.locale import map_terms_to_schema

# Router
from app.agents.sql.router import route_agents

# Tek tablo sub-agent graph (subquestion + column selector)
from app.agents.sql.customer_agent import graph_final

# THY-özel OpenAI zincirleri (NL→SQL)
from app.agents.sql.sql_agent_thy import (
    chain_filter_extractor,
    chain_range_date_extractor,
    chain_query_extractor,
    chain_query_validator,
)



# Birleşik sentezleyici (SQL-only / Policy-only / Hybrid)
from app.agents.synthesis.synthesizer import synthesize_unified

# Fuzzy (DB DISTINCT’e map) + normalize edilmiş yes-list
from app.agents.sql.fuzzy_wuzzy import call_match, normalize_filters

# Web/Policy router (LLM-first, timeout→regex fallback)
from app.agents.dispatcher import classify_intent

# Web/Policy agent (RAG + Wikipedia fallback)
from app.agents.policy_web.policy_agent import answer_policy

# Yalnızca ham SQL’i ayıklamak için
from app.utils.text import extract_sql_only


# ---------------- config / db ----------------
DB_URL = os.getenv("DB_URL", "sqlite:///app/data/thy_ops.db")
engine = create_engine(DB_URL, future=True)

checkpointer = MemorySaver()

# ---------------- helpers ----------------
def _dedup_column_extract(outputs: Dict[str, Dict[str, Any]]) -> List[List[str]]:
    """
    flights_out / complaints_out / refunds_out / weather_out -> column_extract dedup
    """
    seen = set()
    final: List[List[str]] = []
    for v in outputs.values():
        if not v:
            continue
        for item in v.get("column_extract", []):
            key = tuple(item)
            if key not in seen:
                seen.add(key)
                final.append(item)
    return final


def _is_no_filter(val) -> bool:
    """["no"] / [] / hatalı yapı → True"""
    return (
        not isinstance(val, list)
        or len(val) == 0
        or (len(val) == 1 and str(val[0]).lower() == "no")
    )


# ---------------- state ----------------
class FinalState(TypedDict, total=False):
    # raw & normalized
    raw_query: str              # UI’dan gelen orijinal soru
    user_query: str             # normalize edilmiş soru
    term_map: Dict[str, str]    # {"rötar":"delay", ...} audit için

    # WEB/POLICY routing meta
    use_web: bool
    want_sql: bool
    time_window_days: int
    routing_reason: str
    policy_score: float
    sql_score: float

    # web/policy answer
    web_answer: str
    policy_citations: List[Dict[str, str]]

    # router (SQL domain içi)
    router_out: List[str]

    # single-table agents output
    flights_out: Dict[str, Any]
    complaints_out: Dict[str, Any]
    refunds_out: Dict[str, Any]
    weather_out: Dict[str, Any]

    # filters
    filtered_col: str           # str(list-of-lists)
    filter_extractor: list      # ["yes", ["table","col","values"], ...] | ["no"]
    fuzz_match: list            # [["table name:..","column_name:..","filter_value:.."], ...]
    fuzz_norm: list             # ["yes", ["table","col","v1, v2"], ...]
    range_filters: list         # ["ranges", ...] | ["dates", ...] | ["none"]

    # sql + exec + synthesis
    sql_query: str
    final_query: str
    preview_rows: int
    rows: List[Dict[str, Any]]
    columns: List[str]
    analysis_text: str
    headline_metrics: List[Dict[str, Any]]
    vega_lite_spec: Dict[str, Any]
    want_analysis: bool
    want_chart: bool

    # join/barrier kontrol
    policy_done: bool
    sql_done: bool
    arrived: List[str]


# ---------------- nodes ----------------
def normalize_node(state: FinalState):
    raw = state.get("raw_query", "") or state.get("user_query", "")
    q_norm, applied = map_terms_to_schema(raw)
    return {"user_query": q_norm, "raw_query": raw, "term_map": applied}


# app/agents/sql/sql_langraph.py (policy_gate)
import re  # dosyanın başında

def policy_gate(state: FinalState):
    q = state["user_query"]
    res = classify_intent(q)

    POLICY_KWS = [
        r"\bpolitika\w*\b", r"\bkoşul(lar)?\b", r"\biade\w*\b", r"\bücret\w*\b",
        r"\bkur(a|u)l\w*\b", r"\brefund\b", r"\bpolicy\b", r"\bfazla\s*bagaj\b", r"\bbagaj\b", r"\bbaggage\b"
    ]
    # 👇 “geciken” ve “en çok” eklendi
    SQL_KWS = [
        r"\bortalama\w*\b", r"\bavg\b", r"\btoplam\b", r"\bsum\b",
        r"\boran\w*\b", r"\btrend\w*\b", r"\bgecikme\w*\b", r"\bgecik\w*\b", r"\brötar\b",
        r"\bdelay\b", r"\bsay[ıi]s[ıi]\b", r"\ben\schok|en çok\b", r"\bmax\b", r"\bmin\b"
    ]

    p_hit = any(re.search(p, q, flags=re.IGNORECASE) for p in POLICY_KWS)
    s_hit = any(re.search(p, q, flags=re.IGNORECASE) for p in SQL_KWS)

    # LLM kararı temel alınır
    use_web  = bool(res.get("want_policy") or p_hit)
    want_sql = bool(res.get("want_sql")   or s_hit)

    # ❗ Force policy-only SADECE LLM de SQL istemiyorsa
    if use_web and not s_hit and not bool(res.get("want_sql")):
        want_sql = False

    return {
        "use_web": use_web,
        "want_sql": want_sql,
        "time_window_days": res.get("window_days", 60),
        "routing_reason": res.get("reason", ""),
        "policy_score": float(res.get("policy_score", 0.0)) + (0.2 if p_hit else 0.0),
        "sql_score": float(res.get("sql_score", 0.0)) + (0.2 if s_hit else 0.0),
    }

def policy_condition(state: FinalState):
    use_web = bool(state.get("use_web"))
    want_sql = bool(state.get("want_sql"))
    if use_web and want_sql: return "hybrid"
    if use_web:              return "web"
    return "sql"
import re

def policy_rewrite(state: FinalState):
    """
    Hibrit/policy isteklerde, user_query içinden politikayla ilgili parçayı ayıklar.
    Zaman penceresi ve SQL/analitik terimleri temizlenir.
    """
    if not state.get("use_web"):
        return {}

    q = state.get("user_query", "") or ""
    if not q:
        return {}

    # 1) Parçalara ayır (bağlaçlar)
    parts = re.split(r"\b(?:ve|ile|,|;|/|&)\b", q, flags=re.IGNORECASE)

    POLICY_KWS = r"(politika|policy|iade|ücret|kural|kur(a|u)l|bagaj|baggage|refund)"
    policy_parts = [p.strip() for p in parts if re.search(POLICY_KWS, p, re.IGNORECASE)]

    cand = policy_parts[0] if policy_parts else q

    # 2) Zaman penceresi & metrik temizliği
    cand = re.sub(r"\bson\s+\d+\s+(gün|hafta|ay|yıl)\b", "", cand, flags=re.IGNORECASE)
    cand = re.sub(r"\bge(çen|çtiğimiz)\s+(gün|hafta|ay|yıl)\b", "", cand, flags=re.IGNORECASE)
    cand = re.sub(r"\b\d+\s*(dk|dakika|saat|gün)\b", "", cand, flags=re.IGNORECASE)

    # Analitik/metrik kelimeleri at
    cand = re.sub(r"\b(ortalama|avg|toplam|sum|oran|trend|gecikme|rötar|delay|sayısı?)\b", "", cand, flags=re.IGNORECASE)

    # Temizle
    cand = re.sub(r"\s+", " ", cand).strip()
    if cand and not cand.endswith("?"):
        cand += "?"

    return {"policy_query": cand}


def web_policy_node(state: FinalState):
    q = state.get("policy_query") or state["user_query"] 
    out = answer_policy(q)
    return {
        "web_answer": out.get("answer", ""),
        "policy_citations": out.get("citations", []),
        "policy_done": True,
    }



def hybrid_fork(state: FinalState):
    """
    Paralel başlatıcı. LangGraph bu sürümde dict bekler; burada bayrak yazıp
    graf üzerinde iki ayrı edge ile web_policy ve router'ı tetikleriz.
    """
    return {"hybrid_started": True}

# def hybrid_fork(state: FinalState):
#     # sadece SEND! kenar ekleme yok.
#     return [Send("policy_rewrite", {}), Send("router", {})]

# ------ SQL PATH ------
def router(state: FinalState):
    return {"router_out": route_agents(state["user_query"])}


def route_request(state: FinalState):
    valid = {"flights", "complaints", "refunds", "weather"}
    return [r for r in state.get("router_out", []) if r in valid]


def flights_agent(state: FinalState):
    sub = graph_final.invoke({"user_query": state["user_query"], "table_lst": ["flights"]})
    return {"flights_out": sub}


def complaints_agent(state: FinalState):
    sub = graph_final.invoke({"user_query": state["user_query"], "table_lst": ["complaints"]})
    return {"complaints_out": sub}


def refunds_agent(state: FinalState):
    sub = graph_final.invoke({"user_query": state["user_query"], "table_lst": ["refunds"]})
    return {"refunds_out": sub}


def weather_agent(state: FinalState):
    sub = graph_final.invoke({"user_query": state["user_query"], "table_lst": ["weather"]})
    return {"weather_out": sub}


def filter_check(state: FinalState):
    q = state["user_query"]
    collected = {
        "flights_out": state.get("flights_out", {}),
        "complaints_out": state.get("complaints_out", {}),
        "refunds_out": state.get("refunds_out", {}),
        "weather_out": state.get("weather_out", {}),
    }
    col_details = _dedup_column_extract(collected)
    resp = chain_filter_extractor.invoke({"columns": str(col_details), "query": q})
    resp = resp.replace("```", "").replace("\n", "").strip()
    try:
        filt = ast.literal_eval(resp)
    except Exception:
        filt = ["no"]
    return {"filter_extractor": filt, "filtered_col": str(col_details)}


def filter_condition(state: FinalState):
    return "no" if _is_no_filter(state.get("filter_extractor")) else "yes"


def fuzz_filter(state: FinalState):
    val = state.get("filter_extractor", ["no"])
    triples = call_match(val)
    norm = normalize_filters(val)
    return {"fuzz_match": triples, "fuzz_norm": norm}


def query_generation(state: FinalState):
    q = state["user_query"]
    col_str = state.get("filtered_col", "[]")

    cat_filters = state.get("fuzz_norm", ["no"])
    if _is_no_filter(cat_filters):
        cat_filters = state.get("filter_extractor", ["no"])

    rng_raw = chain_range_date_extractor.invoke({"columns": col_str, "query": q})
    rng_raw = rng_raw.replace("```", "").replace("\n", "").strip()
    try:
        range_filters = ast.literal_eval(rng_raw)
    except Exception:
        range_filters = ["none"]

    sql_raw = chain_query_extractor.invoke({
        "columns": col_str,
        "query": q,
        "filters": cat_filters,
        "range_filters": range_filters
    })
    sql_only = extract_sql_only(sql_raw)
    return {"sql_query": sql_only, "range_filters": range_filters}


def query_validation(state: FinalState):
    q = state["user_query"]
    col_str = state.get("filtered_col", "[]")
    sql_in = state.get("sql_query", "")

    cat_filters = state.get("fuzz_norm", ["no"])
    if _is_no_filter(cat_filters):
        cat_filters = state.get("filter_extractor", ["no"])

    rng = state.get("range_filters", ["none"])

    sql_final_raw = chain_query_validator.invoke({
        "columns": col_str,
        "query": q,
        "filters": cat_filters,
        "range_filters": rng,
        "sql_query": sql_in
    })
    return {"final_query": extract_sql_only(sql_final_raw) or sql_in}


def execute_sql(state: FinalState):
    sql = state.get("final_query") or state.get("sql_query") or ""
    pr = int(state.get("preview_rows", 100) or 0)

    rows: List[Dict[str, Any]] = []
    cols: List[str] = []

    if not sql or pr <= 0:
        return {"rows": rows, "columns": cols, "sql_done": True}

    try:
        with engine.connect() as conn:
            res = conn.execute(text(sql))
            rows = [dict(r._mapping) for r in res.fetchmany(pr)]
            cols = list(rows[0].keys()) if rows else []
            if not rows:
                cnt = conn.execute(text(f"SELECT COUNT(*) AS c FROM ({sql}) t")).scalar() or 0
            else:
                cnt = len(rows)
            
    except SQLAlchemyError:
        rows, cols,cnt = [], [],0

    return {"rows": rows, "columns": cols,"rowcount": cnt, "sql_done": True,"executed_sql": sql}


def policy_safety(state: FinalState):
    """
    Eğer hibrit/policy sinyali kaçmışsa ama soru policy anahtarları içeriyorsa,
    tek seferlik policy_agent çalıştır.
    """
    if state.get("web_answer") or state.get("policy_done"):
        return {}
    q = state.get("user_query", "") or ""
    POLICY_KWS = [r"\bpolitika\w*\b", r"\biade\w*\b", r"\bücret\w*\b", r"\brefund\b", r"\bpolicy\b", r"\bfazla\s*bagaj\b", r"\bbagaj\b"]
    if any(re.search(p, q, flags=re.IGNORECASE) for p in POLICY_KWS):
        out = answer_policy(q)
        return {
            "web_answer": out.get("answer", ""),
            "policy_citations": out.get("citations", []),
            "policy_done": True,
            "use_web": True,
        }
    return {}


# --------- JOIN / BARRIER (tek synthesizer tetiklemesi) ----------
def join_node(state: FinalState):
    arrived = set(state.get("arrived", []))
    if state.get("policy_done"):
        arrived.add("policy")
    if state.get("sql_done"):
        arrived.add("sql")
    return {"arrived": list(arrived)}


def join_condition(state: FinalState):
    hybrid = bool(state.get("use_web") and state.get("want_sql"))
    if not hybrid:
        return "go"
    arrived = set(state.get("arrived", []))
    return "go" if arrived.issuperset({"sql", "policy"}) else "wait"

def synthesize_node(state: FinalState):
    q = state["user_query"]
    sql_for_payload = (
        state.get("executed_sql")
        or state.get("final_query")
        or state.get("sql_query")
        or ""
    )
    rows = state.get("rows", []) or []
    web  = state.get("web_answer", "") or ""
    citations = state.get("policy_citations", []) or []
    want_chart = bool(state.get("want_chart", False))

    syn = synthesize_unified(
        question=q,
        sql=sql_for_payload,
        rows=rows,
        web_answer=web,
        web_citations=citations,
        want_chart=want_chart,
        llm_merge=True,
        return_sql=True,
    )

    #  Güvenli fallback zinciri 
    final_answer = (syn.get("final_answer") or "").strip()
    if not final_answer:
        final_answer = (syn.get("analysis_text") or "").strip()
    if not final_answer:
        final_answer = web.strip()
    if not final_answer:
        if rows:
            final_answer = "Sorgu çalıştı ve satırlar döndü; kısa özet üretilemedi."
        elif sql_for_payload:
            final_answer = "SQL hazır; önizleme satırı bulunamadı."
        else:
            final_answer = "Politika / SQL cevabı üretilemedi."

    final_sql    = syn.get("final_sql") or sql_for_payload or ""
    rows_preview = syn.get("rows_preview") or (rows[:10] if isinstance(rows, list) else [])

    return {
        "final_answer": final_answer,
        "analysis_text": syn.get("analysis_text") or final_answer,
        "headline_metrics": syn.get("headline_metrics") or [],
        "vega_lite_spec": syn.get("vega_lite_spec"),
        "citations": syn.get("citations", []) or citations,
        "final_sql": final_sql,
        "rows_preview": rows_preview,
    }




# ---------------- graph build ----------------
builder = StateGraph(FinalState)

# nodes
builder.add_node("normalize", normalize_node)
builder.add_node("policy_gate", policy_gate)
builder.add_node("policy_rewrite", policy_rewrite)
builder.add_node("web_policy", web_policy_node)
builder.add_node("hybrid_fork", hybrid_fork)

builder.add_node("router", router)
builder.add_node("flights", flights_agent)
builder.add_node("complaints", complaints_agent)
builder.add_node("refunds", refunds_agent)
builder.add_node("weather", weather_agent)

builder.add_node("filter_check", filter_check)
builder.add_node("fuzz_filter", fuzz_filter)
builder.add_node("query_generator", query_generation)
builder.add_node("query_validation", query_validation)
builder.add_node("execute_sql", execute_sql)
builder.add_node("policy_safety", policy_safety)

# join & synth
builder.add_node("join", join_node)
builder.add_node("synthesize", synthesize_node)

# edges
builder.add_edge(START, "normalize")
builder.add_edge("normalize", "policy_gate")



builder.add_conditional_edges(
    "policy_gate",
    policy_condition,
    {
        "web": "policy_rewrite",
        "sql": "router",
        "hybrid": "hybrid_fork",
    }
)

# HYBRID: fork düğümünden iki kola çık
builder.add_edge("hybrid_fork", "policy_rewrite")
builder.add_edge("policy_rewrite", "web_policy")
builder.add_edge("web_policy", "join")
builder.add_edge("hybrid_fork", "router")


# SQL path
builder.add_conditional_edges(
    "router",
    route_request,
    ["flights", "complaints", "refunds", "weather"]
)
builder.add_edge("flights", "filter_check")
builder.add_edge("complaints", "filter_check")
builder.add_edge("refunds", "filter_check")
builder.add_edge("weather", "filter_check")

builder.add_conditional_edges(
    "filter_check",
    filter_condition,
    {"no": "query_generator", "yes": "fuzz_filter"}
)
builder.add_edge("fuzz_filter", "query_generator")
builder.add_edge("query_generator", "query_validation")
builder.add_edge("query_validation", "execute_sql")


builder.add_edge("execute_sql", "policy_safety")
builder.add_edge("policy_safety", "join")

# Join kararı
builder.add_conditional_edges(
    "join",
    join_condition,
    {"go": "synthesize", "wait": "join"}
)

builder.add_edge("synthesize", END)

graph_main = builder.compile(checkpointer=checkpointer)
