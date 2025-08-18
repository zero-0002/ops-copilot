# app/agents/dispatcher.py
from __future__ import annotations
import os, re, json, asyncio
from typing import Dict, Any

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# ------------ Config ------------
ROUTER_LLM_MODEL      = os.getenv("ROUTER_LLM_MODEL", "gpt-4o-mini")
ROUTER_LLM_TIMEOUT_S  = float(os.getenv("ROUTER_LLM_TIMEOUT_S", "0.6"))  # 400–600 ms ideal
ROUTER_THRESHOLD      = float(os.getenv("ROUTER_THRESHOLD", "0.6"))      # karar eşiği

# ------------ Hızlı Regex Rules (fallback) ------------
POLICY_PAT = [
    r"\b(ücret|ucret|tarife|fare rule|kural|bilet|iade|kosul|koşul|hak|bagaj|fazla bagaj|el bagaji|check[- ]?in|no[- ]show|iade|iptal|değişiklik|refund|cancellation|bilet sınıfı)\b",
    r"\b(politika|policy|şart|bilgi|sart|sikayet)\b",
    r"\b(hayvan|evcil|spor ekipmani|medikal|hamile|özel yardim)\b",
]
SQL_PAT = [
    r"\b(uçuş|flight|rota|sefer|pax|gun|hangi)\b",
    r"\b(gecikme|geciken|gecik\w+|delay|dakika|rötar|crew_issues|weather)\b",
    r"\b(şikayet|complaint|kategori|resolved|open|pending)\b",
    r"\b(ortalama|trend|en çok|top|sayısı|count|oran|max|min)\b",
]

HYBRID_HINTS = [
    r"\b(hem|ve)\b.*\b(politika|policy|[üu]cret|kural|ko[sş]ul|bagaj|iade|refund|iptal)\b.*\b("
    r"[şs]ikayet\s+oran[ıi]|gecikme|istatistik|trend|say[ıi]s[ıi]|oran|y[üu]zde|%)\b"
]

def _hit(pats, text): 
    t = text.lower()
    return any(re.search(p, t, re.I) for p in pats)

def _fallback_rules(q: str) -> Dict[str, Any]:
    pol = _hit(POLICY_PAT, q); sql = _hit(SQL_PAT, q); hyb = _hit(HYBRID_HINTS, q)
    p, s = (1.0 if pol else 0.0), (1.0 if sql else 0.0)
    if hyb: p += 0.5; s += 0.5
    th = ROUTER_THRESHOLD
    if p >= th and s >= th: mode, wp, ws = "hybrid", True, True
    elif p >= th:           mode, wp, ws = "policy",  True, False
    elif s >= th:           mode, wp, ws = "sql",     False, True
    else:                   mode, wp, ws = "hybrid",  True, True  # belirsizlikte kapsayıcı
    return {
        "mode": mode, "want_policy": wp, "want_sql": ws,
        "window_days": 60,  # basit default
        "reason": f"fallback_rules: policy={pol}, sql={sql}, hybrid_hint={hyb}",
        "policy_score": p, "sql_score": s, "llm_used": False,
    }

# ------------ LLM Classifier (JSON) ------------
ROUTER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a routing classifier for an airline ops assistant. "
     "Respond ONLY as strict JSON with keys: "
     "{\"want_policy\":bool, \"want_sql\":bool, \"confidence\":0-1, "
     "\"window_days\":int|null, \"reason\":str}. "
     "Policy = baggage/fees/refund rules; SQL = flights/delays/complaints/refunds analytics. "
     "If the question mixes both, set both booleans true (hybrid)."),
    ("human", "Question: {q}")
])

async def _llm_route_async(q: str) -> Dict[str, Any]:
    llm = ChatOpenAI(temperature=0, model=ROUTER_LLM_MODEL)
    msg = ROUTER_PROMPT.invoke({"q": q})
    out = await llm.ainvoke(msg.to_messages())
    data = json.loads(out.content if hasattr(out, "content") else str(out))

    want_policy = bool(data.get("want_policy"))
    want_sql    = bool(data.get("want_sql"))
    conf        = float(data.get("confidence", 0.5))
    window      = data.get("window_days") or 60
    reason      = f"llm: {data.get('reason','')} (conf={conf:.2f})"

    # eşik altına düşerse belirsizlik: hybrid
    if conf < ROUTER_THRESHOLD:
        want_policy, want_sql = True, True

    mode = "hybrid" if (want_policy and want_sql) else ("policy" if want_policy else "sql")
    return {
        "mode": mode, "want_policy": want_policy, "want_sql": want_sql,
        "window_days": window, "reason": reason,
        "policy_score": 1.0 if want_policy else 0.0,
        "sql_score": 1.0 if want_sql else 0.0,
        "llm_used": True,
    }

# ------------ Public API ------------
def classify_intent(q: str) -> Dict[str, Any]:
    """
    LLM-first: kısa timeout ile LLM; timeout/hata → regex fallback.
    """
    try:
        res = asyncio.run(asyncio.wait_for(_llm_route_async(q), timeout=ROUTER_LLM_TIMEOUT_S))
        return res
    except Exception as e:
        fb = _fallback_rules(q)
        fb["reason"] += f" | llm_timeout_or_error: {e}"
        return fb


def is_policy_query(q: str) -> bool:
    return classify_intent(q)["want_policy"]
