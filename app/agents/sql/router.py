# app/agents/sql/router.py
"""
THY Text-to-SQL Router
- Input : Turkish/NL user question
- Output: subset of ['flights','complaints','refunds','weather']
"""

import os
import ast
import re
from typing import List

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableMap

from app.i18n.locale import map_terms_to_schema

# -------- Model (low temp for determinism) --------
_model = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    temperature=0.0
)

# -------- Prompt --------
_template = ChatPromptTemplate.from_messages([
    ("system", """
You are a precise router in a text-to-SQL system. Read a Turkish user question
about Turkish Airlines ops and decide which AGENTS are required.

OUTPUT MUST BE ONLY a valid Python list of strings (no text around it), e.g.:
['flights'] or ['flights','complaints'].

AGENT CATALOG:
- 'flights'    : flight_number, departure_time, delay_minutes, aircraft_type, crew_issues (0/1), weather_impact (0/1)
- 'complaints' : complaint_id, flight_number, customer_id, category, status, created_at
- 'refunds'    : ticket_id, flight_number, cancel_reason, refund_status, customer_id
- 'weather'    : flight_number, condition (storm/rain/fog/clear)

JOIN HINTS:
- flights <-> complaints <-> refunds via flight_number
- complaints <-> refunds via customer_id
- flights <-> weather via flight_number

ROUTING RULES (think internally; output only the list):
1) Split the question into minimal sub-questions.
2) Map sub-questions to agents by required columns:
   - delay/aircraft/crew/weather_impact → 'flights'
   - complaint category/status/count/time → 'complaints'
   - refund/cancel reason/cancellation/status → 'refunds'
   - weather condition/storm/rain/fog → 'weather'
3) If join is implied, include all relevant agents.
4) If single-table aggregate suffices, return only that agent.
"""),
    ("human", "User question:\n{question}\n\nReturn only a Python list of strings.")
])

_chain = (
    RunnableMap({"question": lambda x: x["question"]})
    | _template
    | _model
    | StrOutputParser()
)

# Robust list parser
_LIST_RE = re.compile(r"\[[^\]]*\]")

def _safe_parse_list(text: str) -> List[str]:
    if not text:
        return []
    m = _LIST_RE.search(text)
    candidate = m.group(0) if m else text.strip()
    for cand in (candidate, candidate.replace('"', "'")):
        try:
            out = ast.literal_eval(cand)
            if isinstance(out, list) and all(isinstance(x, str) for x in out):
                # only allow known agent names
                out = [s for s in out if s in {"flights", "complaints", "refunds", "weather"}]
                # dedup, keep order
                seen, cleaned = set(), []
                for s in out:
                    if s not in seen:
                        seen.add(s); cleaned.append(s)
                return cleaned
        except Exception:
            continue
    return []

# Keyword fallback (deterministic)
def _fallback_router(q: str) -> List[str]:
    ql = q.lower()
    agents = set()

    if any(w in ql for w in ["complaint", "şikayet", "sikayet", "category", "kategori", "trend", "günlük", "gunluk"]):
        agents.add("complaints")

    if any(w in ql for w in ["delay", "gecikme", "uçuş", "ucus", "flight", "aircraft", "crew", "weather_impact"]):
        agents.add("flights")

    if any(w in ql for w in ["refund", "iade", "iptal", "cancel","cancellation","ucus iptali"]):
        agents.add("refunds")

    if any(w in ql for w in ["weather", "hava", "storm", "rain", "fog", "clear"]):
        agents.add("weather")

    if not agents:
        # very safe default: complaints is common for business questions
        agents.add("complaints")

    return list(agents)

# -------- Public API --------
def route_agents(user_q: str) -> List[str]:
    # i18n normalize (TR→canonical tokens)
    q_norm, _applied = map_terms_to_schema(user_q or "")
    try:
        raw = _chain.invoke({"question": q_norm}) or ""
        routed = _safe_parse_list(raw)
    except Exception:
        routed = []
    if routed:
        return routed
    # fallback if LLM returns nothing / invalid
    return _fallback_router(q_norm)

# backward compat
def agent_2(q: str) -> List[str]:
    return route_agents(q)
