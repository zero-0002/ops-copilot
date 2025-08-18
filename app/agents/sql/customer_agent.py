# app/agents/sql/customer_agent.py
import os, re, ast
from typing import TypedDict, Annotated, List, Dict, Any
from operator import add
from app.agents.sql.sql_agent_thy import (
    chain_subquestion,
    chain_column_extractor,
    chain_filter_extractor,
    chain_range_date_extractor,
    chain_query_extractor,
    chain_query_validator,
)

from dotenv import load_dotenv
load_dotenv()


from langgraph.graph import StateGraph, START, END

# Bu iki zincir: senin THY-özel prompt zincirlerin (OpenAI sürümü)
from app.agents.sql.sql_agent_thy import (
    chain_subquestion,
    chain_column_extractor,
)

# ---------- Basit KB (kb.pkl yoksa) ----------
loaded_dict_thy: Dict[str, List[str]] = {
    "flights": [
        "Uçuş performans metrikleri (gecikme, uçak tipi, ekip/hava etkisi)",
        """flights.flight_number (STRING)...
flights.departure_time (DATETIME/TEXT ISO)...
flights.delay_minutes (INT)...
flights.aircraft_type (STRING)...
flights.crew_issues (INT:0/1)...
flights.weather_impact (INT:0/1)..."""
    ],
    "complaints": [
        "Müşterilerin uçuşa bağlı şikayet kayıtları",
        """complaints.complaint_id (STRING)...
complaints.flight_number (STRING)...
complaints.customer_id (STRING)...
complaints.category (STRING)...
complaints.status (STRING)...
complaints.created_at (DATETIME/TEXT ISO)..."""
    ],
    "refunds": [
        "Bilet iptal/iade süreç kayıtları",
        """refunds.ticket_id (STRING)...
refunds.flight_number (STRING)...
refunds.cancel_reason (STRING)...
refunds.refund_status (STRING)...
refunds.customer_id (STRING)..."""
    ],
    "weather": [
        "Uçuş numarasına göre hava durumu özeti (mock)",
        """weather.flight_number (STRING)...
weather.condition (STRING: storm, rain, fog, clear)..."""
    ],
}

def try_load_kb() -> Dict[str, Any]:
    path = os.path.join(os.getcwd(), "kb.pkl")
    if os.path.exists(path):
        try:
            import pickle
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception:
            pass
    return loaded_dict_thy

loaded_dict: Dict[str, Any] = try_load_kb()

# ---------- State ----------
class OverallState(TypedDict):
    user_query: str
    table_lst: List[str]
    table_extract: Annotated[list, add]
    column_extract: Annotated[list, add]

LIST_PATTERN = re.compile(r"\[\s*\[.*?\]\s*(?:,\s*\[.*?\]\s*)*\]", re.DOTALL)

def _extract_list_literal(text: str) -> list:
    text = (text or "").replace("```", "").strip()
    m = LIST_PATTERN.search(text)
    if not m:
        return []
    try:
        return ast.literal_eval(m.group(0))
    except Exception:
        return []

def agent_subquestion(user_q: str, tables_kv_str: str) -> list:
    resp = chain_subquestion.invoke({"tables": tables_kv_str, "user_query": user_q})
    return _extract_list_literal(resp)

def solve_subquestion(user_q: str, table_names: List[str]) -> list:
    kv = {t: loaded_dict.get(t, ["", ""])[0] for t in table_names}
    return agent_subquestion(user_q, str(kv))

def agent_column_selection(main_q: str, sub_q: str, columns_text: str) -> list:
    resp = chain_column_extractor.invoke({
        "columns": columns_text,
        "query": sub_q,
        "main_question": main_q
    })
    out = _extract_list_literal(resp)
    return out if out else [[]]

def solve_column_selection(main_q: str, subq_list: list) -> list:
    final_cols: List[List[str]] = []
    for item in subq_list:
        if not item: continue
        table_name = item[-1]
        subq_text = "; ".join(item[:-1]) if len(item) > 2 else item[0]
        if table_name not in loaded_dict: continue
        columns_text = loaded_dict[table_name][1]
        selected_cols = agent_column_selection(main_q, subq_text, str(columns_text))
        for col in selected_cols:
            if not col: continue
            final_cols.append(["name of table:" + table_name] + col)
    return final_cols

def sq_node(state: OverallState):
    user_q = state["user_query"]
    tables = state["table_lst"] or ["flights","complaints","refunds","weather"]
    return {"table_extract": solve_subquestion(user_q, tables)}

def column_node(state: OverallState):
    return {"column_extract": solve_column_selection(state["user_query"], state.get("table_extract", []))}

# ---------- Graph ----------
builder = StateGraph(OverallState)
builder.add_node("subquestion", sq_node)
builder.add_node("column_e", column_node)
builder.add_edge(START, "subquestion")
builder.add_edge("subquestion", "column_e")
builder.add_edge("column_e", END)
graph_final = builder.compile()
