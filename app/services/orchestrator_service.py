# app/services/orchestrator_service.py
from typing import Any, Dict, List
import time, uuid

from app.agents.sql.sql_langraph import graph_main

def _columns_from_rows(rows: List[dict] | None) -> list[str]:
    if not rows: return []
    return list(rows[0].keys())

async def try_graph(q, preview_rows, return_chart, return_rows, session_id: str | None = None):
    thread_id = (session_id or "").strip() or f"sess-{uuid.uuid4()}"
    cfg = {"configurable": {"thread_id": thread_id}}   # ✅ sade

    st = graph_main.invoke(
        {"raw_query": q, "preview_rows": preview_rows, "want_chart": bool(return_chart)},
        config=cfg,
    )

    final_sql = (
        st.get("final_sql") or st.get("executed_sql") or st.get("final_query") or st.get("sql_query") or ""
    )
    rows = st.get("rows_preview") or st.get("rows") or []

    return {
        "use_web": bool(st.get("use_web")),
        "want_sql": bool(st.get("want_sql")),
        "final_answer": (st.get("final_answer") or st.get("analysis_text") or "").strip(),
        "analysis_text": st.get("analysis_text"),
        "headline_metrics": st.get("headline_metrics") or [],
        "vega_lite_spec": st.get("vega_lite_spec") if return_chart else None,
        "sql": final_sql or None,
        "rows": (rows if return_rows else []),
        "columns": list(rows[0].keys()) if (return_rows and rows) else [],
        "citations": st.get("citations") or st.get("policy_citations") or [],
    }