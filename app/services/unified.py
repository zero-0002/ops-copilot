# app/services/unified.py
from __future__ import annotations
from typing import Any, Dict, Optional
from app.services.cache_service import try_cache, write_through_if_needed
from app.services.orchestrator_service import try_graph

async def answer_unified(
    question: str,
    preview_rows: int,
    return_rows: bool,
    return_chart: bool,
    top_k: int,
    threshold: float = 0.85,
    session_id: str | None = None,

) -> Dict[str, Any]:

    # 1) FAST PATH (10–50ms)
    cache_out = await try_cache(
        question, top_k, threshold, preview_rows, return_rows, return_chart, session_id=session_id,
    )
    if cache_out:
        cache_out["source"] = "cache"
        cache_out.setdefault("used_cache", True)
        return cache_out
    



    # 2) GRAPH PATH
    graph_out = await try_graph(question, preview_rows, return_chart, return_rows, session_id=session_id)
    graph_out["source"] = "graph"
    graph_out.setdefault("used_cache", False)

    # 3) write-through (yalnız graph kazandığında)
    try:
        await write_through_if_needed(question, graph_out, session_id=session_id)
    except Exception:
        pass

    return graph_out
