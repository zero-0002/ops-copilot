# app/orchestrators/arbiter.py
import asyncio
from app.tools.retriever.search import hybrid_search, is_fresh
from app.agents.sql.sql_langraph import graph_main
from app.tools.retriever.ingest import upsert_cache
from app.orchestrators.arbiter_utils import (
    slot_coverage,
    derive_intent_signature,
    canonicalize,
    detect_dialect,
)

RETRIEVER_SCORE_MIN = 0.85
SLOT_COVERAGE_MIN   = 0.90

async def cache_lookup(q, slots):
    # 1) BM25 + KNN hibrit arama
    hits = hybrid_search(q, k=5)  # [(hybrid_score, doc), ...]
    if not hits:
        return None
    score, doc = hits[0]

    # 2) slot uyumu + tazelik
    if (
        score >= RETRIEVER_SCORE_MIN
        and is_fresh(doc)
        and slot_coverage(doc.get("slots", {}), slots) >= SLOT_COVERAGE_MIN
    ):
        return {
            "source": "cache",
            "payload": {
                "text": doc["sql_text"],
                "dialect": doc.get("dialect", "sqlite"),
                "slots": doc.get("slots", {}),
                "intent_signature": doc.get("intent_signature"),
            },
        }
    return None

async def run_sql_pipeline(q, slots):
    # LangGraph: router → agents → filter → sql gen → validate
    res = graph_main.invoke({"user_query": q})
    final_sql = (res.get("final_query") or res.get("sql_query") or "").strip()

    out = {
        "source": "fresh",
        "payload": {
            "text": final_sql,
            "slots": slots,
            "intent_signature": derive_intent_signature(q, slots),
        },
    }

    # write-through cache (fire & forget)
    try:
        upsert_cache({
            "prompt": q,
            "prompt_norm": canonicalize(q),
            "intent_signature": out["payload"]["intent_signature"],
            "slots": slots,
            "sql_text": final_sql,
            "dialect": detect_dialect(),  # sqlite/mysql
            "rowcount": 0,
        })
    except Exception:
        pass

    return out

async def answer_sql(q: str, slots: dict):
    # Paralel yarış: cache vs sql
    t_cache = asyncio.create_task(cache_lookup(q, slots))
    t_sql   = asyncio.create_task(run_sql_pipeline(q, slots))

    done, pending = await asyncio.wait({t_cache, t_sql}, return_when=asyncio.FIRST_COMPLETED)
    first = list(done)[0].result()

    # Cache boşa dönerse SQL'i bekle
    if not first:
        first = await t_sql

    # Diğer task’ı iptal et
    for p in pending:
        p.cancel()

    return first
