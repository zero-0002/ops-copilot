# app/agents/synthesis/synthesizer.py
from __future__ import annotations
import os, json, re
from typing import Any, Dict, List, Optional

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ---------------------------------------------------------
# Ortak LLM config
# ---------------------------------------------------------
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
LLM_MERGE_HYBRID = os.getenv("LLM_MERGE_HYBRID", "false").lower() == "true"

_llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0)

# ---------------------------------------------------------
# Policy zayıflık kontrolü
# ---------------------------------------------------------
def _policy_is_weak(text: str) -> bool:
    if not text or not text.strip():
        return True
    core = text.split("\nKaynaklar:")[0]
    t = " ".join(core.lower().split())
    if "güncel ve resmi bilgi için thy bilgi edin sayfasına bakınız" in t:
        return True
    if re.search(r"\byeterli\s+bilgi\s+(bulunmamaktad[ıi]r|yok)\b", t):
        return True
    if re.search(r"\bu(y|ğ)gun\s+bilgi\s+(bulunmamaktad[ıi]r|yok)\b", t):
        return True
    return False

# ---------------------------------------------------------
# SQL-only sentezleyici
# ---------------------------------------------------------
_sql_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a senior data analyst. Return ONLY one compact JSON with keys:"
     '\n- "analysis_text": concise Turkish natural-language insight (no hallucinations),'
     '\n- "headline_metrics": array of objects like {"name": str, "value": number|string},'
     '\n- "vega_lite_spec": a Vega-Lite spec object if a chart makes sense, otherwise null.'
     "\nRules:"
     "\n- Never invent columns/metrics that do not exist in the provided rows."
     "\n- If data is insufficient, say so in analysis_text and set vega_lite_spec = null."
     "\n- Output must be pure JSON (no markdown, no code fences)."),
    ("human",
     "User question:\n{{ question }}\n\n"
     "SQL:\n```sql\n{{ sql }}\n```\n\n"
     "Sample rows (JSON, first N):\n```json\n{{ rows_json }}\n```\n\n"
     "Notes:\n"
     "- If a categorical breakdown is implied, consider a simple bar/line chart (x = category, y = metric).\n"
     "- If percentages are present, keep values in the 0–100 range.\n"
     "- Chart desired flag: {{ want_chart }} (set vega_lite_spec to null if a chart is not appropriate).\n"
     'Return ONLY a JSON object like: {"analysis_text": "...", "headline_metrics": [], "vega_lite_spec": null}')
], template_format="jinja2")

_sql_chain = _sql_prompt | _llm | StrOutputParser()

def synthesize_sql(
    question: str,
    sql: str,
    rows: Any,
    want_chart: bool = False,
    max_rows: int = 150,
) -> Dict[str, Any]:
    if isinstance(rows, list):
        rows_list: List[Dict[str, Any]] = rows
    elif isinstance(rows, (str, bytes)):
        try:
            parsed = json.loads(rows)
            rows_list = parsed if isinstance(parsed, list) else []
        except Exception:
            rows_list = []
    else:
        rows_list = []
    rows_cut = rows_list[:max_rows]

    out_txt: str = _sql_chain.invoke({
        "question": question,
        "sql": sql or "",
        "rows_json": json.dumps(rows_cut, ensure_ascii=False),
        "want_chart": "true" if want_chart else "false",
    })

    try:
        data = json.loads(out_txt)
        return {
            "analysis_text": (data.get("analysis_text") or "").strip(),
            "headline_metrics": data.get("headline_metrics") or [],
            "vega_lite_spec": data.get("vega_lite_spec"),
        }
    except Exception:
        return {
            "analysis_text": (out_txt or "").strip(),
            "headline_metrics": [],
            "vega_lite_spec": None,
        }

# ---------------------------------------------------------
# Hybrid birleştirme (policy + sql)
# ---------------------------------------------------------
_merge_text_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Aşağıdaki iki kaynağı tek, tutarlı ve kısa bir Türkçe cevapta birleştir.\n"
     "- SQL Analiz JSON'undaki metrikleri koru; yeni rakam uydurma.\n"
     "- Politika metnini kısaca entegre et; tekrarı azalt.\n"
     "- Başlık yazma; düz metin döndür.\n"
     "- Politika metninin sonunda 'Kaynaklar:' bloğu varsa, en sonda aynen bırak."),
    ("human",
     "Kullanıcı sorusu:\n{{ question }}\n\n"
     "SQL Analiz JSON:\n```json\n{{ sql_json }}\n```\n\n"
     "Politika Metni:\n{{ policy_text }}\n")
], template_format="jinja2")

_merge_text_chain = _merge_text_prompt | ChatOpenAI(model=OPENAI_MODEL, temperature=0) | StrOutputParser()

def _merge_to_text(question: str, sql_part: Dict[str, Any], policy_text: str) -> str:
    txt = _merge_text_chain.invoke({
        "question": question,
        "sql_json": json.dumps(sql_part or {}, ensure_ascii=False),
        "policy_text": policy_text or "",
    }).strip()
    if not txt:
        a = (sql_part or {}).get("analysis_text", "") or ""
        b = policy_text or ""
        txt = (a + ("\n\n" if a and b else "") + b).strip()
    return txt


def synthesize_unified(
    question: str,
    sql: str,
    rows: Any,
    web_answer: str = "",
    web_citations: Optional[List[Dict[str, Any]]] = None,
    want_chart: bool = False,
    llm_merge: Optional[bool] = None,
    return_sql: bool = False,
) -> Dict[str, Any]:
    web_citations = web_citations or []
    if llm_merge is None:
        llm_merge = LLM_MERGE_HYBRID

    def _with_sql(payload: Dict[str, Any]) -> Dict[str, Any]:
        if return_sql:
            payload["final_sql"] = sql if isinstance(sql, str) else ""
            payload["rows_preview"] = rows[:10] if isinstance(rows, list) else []
        return payload  # <-- FIX

    # Sadece satır varsa SQL analizi üret (boş SQL policy'yi gölgelemesin)
    has_rows = isinstance(rows, list) and len(rows) > 0
    sql_part: Optional[Dict[str, Any]] = None
    if sql and has_rows:
        sql_part = synthesize_sql(question=question, sql=sql, rows=rows, want_chart=want_chart)

    # Policy zayıfsa ve SQL tarafı VARSA policy'yi at
    policy_text = (web_answer or "").strip()
    if sql_part and _policy_is_weak(policy_text):
        web_answer = ""
        web_citations = []

    # Policy-only
    if web_answer and not sql_part:
        out = {
            "analysis_text": web_answer.strip(),
            "headline_metrics": [],
            "vega_lite_spec": None,
            "citations": web_citations,
        }
        out["final_answer"] = out["analysis_text"]
        return _with_sql(out)

    # SQL-only
    if sql_part and not web_answer:
        out = {**sql_part, "citations": web_citations}
        out["final_answer"] = out["analysis_text"]
        return _with_sql(out)

    # Hiçbiri yoksa
    if not sql_part and not web_answer:
        out = {
            "analysis_text": "Sorgudan veri veya politika cevabı üretilemedi.",
            "headline_metrics": [],
            "vega_lite_spec": None,
            "citations": [],
        }
        out["final_answer"] = out["analysis_text"]
        return _with_sql(out)

    # HYBRID
    if llm_merge:
        sql_len = len((sql_part or {}).get("analysis_text", ""))
        pol_len = len(web_answer or "")
        if sql_len < 160 and pol_len < 320:
            llm_merge = False
        if llm_merge:
            merged = _merge_to_text(question, sql_part or {}, web_answer)
            out = {
                "analysis_text": merged,
                "headline_metrics": (sql_part or {}).get("headline_metrics", []),
                "vega_lite_spec": (sql_part or {}).get("vega_lite_spec"),
                "citations": web_citations,
            }
            out["final_answer"] = merged
            return _with_sql(out)
        else:
            merged = f"{(sql_part or {}).get('analysis_text','').strip()}\n\n{(web_answer or '').strip()}".strip()

    merged = f"{(sql_part or {}).get('analysis_text','').strip()}\n\n{web_answer.strip()}".strip()
    out = {
        "analysis_text": merged,
        "headline_metrics": (sql_part or {}).get("headline_metrics", []),
        "vega_lite_spec": (sql_part or {}).get("vega_lite_spec"),
        "citations": web_citations,
    }
    out["final_answer"] = merged
    return _with_sql(out)
