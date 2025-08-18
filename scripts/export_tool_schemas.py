# scripts/export_tool_schemas.py
# -*- coding: utf-8 -*-
import json, os, sys
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field

# -------------------------
# Pydantic param modelleri
# -------------------------
class CacheLookupParams(BaseModel):
    question: str = Field(..., description="Kullanıcının doğal dilde sorusu.")
    top_k: int = Field(8, ge=1, le=50, description="Cache içerisinde bakılacak maksimum aday sayısı.")
    threshold: float = Field(0.85, ge=0, le=1, description="Cache eşik skoru.")
    preview_rows: int = Field(100, ge=1, le=2000, description="Önizleme olarak döndürülecek satır adedi.")
    return_rows: bool = Field(True, description="Satır önizlemesini döndür.")
    return_chart: bool = Field(False, description="Grafik spec (varsa) döndür.")
    prefer_tag: str = Field("approved|seed", description="Tag filtre (örn. approved|seed).")
    cache_only: bool = Field(True, description="True ise cache miss olursa graph'a düşme.")
    session_id: Optional[str] = Field(None, description="İstemci oturum ID (izleme için).")

class GraphQueryParams(BaseModel):
    question: str = Field(..., description="Kullanıcının doğal dilde sorusu.")
    preview_rows: int = Field(100, ge=1, le=2000, description="Önizleme olarak döndürülecek satır adedi.")
    return_rows: bool = Field(True, description="Satır önizlemesini döndür.")
    return_chart: bool = Field(False, description="Grafik spec (varsa) döndür.")
    session_id: Optional[str] = Field(None, description="İstemci oturum ID (izleme için).")

class PolicyRagParams(BaseModel):
    question: str = Field(..., description="Politika/knowledge kaynaklarına yönelik soru.")
    top_k: int = Field(6, ge=1, le=50, description="En iyi kaç kaynağı değerlendirileceği.")
    return_citations: bool = Field(True, description="Kaynak/citation listesi döndür.")
    session_id: Optional[str] = Field(None, description="İstemci oturum ID (izleme için).")

# -------------------------
# OpenAI function-calling sarmalayıcı
# -------------------------
def as_openai_tool(name: str, description: str, model: type[BaseModel]) -> dict:
    schema = model.model_json_schema()
    # OpenAI tools: {"type":"function","function":{...}}
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": schema,
        },
    }

TOOL_DEFS = [
    (
        "cache_lookup",
        "Önbellekten (onaylı/seed etiketli) mevcut SQL planını ve örnek satırları döndürür; hızlı cevap için.",
        CacheLookupParams,
        "tools/cache_lookup.json",
    ),
    (
        "graph_query",
        "NL→SQL graph orkestratörüyle soruyu çözer; uygun tabloyu, SQL'i ve satır önizlemesini üretir.",
        GraphQueryParams,
        "tools/graph_query.json",
    ),
    (
        "policy_rag",
        "Politika/knowledge kaynakları (PDF/CSV/Index) üzerinde RAG ile kısa metin yanıtı ve gerekirse kaynaklar döndürür.",
        PolicyRagParams,
        "tools/policy_rag.json",
    ),
]

def main():
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = repo_root / "tools"
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, desc, mdl, relpath in TOOL_DEFS:
        tool = as_openai_tool(name, desc, mdl)
        out_path = repo_root / relpath
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(tool, f, ensure_ascii=False, indent=2)
        print(f"✓ wrote {out_path.relative_to(repo_root)}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERR: {e}", file=sys.stderr)
        sys.exit(1)
