# app/api/main.py
import os, time, uuid
from typing import Optional, List, Dict
from fastapi import FastAPI, Body, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager

from app.services.unified import answer_unified
from app.tools.retriever.schema import ensure_index, IDX_NAME
from app.tools.retriever.seed_examples import seed as seed_jsonl

APP_TITLE = os.getenv("APP_TITLE", "THY Ops Copilot API")
APP_VER   = os.getenv("APP_VER", "0.5.0")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # RediSearch index hazırla
    ensure_index()
    # İsteğe bağlı: seed JSONL yükle
    seed_path = os.getenv("RETRIEVER_SEED_JSONL")
    if seed_path:
        try:
            seed_jsonl(seed_path)
        except Exception as e:
            print(f"[seed] skipped: {e}")
    yield

app = FastAPI(title=APP_TITLE, version=APP_VER, lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AskUnifiedRequest(BaseModel):
    question: str = Field(..., example="Son 7 günde ortalama gecikme ve fazla bagaj politikası nedir?")
    preview_rows: int = 100
    return_rows: bool = True
    return_chart: bool = True
    top_k: int = 8
    threshold: float = float(os.getenv("CACHE_SCORE_THRESHOLD", "0.85"))
           

class AskUnifiedResponse(BaseModel):
    used_cache: bool
    use_web: bool
    want_sql: bool
    final_answer: Optional[str] = ""   # kritik
    analysis_text: Optional[str] = None
    headline_metrics: Optional[List[Dict]] = None
    vega_lite_spec: Optional[Dict] = None
    sql: Optional[str] = None
    rows: Optional[List[Dict]] = None
    columns: Optional[List[str]] = None
    citations: Optional[List[Dict]] = None
    execution_ms: int
    source: str  # "cache" | "graph"


@app.get("/healthz")
def healthz():
    return {"ok": True, "version": APP_VER, "index": IDX_NAME}

@app.post("/ask_unified", response_model=AskUnifiedResponse, response_model_exclude_none=True)
async def ask_unified(req: AskUnifiedRequest = Body(...),
                      x_session_id: Optional[str] = Header(default=None)):  # None olsun

    t0 = time.time()
    session_id = (x_session_id or "").strip() or f"sess-{uuid.uuid4()}"  # benzersiz

    out = await answer_unified(
        question=req.question,
        preview_rows=req.preview_rows,
        return_rows=req.return_rows,
        return_chart=req.return_chart,
        top_k=req.top_k,
        threshold=req.threshold,
        session_id=session_id, 
    )
    out["execution_ms"] = int((time.time() - t0) * 1000)
    return out
