# app/tools/retriever/ingest.py
import os, time, json, hashlib
from typing import Dict, Any
import numpy as np
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from app.tools.retriever.schema import get_redis, PREFIX
from app.tools.retriever.fingerprint import dataset_fingerprint
from app.i18n.locale import map_terms_to_schema, normalize_text

load_dotenv()

EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
EMBED_DIM   = int(os.getenv("EMBED_DIM", "3072"))
emb = OpenAIEmbeddings(model=EMBED_MODEL)

def _float32_bytes(vec):
    arr = np.array(vec, dtype=np.float32)
    return arr.tobytes(order="C")


def make_signature(intent_signature:str, prompt_norm:str)->str:
    return hashlib.sha1((intent_signature + "||" + prompt_norm).encode("utf-8")).hexdigest()

def upsert_cache(doc: Dict[str, Any], ttl_sec:int=30*24*3600) -> str:
    r = get_redis()

    prompt_norm = (doc.get("prompt_norm") or doc["prompt"]).strip().lower()
    intent_sig  = doc.get("intent_signature", "")
    if not intent_sig:
        m = hashlib.sha1()
        m.update(prompt_norm.encode("utf-8"))
        m.update(b"||")
        m.update((doc.get("sql_text") or "").encode("utf-8"))
        intent_sig = m.hexdigest()

    sig = make_signature(intent_sig, prompt_norm)
    key = f"{PREFIX}{sig}"

    vec = emb.embed_query(prompt_norm)
    vec_bytes = _float32_bytes(vec)

    fp  = dataset_fingerprint()
    now = int(time.time())

    tags_field = ""

    tags_in = doc.get("tags", [])
    if tags_in:
        if isinstance(tags_in, list):
            tags_field = "|".join(tags_in)
        elif isinstance(tags_in, str):
            tags_field = tags_in

    payload = {
        "prompt": (doc.get("prompt","") or "").encode("utf-8"),
        "prompt_norm": prompt_norm.encode("utf-8"),
        "intent_signature": intent_sig.encode("utf-8"),
        "slots": json.dumps(doc.get("slots", {}), ensure_ascii=False).encode("utf-8"),
        "sql_text": (doc.get("sql_text","") or "").encode("utf-8"),
        "sql_params_schema": json.dumps(doc.get("sql_params_schema", {}), ensure_ascii=False).encode("utf-8"),
        "dialect": (doc.get("dialect","sqlite") or "sqlite").encode("utf-8"),
        "rowcount": str(int(doc.get("rowcount", 0))).encode("utf-8"),
        "sample_preview": json.dumps(doc.get("sample_preview", []), ensure_ascii=False).encode("utf-8"),
        "dataset_fingerprint": fp.encode("utf-8"),
        "created_at": str(now).encode("utf-8"),
        "ttl_sec": str(int(ttl_sec)).encode("utf-8"),
        "tags": tags_field.encode("utf-8"),                  # ← düz string
        "embedding": vec_bytes,
    }

    r.hset(key, mapping=payload)
    if ttl_sec > 0:
        r.expire(key, ttl_sec)
    return key
