# app/tools/retriever/seeds_examples.py
import os, sys, json, time, hashlib
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from dotenv import load_dotenv
import numpy as np

from app.tools.retriever.schema import get_redis, PREFIX
from app.tools.retriever.fingerprint import dataset_fingerprint
from app.i18n.locale import map_terms_to_schema, normalize_text
from langchain_openai import OpenAIEmbeddings
from app.tools.retriever.search import canonicalize_prompt
load_dotenv()

EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
emb = OpenAIEmbeddings(model=EMBED_MODEL)

TTL_SEC_DEFAULT = int(os.getenv("CACHE_TTL_SEC", str(30*24*3600)))  # 30 gün
DIALECT_DEFAULT = os.getenv("CACHE_SQL_DIALECT", "sqlite")

def _to_bytes_float32(vec: List[float]) -> bytes:
    return np.array(vec, dtype=np.float32).tobytes(order="C")

def _canonicalize_prompt(q: str) -> str:
    # Seed ve arama tarafında aynı normalize hattı:
    # 1) TR terimlerini şema terimlerine map et
    # 2) unidecode + lower + noktalama temizliği
    q_norm, _applied = map_terms_to_schema(q or "")
    return " ".join(normalize_text(q_norm).split())

def seed_jsonl(path: str):
    return seed(path)

def _signature(prompt_norm: str, sql_text: str) -> str:
    m = hashlib.sha1()
    m.update(prompt_norm.encode("utf-8"))
    m.update(b"||")
    m.update(normalize_text(sql_text).encode("utf-8"))
    return m.hexdigest()

@dataclass
class QAExample:
    prompt: str
    sql_text: str
    sql_params_schema: Dict[str, Any]
    intent_signature: Optional[str] = None
    slots: Optional[Dict[str, Any]] = None
    dialect: str = DIALECT_DEFAULT
    tags: Optional[List[str]] = None
    ttl_sec: int = TTL_SEC_DEFAULT
    prompt_norm: Optional[str] = None  # JSONL'den gelebilir ama yine de normalize edeceğiz

def _upsert_example(r, ex: QAExample):
    now = int(time.time())

    # JSONL'de 'prompt_norm' olsa bile yeniden kanonikleştir:
    prompt_norm = _canonicalize_prompt(ex.prompt)

    sig = ex.intent_signature or _signature(prompt_norm, ex.sql_text)
    key = f"{PREFIX}{sig}"

    vec = emb.embed_query(prompt_norm)
    vec_bytes = _to_bytes_float32(vec)

    fp = dataset_fingerprint()

    # TagField(sep='|') ile uyumlu biçim: "approved|seed"
    tags_str = ""
    if ex.tags:
        if isinstance(ex.tags, list):
            tags_str = "|".join(ex.tags)
        elif isinstance(ex.tags, str):
            tags_str = ex.tags

    payload = {
        "prompt": ex.prompt,
        "prompt_norm": prompt_norm,  # <-- normalize edilmiş ASCII/token sürümü
        "intent_signature": ex.intent_signature or "",
        "slots": json.dumps(ex.slots or {}, ensure_ascii=False),
        "sql_text": ex.sql_text,
        "sql_params_schema": json.dumps(ex.sql_params_schema or {}, ensure_ascii=False),
        "dialect": ex.dialect,
        "rowcount": "0",
        "sample_preview": "[]",
        "dataset_fingerprint": fp,
        "created_at": str(now),
        "ttl_sec": str(ex.ttl_sec),
        "tags": tags_str,            # <-- düz string (JSON değil)
        "embedding": vec_bytes,
    }

    r.hset(key, mapping=payload)
    if ex.ttl_sec > 0:
        r.expire(key, ex.ttl_sec)

    print(f"[OK] upsert {key} | dialect={ex.dialect} | prompt_norm='{prompt_norm[:80]}…' | tags='{tags_str}'")

def _load_jsonl(path: str) -> List[QAExample]:
    out: List[QAExample] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            out.append(QAExample(
                prompt=obj["prompt"],
                sql_text=obj["sql_text"],
                sql_params_schema=obj.get("sql_params_schema", {}),
                intent_signature=obj.get("intent_signature"),
                slots=obj.get("slots"),
                dialect=obj.get("dialect", DIALECT_DEFAULT),
                tags=obj.get("tags"),
                ttl_sec=int(obj.get("ttl_sec", TTL_SEC_DEFAULT)),
                prompt_norm=obj.get("prompt_norm"),  # gelse de yeniden normalize edeceğiz
            ))
    return out

def seed(from_jsonl: Optional[str] = None):
    r = get_redis()
    items = _load_jsonl(from_jsonl) if from_jsonl else []
    for ex in items:
        _upsert_example(r, ex)

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else None
    seed(path)
# app/tools/retriever/seed_examples.py
import os, sys, json, time, hashlib
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from dotenv import load_dotenv
import numpy as np

from app.tools.retriever.schema import get_redis, PREFIX
from app.tools.retriever.fingerprint import dataset_fingerprint
from app.tools.retriever.search import canonicalize_prompt   # ← TEK KAYNAK
from langchain_openai import OpenAIEmbeddings

load_dotenv()

EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
emb = OpenAIEmbeddings(model=EMBED_MODEL)

TTL_SEC_DEFAULT = int(os.getenv("CACHE_TTL_SEC", str(30*24*3600)))
DIALECT_DEFAULT = os.getenv("CACHE_SQL_DIALECT", "sqlite")

def _to_bytes_float32(vec: List[float]) -> bytes:
    return np.array(vec, dtype=np.float32).tobytes(order="C")

def _signature(prompt_norm: str, sql_text: str) -> str:
    m = hashlib.sha1()
    m.update(prompt_norm.encode("utf-8"))
    m.update(b"||")
    m.update(sql_text.strip().lower().encode("utf-8"))
    return m.hexdigest()

@dataclass
class QAExample:
    prompt: str
    sql_text: str
    sql_params_schema: Dict[str, Any]
    intent_signature: Optional[str] = None
    slots: Optional[Dict[str, Any]] = None
    dialect: str = DIALECT_DEFAULT
    tags: Optional[List[str]] = None
    ttl_sec: int = TTL_SEC_DEFAULT
    prompt_norm: Optional[str] = None  # GELSE BİLE YOK SAYACAĞIZ

def _upsert_example(r, ex: QAExample):
    now = int(time.time())

    # JSONL'deki 'prompt_norm' GELSE BİLE asla kullanma!
    prompt_norm = canonicalize_prompt(ex.prompt)

    sig = ex.intent_signature or _signature(prompt_norm, ex.sql_text)
    key = f"{PREFIX}{sig}"

    vec = emb.embed_query(prompt_norm)
    vec_bytes = _to_bytes_float32(vec)

    fp = dataset_fingerprint()

    tags_str = ""
    if ex.tags:
        tags_str = "|".join(ex.tags) if isinstance(ex.tags, list) else str(ex.tags)

    payload = {
        "prompt": ex.prompt,
        "prompt_norm": prompt_norm,
        "intent_signature": ex.intent_signature or "",
        "slots": json.dumps(ex.slots or {}, ensure_ascii=False),
        "sql_text": ex.sql_text,
        "sql_params_schema": json.dumps(ex.sql_params_schema or {}, ensure_ascii=False),
        "dialect": ex.dialect,
        "rowcount": "0",
        "sample_preview": "[]",
        "dataset_fingerprint": fp,
        "created_at": str(now),
        "ttl_sec": str(ex.ttl_sec),
        "tags": tags_str,                 # "approved|seed"
        "embedding": vec_bytes,           # FLOAT32 bytes
    }

    r.hset(key, mapping=payload)
    if ex.ttl_sec > 0:
        r.expire(key, ex.ttl_sec)
    print(f"[OK] upsert {key} | prompt_norm='{prompt_norm[:80]}…' | tags='{tags_str}'")

def _load_jsonl(path: str) -> List[QAExample]:
    out: List[QAExample] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                out.append(QAExample(
                    prompt=obj["prompt"],
                    sql_text=obj["sql_text"],
                    sql_params_schema=obj.get("sql_params_schema", {}),
                    intent_signature=obj.get("intent_signature"),
                    slots=obj.get("slots"),
                    dialect=obj.get("dialect", DIALECT_DEFAULT),
                    tags=obj.get("tags"),
                    ttl_sec=int(obj.get("ttl_sec", TTL_SEC_DEFAULT)),
                    prompt_norm=obj.get("prompt_norm"),  # GELSE BİLE KULLANMIYORUZ
                ))
    return out

def seed(from_jsonl: Optional[str] = None):
    r = get_redis()
    items = _load_jsonl(from_jsonl) if from_jsonl else []
    for ex in items:
        _upsert_example(r, ex)

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else None
    seed(path)
