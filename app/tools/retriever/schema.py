# app/tools/retriever/schema.py
import os
from typing import Optional
from redis import Redis
from redis.commands.search.field import TextField, NumericField, VectorField, TagField
from redis.commands.search.index_definition import IndexDefinition, IndexType


# ---- Constants ----
IDX_NAME   = os.getenv("REDIS_INDEX", "sql_cache")
PREFIX     = os.getenv("REDIS_PREFIX", "doc:")
REDIS_URL  = os.getenv("REDIS_URL", "redis://localhost:6379")
EMBED_DIM  = int(os.getenv("EMBED_DIM", "3072"))  # text-embedding-3-large için 3072

# ---- Connection ----
_redis: Optional[Redis] = None
SEARCH_AVAILABLE = False  

def get_redis() -> Redis:
    """
    Singleton pattern ile Redis bağlantısını döner.
    decode_responses=False -> embedding vektörlerini bytes olarak saklamak için uygun.
    """
    global _redis
    if _redis is None:
        _redis = Redis.from_url(REDIS_URL, decode_responses=False)
    return _redis

def _has_search(r: Redis) -> bool:
    try:
        r.execute_command("FT._LIST")
        return True
    except Exception:
        return False

def ensure_index():
    """
    Redis Stack RediSearch index oluşturur.
    VectorField desteklenmezse yalnızca text + numeric alanları ile devam eder.
    """
    r = get_redis()
    global SEARCH_AVAILABLE

    if not _has_search(r):
        SEARCH_AVAILABLE = False
        print("[ensure_index] RediSearch not available — cache disabled (graph path still works).")
        return

    SEARCH_AVAILABLE = True
    try:
        r.ft(IDX_NAME).info()
        print(f"[ensure_index] Index already exists: {IDX_NAME}")
        return
    except Exception:
        pass

    # ---- Alanlar ----
    fields = [
        TextField("prompt"),
        TextField("prompt_norm"),
        TextField("intent_signature"),
        TextField("slots"),
        TextField("sql_text"),
        TextField("sql_params_schema"),
        TextField("dialect"),
        TextField("dataset_fingerprint"),
        NumericField("rowcount"),
        NumericField("created_at"),
        NumericField("ttl_sec"),
        TagField("tags", separator="|")
    ]

    # ---- Vector alanı (varsa) ----
    try:
        fields.append(
            VectorField(
                "embedding",
                "HNSW", {
                    "TYPE": "FLOAT32",
                    "DIM": EMBED_DIM,
                    "DISTANCE_METRIC": "COSINE",
                    "M": 16,
                    "EF_CONSTRUCTION": 200,
                }
            )
        )
        print(f"[ensure_index] Added VectorField (DIM={EMBED_DIM})")
    except Exception as e:
        print(f"[ensure_index] Warning: VectorField eklenemedi -> {e}")

    # ---- Index tanımı ----
    definition = IndexDefinition(prefix=[PREFIX], index_type=IndexType.HASH)

    try:
        definition = IndexDefinition(prefix=[PREFIX], index_type=IndexType.HASH)
        r.ft(IDX_NAME).create_index(fields=fields, definition=definition)
        print(f"[ensure_index] Created index: {IDX_NAME}")
    except Exception as e:
        if "Index already exists" in str(e):
            print(f"[ensure_index] Index already exists: {IDX_NAME}")
        else:
            raise e
