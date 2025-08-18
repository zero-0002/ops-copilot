# app/tools/retriever/search.py
import os, json
from typing import Tuple, Optional, Dict, Any, List
import numpy as np
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
import app.tools.retriever.schema as schema
get_redis = schema.get_redis
IDX_NAME  = schema.IDX_NAME

from app.tools.retriever.fingerprint import dataset_fingerprint
from app.i18n.locale import map_terms_to_schema, normalize_text

load_dotenv()

EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
emb = OpenAIEmbeddings(model=EMBED_MODEL)

W_VEC  = float(os.getenv("HYBRID_W_VEC", "0.7"))
W_BM25 = float(os.getenv("HYBRID_W_BM25", "0.3"))
TOP_K  = int(os.getenv("HYBRID_TOP_K", "8"))

TAG_FILTER = os.getenv("RETRIEVER_TAG_FILTER", "").strip()  # e.g. "approved|seed"

def _to_bytes_float32(vec: List[float]) -> bytes:
    return np.array(vec, dtype=np.float32).tobytes(order="C")

def canonicalize_prompt(q: str) -> str:
    q_norm, _ = map_terms_to_schema(q)
    return " ".join(normalize_text(q_norm).split())

def _tokenize_bm25(q: str) -> str:
    q = canonicalize_prompt(q)
    toks = [t for t in q.replace("'", " ").replace('"', " ").replace(":", " ").split() if t]
    return "(" + "|".join(toks) + ")" if toks else "*"

def _parse_hash(r, key: bytes) -> Dict[str, Any]:
    h = r.hgetall(key)
    out: Dict[str, Any] = {}
    for k, v in h.items():
        kd = k.decode() if isinstance(k, (bytes, bytearray)) else k
        if kd in ("embedding",):
            out[kd] = v
        elif kd in ("slots","sql_params_schema","sample_preview"):
            out[kd] = json.loads(v.decode() or ("{}" if kd != "sample_preview" else "[]"))
        elif kd == "tags":
            out[kd] = v.decode()
        elif kd in ("rowcount","created_at","ttl_sec"):
            out[kd] = int(v.decode() or "0")
        else:
            out[kd] = v.decode()
    return out

def _tag_clause() -> str:
    if not TAG_FILTER:
        return ""
    return f"@tags:{{{TAG_FILTER}}}"

# 🔹 EK: ultra hızlı EXACT MATCH (embedding çağırmadan)
def search_exact(q: str) -> Optional[Dict[str, Any]]:
    r = get_redis()
    qn = canonicalize_prompt(q)
    tagc = _tag_clause()
    exact_q = f'@prompt_norm:"{qn}"'
    if tagc:
        exact_q = f"({exact_q} {tagc})"
    res = r.execute_command("FT.SEARCH", IDX_NAME, exact_q, "NOCONTENT", "LIMIT", "0", "1")
    if res and res[0] >= 1:
        key = res[1]
        return _parse_hash(r, key)
    return None

def hybrid_search(q: str, k:int=TOP_K) -> List[Tuple[float, Dict[str,Any]]]:
    r = get_redis()
    q_norm = canonicalize_prompt(q)

    # 0) EXACT MATCH kısayolu
    tagc = _tag_clause()
    exact_q = f'@prompt_norm:"{q_norm}"'
    exact_q = f"({exact_q} {tagc})" if tagc else exact_q
    exact_res = r.execute_command("FT.SEARCH", IDX_NAME, exact_q, "NOCONTENT", "LIMIT", "0", "1")
    if exact_res and exact_res[0] >= 1:
        key = exact_res[1]
        doc = _parse_hash(r, key)
        return [(1.10, doc)]

    # 1) KNN
    vec = emb.embed_query(q_norm)
    vec_bytes = _to_bytes_float32(vec)
    base_filter = tagc if tagc else "*"
    knn_query = f"{base_filter}=>[KNN {k} @embedding $vec AS __vec_score]"
    knn_cmd = [
        "FT.SEARCH", IDX_NAME, knn_query,
        "PARAMS", "2", "vec", vec_bytes,
        "SORTBY", "__vec_score",
        "RETURN", "1", "__vec_score",
        "DIALECT", "2",
        "LIMIT", "0", str(k)
    ]
    knn_res = r.execute_command(*knn_cmd)

    vec_dist: Dict[bytes, float] = {}
    if knn_res and knn_res[0] > 0:
        for i in range(1, len(knn_res), 2):
            key = knn_res[i]
            fields = knn_res[i+1]
            d = None
            for j in range(0, len(fields), 2):
                if fields[j].decode() == "__vec_score":
                    fval = fields[j+1]
                    d = float(fval.decode() if isinstance(fval, (bytes, bytearray)) else fval)
                    break
            if d is not None:
                vec_dist[key] = d

    vec_sim: Dict[bytes, float] = {}
    if vec_dist:
        dists = list(vec_dist.values())
        vmin, vmax = min(dists), max(dists)
        rng = (vmax - vmin) if (vmax > vmin) else 1e-12
        for kkey, d in vec_dist.items():
            vec_sim[kkey] = 1.0 - (d - vmin)/rng

    # 2) BM25
    bm25_text = _tokenize_bm25(q_norm)
    bm25_q = f"(@prompt_norm:{bm25_text})"
    if tagc:
        bm25_q = f"({bm25_q} {tagc})"
    bm25_cmd = ["FT.SEARCH", IDX_NAME, bm25_q, "WITHSCORES", "NOCONTENT", "LIMIT", "0", str(k)]
    bm25_res = r.execute_command(*bm25_cmd)

    bm25_raw: Dict[bytes, float] = {}
    if bm25_res and bm25_res[0] > 0:
        for i in range(1, len(bm25_res), 2):
            key = bm25_res[i]
            score = bm25_res[i+1]
            bm25_raw[key] = float(score.decode() if isinstance(score, (bytes, bytearray)) else score)

    bm25_norm: Dict[bytes, float] = {}
    if bm25_raw:
        vals = list(bm25_raw.values())
        smin, smax = min(vals), max(vals)
        srng = (smax - smin) if (smax > smin) else 1e-12
        for kkey, s in bm25_raw.items():
            bm25_norm[kkey] = (s - smin)/srng

    # 3) Combine
    keys = set(vec_sim.keys()) | set(bm25_norm.keys())
    scored: List[Tuple[float, Dict[str,Any]]] = []
    for key in keys:
        v = vec_sim.get(key, 0.0)
        b = bm25_norm.get(key, 0.0)
        hybrid = W_VEC * v + W_BM25 * b
        doc = _parse_hash(r, key)
        doc["__vec_sim"] = v
        doc["__bm25_norm"] = b
        doc["__hybrid"] = hybrid
        scored.append((hybrid, doc))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored

def is_fresh(doc: Dict[str, Any]) -> bool:
    return doc.get("dataset_fingerprint") == dataset_fingerprint()
