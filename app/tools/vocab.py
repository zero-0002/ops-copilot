import os, json

VOCAB_JSON_PATH = os.getenv("VOCAB_JSON_PATH", "app/data/vocab.json")

def load_vocab():
    try:
        with open(VOCAB_JSON_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"tables": {}}

def render_vocab_text(vocab: dict, max_vals_per_col=12, max_lines=160) -> str:
    """
    CATEGORICALS: gerçek unique değerlerden kısa liste
    ID_PREVIEW:  uniq sayısı + örnek ilk K
    """
    lines = []
    tables = (vocab or {}).get("tables", {})
    for t, info in tables.items():
        lines.append(f"- {t}:")
        cats = (info or {}).get("categoricals", {})
        for cname, meta in cats.items():
            vals = (meta or {}).get("values", [])[:max_vals_per_col]
            suffix = "…" if len((meta or {}).get("values", [])) > max_vals_per_col else ""
            joined = ", ".join([repr(x) for x in vals])
            lines.append(f"  {cname} (uniq={meta.get('uniq')}): {joined}{suffix}")

        ids = (info or {}).get("id_preview", {})
        for cname, meta in ids.items():
            ex = (meta or {}).get("examples", [])[:max_vals_per_col]
            suffix = "…" if len((meta or {}).get("examples", [])) > max_vals_per_col else ""
            joined = ", ".join([repr(x) for x in ex])
            lines.append(f"  {cname} [ID] (uniq~{meta.get('uniq')}): examples: {joined}{suffix}")

        for cname, r in (info or {}).get("date_cols", {}).items():
            if r and (r.get("min") or r.get("max")):
                lines.append(f"  {cname}_range: {r.get('min')} → {r.get('max')}")

        if len(lines) >= max_lines:
            break
    return "\n".join(lines[:max_lines])

_V = load_vocab()
VOCAB_TEXT = render_vocab_text(_V)
