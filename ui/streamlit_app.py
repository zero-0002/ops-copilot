# ui/streamlit_app.py
# -*- coding: utf-8 -*-
import os, uuid, re
from pathlib import Path
from typing import List, Dict, Any, Tuple, cast

import requests
import pandas as pd
import streamlit as st

# Kaç tur geçmiş tutulsun?
MAX_HISTORY = 3

# =========================================================
# Page config
# =========================================================
st.set_page_config(
    page_title="THY Ops Copilot",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# CSS — STICKY TOPBAR (başlık + QA)
# =========================================================
st.markdown(
    """
<style>
/* Ana container boşluk: üstte minimal, altta rahat */
.block-container { padding-top: 0.6rem; padding-bottom: 1.0rem; }

/* STICKY TOPBAR: her zaman görünür */
#topbar {
  position: sticky;
  top: 0;
  z-index: 9999;
  background: rgba(255,255,255,0.96);
  backdrop-filter: saturate(180%) blur(6px);
  border-bottom: 1px solid #eef2f7;
  padding: 10px 6px 12px 6px;
  margin: -6px 0 10px 0;
  box-shadow: 0 1px 0 rgba(0,0,0,.03);
}

/* İç genişlik — main içeriğe hizala */
#topbar .inner { max-width: 1200px; margin: 0 auto; padding: 0 12px; }

/* Başlık ve alt başlık */
.top-title { font-size: 40px; font-weight: 800; line-height: 1.05; margin: 0; }
.top-sub   { font-size: 14px; color: #6b7280; margin: 4px 0 10px 0; }

/* Quick Analysis etiket */
.qa-badge {
  display:inline-flex; align-items:center; gap:6px;
  padding:2px 8px; border-radius:999px; background:#fff; border:1px solid #e5e7eb;
  margin-bottom:8px;
}
.qa-emoji { font-size:12px; }

/* Butonları “chip” gibi */
.stButton > button {
  width: 100%; padding: 8px 10px;
  border:1px solid #e5e7eb; background:#fff; border-radius:999px; cursor:pointer;
}
.stButton > button:hover { background:#f8fafc; }

/* Dark mode */
@media (prefers-color-scheme: dark) {
  #topbar { background: rgba(17,17,17,0.92); border-bottom-color:#1f2937; }
  .top-sub { color:#9ca3af; }
}
</style>
""",
    unsafe_allow_html=True,
)

# =========================================================
# Helpers
# =========================================================
_SRC_BLOCK_RE = re.compile(r"(?:\n|^)\s*Kaynaklar\s*:\s*(?:\n|$).*", re.IGNORECASE | re.DOTALL)

def _strip_sources_block(txt: str) -> str:
    """Cevap içindeki 'Kaynaklar:' bloğunu sohbetten gizle (sekmede göstereceğiz)."""
    if not isinstance(txt, str):
        return ""
    return _SRC_BLOCK_RE.sub("", txt).rstrip()

def _get_api_base_default() -> str:
    try:
        return st.secrets["api_base_url"]
    except Exception:
        return os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

def ping_api(base: str) -> Tuple[bool, str]:
    try:
        r = requests.get(f"{base.rstrip('/')}/healthz", timeout=5)
        return (r.status_code == 200, "OK" if r.status_code == 200 else f"HTTP {r.status_code}")
    except Exception as e:
        return False, str(e)

def call_backend(question: str, base_url: str, session_id: str) -> Dict[str, Any]:
    payload = {
        "question": question, "preview_rows": 100,
        "return_rows": True, "return_chart": False, "top_k": 8,
    }
    r = requests.post(
        f"{base_url.rstrip('/')}/ask_unified",
        json=payload, headers={"X-Session-Id": session_id}, timeout=60
    )
    r.raise_for_status()
    return r.json()

def call_backend_quick(question: str, base_url: str, session_id: str, prefer_tag: str) -> Dict[str, Any]:
    """Quick Analysis tıklarında isteği cache-only gibi gönderiyoruz (ek alanlar API tarafında görmezden gelinir)."""
    payload = {
        "question": question, "preview_rows": 100,
        "return_rows": True, "return_chart": False, "top_k": 8,
        "prefer_tag": prefer_tag, "cache_only": True,
    }
    r = requests.post(
        f"{base_url.rstrip('/')}/ask_unified",
        json=payload, headers={"X-Session-Id": session_id}, timeout=60
    )
    r.raise_for_status()
    return r.json()

# Geçmiş yardımcıları
ExList = List[Dict[str, Any]]
def get_exchanges() -> ExList:
    return cast(ExList, st.session_state.setdefault("exchanges", []))

def push_exchange(item: Dict[str, Any]) -> None:
    lst = get_exchanges()
    lst.append(item)
    st.session_state["exchanges"] = lst[-MAX_HISTORY:]

def list_last_questions(n: int = 3) -> List[str]:
    """Son n soruyu (en yeni en üstte) ver."""
    items = get_exchanges()
    return [ex.get("q", "") for ex in reversed(items)][:n]

def handle_local_commands(prompt: str) -> bool:
    """'bir önceki soru', 'son 3 soru', 'geçmişi sil' gibi komutları backend'e gitmeden ele al."""
    t = (prompt or "").strip().lower()

    # Son 3 soru
    if t in ("son 3 soru", "son üç soru", "last 3 questions", "history"):
        qs = list_last_questions(3)
        with st.chat_message("assistant"):
            if not qs:
                st.info("Henüz geçmiş yok.")
            else:
                st.markdown("**Son 3 soru:**")
                for i, q in enumerate(qs, start=1):
                    st.markdown(f"{i}. {q}")
        return True

    # Bir önceki soru
    if t in ("bir önceki soru neydi", "bir önceki soru", "önceki soru", "last question", "previous question"):
        items = get_exchanges()
        with st.chat_message("assistant"):
            if not items:
                st.info("Henüz geçmiş yok.")
            else:
                st.markdown(f"**Bir önceki soru:** {items[-1].get('q','')}")
        return True

    # Geçmişi temizle
    if t in ("geçmişi sil", "geçmişi temizle", "clear history", "reset"):
        st.session_state["exchanges"] = []
        with st.chat_message("assistant"):
            st.success("Geçmiş temizlendi.")
        return True

    return False

# =========================================================
# Session state
# =========================================================
if "sid" not in st.session_state:
    st.session_state.sid = f"sess-{uuid.uuid4()}"
if "api_base" not in st.session_state:
    st.session_state.api_base = _get_api_base_default()
_ = get_exchanges()

# =========================================================
# Sidebar
# =========================================================
BASE_DIR = Path(__file__).parent
LOGO = BASE_DIR / "assets" / "thy_logo_2.png"

with st.sidebar:
    if LOGO.exists():
        st.image(str(LOGO), use_container_width=True)
    st.subheader("Veri Kaynakları")
    st.markdown(
        "- **Uçuş Durumu Raporu** · `flights`\n"
        "- **Yolcu Şikayetleri** · `complaints`\n"
        "- **İptal/İade** · `refunds`\n"
        "- **Policy PDF/CSV** · `company policy`\n"
    )
    st.divider()
    st.subheader("API")
    st.session_state.api_base = st.text_input(
        "Base URL", value=st.session_state.api_base, key="api_base_input",
    )
    ok, info = ping_api(st.session_state.api_base)
    st.caption(f"Health: {'✅' if ok else '❌'} {info}")
    if st.button("Geçmişi temizle", key="clear_history_btn"):
        st.session_state["exchanges"] = []
        st.rerun()

# =========================================================
# STICKY TOPBAR (Başlık + QA)
# =========================================================
st.markdown('<div id="topbar"><div class="inner">', unsafe_allow_html=True)

st.markdown('<div class="top-title">THY Ops Copilot ✈️</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="top-sub">Agentic Orchestrator (LangGraph) • SQL + Policy RAG • Unified Answer</div>',
    unsafe_allow_html=True,
)
st.markdown('<div class="qa-badge"><span class="qa-emoji">⚡</span> Quick Analysis</div>', unsafe_allow_html=True)

qa_items = [
    "Son 7 günde ortalama gecikme nedir?",
    "Son 14 günde hava kaynaklı gecikmesi olan uçuşlar: en çok etkilenen ilk 10 uçuş (toplam ve ortalama gecikme).",
    "Gecikme trendi (haftalık) nedir?",
    "Fazla bagaj politikası nedir?",
    "Aynı müşterinin birden fazla şikayeti var mı? (top tekrar eden müşteriler)",
    "Hangi günlerde şikayetler yoğunlaşıyor? (haftanın günü)",
    "Son 30 günde weather_impact=1 uçuşlarda; kategoriye göre şikayet sayısı, müşteri sayısı ve refund oranı nedir?",
]
qa_cols = st.columns(len(qa_items))
_chosen = None
for i, (c, q) in enumerate(zip(qa_cols, qa_items), start=1):
    with c:
        if st.button(q, key=f"qa_{i}", use_container_width=True):
            _chosen = q

st.markdown('</div></div>', unsafe_allow_html=True)  # /.inner, /#topbar

# =========================================================
# Chat history (yalnızca son MAX_HISTORY)
# =========================================================
for ex in get_exchanges():
    with st.chat_message("user"):
        st.write(ex["q"])
    with st.chat_message("assistant"):
        used_cache = ex.get("used_cache"); use_web = ex.get("use_web")
        want_sql = ex.get("want_sql");     source = ex.get("source") or ""
        st.caption(
            f"used_cache: {'✅' if used_cache else '❌'} · "
            f"use_web: {'✅' if use_web else '❌'} · "
            f"want_sql: {'✅' if want_sql else '❌'} · "
            f"source: {source or '-'}"
        )
        st.write(_strip_sources_block(ex.get("answer") or ""))
        tabs = st.tabs(["Tablo", "SQL", "Kaynaklar"])
        with tabs[0]:
            rows = ex.get("rows") or []
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True)
            else:
                st.caption("Tablo verisi yok.")
        with tabs[1]:
            sql_text = ex.get("sql") or ""
            if sql_text:
                st.code(sql_text, language="sql")
            else:
                st.caption("SQL yok.")
        with tabs[2]:
            cits = ex.get("citations") or []
            if cits:
                for i, c in enumerate(cits, 1):
                    title = (c.get("title") or "").strip()
                    url = (c.get("url") or "").strip()
                    st.markdown(f"{i}. [{title}]({url})" if url else f"{i}. {title}")
            else:
                st.caption("Kaynak yok.")

# =========================================================
# Input + run
# =========================================================
prompt = st.chat_input("Sorunu yaz (örn: 'Son 7 günde ortalama gecikme ve fazla bagaj politikası?')")
if _chosen and not prompt:
    prompt = _chosen

if prompt:
    # Önce yerel komutları ele al (backend'e gitmeden)
    if handle_local_commands(prompt):
        st.stop()

    with st.chat_message("user"):
        st.write(("⚡ " if prompt in qa_items else "") + prompt)

    try:
        if prompt in qa_items:
            prefer_tag = os.getenv("RETRIEVER_TAG_FILTER", "").strip() or "approved|seed"
            data = call_backend_quick(prompt, st.session_state.api_base, st.session_state.sid, prefer_tag)
        else:
            data = call_backend(prompt, st.session_state.api_base, st.session_state.sid)
    except Exception as e:
        with st.chat_message("assistant"):
            st.error(f"Sunucuya erişilemedi: {e}")
    else:
        answer_raw = (data.get("final_answer") or data.get("analysis_text") or "").strip()
        answer = _strip_sources_block(answer_raw)
        sql = data.get("sql") or ""
        rows = data.get("rows") or []
        citations = data.get("citations") or []
        used_cache = bool(data.get("used_cache"))
        use_web = bool(data.get("use_web"))
        want_sql = bool(data.get("want_sql"))
        source = (data.get("source") or "").strip()

        with st.chat_message("assistant"):
            st.caption(
                f"used_cache: {'✅' if used_cache else '❌'} · "
                f"use_web: {'✅' if use_web else '❌'} · "
                f"want_sql: {'✅' if want_sql else '❌'} · "
                f"source: {source or '-'}"
            )
            st.write(answer or "_(boş yanıt)_")
            tabs = st.tabs(["Tablo", "SQL", "Kaynaklar"])
            with tabs[0]:
                if rows:
                    st.dataframe(pd.DataFrame(rows), use_container_width=True)
                else:
                    st.caption("Tablo verisi yok.")
            with tabs[1]:
                if sql:
                    st.code(sql, language="sql")
                else:
                    st.caption("SQL yok.")
            with tabs[2]:
                if citations:
                    for i, c in enumerate(citations, 1):
                        title = (c.get("title") or "").strip()
                        url = (c.get("url") or "").strip()
                        st.markdown(f"{i}. [{title}]({url})" if url else f"{i}. {title}")
                else:
                    st.caption("Kaynak yok.")

        # Geçmişe ekle (sadece son 3 tur kalsın)
        push_exchange({
            "q": prompt, "answer": answer, "sql": sql,
            "rows": rows, "citations": citations,
            "used_cache": used_cache, "use_web": use_web, "want_sql": want_sql, "source": source,
        })
