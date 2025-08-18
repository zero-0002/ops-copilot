# app/agents/policy_web/policy_agent.py
# -----------------------------------------------------------------------------
# THY Policy QA (final, full-featured)
#   • Offline-first RAG: CSV + PDF → FAISS (MMR) → kısa TR cevap + [1],[2]...
#   • Wikipedia sadece FALLBACK (Tavily Extract, tek URL whitelist)
#   • Gelişmiş chunking (tiktoken + heading-aware + overlap + tiny-merge)
#   • Kaynaklar k=3 ile sınırlandı, CSV başlığı zenginleştirildi
#   • "Kaynaklar" metne gömülü döner (tek print ile görünür)
# -----------------------------------------------------------------------------

from __future__ import annotations
import os
import asyncio
import datetime
from typing import List, Dict, Any, Tuple
from langchain_tavily import TavilyExtract
import pandas as pd
from dotenv import load_dotenv, find_dotenv
import re  
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader

# --- .env yükle ---
env_file = find_dotenv(usecwd=True)
load_dotenv(dotenv_path=env_file, override=True)




OPENAI_MODEL        = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL     = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")

POLICY_CSV          = os.getenv("POLICY_CSV", "app/data/thy_qa_policy.csv")
POLICY_PDFS         = [p.strip() for p in os.getenv("POLICY_PDFS", "").split(",") if p.strip()]
POLICY_INDEX_DIR    = os.getenv("POLICY_INDEX_DIR", "app/data/thy_policy_index")

# Kaynak sayısını kısıtla
RETRIEVE_K          = int(os.getenv("POLICY_RETRIEVE_K", "3"))

# Chunking (token-aware)
CHUNK_TOKENS        = int(os.getenv("POLICY_CHUNK_TOKENS", "900"))
CHUNK_OVERLAP       = int(os.getenv("POLICY_CHUNK_OVERLAP", "150"))

# Wikipedia (fallback)
WIKI_URL            = os.getenv("WIKI_URL", "https://w.wiki/F3kw")
try:
    from langchain_tavily import TavilyExtract
    _HAS_TAVILY = True
except Exception:
    _HAS_TAVILY = False

# --- tiktoken chunk splitter'lar ---
try:
    from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
except Exception:
    from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter  # type: ignore



THY_INFO_URL = "https://www.turkishairlines.com/tr-int/bilgi-edin"

def _is_redirect_only(text: str) -> bool:
    """YANIT: ... THY Bilgi Edin ... minimal yönlendirme mesajı mı? (case/whitespace tolerant)"""
    if not text:
        return False
    t = " ".join(text.strip().split()).lower()
    if t.startswith("yanit:"):
        t = t[len("yanit:"):].strip()
    # hedef cümle
    return t == "güncel ve resmi bilgi için thy bilgi edin sayfasına bakınız."



WIKI_FIRST_PATTERNS = [
   r"\bkurul\w*\b",                     
   r"\btarihçe\b|\bhistory\b",
   r"\bgenel\s*merkez\b|\bmerkez[iı]\b",
   r"\bceo\b|\bgenel\s*müdür\b|\byönetim\b",
   r"\bfilo\b|\buçak\s*sayısı\b|\bhub\b|\bmerkez\s*havaalan[ıi]\b",
   r"\bstar\s*alliance\b|\bittifak\b|\büyeli[kğ]\b",
   r"\bsahip\b|\bortaklık\b|\bhisse\b",
  r"\bslogan\b|\blogo\b"]

_WIKI_FIRST_COMPILED = [re.compile(p, re.IGNORECASE) for p in WIKI_FIRST_PATTERNS]
def _should_use_wiki_first(q: str) -> bool:
   t = (q or "")
   return any(p.search(t) for p in _WIKI_FIRST_COMPILED)



# -------------------------- yardımcılar --------------------------
def _today_str() -> str:
    return datetime.datetime.utcnow().strftime("%Y-%m-%d")

def _normalize_source_url(src: str | None) -> str:
    """http/file/csv şemalarını koru; path ise file:// yap."""
    if not src:
        return ""
    s = str(src).strip()
    if s.startswith(("http", "csv:", "file://")):
        return s
    if s.endswith(".pdf") or s.startswith(("/", "./")) or os.path.exists(s):
        return f"file://{os.path.abspath(s)}"
    return s

def _render_with_citations(text: str, citations: List[Dict[str, str]]) -> str:
    """Cevabın sonuna 'Kaynaklar:' bloğunu ekler."""
    if not citations:
        return text
    lines = ["\nKaynaklar:"]
    for i, c in enumerate(citations, 1):
        title = (c.get("title") or "").strip()
        url   = _normalize_source_url(c.get("url") or "").strip()
        lines.append(f"[{i}] {title} {url}")
    return text + "\n" + "\n".join(lines)


# -------------------------- CSV / PDF yükleyiciler --------------------------
def _docs_from_csv(csv_path: str) -> List[Document]:
    """
    Beklenen CSV: ; ile ayrılmış kolonlar
      id;question;answer;source_url;last_updated (id/source/last_updated opsiyonel)
    """
    if not os.path.exists(csv_path):
        return []

    df = pd.read_csv(csv_path, sep=";", encoding="utf-8-sig")
    norm = {c.strip().lower(): c for c in df.columns}
    if "question" not in norm or "answer" not in norm:
        raise ValueError("CSV must contain 'question' and 'answer' columns")

    docs: List[Document] = []
    for i, row in df.iterrows():
        q = str(row[norm["question"]]).strip()
        a = str(row[norm["answer"]]).strip()
        # Başlığı sorudan kısaltarak üret
        title = f"THY Politika (CSV): {q[:40]}..."

        src = ""
        if "source_url" in norm:
            try:
                src = str(row[norm["source_url"]]).strip()
            except Exception:
                src = ""
        src = _normalize_source_url(src) or f"csv:row:{i}"

        last_upd = ""
        if "last_updated" in norm:
            try:
                last_upd = str(row[norm["last_updated"]]).strip()
            except Exception:
                last_upd = ""

        meta = {
            "kind": "csv",
            "title": title,
            "source": src,
            "last_updated": last_upd,
            "row": int(i),
        }
        docs.append(Document(page_content=f"Q: {q}\nA: {a}", metadata=meta))
    return docs


def _docs_from_pdfs(pdf_paths: List[str]) -> List[Document]:
    """Her PDF sayfasını ayrı Document olarak döndür; kaynak URL: file://...#page=N"""
    out: List[Document] = []
    for path in pdf_paths:
        if not path or not os.path.exists(path):
            continue
        loader = PyPDFLoader(path)
        pages = loader.load()
        title = os.path.basename(path)
        for d in pages:
            src = d.metadata.get("source", path)
            page_no = d.metadata.get("page", 1)
            d.metadata.update({
                "kind": "pdf",
                "title": title,
                "source": f"file://{os.path.abspath(src)}#page={page_no}",
            })
        out.extend(pages)
    return out


# -------------------------- gelişmiş chunking --------------------------
def _tiny_merge(parts: List[str], min_chars: int = 300) -> List[str]:
    """Çok kısa parçaları komşularla birleştir (FAQ kırpıntılarını azaltır)."""
    if not parts:
        return parts
    buf = ""
    out: List[str] = []
    for p in parts:
        if not p.strip():
            continue
        if len(buf) < min_chars:
            buf = (buf + "\n\n" + p).strip() if buf else p
        else:
            out.append(buf)
            buf = p
    if buf:
        out.append(buf)
    return out

def _split_documents_advanced(docs: List[Document]) -> List[Document]:
    """
    CSV’ler doğrudan; PDF/uzun metinler: heading-aware → token-aware (tiktoken)
    """
    out: List[Document] = []

    # CSV kısa içerik: split etme
    for d in docs:
        if d.metadata.get("kind") == "csv":
            out.append(d)

    long_docs = [d for d in docs if d.metadata.get("kind") != "csv"]
    headers = [("#", "h1"), ("##", "h2"), ("###", "h3")]
    rc = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=CHUNK_TOKENS,
        chunk_overlap=CHUNK_OVERLAP,
        disallowed_special=(),
    )

    for d in long_docs:
        raw = d.page_content or ""
        try:
            splitter = MarkdownHeaderTextSplitter(headers=headers, strip_headers=False)
            sections = splitter.split_text(raw)
        except Exception:
            sections = [d]

        for sec in sections:
            sec_text = (sec.page_content or "").strip()
            if not sec_text:
                continue
            parts = rc.split_text(sec_text)
            parts = _tiny_merge(parts, min_chars=300)

            base = dict(d.metadata)
            meta_sec = getattr(sec, "metadata", {}) or {}
            path = " > ".join([meta_sec.get(k, "") for _, k in headers if meta_sec.get(k)])
            if path:
                base["section_path"] = path

            for idx, p in enumerate(parts):
                out.append(Document(page_content=p, metadata={**base, "chunk_id": idx}))
    return out


# -------------------------- index inşa / yükle --------------------------
def _build_index_csv_pdf(csv_path: str, pdf_paths: List[str], index_dir: str) -> FAISS:
    docs = _docs_from_csv(csv_path) + _docs_from_pdfs(pdf_paths)
    if not docs:
        raise FileNotFoundError("Policy index için CSV/PDF bulunamadı.")
    chunks = _split_documents_advanced(docs)
    vs = FAISS.from_documents(chunks, OpenAIEmbeddings(model=EMBEDDING_MODEL))
    os.makedirs(index_dir, exist_ok=True)
    vs.save_local(index_dir)
    return vs

def _load_or_build_index() -> FAISS:
    emb = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    try:
        return FAISS.load_local(POLICY_INDEX_DIR, emb, allow_dangerous_deserialization=True)
    except Exception:
        return _build_index_csv_pdf(POLICY_CSV, POLICY_PDFS, POLICY_INDEX_DIR)

_VS: FAISS | None = None
def _get_vs() -> FAISS:
    global _VS
    if _VS is None:
        _VS = _load_or_build_index()
    return _VS

def rebuild_policy_index() -> None:
    """İndeksi sil-yeniden kur (CSV/PDF güncellendiğinde)."""
    import shutil
    if os.path.isdir(POLICY_INDEX_DIR):
        shutil.rmtree(POLICY_INDEX_DIR, ignore_errors=True)
    _build_index_csv_pdf(POLICY_CSV, POLICY_PDFS, POLICY_INDEX_DIR)


# -------------------------- QA Prompts --------------------------
_QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Sen bir havayolu politika asistanısın. SADECE verilen içeriklere dayanarak cevap ver. "
     "Soru karma (zaman penceresi + politika) ise ZAMAN PENCERESİNİ YOK SAY ve genel politikayı açıkla. "
     "İçerik yeterliyse YÖNLENDİRME CÜMLESİ EKLEME. Ancak içerik yetersizse, "
     "'Güncel ve resmi bilgi için THY Bilgi Edin sayfasına bakınız' diye yönlendir. "
     "Eğer içerik bir güncelleme tarihi içeriyorsa 'Son güncelleme: YYYY-MM-DD' olarak belirt. "
     "Cevap Türkçe ve kısa olsun; sonda [1], [2] gibi numaralı kaynaklar bulunsun."),
    ("human",
     "Soru (TR):\n{question}\n\n"
     "Bağlam (her satır bir kaynak, [n] ile):\n{context}\n")
])

_WIKI_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Aşağıdaki Vikipedi içeriğini kısaca ve doğru şekilde özetle. "
     "Politika/ücret/koşul arıyorsan ve içerikte yoksa THY'nin resmi sayfasına yönlendir. "
     "Türkçe yanıt ver. Sonda '[Kaynak: Wikipedia]' ekle."),
    ("human", "Soru (TR):\n{question}\n\nVikipedi içeriği (kırpılmış):\n{content}\n")
])


# -------------------------- RAG: CSV+PDF --------------------------
def _format_context(docs: List[Document]) -> Tuple[str, List[str], List[str]]:
    lines, urls, titles = [], [], []
    for i, d in enumerate(docs, start=1):
        src = _normalize_source_url(d.metadata.get("source"))
        title = d.metadata.get("title") or ("CSV" if d.metadata.get("kind") == "csv" else "PDF")
        last_upd = (d.metadata.get("last_updated") or "").strip()
        snippet = d.page_content.replace("\n", " ").strip()[:650]
        suffix = f"  (src: {src})"
        if last_upd:
            suffix = f"  (src: {src}; günc: {last_upd})"
        lines.append(f"[{i}] {snippet}{suffix}")
        urls.append(src or "")
        titles.append(str(title))
    return "\n".join(lines), urls, titles

async def _answer_from_csv_pdf_rag(question: str) -> Dict[str, Any]:
    vs = _get_vs()
    retriever = vs.as_retriever(
        search_type="mmr",
        search_kwargs={"k": RETRIEVE_K, "fetch_k": max(RETRIEVE_K * 4, 12), "lambda_mult": 0.5},
    )
    docs = retriever.invoke(question)   # modern API
    if not docs:
        return {"answer": "", "citations": [], "channel": "csv_pdf_rag"}

    context_text, src_urls, src_titles = _format_context(docs)
    llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0)
    msg = _QA_PROMPT.invoke({"question": question, "context": context_text})
    out = await llm.ainvoke(msg.to_messages())
    text = out.content if hasattr(out, "content") else str(out)

    # Not: numaralandırma doğrudan doküman sırasına göre; uniq yapmıyoruz ki [n] eşleşmesi bozulmasın.
    citations = [{"title": t or "", "url": u} for u, t in zip(src_urls, src_titles)]
    return {"answer": text, "citations": citations, "channel": "csv_pdf_rag"}

def _answer_from_csv_pdf_rag_sync(question: str) -> Dict[str, Any]:
    vs = _get_vs()
    retriever = vs.as_retriever(
        search_type="mmr",
        search_kwargs={"k": RETRIEVE_K, "fetch_k": max(RETRIEVE_K * 4, 12), "lambda_mult": 0.5},
    )
    docs = retriever.invoke(question)
    if not docs:
        return {"answer": "", "citations": [], "channel": "csv_pdf_rag"}

    context_text, src_urls, src_titles = _format_context(docs)
    llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0)
    msg = _QA_PROMPT.invoke({"question": question, "context": context_text})
    out = llm.invoke(msg.to_messages())
    text = out.content if hasattr(out, "content") else str(out)

    citations = [{"title": t or "", "url": u} for u, t in zip(src_urls, src_titles)]
    return {"answer": text, "citations": citations, "channel": "csv_pdf_rag"}



# -------------------------- Wikipedia (yalnızca fallback) --------------------------
async def _answer_from_wiki(question: str) -> Dict[str, Any]:
    if not (_HAS_TAVILY and WIKI_URL):
        return {"answer": "", "citations": [], "channel": "wiki"}
    try:
        from os import getenv
        extractor = TavilyExtract(tavily_api_key=getenv("TAVILY_API_KEY"))
        page = extractor.invoke({"urls": [WIKI_URL]})
        content = ""
        if isinstance(page, dict) and "results" in page and page["results"]:
            content = (page["results"][0].get("content") or page["results"][0].get("markdown") or "")[:4000]
    except Exception:
        content = ""

    llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0)
    msg = _WIKI_PROMPT.invoke({"question": question, "content": content})
    out = await llm.ainvoke(msg.to_messages())
    text = out.content if hasattr(out, "content") else str(out)

    return {
        "answer": text,
        "citations": [{"title": "Wikipedia - Türk Hava Yolları", "url": WIKI_URL}],
        "channel": "wiki",
    }

def _answer_from_wiki_sync(question: str) -> Dict[str, Any]:
    if not (_HAS_TAVILY and WIKI_URL):
        return {"answer": "", "citations": [], "channel": "wiki"}
    try:
        from os import getenv
        extractor = TavilyExtract(tavily_api_key=getenv("TAVILY_API_KEY"))
        page = extractor.invoke({"urls": [WIKI_URL]})
        content = ""
        if isinstance(page, dict) and "results" in page and page["results"]:
            content = (page["results"][0].get("content") or page["results"][0].get("markdown") or "")[:4000]
    except Exception:
        content = ""

    llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0)
    msg = _WIKI_PROMPT.invoke({"question": question, "content": content})
    out = llm.invoke(msg.to_messages())
    text = out.content if hasattr(out, "content") else str(out)

    return {
        "answer": text,
        "citations": [{"title": "Wikipedia - Türk Hava Yolları", "url": WIKI_URL}],
        "channel": "wiki",
    }


# -------------------------- Orkestrasyon --------------------------
async def _answer_policy_async(query: str) -> Dict[str, Any]:
    # 0) Eğer soru 'kuruluş/tarihçe/genel bilgi' ise Wikipedia'yı önce dene
    if _should_use_wiki_first(query):
        wiki = await _answer_from_wiki(query)
        if wiki.get("answer"):
            final_text = f"{wiki['answer']}\n\n(Son kontrol: {_today_str()})"
            final_text = _render_with_citations(final_text, wiki.get("citations", []))
            return {
                "answer": final_text,
                "citations": wiki.get("citations", []),
                "last_checked": _today_str(),
                "source_channel": "wiki",
            }
        # wiki boş dönerse RAG'e düş

    # 1) Önce offline RAG
    rag = await _answer_from_csv_pdf_rag(query)
    if rag.get("answer"):
        ans = rag["answer"].strip()

        # 🔴 Özel kural: Sadece Bilgi Edin'e yönlendiren minimal cevap
        if _is_redirect_only(ans):
            only_cit = [{"title": "THY - Bilgi Edin", "url": THY_INFO_URL}]
            final_text = _render_with_citations(f"{ans}\n\n(Son kontrol: {_today_str()})", only_cit)
            return {
                "answer": final_text,
                "citations": only_cit,
                "last_checked": _today_str(),
                
            }

        # Normal akış
        final_text = f"{ans}\n\n(Son kontrol: {_today_str()})"
        final_text = _render_with_citations(final_text, rag.get("citations", []))
        return {
            "answer": final_text,
            "citations": rag.get("citations", []),
            "last_checked": _today_str(),
            "source_channel": "csv_pdf_rag",
        }

    # 2) RAG boşsa Wikipedia fallback
    wiki = await _answer_from_wiki(query)
    final_text = f"{wiki.get('answer','')}\n\n(Son kontrol: {_today_str()})"
    final_text = _render_with_citations(final_text, wiki.get("citations", []))
    return {
        "answer": final_text,
        "citations": wiki.get("citations", []),
        "last_checked": _today_str(),
        "source_channel": "wiki",
    }

def _answer_policy_sync(query: str) -> Dict[str, Any]:
    # 0) Wiki-first gate
    if _should_use_wiki_first(query):
        wiki = _answer_from_wiki_sync(query)
        if wiki.get("answer"):
            final_text = f"{wiki['answer']}\n\n(Son kontrol: {_today_str()})"
            final_text = _render_with_citations(final_text, wiki.get("citations", []))
            return {
                "answer": final_text,
                "citations": wiki.get("citations", []),
                "last_checked": _today_str(),
                "source_channel": "wiki",
            }
        # wiki boşsa RAG'e geç

    rag = _answer_from_csv_pdf_rag_sync(query)
    if rag.get("answer"):
        ans = rag["answer"].strip()

        # 🔴 Özel kural (sync)
        if _is_redirect_only(ans):
            only_cit = [{"title": "THY - Bilgi Edin", "url": THY_INFO_URL}]
            final_text = _render_with_citations(f"{ans}\n\n(Son kontrol: {_today_str()})", only_cit)
            return {
                "answer": final_text,
                "citations": only_cit,
                "last_checked": _today_str(),
                
            }

        final_text = f"{ans}\n\n(Son kontrol: {_today_str()})"
        final_text = _render_with_citations(final_text, rag.get("citations", []))
        return {
            "answer": final_text,
            "citations": rag.get("citations", []),
            "last_checked": _today_str(),
            "source_channel": "csv_pdf_rag",
        }


    wiki = _answer_from_wiki_sync(query)
    final_text = f"{wiki.get('answer','')}\n\n(Son kontrol: {_today_str()})"
    final_text = _render_with_citations(final_text, wiki.get("citations", []))
    return {
        "answer": final_text,
        "citations": wiki.get("citations", []),
        "last_checked": _today_str(),
        "source_channel": "wiki",
    }

def answer_policy(query: str) -> Dict[str, Any]:
    """
    Loop içindeysek (FastAPI), ASLA asyncio.run() çağırma.
    Sync yolu kullan. Loop yoksa async yolu koştur.
    """
    try:
        asyncio.get_running_loop()
        # event loop var → sync yol
        return _answer_policy_sync(query)
    except RuntimeError:
        # event loop yok → async yolu güvenle çalıştır
        return asyncio.run(_answer_policy_async(query))

# Soğuk başlangıç gecikmesini azalt
if os.getenv("POLICY_WARM", "1") == "1":
    try:
        _ = _get_vs()  # FAISS index'ini load et
    except Exception:
        pass
