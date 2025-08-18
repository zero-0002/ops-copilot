# Prompt İşleme Adımları

1) UI → API
   - Kullanıcı prompt’u Streamlit’ten `/ask_unified`’a gider.
   - Payload: question, preview_rows, return_rows, return_chart, top_k.

2) Cache Path (fast)
   - `try_cache(question, ...)`:
     - RediSearch exact match (escape + DIALECT 3).
     - Tag ve tazelik kontrolü.
     - Hit → sample/execute → `synthesize_sql()` → final_answer.
     - Miss → Graph path.

3) Graph Path (agentic)
   - `try_graph()` LangGraph’i çağırır:
     - Tablo seçimi / alt ajan (complaints/refunds/flights).
     - SQL oluşturma, güvenli LIMIT/preview.
     - Policy RAG (FAISS) ve gerekirse birleştirme.

4) Synthesis / Merge
   - `synthesize_sql()` satırdan kısa analiz + (opsiyonel) Vega-Lite.
   - `synthesize_unified()` policy + sql tek nihai cevap.

5) Response
   - `used_cache`, `use_web`, `want_sql`, `source`, `sql`, `rows`, `citations` ile UI’ya döner.
   - UI sekmelerde Tablo/SQL/Kaynaklar gösterir; metindeki “Kaynaklar:” bloğu strip edilir.
