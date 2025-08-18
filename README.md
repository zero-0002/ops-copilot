# THY Ops Copilot ✈️ — Agentic AI Case Study

Operasyon ekiplerinin **doğal dilde** (TR/EN) veri talebi yazdığı; **cache → graph → policy RAG** orkestrasyonuyla **SQL sorgusu / tablo / grafik** veya **politika metni** üreten bir *Agentic AI* demo sistemi.

---

## İçindekiler

* [Özellikler](#özellikler)
* [Mimari (özet)](#mimari-özet)
* [Proje Yapısı](#proje-yapısı)
* [Hızlı Başlangıç (Docker Compose)](#hızlı-başlangıç-docker-compose)
* [Lokal Geliştirme (Python)](#lokal-geliştirme-python)
* [Ortam Değişkenleri](#ortam-değişkenleri)
* [OpenAPI & Tool Schemas](#openapi--tool-schemas)
* [Prompt İşleme Adımları](#prompt-i̇şleme-adımları)
* [UI Özeti](#ui-özeti)
* [Smoke Test](#smoke-test)
* [Lisans / Notlar](#lisans--notlar)

---

## Özellikler

* **Cache-first**: Onaylı/`seed` etiketli hazır SQL planlarını ve örnek satırları **milisaniyede** döndürür.
* **Graph fallback**: Cache miss olursa **LangGraph** tabanlı NL→SQL orkestratörü devreye girer.
* **Policy RAG**: CSV/PDF politikalarından kısa metin yanıt + kaynaklar.
* **Tek cevap**: SQL analiz + politika metnini tek, kısa ve tutarlı Türkçe cevapta birleştirme.
* **UI**: Streamlit arayüzü, “Quick Analysis” aksiyonları, tablo/SQL/kaynak sekmeleri.
* **Geçmiş**: UI içinde son 3 soru-cevap tutulur (configurable).
* **Write-through cache**: Graph kazanınca cache’e yazabilme (opsiyonel).

---

## Mimari (özet)

```
![Mimari Diyagramı](docs/images/Agentic Workflow.png)

```

**Detay**: `docs/architecture.md` ve `docs/prompt_flow.md` (önerilen).

---

## Proje Yapısı

```
Proje Yapısı
thy_agentic_sql/
├─ app/
│  ├─ api/                   # FastAPI (healthz, ask_unified)
│  ├─ agents/                # LangGraph / synthesis
│  ├─ services/              # cache / orchestrator / unified
│  └─ tools/                 # retriever, seed, index, policy vb.
├─ ui/
│  └─ streamlit_app.py       # UI (Quick Analysis + chat + sekmeler)
├─ docs/
│  ├─ README_RUN.md          # (opsiyonel) hızlı çalışma yönergeleri
│  ├─ architecture.md        # mimari şema + açıklamalar
│  ├─ tool_schemas.yaml      # tool schema örnekleri (YAML)
│  ├─ tool_schemas.json      # tool schema örnekleri (JSON)
│  └─ prompt_flow.md         # prompt/LLM zincir akışı
├─ scripts/
│  ├─ export_tool_schemas.py # tool schema’ları JSON’a yazar
│  └─ smoke.sh               # basit çalışma testi
├─ .env.example              # örnek env (secret yok)
├─ .gitignore
├─ Dockerfile
├─ docker-compose.yml
├─ requirements.txt
└─ README.md            
```

---

## Hızlı Başlangıç (Docker Compose)

> **Önkoşul**: Docker Engine + Compose v2 (WSL/Linux/Windows/Mac).

```bash
# İlk kurulum / build
docker compose up --build -d

# Durumu kontrol
docker compose ps

# Loglar
docker logs -f thy-api
docker logs -f thy-ui
```

* UI: `http://localhost:8501`
* API: `http://localhost:8000/docs` (Swagger UI)

> **Redis port çakışması** yaşıyorsanız, `redis` servisini sadece iç ağda kullanın (compose’ta `ports:` bölümünü kaldırın) ve `REDIS_URL=redis://thy-redis:6379/0` olarak ayarlayın.

---

## Lokal Geliştirme (Python)

```bash
conda create -n agentic python=3.10 -y
conda activate agentic
pip install -r requirements.txt

# Çevre değişkenleri
cp .env.example .env

# API
uvicorn api.main:app --reload --port 8000

# UI (ayrı terminal)
streamlit run ui/streamlit_app.py --server.port 8501
```

---

## Ortam Değişkenleri

`.env.example` önerisi:

```env
# Database
DB_URL=sqlite:///app/data/thy_ops.db

# OpenAI
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
OPENAI_MODEL_SECONDARY=gpt-4o-mini

# Router
ROUTER_LLM_MODEL=gpt-4o-mini
ROUTER_LLM_TIMEOUT_S=0.6
ROUTER_THRESHOLD=0.6

# Synthesis
LLM_MERGE_HYBRID=false

# Cache
REDIS_URL=redis://thy-redis:6379/0

# Policy RAG
POLICY_CSV=app/data/policy.csv
POLICY_PDFS=app/data/policies/*.pdf
POLICY_INDEX_DIR=app/data/policy_index
```

> **Not**: Production’da gizli değerleri secret manager ile yönetin.

---

## OpenAPI & Tool Schemas

**FastAPI** uç noktası (özet):

* `POST /ask_unified`
* Body:

```json
{
  "query": "Hava şartlarından dolayı uçuşlar en çok hangi günlerde iptal ediliyor?",
  "preview_rows": 20,
  "want_chart": true
}
```

* Response (örnek):

```json
{
  "final_answer": "En çok iptal 2025-06-28 ve 2025-07-03 tarihlerinde...",
  "executed_sql": "WITH cancels AS ( ... ) SELECT day, cancel_cnt ...",
  "rows_preview": [{"day":"2025-06-28","cancel_cnt":4}],
  "citations": [{"title":"THY Politika PDF","url":"..."}]
}
```

**Tool schema** örnekleri (yönerge amaçlı):

```yaml
# reports-mcp (örnek)
name: reports-mcp
functions:
  - name: flight_status
    description: Return flight performance metrics
    parameters:
      type: object
      properties:
        flight_number: { type: string }
      required: [flight_number]
  - name: complaints
    description: Return complaint stats
    parameters:
      type: object
      properties:
        window_days: { type: integer, default: 30 }
```

---

## Prompt İşleme Adımları

Agentik akış (LangGraph):

1. **normalize** → TR terimleri kavramsal eşleme (`i18n/locale.py`).
2. **policy\_gate** → `web` / `sql` / `hybrid` kararı (LLM + regex fallback).
3. **policy\_rewrite** → policy partını ayıkla (hybrid).
4. **router** → tablo alt seti: `flights|complaints|refunds|weather`.
5. **customer\_agent** → *subquestion* & *column selection* (tek tablo ajanları).
6. **filter\_check** → kategorik filtreler (KNOWN VALUES).
7. **fuzz\_filter** → fuzzy eşleştirme & normalize.
8. **range\_date** → sayısal/tarih pencereleri.
9. **query\_generator** → şema bağlı SQL üretimi.
10. **query\_validation** → onarım/guard (iptal ≠ gecikme vb.).
11. **execute\_sql** → örnek satırlar (LIMIT, güvenli).
12. **policy\_safety** → kaçan policy sinyali varsa tek atım politikayı getir.
13. **join** → hybrid bekleme bariyeri.
14. **synthesize** → SQL JSON + Policy → **tek, kısa TR cevap**.

> **Domain guard örneği**: “İptaller” sorularında **`refunds.cancel_reason='weather'` + `flights.departure_time`** ile gün/hafta gruplama; `delay_minutes`/`weather_impact` ile iptal çıkarımı **yapma**.

---

## UI Özeti

* **Sticky header** + hızlı aksiyonlar (Quick Analysis)
* Sekmeler: **Tablo**, **SQL**, **Kaynaklar**
* Son **3** etkileşimi lokal state’te tutar
* Vega-Lite chart spec desteği (opsiyonel)

---

## Smoke Test

```bash
# API üzerinden basit test
curl -s http://localhost:8000/ask_unified \
  -H 'Content-Type: application/json' \
  -d '{"query":"Hava şartlarından dolayı uçuşlar en çok hangi günlerde iptal ediliyor?","preview_rows":20}' | jq .

# Beklenen: refunds + flights join’li, gün bazında iptal sayımı yapan SQL & satırlar
```

---

## Lisans / Notlar

* MIT (örnek). Kurum içi kullanımda şirket politikanıza göre güncelleyin.
* Demo veri şeması: `flights / complaints / refunds / weather` (mock/örnek). Gerçek ortamlarda kimlik verilerini maskeleyin.

---




     

