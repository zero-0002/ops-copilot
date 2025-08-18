# app/agents/sql/sql_agent_thy.py
"""
THY-specific Text-to-SQL chains (OpenAI)
- chain_subquestion           : (optional) lightweight router helper
- chain_column_extractor      : minimal column selection aid
- chain_filter_extractor      : categorical filters (STRING + 0/1 flags) using KNOWN VALUES
- chain_range_date_extractor  : numeric/date windows
- chain_query_extractor       : SQL generation (schema-bound)
- chain_query_validator       : SQL validate/repair + domain guard + safe fallback

Design notes
------------
• No hard-coded rules like "A321 ⇒ aircraft_type". We use KNOWN VALUES + regex patterns
  to map literals to the correct column domains dynamically.
• Never invent tables/columns/enums. If unmappable, produce a safe empty SQL that returns 0 rows.
• Return only raw, executable SQL (no fences / commentary).
"""

import os
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableMap
from sqlalchemy.engine.url import make_url

# KNOWN VALUES / VOCAB snapshot (categoricals + identifier previews)
from app.tools.vocab import VOCAB_TEXT

# ---------------- Models ----------------
chat_model_primary = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0.2
)
chat_model_secondary = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL_SECONDARY", os.getenv("OPENAI_MODEL", "gpt-4o-mini")),
    temperature=0.0
)

# ---------------- Dialect hint ----------------
def dialect_hint() -> str:
    url = os.getenv("DB_URL", "sqlite:///app/data/thy_ops.db")
    try:
        backend = make_url(url).get_backend_name()
    except Exception:
        backend = "sqlite"
    if backend == "mysql":
        return "Target SQL dialect: MySQL 8+"
    return "Target SQL dialect: SQLite (3.35+)"

# ---------------- Catalog text (THY) ----------------
TABLES_TEXT = """
- flights (Ucus Durumu): Ucus performans metrikleri.
  Kolonlar: flight_number (STRING), departure_time (DATETIME/TEXT ISO), delay_minutes (INT),
            aircraft_type (STRING), crew_issues (INT: 0/1), weather_impact (INT: 0/1)

- complaints (Yolcu Sikayetleri): Müşteri sikayet kayitları (ucus bagli).
  Kolonlar: complaint_id (STRING), flight_number (STRING), customer_id (STRING),
            category (STRING), status (STRING), created_at (DATETIME/TEXT ISO)

- refunds (Bilet İptal/İade): İptal ve iade süreçleri.
  Kolonlar: ticket_id (STRING), flight_number (STRING), cancel_reason (STRING),
            refund_status (STRING), customer_id (STRING)

- weather (Hava Durumu Mock): Ucus numarasina göre hava durumu özeti.
  Kolonlar: flight_number (STRING), condition (STRING: storm, rain, fog, clear)
"""

COLUMNS_TEXT = """
flights.flight_number (STRING): Ucus kodu; join anahtari. ör: TK102, TK345, ...
flights.departure_time (DATETIME/TEXT ISO): Kalkis zamani; tarih/saat filtreleri için.
flights.delay_minutes (INT): Gecikme dakikasi; agregasyon/kosul icin.
flights.aircraft_type (STRING): Ucak modeli; ör: A320, A321, B737, A330.
flights.crew_issues (INT: 0/1): Ekip kaynakli sorun flag’i (1=var).
flights.weather_impact (INT: 0/1): Hava etkisi flag’i (1=etkilenmiş).

complaints.complaint_id (STRING)
complaints.flight_number (STRING)
complaints.customer_id (STRING)
complaints.category (STRING): service | delay | baggage | seat | other
complaints.status   (STRING): open | resolved | pending
complaints.created_at (DATETIME/TEXT ISO)

refunds.ticket_id (STRING)
refunds.flight_number (STRING)
refunds.cancel_reason (STRING)
refunds.refund_status (STRING)
refunds.customer_id (STRING)

weather.flight_number (STRING)
weather.condition (STRING): storm | rain | fog | clear
"""

# ===================== 1) Subquestion Agent (optional) ==================
template_subquestion = ChatPromptTemplate.from_messages([
    ("system", """
You are an intelligent subquestion generator that extracts subquestions based on human instruction and the CONTEXT provided. You are part of a Text-to-SQL agent.
"""),
    ("human", '''
CONTEXT:
This dataset pertains to Turkish Airlines operations analytics. Business reports come from four tables:
- flights (delays, aircraft type, crew issues, weather_impact),
- complaints (category, status, timestamps),
- refunds (cancel_reason, refund_status),
- weather (condition).

You are given:
- A user question
- A list of table names with descriptions

Instructions:
Think like a Text-to-SQL agent. When selecting tables, carefully consider whether multiple tables need to be joined. Only select the tables necessary to answer the user question.
*** A table might not answer a subquestion, but adding it might act as a link with another table selected by different agent. If selected table has all information, ignore other tables.

SPECIAL DOMAIN RULE:
- If the main question is about cancellations by day/week, prefer **refunds** for the cancellation cause (via `cancel_reason`) and **flights** for the date (via `departure_time`) by joining on `flight_number`. Do not use delay-based fields to infer cancellations.
     

Your task:
1. Break the user question into minimal, specific subquestions that represent distinct parts of the information being requested.
2. For each subquestion, identify a **single table** whose **description** clearly indicates it contains the needed information.
3. Ignore subquestions that cannot be answered using the provided tables.
4. Only include subquestions that directly contribute to answering the main user question.
5. If a subquestion can be answered using multiple tables, choose the single most appropriate table.
6. Be specific and avoid redundancy.

Output format (list of lists; use double quotes):
[["subquestion1", "table name 1"], ["subquestion2", "table name 2"]]
If multiple subquestions map to same table:
[["subq1", "subq2", "table name"]]
If only one valid subquestion:
[["subquestion1", "table name"]]
If none:
[[]]

Table List:
{tables}

User question:
{user_query}
''')
])

chain_subquestion = (
    RunnableMap({
        "tables": lambda x: x["tables"],
        "user_query": lambda x: x["user_query"],
    })
    | template_subquestion
    | chat_model_primary
    | StrOutputParser()
)

# ===================== 2) Column Selector Agent ==================
template_column = ChatPromptTemplate.from_messages([
    ("system", """
You are an intelligent data column selector that chooses the most relevant columns from a list of available column descriptions to help answer a subquestion ONLY.
Your selections will be used by a SQL generation agent, so choose only those columns that will help write the correct SQL query for a subquestion based on main question.
"""),
    ("human", '''
HOW TO THINK:
- For each subquestion below, evaluate columns by their descriptions; include identifiers (flight_number, complaint_id, ticket_id, customer_id) when needed for grouping/joins.
- Consider dependencies (e.g., aggregations across multiple rows need grouping keys).
- After finishing subquestion coverage, check the main question once for any extra necessary columns.

RULES:
1) Include unique identifiers when relevant.
2) Describe how each selected column contributes, quoting its catalog description.
3) Output strictly as:
[["<column name 1>", "<description + why needed>"], ["<column name 2>", "<...>"]]
4) **Cancellation questions:** Always consider selecting the trio below, unless truly irrelevant:
   - refunds.cancel_reason → maps the cause of cancellation (e.g., weather).
   - flights.flight_number → join key between refunds and flights.
   - flights.departure_time → provides the date/time for day/week grouping.

Column list:
{columns}

subquestion:
{query}

Main question:
{main_question}
''')
])

chain_column_extractor = (
    RunnableMap({
        "columns": lambda x: x["columns"],
        "query": lambda x: x["query"],
        "main_question": lambda x: x["main_question"],
    })
    | template_column
    | chat_model_primary
    | StrOutputParser()
)

# ===================== 3) Categorical Filter Agent ==================
template_filter_check = ChatPromptTemplate.from_messages([
    ("system", """
Decide categorical WHERE filters (STRING + 0/1 flags only).
DO NOT include numeric ranges or date windows here.

Use KNOWN VALUES to map literals to correct column domains.
Never invent columns or values.
If the question mentions cancellation + weather/hava şartları, prefer mapping to refunds.cancel_reason='weather'.
     
Patterns to help:
- flight_number ≈ ^[A-Z]{{2}}[0-9]{{3,4}}$
- aircraft_type ≈ ^[AB][0-9]{{3}}$

If a literal cannot be confidently mapped to a known domain, do not force it.
You decide if WHERE filters are needed based on categorical columns only.

Include filters for:
- STRING columns (e.g., flight_number, aircraft_type, category, status, refund_status, cancel_reason, weather.condition)
- Binary flag INT columns used categorically (e.g., crew_issues, weather_impact; values 0 or 1)
     
Output strictly:
- If filters: ["yes", ["<table>","<column>","<values>"], ...]
- Else: ["no"]
"""),
    ("human", '''
User question:
{query}

Available columns (with examples):
{columns}

KNOWN VALUES (categoricals + identifier previews):
{vocab_text}
''')
])

chain_filter_extractor = (
    RunnableMap({
        "columns": lambda x: x["columns"],
        "query": lambda x: x["query"],
        "vocab_text": lambda x: VOCAB_TEXT,
    })
    | template_filter_check
    | chat_model_secondary
    | StrOutputParser()
)

# ===================== 3b) Range/Date Filter Agent ==================
template_range_date = ChatPromptTemplate.from_messages([
    ("system", """
Extract explicit numeric range and date/time windows from the question.

Rules:
- Numeric examples: delay_minutes >= 15
- Date/time examples: "last 7 days", "between 2025-07-01 and 2025-07-10"
- Map date windows to the correct date column by table:
  * flights  → departure_time
  * complaints → created_at
- Return exactly one of:
  ["ranges", ["<table>","<column>","<operator/value expr>"], ...]
  ["dates",  ["<table>","<column>","<human range>"], ...]
  ["none"]
Do not invent ranges.
"""),
    ("human", '''
User question:
{query}

Relevant columns (with types):
{columns}
''')
])

chain_range_date_extractor = (
    RunnableMap({
        "columns": lambda x: x["columns"],
        "query": lambda x: x["query"],
    })
    | template_range_date
    | chat_model_secondary
    | StrOutputParser()
)

# ===================== 4) SQL Query Generator ==================
template_sql_query = ChatPromptTemplate.from_messages([
    ("system", """
You are a schema-bound SQL generator.
- Use ONLY the schema below; never invent tables/columns.
- Apply categorical filters if present.
- Apply numeric/date filters if present.
- Use safe aliases (fs, pc, rf, wx). Never use reserved words like "or", "and", "as" as aliases.
- Prefer CTEs for clarity.
- Keep syntax valid for the target dialect.
-If the user asks about cancellations per time period (day/week), and refunds has no date column, JOIN refunds→flights on flight_number and group by date(flights.departure_time) (or week).

If both weather_impact and cancel_reason are candidates, and the user explicitly says iptal/cancellation, use cancel_reason (do not use delay_minutes or weather_impact).
If the user request requires unavailable fields or unmappable literals,
produce a SAFE EMPTY SQL that returns 0 rows (e.g. `SELECT * FROM flights WHERE 1=0;`).
No commentary. Return ONLY raw SQL.
"""),
    ("human", '''
{dialect_hint}

SCHEMA (SQLite 3.35+):
{columns}

KNOWN VALUES (categoricals + identifier previews):
{vocab_text}

User question:
{query}

Applicable categorical filters:
{filters}

Applicable numeric/date filters (may be ["none"]):
{range_filters}

Guidelines (SQLite):
- Date windows: date(col) >= date('now','-N day')   (flights→departure_time, complaints→created_at)
- Join on: flight_number (flights↔complaints/refunds/weather), customer_id (complaints↔refunds)
Return only the final SQL.
''')
])

chain_query_extractor = (
    RunnableMap({
        "columns": lambda x: COLUMNS_TEXT,
        "query": lambda x: x["query"],
        "filters": lambda x: x["filters"],
        "range_filters": lambda x: x["range_filters"],
        "dialect_hint": lambda x: dialect_hint(),
        "vocab_text": lambda x: VOCAB_TEXT,
    })
    | template_sql_query
    | chat_model_primary
    | StrOutputParser()
)

# ===================== 5) SQL Validator (self-repair + domain guard) ==================



template_validation = ChatPromptTemplate.from_messages([
    ("system", """
CRITICAL: If the provided SQL already conforms to the schema and uses only valid columns/tables, KEEP IT UNCHANGED.
If there are fixable issues (e.g., wrong date column used for a date window), REPAIR them and return the repaired SQL.
Return ONLY raw SQL. Never return commentary or fences.
Validate/repair the SQL for syntax, logic, and the specified dialect.
- Replace reserved-word aliases if any.
- Prefer subqueries for grouped filters to avoid conflicts.
- Keep only relevant columns in SELECT (but don't break joins/logic).
Return the final SQL (or the same if already correct).
-If SQL uses weather_impact or delay_minutes while the question is about cancellations, replace logic to use refunds.cancel_reason and join flights for dates; group by date
"""),
    ("human", '''
Dialect hint:
{dialect_hint}



KNOWN VALUES (categoricals + identifier previews):
{vocab_text}

User Question:
{query}

Relevant Tables/Columns (docstring):
{columns}

Categorical Filters:
{filters}

Numeric/Date Filters:
{range_filters}

SQL to validate:
{sql_query}
''')
])

chain_query_validator = (
    RunnableMap({
        "columns": lambda x: COLUMNS_TEXT,
        "query": lambda x: x["query"],
        "filters": lambda x: x["filters"],
        "range_filters": lambda x: x["range_filters"],
        "sql_query": lambda x: x["sql_query"],
        "dialect_hint": lambda x: dialect_hint(),
        "vocab_text": lambda x: VOCAB_TEXT,
    })
    | template_validation
    | chat_model_primary.bind(temperature=0.0)
    | StrOutputParser()
)
