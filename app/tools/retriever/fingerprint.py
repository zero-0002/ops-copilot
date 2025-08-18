import os
from typing import Dict
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
load_dotenv()

DB_URL = os.getenv("DB_URL") or f"sqlite:///{os.getenv('DB_PATH','app/data/thy_ops.db')}"

def dataset_fingerprint()->str:
    """
    Basit parmak izi: her tablonun max(tarih) veya rowcount’larından bir özet.
    SQLite seed şemana göre tarih kolonları:
      - flights.departure_time
      - complaints.created_at
      - refunds (seed.py'de tarih yoksa COUNT bazlı)
      - weather (statik)
    """
    eng = create_engine(DB_URL, future=True)
    with eng.connect() as c:
        # flights
        max_dep = c.execute(text("SELECT COALESCE(MAX(departure_time),'') FROM flights")).scalar()
        # complaints
        max_comp = c.execute(text("SELECT COALESCE(MAX(created_at),'') FROM complaints")).scalar()
        # refunds (seed.py’de tarih yoksa COUNT)
        rc_ref = c.execute(text("SELECT COUNT(*) FROM refunds")).scalar()
        # weather
        rc_wx = c.execute(text("SELECT COUNT(*) FROM weather")).scalar()
    return f"flights:max_dep={max_dep}|complaints:max={max_comp}|refunds:rows={rc_ref}|weather:rows={rc_wx}"
