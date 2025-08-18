import os, uuid, random
import numpy as np
import pandas as pd
import datetime as dt
import sqlite3

BASE = os.path.dirname(__file__)
DB_PATH = os.path.join(BASE, "thy_ops.db")
np.random.seed(42)
random.seed(42)

def rand_date(days=60):
    today = dt.date.today()
    return today - dt.timedelta(days=int(np.random.randint(0, days)))

def build():
    # Flights
    flights = []
    aircraft_types = ["A320","A321","B737","A330"]
    for _ in range(200):
        fn = f"TK{np.random.randint(100,999)}"
        date = rand_date(60)
        dep = dt.datetime.combine(date, dt.time(hour=int(np.random.randint(0,24)),
                                                minute=int(np.random.choice([0,15,30,45]))))
        delay = max(0, int(np.random.normal(20,30))) if np.random.rand()<0.6 else 0
        aircraft = random.choice(aircraft_types)
        crew_issue = 1 if (np.random.rand()<0.1 and delay>0) else 0
        weather_impact = 1 if (np.random.rand()<0.2 and delay>0) else 0
        flights.append([fn, dep.isoformat(), delay, aircraft, crew_issue, weather_impact])
    df_f = pd.DataFrame(flights, columns=[
        "flight_number","departure_time","delay_minutes","aircraft_type","crew_issues","weather_impact"
    ])

    # Complaints
    complaints = []
    categories = ["service","delay","baggage","seat","other"]
    for _ in range(400):
        cid = str(uuid.uuid4())[:8]
        fn = random.choice(df_f["flight_number"].tolist())
        cust = f"C{np.random.randint(10000,99999)}"
        cat = np.random.choice(categories, p=[0.2,0.4,0.2,0.1,0.1])
        status = np.random.choice(["open","resolved","pending"], p=[0.2,0.6,0.2])
        created_at = rand_date(60)
        complaints.append([cid, fn, cust, cat, status, created_at.isoformat()])
    df_c = pd.DataFrame(complaints, columns=[
        "complaint_id","flight_number","customer_id","category","status","created_at"
    ])

    # Refunds
    refunds = []
    reasons = ["weather","crew","personal","schedule","other"]
    for _ in range(250):
        tid = str(uuid.uuid4())[:8]
        fn = random.choice(df_f["flight_number"].tolist())
        cust = f"C{np.random.randint(10000,99999)}"
        reason = np.random.choice(reasons, p=[0.25,0.15,0.3,0.2,0.1])
        status = np.random.choice(["approved","rejected","processing"], p=[0.6,0.2,0.2])
        refunds.append([tid, fn, reason, status, cust])
    df_r = pd.DataFrame(refunds, columns=[
        "ticket_id","flight_number","cancel_reason","refund_status","customer_id"
    ])

    # Weather (mock by flight)
    weather = []
    for fn in df_f["flight_number"].unique():
        cond = np.random.choice(["storm","rain","fog","clear"], p=[0.1,0.3,0.1,0.5])
        weather.append([fn, cond])
    df_w = pd.DataFrame(weather, columns=["flight_number","condition"])

    
    df_f.to_csv(os.path.join(BASE,"flights.csv"), index=False)
    df_c.to_csv(os.path.join(BASE,"complaints.csv"), index=False)
    df_r.to_csv(os.path.join(BASE,"refunds.csv"), index=False)
    df_w.to_csv(os.path.join(BASE,"weather.csv"), index=False)

    
    con = sqlite3.connect(DB_PATH)
    df_f.to_sql("flights", con, if_exists="replace", index=False)
    df_c.to_sql("complaints", con, if_exists="replace", index=False)
    df_r.to_sql("refunds", con, if_exists="replace", index=False)
    df_w.to_sql("weather", con, if_exists="replace", index=False)
    con.commit(); con.close()

if __name__=="__main__":
    build()
    print("Seeded:", DB_PATH)
