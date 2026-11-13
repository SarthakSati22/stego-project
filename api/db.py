"""Run log for every embed and extract (PostgreSQL, or SQLite if DATABASE_URL is not set).

Set DATABASE_URL to use PostgreSQL, for example
    postgresql://user:password@localhost:5432/stego
The secret message and the passphrase are never stored."""
import json
import os
import sqlite3
import time
import traceback
from pathlib import Path

URL = os.environ.get("DATABASE_URL", "").strip()
PG = URL.startswith("postgres")
SQLITE = Path(__file__).resolve().parents[1] / "results" / "stego.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs(
    job_id TEXT PRIMARY KEY, created TEXT, clip_id TEXT, plan TEXT,
    k INTEGER, n INTEGER, bits_total INTEGER, bits_by_carrier TEXT,
    shares TEXT, risk_driven INTEGER, quality TEXT);
CREATE TABLE IF NOT EXISTS extractions(
    id {pk}, created TEXT, job_id TEXT, scenario TEXT, recovered INTEGER, carrier_status TEXT);
"""


def backend():
    return "PostgreSQL" if PG else "SQLite"


def _con():
    if PG:
        import psycopg2
        con = psycopg2.connect(URL)
        pk = "SERIAL PRIMARY KEY"
    else:
        SQLITE.parent.mkdir(exist_ok=True)
        con = sqlite3.connect(SQLITE)
        pk = "INTEGER PRIMARY KEY AUTOINCREMENT"
    cur = con.cursor()
    for stmt in SCHEMA.format(pk=pk).split(";"):
        if stmt.strip():
            cur.execute(stmt)
    con.commit()
    return con


def _q(sql):
    return sql.replace("?", "%s") if PG else sql


def _run(sql, args=(), fetch=False):
    con = _con()
    try:
        cur = con.cursor()
        cur.execute(_q(sql), args)
        rows = None
        if fetch:
            cols = [d[0] for d in cur.description]
            rows = [dict(zip(cols, r)) for r in cur.fetchall()]
        con.commit()
        return rows
    finally:
        con.close()


def _i(x):
    return None if x is None else int(x)


def _j(x):
    return json.dumps(x, default=lambda o: o.item() if hasattr(o, "item") else str(o))


def log_job(r, clip_id):
    try:
        _run("DELETE FROM jobs WHERE job_id=?", (r["job_id"],))
        _run("INSERT INTO jobs VALUES(?,?,?,?,?,?,?,?,?,?,?)", (
            str(r["job_id"]), time.strftime("%Y-%m-%d %H:%M:%S"), str(clip_id), str(r["plan"]),
            _i(r["k"]), _i(r["n"]), _i(r["bits_total"]), _j(r.get("bits_by_carrier")),
            _j(r.get("shares")), int(bool(r.get("risk_driven"))),
            _j(r.get("quality"))))
        print("[db] logged job", r["job_id"], "->", backend())
    except Exception:
        print("[db] could not log job:")
        traceback.print_exc()  # logging must never break the demo


def log_extract(job_id, scenario, recovered, carrier_status):
    try:
        _run("INSERT INTO extractions(created,job_id,scenario,recovered,carrier_status)"
             " VALUES(?,?,?,?,?)", (time.strftime("%Y-%m-%d %H:%M:%S"), job_id, scenario,
                                    int(recovered), _j(carrier_status)))
        print("[db] logged extraction", job_id, scenario)
    except Exception:
        print("[db] could not log extraction:")
        traceback.print_exc()


def history(limit=50):
    jobs = _run("SELECT * FROM jobs ORDER BY created DESC LIMIT ?", (limit,), fetch=True)
    for j in jobs:
        j["extractions"] = _run(
            "SELECT scenario, recovered, created FROM extractions WHERE job_id=? ORDER BY id",
            (j["job_id"],), fetch=True)
    return {"database": backend(), "jobs": jobs}
