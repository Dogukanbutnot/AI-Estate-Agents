"""
lead_store.py
-------------
Form token'ları ve doldurulan lead'ler için TEK doğruluk kaynağı (SQLite).

Neden var: Daha önce FormAgent token'ları kendi belleğinde, form_server ise
kendi tablosunda tutuyordu; birbirlerini tanımıyorlardı. Artık FormAgent
token'ı buraya kaydeder, sunucu formu buradan doğrular ve buraya yazar.
"""

import os
import sqlite3
import time
from contextlib import closing

SCHEMA = """
CREATE TABLE IF NOT EXISTS form_tokens (
    token        TEXT PRIMARY KEY,
    username     TEXT NOT NULL,
    utm_source   TEXT,
    utm_campaign TEXT,
    score        INTEGER,
    urgency      TEXT,
    created_at   REAL NOT NULL,
    completed_at REAL
);
CREATE TABLE IF NOT EXISTS leads (
    token         TEXT PRIMARY KEY REFERENCES form_tokens(token),
    username      TEXT,
    name          TEXT,
    phone         TEXT,
    email         TEXT,
    budget        TEXT,
    location      TEXT,
    property_type TEXT,
    timeline      TEXT,
    kvkk_consent  INTEGER NOT NULL,
    submitted_at  REAL NOT NULL
);
"""


class UnknownToken(Exception):
    """Token hiç üretilmemiş."""


class AlreadySubmitted(Exception):
    """Bu token ile form zaten doldurulmuş."""


class ConsentRequired(Exception):
    """KVKK onayı olmadan kayıt yapılamaz."""


def db_path() -> str:
    return os.environ.get("EMLAIK_DB", "leads.db")


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(db_path())
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def register_token(token: str, username: str, utm_source: str = "",
                   utm_campaign: str = "", score: int = 0, urgency: str = "") -> None:
    with closing(_connect()) as conn, conn:
        conn.execute(
            "INSERT INTO form_tokens (token, username, utm_source, utm_campaign,"
            " score, urgency, created_at) VALUES (?,?,?,?,?,?,?)",
            (token, username, utm_source, utm_campaign, score, urgency, time.time()),
        )


def get_token(token: str):
    with closing(_connect()) as conn:
        row = conn.execute(
            "SELECT * FROM form_tokens WHERE token = ?", (token,)
        ).fetchone()
    return dict(row) if row else None


def save_submission(token: str, data: dict) -> str:
    """
    Formu kaydeder ve token'ın sahibi olan username'i döndürür.
    Token kullanımı tek seferliktir; ikinci deneme AlreadySubmitted fırlatır.
    """
    if not data.get("kvkk_consent"):
        raise ConsentRequired("KVKK onayı verilmedi")

    now = time.time()
    with closing(_connect()) as conn, conn:   # `with conn` = transaction
        cur = conn.execute(
            "UPDATE form_tokens SET completed_at = ? "
            "WHERE token = ? AND completed_at IS NULL",
            (now, token),
        )
        if cur.rowcount == 0:
            exists = conn.execute(
                "SELECT 1 FROM form_tokens WHERE token = ?", (token,)
            ).fetchone()
            raise AlreadySubmitted(token) if exists else UnknownToken(token)

        username = conn.execute(
            "SELECT username FROM form_tokens WHERE token = ?", (token,)
        ).fetchone()["username"]

        conn.execute(
            "INSERT INTO leads (token, username, name, phone, email, budget,"
            " location, property_type, timeline, kvkk_consent, submitted_at)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (token, username, data.get("name"), data.get("phone"),
             data.get("email"), data.get("budget"), data.get("location"),
             data.get("property_type"), data.get("timeline"), 1, now),
        )
    return username


def list_completed() -> list:
    with closing(_connect()) as conn:
        rows = conn.execute(
            "SELECT l.token, l.username, l.name, l.phone, l.email, l.budget,"
            " l.location, l.property_type, l.timeline, l.kvkk_consent,"
            " l.submitted_at, t.score, t.urgency, t.utm_source, t.utm_campaign"
            " FROM leads l JOIN form_tokens t ON t.token = l.token"
            " ORDER BY l.submitted_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]
