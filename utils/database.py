"""
utils/database.py
-------------------
Lightweight SQLite persistence layer for the AI Resume Builder.
Stores generated resumes and analysis history so the Resume Ranking
page can compare resumes uploaded/created across sessions.

Swap DB_PATH's sqlite3 connection for a MongoDB client if you prefer
MongoDB (see README for notes) -- the function signatures here are the
integration surface the app.py calls into.
"""

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from typing import Dict, List, Optional

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "resume_builder.db")


@contextmanager
def get_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS resumes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                source TEXT,                 -- 'builder' or 'upload'
                resume_text TEXT,
                predicted_category TEXT,
                category_confidence REAL,
                ats_score REAL,
                job_match_pct REAL,
                skills_json TEXT,
                created_at TEXT
            )
        """)


def save_resume_record(
    name: str,
    resume_text: str,
    source: str = "builder",
    predicted_category: Optional[str] = None,
    category_confidence: Optional[float] = None,
    ats_score: Optional[float] = None,
    job_match_pct: Optional[float] = None,
    skills: Optional[List[str]] = None,
) -> int:
    init_db()
    with get_connection() as conn:
        cur = conn.execute(
            """INSERT INTO resumes
               (name, source, resume_text, predicted_category, category_confidence,
                ats_score, job_match_pct, skills_json, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                name, source, resume_text, predicted_category, category_confidence,
                ats_score, job_match_pct, json.dumps(skills or []),
                datetime.utcnow().isoformat(),
            ),
        )
        return cur.lastrowid


def get_all_resumes() -> List[Dict]:
    init_db()
    with get_connection() as conn:
        rows = conn.execute("SELECT * FROM resumes ORDER BY created_at DESC").fetchall()
        return [dict(row) for row in rows]


def clear_all_resumes():
    init_db()
    with get_connection() as conn:
        conn.execute("DELETE FROM resumes")
