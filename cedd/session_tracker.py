"""
CEDD Session Tracker
====================
Cross-session tracking to detect longitudinal emotional drift.
Uses SQLite (included in the Python standard library — no install required).

Suivi inter-sessions pour détecter les dérives longitudinales.
Utilise SQLite (inclus dans Python standard, aucune installation requise).
"""

import os
import sqlite3
import uuid
from datetime import datetime

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DB_PATH = os.path.join(_HERE, "data", "cedd_sessions.db")

# ── DDL statements ────────────────────────────────────────────────────────────

_CREATE_SESSIONS = """
CREATE TABLE IF NOT EXISTS sessions (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id          TEXT NOT NULL,
    session_id       TEXT NOT NULL,
    started_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ended_at         TIMESTAMP,
    max_alert_level  INTEGER DEFAULT 0,
    message_count    INTEGER DEFAULT 0
);
"""

_CREATE_ALERT_EVENTS = """
CREATE TABLE IF NOT EXISTS alert_events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id         TEXT NOT NULL,
    session_id      TEXT NOT NULL,
    timestamp       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    alert_level     INTEGER NOT NULL,
    confidence      FLOAT,
    trigger_message TEXT        -- truncated to 500 chars / tronqué à 500 caractères
);
"""


class SessionTracker:
    """
    Cross-session CEDD alert level tracker using SQLite.
    Suivi inter-sessions des niveaux d'alerte CEDD via SQLite.
    """

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self._init_db()

    # ── Database connection / Connexion ───────────────────────────────────────

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        """Create tables if they don't exist. / Crée les tables si nécessaire."""
        with self._connect() as conn:
            conn.execute(_CREATE_SESSIONS)
            conn.execute(_CREATE_ALERT_EVENTS)
            conn.commit()

    # ── Public API ────────────────────────────────────────────────────────────

    def start_session(self, user_id: str) -> str:
        """
        Start a new session and return its session_id (uuid4).
        Démarre une nouvelle session et retourne son session_id (uuid4).
        """
        session_id = str(uuid.uuid4())
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO sessions (user_id, session_id, started_at) VALUES (?, ?, ?)",
                (user_id, session_id, datetime.now().isoformat()),
            )
            conn.commit()
        return session_id

    def log_alert(
        self,
        user_id: str,
        session_id: str,
        alert_level: int,
        confidence: float,
        message: str,
    ):
        """
        Record a CEDD evaluation in alert_events.
        Enregistre une évaluation CEDD dans alert_events.
        """
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO alert_events
                   (user_id, session_id, timestamp, alert_level, confidence, trigger_message)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    user_id, session_id, datetime.now().isoformat(),
                    alert_level, confidence,
                    (message or "")[:500],
                ),
            )
            conn.commit()

    def end_session(
        self,
        user_id: str,
        session_id: str,
        max_level: int,
        message_count: int,
    ):
        """
        Close a session with its max alert level and message count.
        Clôture une session avec son niveau max et le nombre de messages.
        """
        with self._connect() as conn:
            conn.execute(
                """UPDATE sessions
                   SET ended_at = ?, max_alert_level = ?, message_count = ?
                   WHERE user_id = ? AND session_id = ?""",
                (
                    datetime.now().isoformat(), max_level, message_count,
                    user_id, session_id,
                ),
            )
            conn.commit()

    def get_user_history(self, user_id: str, last_n_sessions: int = 7) -> list:
        """
        Return the N most recent completed sessions (chronological order).
        Retourne les N dernières sessions (ordre chronologique croissant).
        """
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT session_id, started_at, ended_at,
                          max_alert_level, message_count
                   FROM sessions
                   WHERE user_id = ? AND ended_at IS NOT NULL
                   ORDER BY started_at DESC
                   LIMIT ?""",
                (user_id, last_n_sessions),
            ).fetchall()
        # Reverse to chronological order (oldest first) / Ordre chronologique
        return [dict(r) for r in reversed(rows)]

    def get_longitudinal_risk(self, user_id: str) -> dict:
        """
        Compute the longitudinal risk profile over the last 7 sessions.
        Calcule le profil de risque longitudinal sur les 7 dernières sessions.

        Returns:
            dict with / dict avec :
                risk_score (float 0-1): weighted average of max levels
                trend (str): "stable" | "worsening" | "improving"
                consecutive_high_sessions (int): consecutive sessions >= Orange
                recommendation (str): action suggested for a health professional
                sessions_analyzed (int): number of sessions considered
        """
        sessions = self.get_user_history(user_id, last_n_sessions=7)
        n = len(sessions)

        if n == 0:
            return {
                "risk_score": 0.0,
                "trend": "stable",
                "consecutive_high_sessions": 0,
                # Internal key — translated in the UI layer / Clé interne traduite dans l'UI
                "recommendation": "Suivi normal",
                "sessions_analyzed": 0,
            }

        levels = [s["max_alert_level"] for s in sessions]

        # Weighted average (recent sessions have higher weight)
        # Moyenne pondérée (sessions récentes = poids plus élevé)
        weights = list(range(1, n + 1))
        weighted_sum = sum(lvl * w for lvl, w in zip(levels, weights))
        risk_score = min(weighted_sum / (sum(weights) * 3), 1.0)

        # Consecutive high-level sessions from the most recent
        # Sessions consécutives haute (>= Orange) depuis la plus récente
        consecutive_high = 0
        for lvl in reversed(levels):
            if lvl >= 2:
                consecutive_high += 1
            else:
                break

        # Trend: average of last 3 vs previous 3
        # Tendance : moyenne des 3 dernières vs 3 précédentes
        if n >= 6:
            recent_avg = sum(levels[-3:]) / 3
            older_avg  = sum(levels[-6:-3]) / 3
            if recent_avg > older_avg + 0.3:
                trend = "worsening"
            elif recent_avg < older_avg - 0.3:
                trend = "improving"
            else:
                trend = "stable"
        elif n >= 2:
            if levels[-1] > levels[0]:
                trend = "worsening"
            elif levels[-1] < levels[0]:
                trend = "improving"
            else:
                trend = "stable"
        else:
            trend = "stable"

        # Recommendation (internal French key — translated in the UI layer)
        # Recommandation (clé interne en français, traduite dans l'interface)
        if consecutive_high >= 3 or risk_score > 0.8:
            recommendation = "Intervention prioritaire recommandée"
        elif risk_score > 0.6:
            recommendation = "Consultation professionnelle suggérée"
        elif risk_score > 0.3:
            recommendation = "Attention soutenue recommandée"
        else:
            recommendation = "Suivi normal"

        return {
            "risk_score": round(risk_score, 3),
            "trend": trend,
            "consecutive_high_sessions": consecutive_high,
            "recommendation": recommendation,
            "sessions_analyzed": n,
        }
