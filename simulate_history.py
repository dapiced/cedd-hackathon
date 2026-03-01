"""
DDEC — Simulation d'historique pour la démo au jury.
Insère 7 sessions fictives avec une trajectoire réaliste croissante.
Aucune dépendance externe : sqlite3 standard.
"""

import os
import random
import sqlite3
import sys
import uuid
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ddec.session_tracker import SessionTracker, DEFAULT_DB_PATH

USER_ID = "demo_user"

# (max_level, [niveaux des évaluations], heure_début, nb_messages)
SESSIONS_PLAN = [
    (0, [0, 0, 0, 0, 0],      20, 5),   # Jour 1 — Vert stable
    (1, [0, 0, 1, 1, 0],      21, 5),   # Jour 2 — Jaune ponctuel
    (1, [0, 1, 1, 1, 1],      20, 5),   # Jour 3 — Jaune persistant
    (2, [1, 1, 2, 2, 1],      22, 5),   # Jour 4 — Orange détecté
    (2, [1, 2, 2, 2, 2],      21, 6),   # Jour 5 — Orange dominant
    (3, [2, 2, 3, 3, 2],      23, 6),   # Jour 6 — Rouge détecté
    (3, [2, 3, 3, 3, 3],      22, 7),   # Jour 7 — Rouge dominant
]

SAMPLE_MESSAGES = {
    0: [
        "Bonne journée aujourd'hui !",
        "J'ai passé du temps avec mes amis.",
        "Ça va bien en ce moment.",
        "J'ai hâte au week-end avec ma famille.",
    ],
    1: [
        "Je suis un peu fatigué ces temps-ci.",
        "L'école c'est stressant des fois.",
        "J'ai du mal à dormir.",
        "Je me sens un peu seul parfois.",
    ],
    2: [
        "Je me sens vide parfois.",
        "Personne ne comprend vraiment ce que je vis.",
        "Tout est lourd ces temps-ci.",
        "Je pleure souvent sans raison.",
    ],
    3: [
        "À quoi ça sert de continuer comme ça.",
        "Je suis un fardeau pour tout le monde.",
        "Plus envie de rien du tout.",
        "J'ai juste envie de partir et disparaître.",
    ],
}

EMOJI = {0: "🟢", 1: "🟡", 2: "🟠", 3: "🔴"}
LABEL = {0: "Vert  ", 1: "Jaune ", 2: "Orange", 3: "Rouge "}


def main():
    # Effacer l'historique précédent de demo_user
    conn = sqlite3.connect(DEFAULT_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM alert_events WHERE user_id = 'demo_user'")
    cursor.execute("DELETE FROM sessions WHERE user_id = 'demo_user'")
    conn.commit()
    conn.close()
    print("  Historique précédent effacé.")

    # Initialiser le tracker (crée la DB si nécessaire)
    tracker = SessionTracker(db_path=DEFAULT_DB_PATH)

    print("=" * 62)
    print("  DDEC — Simulation d'historique (7 sessions / 7 jours)")
    print("=" * 62)
    print(f"  Utilisateur  : {USER_ID}")
    print(f"  Base SQLite  : {DEFAULT_DB_PATH}")
    print()

    base_date = datetime.now() - timedelta(days=7)
    inserted_sessions = []

    for day_idx, (max_level, alert_levels, hour, n_msgs) in enumerate(SESSIONS_PLAN):
        session_date = (base_date + timedelta(days=day_idx)).replace(
            hour=hour,
            minute=random.randint(0, 30),
            second=0,
            microsecond=0,
        )
        session_id = str(uuid.uuid4())
        ended_at = session_date + timedelta(minutes=random.randint(20, 55))

        # Insérer la session avec timestamps simulés
        with sqlite3.connect(DEFAULT_DB_PATH) as conn:
            conn.execute(
                """INSERT INTO sessions
                   (user_id, session_id, started_at, ended_at,
                    max_alert_level, message_count)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    USER_ID, session_id,
                    session_date.isoformat(), ended_at.isoformat(),
                    max_level, n_msgs,
                ),
            )
            for i, lvl in enumerate(alert_levels):
                msg_time = session_date + timedelta(minutes=i * random.randint(3, 8))
                conn.execute(
                    """INSERT INTO alert_events
                       (user_id, session_id, timestamp,
                        alert_level, confidence, trigger_message)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (
                        USER_ID, session_id,
                        msg_time.isoformat(),
                        lvl,
                        round(random.uniform(0.52, 0.95), 2),
                        random.choice(SAMPLE_MESSAGES[lvl]),
                    ),
                )
            conn.commit()

        inserted_sessions.append((session_date, max_level, n_msgs, session_id))
        print(
            f"  Jour {day_idx + 1}  [{session_date.strftime('%Y-%m-%d %Hh%M')}]  "
            f"{EMOJI[max_level]} {LABEL[max_level]} — {n_msgs} msgs"
            f"  (id: {session_id[:8]}…)"
        )

    print()
    print("-" * 62)
    print("  ANALYSE LONGITUDINALE")
    print("-" * 62)

    risk = tracker.get_longitudinal_risk(USER_ID)
    history = tracker.get_user_history(USER_ID)

    trend_labels = {
        "stable":    "→  Stable",
        "worsening": "↗  En hausse",
        "improving": "↘  En amélioration",
    }

    print(f"  Sessions analysées        : {risk['sessions_analyzed']}")
    print(f"  Score de risque           : {risk['risk_score']:.1%}")
    print(f"  Tendance                  : {trend_labels[risk['trend']]}")
    print(f"  Sessions hautes conséc.   : {risk['consecutive_high_sessions']}")
    print(f"  Recommandation            : ⚠️  {risk['recommendation']}")
    print()
    print("  Niveaux max par session :")
    for s in history:
        lvl = s["max_alert_level"]
        date_str = s["started_at"][:16].replace("T", " ")
        bar = "█" * (lvl + 1)
        print(f"    {date_str}   {EMOJI[lvl]} niveau {lvl}  {bar:<4}  ({s['message_count']} msgs)")

    print()
    print("  Simulation terminée.")
    print("  Relancez l'app Streamlit pour voir l'historique longitudinal.")
    print("=" * 62)


if __name__ == "__main__":
    main()
