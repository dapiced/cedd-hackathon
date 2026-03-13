"""
CEDD — History Simulation / Simulation d'historique
====================================================
Inserts 7 fictitious sessions with a realistic worsening trajectory for jury demos.
Insère 7 sessions fictives avec une trajectoire réaliste croissante pour la démo au jury.

No external dependencies — standard sqlite3 only.
Aucune dépendance externe : sqlite3 standard.

Usage / Utilisation :
    python simulate_history.py           # French / Français (default)
    python simulate_history.py --lang en # English / Anglais
"""

import argparse
import os
import random
import sqlite3
import sys
import uuid
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cedd.session_tracker import SessionTracker, DEFAULT_DB_PATH

# Per-user session plans: user_name → list of (max_level, [alert_levels], start_hour, n_messages)
# Plans par utilisateur : nom → liste de (niveau_max, [niveaux_alertes], heure_début, nb_messages)
USERS_PLANS = {
    "Shuchita": [
        # Stable green — healthy user, occasional yellow / Vert stable — utilisatrice en bonne santé
        (0, [0, 0, 0, 0, 0], 18, 5),
        (0, [0, 0, 0, 0, 0], 19, 5),
        (0, [0, 0, 0, 0, 0], 20, 5),
        (1, [0, 0, 1, 0, 0], 18, 5),
        (0, [0, 0, 0, 0, 0], 21, 5),
        (0, [0, 0, 0, 0, 0], 19, 5),
        (0, [0, 0, 0, 0, 0], 20, 5),
    ],
    "Priyanka": [
        # Gradual improvement — started rough, getting better / Amélioration graduelle
        (2, [1, 2, 2, 1, 1], 20, 5),
        (2, [1, 1, 2, 1, 1], 21, 5),
        (1, [1, 1, 1, 1, 0], 19, 5),
        (1, [0, 1, 1, 0, 0], 20, 5),
        (0, [0, 0, 0, 1, 0], 18, 5),
        (0, [0, 0, 0, 0, 0], 19, 5),
        (0, [0, 0, 0, 0, 0], 20, 5),
    ],
    "Amanda": [
        # Fluctuating — up and down, no clear trend / Fluctuant — sans tendance nette
        (0, [0, 0, 0, 0, 0], 19, 5),
        (1, [0, 1, 1, 0, 0], 20, 5),
        (2, [1, 1, 2, 2, 1], 21, 5),
        (1, [1, 1, 0, 0, 0], 19, 5),
        (0, [0, 0, 0, 0, 0], 18, 5),
        (1, [0, 0, 1, 1, 0], 20, 5),
        (0, [0, 0, 0, 0, 0], 21, 5),
    ],
    "Dominic": [
        # Escalating — classic drift toward crisis (best for demo) / Escalade classique
        (0, [0, 0, 0, 0, 0], 20, 5),
        (0, [0, 0, 0, 0, 0], 21, 5),
        (1, [0, 0, 1, 1, 0], 20, 5),
        (1, [0, 1, 1, 1, 1], 22, 5),
        (2, [1, 1, 2, 2, 1], 21, 6),
        (2, [1, 2, 2, 2, 2], 23, 6),
        (3, [2, 3, 3, 3, 3], 22, 7),
    ],
    # Guest: no history generated / Invité : aucun historique généré
}

# Sample messages by level — bilingual / Messages d'exemple par niveau — bilingues
SAMPLE_MESSAGES = {
    "fr": {
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
    },
    "en": {
        0: [
            "Had a great day today!",
            "Spent some time with my friends.",
            "Things are going well right now.",
            "Looking forward to the weekend with my family.",
        ],
        1: [
            "I've been a bit tired lately.",
            "School is stressful sometimes.",
            "I'm having trouble sleeping.",
            "I feel a little alone sometimes.",
        ],
        2: [
            "I feel empty sometimes.",
            "Nobody really understands what I'm going through.",
            "Everything feels heavy lately.",
            "I cry often for no reason.",
        ],
        3: [
            "What's the point of going on like this.",
            "I'm a burden to everyone.",
            "I don't want to do anything anymore.",
            "I just want to leave and disappear.",
        ],
    },
}

EMOJI  = {0: "🟢", 1: "🟡", 2: "🟠", 3: "🔴"}
LABEL  = {
    "fr": {0: "Vert  ", 1: "Jaune ", 2: "Orange", 3: "Rouge "},
    "en": {0: "Green ", 1: "Yellow", 2: "Orange", 3: "Red   "},
}


def main():
    parser = argparse.ArgumentParser(
        description="Simulate CEDD session history for demo. / "
                    "Simule un historique de sessions CEDD pour la démo."
    )
    parser.add_argument(
        "--lang",
        choices=["fr", "en"],
        default="fr",
        help="Language for sample messages (default: fr). / Langue des messages (défaut : fr).",
    )
    args = parser.parse_args()
    lang = args.lang
    messages_by_level = SAMPLE_MESSAGES[lang]
    label_by_level    = LABEL[lang]

    # Clear previous history for all demo users / Effacer l'historique de tous les utilisateurs démo
    all_users = list(USERS_PLANS.keys()) + ["Guest", "demo_user"]
    conn   = sqlite3.connect(DEFAULT_DB_PATH)
    cursor = conn.cursor()
    for uid in all_users:
        cursor.execute("DELETE FROM alert_events WHERE user_id = ?", (uid,))
        cursor.execute("DELETE FROM sessions WHERE user_id = ?", (uid,))
    conn.commit()
    conn.close()
    print("  Previous history cleared for all profiles. / Historique effacé pour tous les profils.")

    # Init tracker (creates DB if needed) / Initialiser le tracker
    tracker = SessionTracker(db_path=DEFAULT_DB_PATH)

    print("=" * 62)
    print("  CEDD — Session history simulation / Simulation d'historique")
    print(f"  {len(USERS_PLANS)} users × 7 sessions  [{lang.upper()}]")
    print("=" * 62)
    print(f"  DB / Base SQLite : {DEFAULT_DB_PATH}")
    print()

    base_date = datetime.now() - timedelta(days=7)

    for user_id, sessions_plan in USERS_PLANS.items():
        print(f"  ── {user_id} {'─' * (50 - len(user_id))}")

        for day_idx, (max_level, alert_levels, hour, n_msgs) in enumerate(sessions_plan):
            session_date = (base_date + timedelta(days=day_idx)).replace(
                hour=hour,
                minute=random.randint(0, 30),
                second=0,
                microsecond=0,
            )
            session_id = str(uuid.uuid4())
            ended_at   = session_date + timedelta(minutes=random.randint(20, 55))

            # Insert session with simulated timestamps / Insérer la session avec timestamps simulés
            with sqlite3.connect(DEFAULT_DB_PATH) as conn:
                conn.execute(
                    """INSERT INTO sessions
                       (user_id, session_id, started_at, ended_at,
                        max_alert_level, message_count)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (
                        user_id, session_id,
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
                            user_id, session_id,
                            msg_time.isoformat(),
                            lvl,
                            round(random.uniform(0.52, 0.95), 2),
                            random.choice(messages_by_level[lvl]),
                        ),
                    )
                conn.commit()

            print(
                f"    Day/Jour {day_idx + 1}  [{session_date.strftime('%Y-%m-%d %Hh%M')}]  "
                f"{EMOJI[max_level]} {label_by_level[max_level]} — {n_msgs} msgs"
                f"  (id: {session_id[:8]}…)"
            )

        print()

    print("-" * 62)
    print("  LONGITUDINAL ANALYSIS / ANALYSE LONGITUDINALE")
    print("-" * 62)

    trend_labels = {
        "fr": {
            "stable":    "→  Stable",
            "worsening": "↗  En hausse",
            "improving": "↘  En amélioration",
        },
        "en": {
            "stable":    "→  Stable",
            "worsening": "↗  Worsening",
            "improving": "↘  Improving",
        },
    }

    labels = {
        "fr": {
            "sessions":     "Sessions analysées",
            "risk_score":   "Score de risque",
            "trend":        "Tendance",
            "consec":       "Sessions hautes consécutives",
            "rec":          "Recommandation",
            "max_by_sess":  "Niveaux max par session",
        },
        "en": {
            "sessions":     "Sessions analyzed",
            "risk_score":   "Risk score",
            "trend":        "Trend",
            "consec":       "Consecutive high sessions",
            "rec":          "Recommendation",
            "max_by_sess":  "Max levels per session",
        },
    }
    L = labels[lang]

    # Show analysis for each user / Afficher l'analyse pour chaque utilisateur
    for user_id in USERS_PLANS:
        risk    = tracker.get_longitudinal_risk(user_id)
        history = tracker.get_user_history(user_id)

        print(f"\n  ── {user_id} {'─' * (50 - len(user_id))}")
        print(f"  {L['sessions']:30s}: {risk['sessions_analyzed']}")
        print(f"  {L['risk_score']:30s}: {risk['risk_score']:.1%}")
        print(f"  {L['trend']:30s}: {trend_labels[lang][risk['trend']]}")
        print(f"  {L['consec']:30s}: {risk['consecutive_high_sessions']}")
        print(f"  {L['rec']:30s}: ⚠️  {risk['recommendation']}")
        print()
        print(f"  {L['max_by_sess']} :")
        for s in history:
            lvl      = s["max_alert_level"]
            date_str = s["started_at"][:16].replace("T", " ")
            bar      = "█" * (lvl + 1)
            print(f"    {date_str}   {EMOJI[lvl]} level/niveau {lvl}  {bar:<4}  ({s['message_count']} msgs)")

    print()
    if lang == "fr":
        print("  Simulation terminée.")
        print("  Relancez l'app Streamlit pour voir l'historique longitudinal.")
    else:
        print("  Simulation complete.")
        print("  Restart the Streamlit app to see the longitudinal history.")
    print("=" * 62)


if __name__ == "__main__":
    main()
