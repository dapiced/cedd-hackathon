"""
CEDD — Streamlit Application
Bilingual demo interface: chat + real-time monitoring dashboard.
Interface de démonstration bilingue : chat + dashboard de surveillance en temps réel.
"""

import os
import sys
import json
import time
from datetime import datetime
import streamlit as st
import plotly.graph_objects as go

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cedd.classifier import CEDDClassifier, LEVEL_LABELS
from cedd.response_modulator import (
    get_llm_response,
    get_level_description,
    get_system_prompt,
    get_handoff_description,
)
from cedd.session_tracker import SessionTracker

# ─── Configuration ─────────────────────────────────────────────────────────────
MODEL_PATH = "models/cedd_model.joblib"

LEVEL_COLORS = {0: "#2ecc71", 1: "#f1c40f", 2: "#e67e22", 3: "#e74c3c"}

# ─── Themes ─────────────────────────────────────────────────────────────────────
THEMES = {
    "light": {
        "bg_main":    "#c8d6e5",
        "bg_card":    "#dbe6f0",
        "bg_input":   "#eaf1f8",
        "bg_chat":    "#dbe6f0",
        "text_main":  "#0d1b2a",
        "text_muted": "#000000",
        "border":     "#9bb5cc",
        "chat_user":  "#a8d5b5",
        "chat_bot":   "#ccdae6",
        "pill_bg":          "#b8d4e8",
        "pill_bord":        "#7aaec8",
        "pill_text":        "#0d3a5c",
        "btn_primary_bg":   "#2e86c1",
        "btn_primary_text": "#ffffff",
    },
    "dark": {
        "bg_main":    "#0e1117",
        "bg_card":    "#1a1d27",
        "bg_input":   "#262b38",
        "bg_chat":    "#161922",
        "text_main":  "#e2e8f0",
        "text_muted": "#ffffff",
        "border":     "#c0c8d8",
        "chat_user":  "#1a3d2b",
        "chat_bot":   "#1e2130",
        "pill_bg":          "#1e2340",
        "pill_bord":        "#3b4680",
        "pill_text":        "#818cf8",
        "btn_primary_bg":   "#3b82f6",
        "btn_primary_text": "#ffffff",
    },
}


def get_theme_css(theme: str) -> str:
    t = THEMES[theme]
    return f"""
<style>
    .stApp, [data-testid="stAppViewContainer"],
    .main .block-container {{
        background-color: {t['bg_main']} !important;
    }}
    [data-testid="stMain"] h1,
    [data-testid="stMain"] h2,
    [data-testid="stMain"] h3,
    [data-testid="stMain"] h4 {{
        color: {t['text_main']} !important;
    }}
    [data-testid="stMain"] p,
    [data-testid="stMain"] li,
    [data-testid="stMain"] label {{
        color: {t['text_main']} !important;
    }}
    hr {{ border-color: {t['border']} !important; opacity: 0.6; }}
    [data-baseweb="input"] {{
        border: 1px solid {t['border']} !important;
        border-radius: 20px !important;
        background-color: {t['bg_input']} !important;
        outline: none !important;
        box-shadow: none !important;
    }}
    [data-baseweb="input"] input {{
        background-color: {t['bg_input']} !important;
        color: {t['text_main']} !important;
        border: none !important;
        outline: none !important;
        box-shadow: none !important;
        caret-color: {t['text_main']} !important;
    }}
    [data-testid="metric-container"] {{
        background-color: {t['bg_card']} !important;
        border: 1px solid {t['border']} !important;
        border-radius: 8px;
    }}
    [data-testid="stMetricValue"], [data-testid="stMetricLabel"] {{
        color: {t['text_main']} !important;
    }}
    [data-testid="stExpander"] {{
        background-color: {t['bg_card']} !important;
        border-color: {t['border']} !important;
    }}
    [data-testid="stExpanderDetails"] {{
        background-color: {t['bg_card']} !important;
    }}
    [data-testid="stExpander"] [data-testid="stIconMaterial"] {{
        color: {t['text_main']} !important;
    }}
    .stButton > button[kind="secondary"],
    [data-testid="stBaseButton-secondary"] {{
        background-color: {t['bg_card']} !important;
        color: {t['text_main']} !important;
        border-color: {t['border']} !important;
        font-size: 0.75rem !important;
    }}
    .stButton > button[kind="secondary"]:hover,
    [data-testid="stBaseButton-secondary"]:hover {{
        border-color: {t['text_muted']} !important;
    }}
    [data-testid="stFormSubmitButton"] > button,
    .stButton > button[kind="primary"],
    [data-testid="stBaseButton-primary"] {{
        background-color: {t['btn_primary_bg']} !important;
        color: {t['btn_primary_text']} !important;
        border-color: {t['btn_primary_bg']} !important;
    }}
    [data-testid="stFormSubmitButton"] > button:hover,
    .stButton > button[kind="primary"]:hover,
    [data-testid="stBaseButton-primary"]:hover {{
        filter: brightness(1.1);
    }}
    [data-testid="stMain"] [data-testid="stCaptionContainer"] p {{
        color: {t['text_muted']} !important;
    }}
    [data-testid="stMain"] [data-testid="stAlert"] {{
        background-color: {t['bg_card']} !important;
        border-color: {t['border']} !important;
    }}
    [data-testid="stMain"] [data-testid="stAlert"] p {{
        color: {t['text_main']} !important;
    }}
    [data-testid="stForm"],
    .stForm,
    [data-testid="stForm"] > div {{
        border: 1px solid {t['border']} !important;
        outline: none !important;
        box-shadow: none !important;
        border-radius: 12px !important;
        background: {t['bg_card']} !important;
        padding: 8px !important;
    }}
    pre, code {{
        background-color: {t['bg_input']} !important;
        color: {t['text_main']} !important;
        border-color: {t['border']} !important;
    }}
    [data-testid="stProgressBar"] > div {{
        background-color: {t['border']} !important;
    }}
    .chat-bubble-user {{
        background-color: {t['chat_user']} !important;
        color: {t['text_main']} !important;
    }}
    .chat-bubble-assistant {{
        background-color: {t['chat_bot']} !important;
        color: {t['text_main']} !important;
    }}
    .chat-container {{
        background: {t['bg_chat']} !important;
        border: 1px solid {t['border']} !important;
    }}
    .feature-pill {{
        background: {t['pill_bg']} !important;
        border-color: {t['pill_bord']} !important;
        color: {t['pill_text']} !important;
    }}
    .metric-card {{
        background: {t['bg_card']} !important;
        border-color: {t['border']} !important;
    }}
    [data-baseweb="select"] > div {{
        background-color: {t['bg_card']} !important;
        border-color: {t['border']} !important;
        color: {t['text_main']} !important;
    }}
    [data-baseweb="select"] [data-testid="stMarkdownContainer"] p,
    [data-baseweb="select"] span {{
        color: {t['text_main']} !important;
    }}
    [data-baseweb="select"] svg {{
        fill: {t['text_main']} !important;
    }}
    [data-baseweb="popover"] {{
        background-color: {t['bg_card']} !important;
    }}
    [data-baseweb="popover"] li {{
        background-color: {t['bg_card']} !important;
        color: {t['text_main']} !important;
    }}
    [data-baseweb="popover"] li:hover {{
        background-color: {t['bg_input']} !important;
    }}
</style>
"""

LLM_SOURCE_INDICATOR = {
    "groq":              ("🟠", "#f97316"),
    "gemini-flash":      ("💎", "#4285f4"),
    "claude-haiku":      ("🟣", "#7c3aed"),
    "fallback-statique": ("⚠️", "#f59e0b"),
}
LLM_DISPLAY_NAMES = {
    "groq": "Groq Llama 3.3 70B",
    "gemini-flash": "Gemini 2.5 Flash",
    "claude-haiku": "Claude Haiku",
}
LEVEL_EMOJIS = {0: "🟢", 1: "🟡", 2: "🟠", 3: "🔴"}
DEMO_USERS = ["Shuchita", "Priyanka", "Amanda", "Dominic", "Guest"]

# ─── Demo autopilot scenarios / Scénarios de démo automatisés ─────────────────
DEMO_SCENARIOS = {
    "fr": [
        "Salut! Ça fait longtemps que j'ai pas parlé à quelqu'un de juste... neutre, tsé? L'école recommence pis là c'est déjà intense. T'as des trucs pour gérer le stress des examens? Parce que je stresse déjà pour la session pis on est même pas rendus aux mi-sessions.",
        "Ouais c'est ça, j'essaie de faire des listes. Ça aide un peu. Hier j'ai quand même réussi à finir mon travail de bio pis à voir mes chums le soir, faque c'était correct. Mon chum Mathieu m'a montré une technique de respiration, je sais pas si ça marche vraiment mais on a ri en la faisant haha. C'est quoi les techniques que tu recommandes en général?",
        "Ouais, les pauses ça marche bien quand j'y pense. Le truc c'est que des fois je pense même pas à en prendre. Je suis genre scotché à mon bureau pis le temps passe. Mais bon, cette semaine ça devrait mieux aller, j'ai moins de cours jeudi-vendredi. Mathieu pis moi on pense aller au cinéma si on a le temps.",
        "Mmh ouais. Cette semaine a été rough. J'ai pas dormi super bien, genre je me réveille à 3h du mat pis j'arrive pu à me rendormir. C'est probablement le stress. J'ai annulé le cinéma avec Mathieu, j'avais trop de trucs à faire.",
        "C'est correct, je gère. C'est juste que des fois j'ai l'impression que tout le monde avance pis moi je tourne en rond. Mathieu il a l'air de trouver ça facile lui. Je sais pas trop. Anyway.",
        "Honnêtement j'sais plus trop. Chez nous c'est tendu depuis quelques semaines, mes parents sont pas dans leur meilleure forme. J'essaie de pas ramener mes affaires là-dedans mais c'est difficile. J'ai l'impression d'avoir nulle part où décompresser vraiment.",
        "Ouais je sais. C'est juste que je me sens fatigué de tout, pas juste de l'école. Comme... fatigué de tenir. Je dis ça pis ça a l'air dramatique mais c'est vraiment comme ça que je me sens ces temps-ci.",
        "Je sais pas. Peut-être. J'ai même plus vraiment envie de sortir. Mes chums me textent pis j'réponds même pu vraiment. C'est trop.",
        "C'est correct. De toute façon.",
    ],
    "en": [
        "Hey, just wanted to talk to someone tonight. Starting university is kind of a lot, honestly? Like it's exciting but also way more intense than I expected. How do people usually handle the adjustment? I feel like everyone else already has their thing figured out.",
        "That's actually helpful. I've been trying to go to the study sessions in my building. Made a couple of friends already — Jordan's been showing me around campus which is great. I had a rough week with assignments piling up but I got through it. Any tips for staying on top of things without burning out?",
        "Yeah the breaks thing makes sense. I keep telling myself I'll do that but then I just end up at my desk for like five hours straight. It's fine, I'm managing. Jordan and I were supposed to hang out this weekend, looking forward to that at least.",
        "Weekend ended up being kind of whatever. Jordan cancelled last minute, it's fine. Been tired lately, not sleeping great. I keep waking up in the middle of the night thinking about stuff. I don't know, probably just adjusting still.",
        "I'm okay. Just feel kind of behind on everything. Like everyone else knows what they're doing and I'm still figuring out how to do laundry. It sounds dumb when I say it out loud. Whatever.",
        "Honestly, I've just been staying in my room more. Home is kind of far so I can't just go back for the weekend. My roommate has his own friends and I don't really fit in their group. It just feels like there's nowhere to recharge, you know?",
        "Yeah, I know. It's just exhausting. Not like tired-exhausted, more like... I don't know how to explain it. Tired of trying to seem okay I guess. I haven't really talked to anyone back home in a while either.",
        "I don't know, maybe. I used to like being around people but now it just takes too much energy. Jordan texted me twice this week. Didn't answer. It's easier that way.",
        "It's fine. I'll figure it out.",
    ],
}

# ─── About CEDD content / Contenu À propos de CEDD ───────────────────────────
ABOUT_CEDD = {
    "fr": """**CEDD** (Conversational Emotional Drift Detection) est une couche de sécurité en temps réel pour les chatbots de santé mentale jeunesse (16-22 ans).

**Comment ça fonctionne :**
- Analyse **67 caractéristiques** par conversation (lexicales, sémantiques, comportementales)
- Surveille la **trajectoire** émotionnelle — pas juste un message, mais l'évolution complète
- **6 portes de sécurité** : les mots-clés de crise surpassent toujours le ML
- **Transfert accompagné** en 5 étapes vers Jeunesse, J'écoute au niveau Rouge

**Ce que vous voyez à droite :**
- 🎯 **Jauge** : niveau d'alerte actuel (Vert → Rouge)
- 📊 **Probabilités** : confiance du modèle par classe
- ⚡ **Signaux** : caractéristiques dominantes qui influencent le niveau
- 📈 **Historique** : évolution du niveau au fil des messages

**Philosophie :** Les faux positifs (sur-alerter) sont toujours préférables aux faux négatifs (manquer une crise).""",
    "en": """**CEDD** (Conversational Emotional Drift Detection) is a real-time safety layer for youth mental health chatbots (ages 16-22).

**How it works:**
- Analyzes **67 features** per conversation (lexical, semantic, behavioral)
- Monitors the emotional **trajectory** — not just one message, but the full evolution
- **6 safety gates**: crisis keywords always override ML predictions
- **5-step warm handoff** to Kids Help Phone at Red level

**What you see on the right:**
- 🎯 **Gauge**: current alert level (Green → Red)
- 📊 **Probabilities**: model confidence per class
- ⚡ **Signals**: dominant features driving the alert level
- 📈 **History**: alert level evolution across messages

**Philosophy:** False positives (over-alerting) are always preferable to false negatives (missing a crisis).""",
}

# ─── Bilingual UI strings / Chaînes d'interface bilingues ─────────────────────
STRINGS = {
    "fr": {
        "lang_btn":            "🇬🇧 English",
        "page_title":          "CEDD - Détection de dérive émotionnelle",
        "app_title":           "🧠 CEDD — Détection de dérive émotionnelle conversationnelle",
        "app_subtitle":        "Hackathon Mila · Sécurité IA en santé mentale des jeunes · Équipe 404HarmNotFound",
        "theme_btn":           "🌙 Sombre",
        "reset_btn":           "🔄 Réinitialiser",
        "chat_header":         "### 💬 Conversation",
        "chat_empty":          "Commence la conversation...",
        "welcome_title":       "Bienvenue sur CEDD",
        "welcome_text":        "Un système de sécurité en temps réel qui surveille la trajectoire émotionnelle de ta conversation — pas juste un message, mais l'évolution complète.",
        "welcome_cta":         "Écris ton premier message ci-dessous ⬇️",
        "input_placeholder":   "Écris ton message ici et appuie sur Entrée",
        "send_btn":            "Envoyer ➤",
        "dashboard_header":    "### 📊 Dashboard CEDD",
        "confidence":          "**Confiance**",
        "proba_header":        "**Probabilités par classe**",
        "signals_header":      "**Signaux actifs**",
        "signals_waiting":     "En attente d'analyse...",
        "history_header":      "**Évolution du niveau**",
        "history_waiting":     "Historique disponible après 2 messages.",
        "longitudinal_header": "### 📊 Historique longitudinal",
        "longitudinal_empty":  "Aucun historique — complétez des sessions pour voir la tendance.",
        "trend_stable":        "→ Stable",
        "trend_worsening":     "↗ En hausse",
        "trend_improving":     "↘ En amélioration",
        "llm_header":          "**LLM conversationnel**",
        "llm_last_call":       "Dernier appel :",
        "mode_header":         "**Mode de réponse actif**",
        "prompt_expander":     "Voir le prompt système complet",
        "stats_header":        "**Statistiques de session**",
        "stat_messages":       "Messages",
        "stat_exchanges":      "Échanges",
        "stat_peak":           "Pic alerte",
        "sessions_caption":    "Sessions analysées : {n}  •  Score longitudinal : {score:.0%}",
        "model_not_found":     "Modèle introuvable : {path}. Lancez d'abord `python train.py`.",
        "gauge_title":         "Niveau d'alerte",
        "level_labels":        {0: "VERT", 1: "JAUNE", 2: "ORANGE", 3: "ROUGE"},
        "gauge_ticks":         ["Vert", "Jaune", "Orange", "Rouge"],
        "proba_names": {
            "green": "Verte", "yellow": "Jaune", "orange": "Orange", "red": "Rouge"
        },
        # Internal recommendation keys from session_tracker (always French)
        # → mapped here for display / Clés internes → affichage traduit
        "rec_normal":       "Suivi normal",
        "rec_attention":    "Attention soutenue recommandée",
        "rec_consultation": "Consultation professionnelle suggérée",
        "rec_intervention": "Intervention prioritaire recommandée",
        "llm_fallback":     "Sans LLM",
        "handoff_title":      "Transfert accompagné",
        "handoff_step_label": "Étape {step}/5 : <b>{desc}</b>",
        "withdrawal_banner":  "Bon retour. Ça fait un moment — comment tu te sens ?",
        "withdrawal_badge":   "Retour après absence",
        "feature_chart_title": "🔍 Signaux détectés",
        "feature_chart_note":  "Score composite = importance du modèle × valeur normalisée. Les barres montrent ce qui influence le plus le niveau d'alerte actuel.",
        "profile_label":       "Profil",
        "demo_btn":            "▶️ Démo",
        "demo_stop_btn":       "⏹️ Arrêter",
        "demo_character_fr":   "Félix, 18 ans, CÉGEP",
        "demo_character_en":   "Alex, 19, université",
        "demo_running":        "Démo en cours — message {n}/9",
        "about_btn":           "ℹ️ À propos",
        "about_title":         "À propos de CEDD",
        "export_btn":          "📥 Exporter",
        "alert_toast_up":      "⚠️ Niveau d'alerte augmenté : {emoji} {label}",
        "compare_btn":         "🔀 Comparer",
        "compare_btn_off":     "🔀 Mode normal",
        "compare_left_header": "### 💬 Sans CEDD",
        "compare_left_sub":    "LLM brut — aucune instruction de sécurité",
        "compare_right_header":"### 🧠 Avec CEDD",
        "compare_right_sub":   "LLM guidé par les instructions CEDD adaptatives",
    },
    "en": {
        "lang_btn":            "🇫🇷 Français",
        "page_title":          "CEDD - Conversational Emotional Drift Detection",
        "app_title":           "🧠 CEDD — Conversational Emotional Drift Detection",
        "app_subtitle":        "Mila Hackathon · AI Safety in Youth Mental Health · Team 404HarmNotFound",
        "theme_btn":           "🌙 Dark",
        "reset_btn":           "🔄 Reset",
        "chat_header":         "### 💬 Conversation",
        "chat_empty":          "Start the conversation...",
        "welcome_title":       "Welcome to CEDD",
        "welcome_text":        "A real-time safety layer that monitors the emotional trajectory of your conversation — not just one message, but the full evolution.",
        "welcome_cta":         "Type your first message below ⬇️",
        "input_placeholder":   "Type your message here and press Enter",
        "send_btn":            "Send ➤",
        "dashboard_header":    "### 📊 CEDD Dashboard",
        "confidence":          "**Confidence**",
        "proba_header":        "**Class probabilities**",
        "signals_header":      "**Active signals**",
        "signals_waiting":     "Waiting for analysis...",
        "history_header":      "**Alert level history**",
        "history_waiting":     "History available after 2 messages.",
        "longitudinal_header": "### 📊 Longitudinal history",
        "longitudinal_empty":  "No history yet — complete sessions to see the trend.",
        "trend_stable":        "→ Stable",
        "trend_worsening":     "↗ Worsening",
        "trend_improving":     "↘ Improving",
        "llm_header":          "**Conversational LLM**",
        "llm_last_call":       "Last call:",
        "mode_header":         "**Active response mode**",
        "prompt_expander":     "View full system prompt",
        "stats_header":        "**Session statistics**",
        "stat_messages":       "Messages",
        "stat_exchanges":      "Exchanges",
        "stat_peak":           "Alert peak",
        "sessions_caption":    "Sessions analyzed: {n}  •  Longitudinal score: {score:.0%}",
        "model_not_found":     "Model not found: {path}. Run `python train.py` first.",
        "gauge_title":         "Alert level",
        "level_labels":        {0: "GREEN", 1: "YELLOW", 2: "ORANGE", 3: "RED"},
        "gauge_ticks":         ["Green", "Yellow", "Orange", "Red"],
        "proba_names": {
            "green": "Green", "yellow": "Yellow", "orange": "Orange", "red": "Red"
        },
        "rec_normal":       "Normal monitoring",
        "rec_attention":    "Sustained attention recommended",
        "rec_consultation": "Professional consultation suggested",
        "rec_intervention": "Priority intervention recommended",
        "llm_fallback":     "Without LLM",
        "handoff_title":      "Warm Handoff",
        "handoff_step_label": "Step {step}/5: <b>{desc}</b>",
        "withdrawal_banner":  "Welcome back. It's been a while — how are you feeling?",
        "withdrawal_badge":   "Returned after absence",
        "feature_chart_title": "🔍 Detected signals",
        "feature_chart_note":  "Composite score = model importance × scaled value. Bars show what drives the current alert level most.",
        "profile_label":       "Profile",
        "demo_btn":            "▶️ Demo",
        "demo_stop_btn":       "⏹️ Stop",
        "demo_character_fr":   "Félix, 18, CÉGEP",
        "demo_character_en":   "Alex, 19, university",
        "demo_running":        "Demo running — message {n}/9",
        "about_btn":           "ℹ️ About",
        "about_title":         "About CEDD",
        "export_btn":          "📥 Export",
        "alert_toast_up":      "⚠️ Alert level increased: {emoji} {label}",
        "compare_btn":         "🔀 Compare",
        "compare_btn_off":     "🔀 Single mode",
        "compare_left_header": "### 💬 Without CEDD",
        "compare_left_sub":    "Raw LLM — no safety instructions",
        "compare_right_header":"### 🧠 With CEDD",
        "compare_right_sub":   "LLM guided by CEDD adaptive instructions",
    },
}

# Maps English recommendation strings (from session_tracker) to STRINGS keys
# Mappe les recommandations anglaises (session_tracker) vers les clés STRINGS
_REC_KEY_MAP = {
    "Normal monitoring":                   "rec_normal",
    "Sustained attention recommended":     "rec_attention",
    "Professional consultation suggested": "rec_consultation",
    "Priority intervention recommended":   "rec_intervention",
}

# ─── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CEDD",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Static CSS (layout only — colors handled by get_theme_css) ─────────────────
st.markdown("""
<style>
    .main { padding-top: 1rem; }
    .chat-bubble-user {
        border-radius: 18px 18px 4px 18px;
        padding: 10px 14px;
        margin: 6px 0 6px 40px;
        max-width: 85%;
        float: right;
        clear: both;
        font-size: 0.95rem;
    }
    .chat-bubble-assistant {
        border-radius: 18px 18px 18px 4px;
        padding: 10px 14px;
        margin: 6px 40px 6px 0;
        max-width: 85%;
        float: left;
        clear: both;
        font-size: 0.95rem;
    }
    .chat-container {
        overflow-y: auto;
        max-height: 420px;
        padding: 10px;
        border-radius: 12px;
        margin-bottom: 12px;
    }
    .chat-time-user {
        font-size: 0.68rem;
        opacity: 0.5;
        text-align: right;
        margin: -2px 0 4px 0;
        clear: both;
    }
    .chat-time-assistant {
        font-size: 0.68rem;
        opacity: 0.5;
        text-align: left;
        margin: -2px 0 4px 0;
        clear: both;
    }
    .llm-badge {
        font-size: 0.68rem;
        opacity: 0.7;
        display: block;
        margin-top: 2px;
    }
    .alert-dot {
        font-size: 0.68rem;
        display: inline-block;
        clear: both;
        float: left;
        margin: 2px 0 6px 0;
    }
    .clearfix::after { content: ""; display: table; clear: both; }
    .alert-badge {
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 1.1rem;
        display: inline-block;
    }
    .metric-card {
        border: 1px solid;
        border-radius: 10px;
        padding: 12px;
        margin: 6px 0;
    }
    .feature-pill {
        border: 1px solid;
        border-radius: 12px;
        padding: 4px 10px;
        font-size: 0.82rem;
        display: inline-block;
        margin: 2px;
    }
    @keyframes alert-flash {
        0%   { opacity: 0; transform: translateY(-10px); }
        15%  { opacity: 1; transform: translateY(0); }
        85%  { opacity: 1; transform: translateY(0); }
        100% { opacity: 0; transform: translateY(-10px); }
    }
    .alert-toast {
        animation: alert-flash 3s ease-in-out forwards;
        position: fixed;
        top: 60px;
        left: 50%;
        transform: translateX(-50%);
        z-index: 9999;
        padding: 10px 24px;
        border-radius: 24px;
        font-weight: 700;
        font-size: 0.95rem;
        box-shadow: 0 4px 16px rgba(0,0,0,0.2);
        pointer-events: none;
    }
    h1 { font-size: 1.4rem !important; }
    h3 { font-size: 1.05rem !important; margin-bottom: 0.4rem !important; }
</style>
""", unsafe_allow_html=True)


# ─── Model & tracker loading ────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model not found / Modèle introuvable : {MODEL_PATH}. Run `python train.py`.")
        st.stop()
    return CEDDClassifier.load(MODEL_PATH)


@st.cache_resource
def load_tracker():
    return SessionTracker()


# ─── Session state initialisation ───────────────────────────────────────────────
def init_state():
    defaults = {
        "messages":      [],
        "alert_history": [],
        "current_alert": {
            "level": 0, "label": "green", "confidence": 0.0,
            "dominant_features": [], "probabilities": {},
        },
        "selected_llm":    "groq",
        "last_llm_source": None,
        "input_key":       0,
        "user_id":         "Guest",
        "session_id":      None,
        "lang":            "en",   # default language / langue par défaut
        "theme":           "light",
        "handoff_step":    0,      # 0 = not in handoff, 1-5 = warm handoff steps
        "withdrawal_detected": False,  # True if user returned after extended absence
        "demo_running":   False,    # True while demo autopilot is active
        "demo_step":      0,        # Current demo message index (0-8)
        "compare_mode":   False,    # True = side-by-side compare mode
        "compare_messages": [],     # "Without CEDD" message list (left side)
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def reset_conversation():
    st.session_state.messages      = []
    st.session_state.alert_history = []
    st.session_state.current_alert = {
        "level": 0, "label": "green", "confidence": 0.0,
        "dominant_features": [], "probabilities": {},
    }
    st.session_state.input_key += 1
    st.session_state.handoff_step = 0
    st.session_state.withdrawal_detected = False
    st.session_state.demo_running = False
    st.session_state.demo_step = 0
    st.session_state.compare_mode = False
    st.session_state.compare_messages = []


# ─── UI components / Composants UI ──────────────────────────────────────────────

def render_chat(S: dict, theme: str = "light", messages: list = None):
    """Display chat bubbles. / Affiche les bulles de conversation."""
    t = THEMES[theme]
    if messages is None:
        messages = st.session_state.messages
    msgs_html = '<div class="chat-container"><div class="clearfix">'
    if not messages:
        msgs_html += (
            f'<div style="text-align:center;margin:30px 16px;">'
            f'<div style="background:{t["bg_card"]};border:1px solid {t["border"]};'
            f'border-radius:14px;padding:24px 20px;display:inline-block;max-width:380px;">'
            f'<div style="font-size:2rem;margin-bottom:6px;">🧠</div>'
            f'<div style="font-size:1.1rem;font-weight:700;color:{t["text_main"]};margin-bottom:8px;">'
            f'{S["welcome_title"]}</div>'
            f'<div style="font-size:0.88rem;color:{t["text_main"]};opacity:0.85;margin-bottom:12px;">'
            f'{S["welcome_text"]}</div>'
            f'<div style="font-size:0.8rem;color:{t["text_muted"]};opacity:0.7;">'
            f'{S["welcome_cta"]}</div>'
            f'</div></div>'
        )
    else:
        for msg in messages:
            role = msg["role"]
            content = (
                msg["content"]
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace("\n", "<br>")
            )
            ts = msg.get("timestamp", "")
            if role == "user":
                msgs_html += f'<div class="chat-bubble-user">{content}</div>'
                if ts:
                    msgs_html += f'<div class="chat-time-user" style="color:{t["text_muted"]};">{ts}</div>'
            else:
                # Assistant bubble + optional LLM badge
                bubble = content
                source = msg.get("source")
                if source and source in LLM_SOURCE_INDICATOR:
                    src_emoji, src_color = LLM_SOURCE_INDICATOR[source]
                    src_name = LLM_DISPLAY_NAMES.get(source, source)
                    bubble += f'<span class="llm-badge" style="color:{src_color};">{src_emoji} {src_name}</span>'
                msgs_html += f'<div class="chat-bubble-assistant">{bubble}</div>'
                # Timestamp + alert dot row
                meta_parts = []
                alert_lvl = msg.get("alert_level")
                if alert_lvl is not None:
                    a_color = LEVEL_COLORS[alert_lvl]
                    a_emoji = LEVEL_EMOJIS[alert_lvl]
                    a_label = LEVEL_LABELS[alert_lvl]
                    meta_parts.append(
                        f'<span class="alert-dot" style="color:{a_color};">{a_emoji} {a_label.capitalize()}</span>'
                    )
                if ts:
                    meta_parts.append(f'<span style="opacity:0.5;">{ts}</span>')
                if meta_parts:
                    msgs_html += (
                        f'<div class="chat-time-assistant" style="color:{t["text_muted"]};">'
                        f'{" &nbsp;·&nbsp; ".join(meta_parts)}</div>'
                    )
    msgs_html += '</div></div>'
    st.markdown(msgs_html, unsafe_allow_html=True)


def render_gauge(level: int, confidence: float, S: dict, theme: str = "light"):
    """Circular alert-level gauge using Plotly. / Jauge circulaire du niveau d'alerte."""
    color = LEVEL_COLORS[level]
    label = S["level_labels"][level]
    emoji = LEVEL_EMOJIS[level]
    font_color = "#000000" if theme == "light" else "#ffffff"

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=level,
        number={"suffix": f" {emoji}", "font": {"size": 28, "color": font_color}},
        title={"text": f'{S["gauge_title"]}<br><b>{label}</b>', "font": {"size": 14, "color": font_color}},
        gauge={
            "axis": {
                "range": [0, 4],
                "tickvals": [0, 1, 2, 3],
                "ticktext": S["gauge_ticks"],
                "tickfont": {"size": 10, "color": font_color},
            },
            "bar": {"color": color, "thickness": 0.3},
            "bgcolor": "white",
            "borderwidth": 2,
            "bordercolor": "#ccc",
            "steps": [
                {"range": [0, 1], "color": "#7dcea0"},
                {"range": [1, 2], "color": "#f7dc6f"},
                {"range": [2, 3], "color": "#f0a959"},
                {"range": [3, 4], "color": "#e74c3c"},
            ],
            "threshold": {
                "line": {"color": color, "width": 4},
                "thickness": 0.75,
                "value": level,
            },
        },
    ))
    fig.update_layout(
        height=220,
        margin=dict(t=30, b=10, l=20, r=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": font_color},
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # Confidence bar / Barre de confiance
    st.markdown(f'{S["confidence"]} : {confidence:.0%}')
    st.progress(confidence)


def render_proba_bars(probabilities: dict, S: dict, level: int = 0):
    """Class probability bars. / Barres de probabilité par classe."""
    if not probabilities:
        color = LEVEL_COLORS[level]
        emoji = LEVEL_EMOJIS[level]
        label = S["level_labels"][level]
        st.markdown(S["proba_header"])
        st.markdown(
            f'<div style="background:{color}22;border-left:4px solid {color};'
            f'padding:6px 10px;border-radius:4px;margin:4px 0;">'
            f'{emoji} <b>{label}</b> — safety rule override</div>',
            unsafe_allow_html=True,
        )
        return
    st.markdown(S["proba_header"])
    for label_name, proba in probabilities.items():
        level_num = {"green": 0, "yellow": 1, "orange": 2, "red": 3}[label_name]
        color     = LEVEL_COLORS[level_num]
        emoji     = LEVEL_EMOJIS[level_num]
        bar_width = int(proba * 100)
        display_name = S["proba_names"].get(label_name, label_name.capitalize())
        st.markdown(
            f'<div style="margin:3px 0;">'
            f'{emoji} <b>{display_name}</b> '
            f'<span style="float:right">{proba:.0%}</span>'
            f'<div style="background:#eee;border-radius:4px;height:8px;margin-top:2px;">'
            f'<div style="background:{color};width:{bar_width}%;height:8px;border-radius:4px;"></div>'
            f'</div></div>',
            unsafe_allow_html=True,
        )


def render_longitudinal_section(tracker: SessionTracker, user_id: str, S: dict, theme: str = "light"):
    """
    Longitudinal history section: bar chart + trend + recommendation.
    Section historique longitudinal : barres + tendance + recommandation.
    """
    history = tracker.get_user_history(user_id, last_n_sessions=7)
    risk    = tracker.get_longitudinal_risk(user_id)
    n       = risk["sessions_analyzed"]

    if n == 0:
        st.caption(S["longitudinal_empty"])
        return

    # ── Bar chart: max alert level per session ────────────────────────────────
    dates      = [s["started_at"][:10] for s in history]
    levels     = [s["max_alert_level"] for s in history]
    bar_colors = [LEVEL_COLORS[lvl] for lvl in levels]

    font_color = "#000000" if theme == "light" else "#ffffff"
    grid_color = "#9bb5cc" if theme == "light" else "#2d3748"

    fig = go.Figure(go.Bar(
        x=dates, y=levels,
        marker_color=bar_colors,
        hovertemplate="Session: %{x}<br>" + S["gauge_title"] + ": %{y}<extra></extra>",
    ))
    fig.update_layout(
        height=160,
        margin=dict(t=10, b=30, l=30, r=10),
        yaxis=dict(
            range=[0, 3.5],
            tickvals=[0, 1, 2, 3],
            ticktext=["🟢", "🟡", "🟠", "🔴"],
            tickfont=dict(size=12, color=font_color),
        ),
        xaxis=dict(tickfont=dict(size=9, color=font_color)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=font_color),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor=grid_color)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # ── Trend / Tendance ──────────────────────────────────────────────────────
    trend_map = {
        "stable":    (S["trend_stable"],    "#6b7280"),
        "worsening": (S["trend_worsening"], "#e74c3c"),
        "improving": (S["trend_improving"], "#2ecc71"),
    }
    trend_label, trend_color = trend_map[risk["trend"]]
    st.markdown(
        f'<span style="color:{trend_color};font-weight:bold;">{trend_label}</span>',
        unsafe_allow_html=True,
    )

    # ── Recommendation / Recommandation ───────────────────────────────────────
    rec_key    = _REC_KEY_MAP.get(risk["recommendation"], "rec_normal")
    rec_display = S[rec_key]
    rec_color_map = {
        "rec_normal":       "#2ecc71",
        "rec_attention":    "#f1c40f",
        "rec_consultation": "#e67e22",
        "rec_intervention": "#e74c3c",
    }
    rc = rec_color_map.get(rec_key, "#6b7280")
    text_color = "#000000" if theme == "light" else "#ffffff"
    st.markdown(
        f'<div style="background:{rc}22;border-left:4px solid {rc};'
        f'padding:6px 10px;border-radius:4px;margin:4px 0;color:{text_color};">'
        f'<b>{rec_display}</b></div>',
        unsafe_allow_html=True,
    )
    st.caption(S["sessions_caption"].format(n=n, score=risk["risk_score"]))


def render_history_chart(S: dict, theme: str = "light"):
    """In-session alert level history chart. / Graphique historique de la session."""
    history = st.session_state.alert_history
    if len(history) < 2:
        st.info(S["history_waiting"])
        return

    font_color = "#000000" if theme == "light" else "#ffffff"
    grid_color = "#9bb5cc" if theme == "light" else "#2d3748"
    x      = list(range(1, len(history) + 1))
    colors = [LEVEL_COLORS[h["level"]] for h in history]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=[h["level"] for h in history],
        mode="lines+markers",
        line=dict(color="#6366f1", width=2),
        marker=dict(color=colors, size=10, line=dict(width=2, color="white")),
        name=S["gauge_title"],
        hovertemplate=f'Msg %{{x}}<br>{S["gauge_title"]}: %{{y}}<extra></extra>',
    ))
    fig.update_layout(
        height=150,
        margin=dict(t=10, b=30, l=30, r=10),
        yaxis=dict(
            range=[-0.2, 3.2],
            tickvals=[0, 1, 2, 3],
            ticktext=["🟢", "🟡", "🟠", "🔴"],
            tickfont=dict(size=14, color=font_color),
        ),
        xaxis=dict(title="Msg", tickfont=dict(size=10, color=font_color), title_font=dict(color=font_color)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        font=dict(color=font_color),
    )
    fig.update_xaxes(showgrid=True, gridcolor=grid_color)
    fig.update_yaxes(showgrid=True, gridcolor=grid_color)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_dominant_features(features: list, S: dict):
    """Display dominant features as pills. / Affiche les features dominantes en pills."""
    if not features:
        st.caption(S["signals_waiting"])
        return
    pills_html = "".join(
        f'<span class="feature-pill">⚡ {f}</span>' for f in features
    )
    st.markdown(pills_html, unsafe_allow_html=True)


# ─── Feature importance chart / Graphique d'importance des signaux ────────────

# Maps raw feature name prefixes to category colors
# Mappe les préfixes de noms bruts aux couleurs de catégorie
def _feature_color(raw_name: str) -> str:
    if raw_name.startswith("finality_score") or raw_name == "crisis_similarity":
        return "#e74c3c"   # red — crisis/finality
    if raw_name.startswith("negative_score") or raw_name.startswith("negation_score"):
        return "#e67e22"   # orange — negative/negation
    if raw_name.startswith("hope_score"):
        return "#27ae60"   # green — hope
    if raw_name.startswith("identity_conflict") or raw_name.startswith("somatization"):
        return "#9b59b6"   # purple — identity/cultural
    if raw_name.startswith(("short_response", "min_topic", "question_response",
                            "embedding_drift", "embedding_slope", "embedding_variance")):
        return "#1abc9c"   # teal — behavioral/coherence
    return "#3498db"       # blue — structural (word_count, punctuation, length_delta)


def render_feature_chart(feature_scores: list, S: dict, theme: str = "light"):
    """Horizontal bar chart of top feature scores. / Barres horizontales des scores de signaux."""
    if not feature_scores:
        return

    names  = [fs["name"] for fs in reversed(feature_scores)]
    scores = [fs["score"] for fs in reversed(feature_scores)]
    colors = [_feature_color(fs["raw_name"]) for fs in reversed(feature_scores)]

    font_color = "#000000" if theme == "light" else "#ffffff"

    fig = go.Figure(go.Bar(
        x=scores,
        y=names,
        orientation="h",
        marker_color=colors,
        hovertemplate="%{y}: %{x:.3f}<extra></extra>",
    ))
    fig.update_layout(
        height=max(120, 40 * len(names)),
        margin=dict(t=5, b=5, l=5, r=5),
        xaxis=dict(
            showticklabels=False,
            showgrid=False,
            zeroline=False,
        ),
        yaxis=dict(
            tickfont=dict(size=11, color=font_color),
            automargin=True,
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=font_color),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.caption(S["feature_chart_note"])


# ─── Main application / Application principale ──────────────────────────────────
def main():
    init_state()

    # Propagate Streamlit secrets to environment for LLM providers
    # Propager les secrets Streamlit vers l'environnement pour les fournisseurs LLM
    for key in ("GROQ_API_KEY", "GEMINI_API_KEY", "ANTHROPIC_API_KEY"):
        if key not in os.environ:
            try:
                os.environ[key] = st.secrets[key]
            except (KeyError, FileNotFoundError):
                pass

    clf     = load_model()
    tracker = load_tracker()
    lang    = st.session_state.lang
    theme   = st.session_state.theme
    S       = STRINGS[lang]

    st.markdown(get_theme_css(theme), unsafe_allow_html=True)

    # Alert transition toast (rendered once, cleared after display)
    # Toast de transition d'alerte (affiché une fois, effacé après)
    toast_level = st.session_state.pop("_alert_toast", None)
    if toast_level is not None:
        t_color = LEVEL_COLORS[toast_level]
        t_emoji = LEVEL_EMOJIS[toast_level]
        t_label = S["level_labels"][toast_level]
        toast_msg = S["alert_toast_up"].format(emoji=t_emoji, label=t_label)
        st.markdown(
            f'<div class="alert-toast" style="background:{t_color};color:#fff;">'
            f'{toast_msg}</div>',
            unsafe_allow_html=True,
        )

    # Start a session if needed (first visit or after reset)
    # Démarrer une session si nécessaire (première visite ou après reset)
    if st.session_state.session_id is None:
        # Check for withdrawal risk before starting new session
        # Vérifier le risque d'abandon avant de démarrer une nouvelle session
        withdrawal = tracker.check_withdrawal_risk(st.session_state.user_id)
        if withdrawal["is_withdrawal"]:
            st.session_state.withdrawal_detected = True
        st.session_state.session_id = tracker.start_session(st.session_state.user_id)

    # ── Header ─────────────────────────────────────────────────────────────────
    col_title, col_profile, col_lang, col_theme, col_reset = st.columns([3, 1.5, 1, 1, 1])
    with col_title:
        st.markdown(f"# {S['app_title']}")
        st.caption(S["app_subtitle"])

    with col_profile:
        st.markdown("<br>", unsafe_allow_html=True)
        current_idx = DEMO_USERS.index(st.session_state.user_id) if st.session_state.user_id in DEMO_USERS else len(DEMO_USERS) - 1
        selected_user = st.selectbox(
            S["profile_label"],
            DEMO_USERS,
            index=current_idx,
            key="profile_selector",
            label_visibility="collapsed",
        )
        if selected_user != st.session_state.user_id:
            # End current session before switching / Clôturer la session avant de changer
            max_lvl = max((h["level"] for h in st.session_state.alert_history), default=0)
            n_user = sum(1 for m in st.session_state.messages if m["role"] == "user")
            tracker.end_session(
                st.session_state.user_id, st.session_state.session_id,
                max_lvl, n_user,
            )
            reset_conversation()
            st.session_state.user_id = selected_user
            st.session_state.session_id = None
            st.rerun()

    with col_lang:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button(S["lang_btn"], use_container_width=True):
            st.session_state.lang = "en" if lang == "fr" else "fr"
            st.rerun()

    with col_theme:
        st.markdown("<br>", unsafe_allow_html=True)
        theme_label = "☀️ Light" if theme == "dark" else "🌙 Dark"
        if st.button(theme_label, use_container_width=True):
            st.session_state.theme = "light" if theme == "dark" else "dark"
            st.rerun()

    with col_reset:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button(S["reset_btn"], use_container_width=True):
            # Close the current session before resetting
            # Clôturer la session courante avant de réinitialiser
            max_lvl  = max((h["level"] for h in st.session_state.alert_history), default=0)
            n_user   = sum(1 for m in st.session_state.messages if m["role"] == "user")
            tracker.end_session(
                st.session_state.user_id, st.session_state.session_id,
                max_lvl, n_user,
            )
            tracker.mark_session_closed(st.session_state.user_id)
            reset_conversation()
            st.session_state.session_id = None
            st.rerun()

    st.divider()

    # ── Main layout: Chat | Dashboard ──────────────────────────────────────────
    col_chat, col_dash = st.columns([3, 2], gap="medium")

    # ── LEFT: Chat interface / GAUCHE : Interface de chat ─────────────────────
    with col_chat:
        st.markdown(S["chat_header"])

        # Action buttons row: Demo, About, Export, Compare / Boutons d'action
        btn_cols = st.columns([1, 1, 1, 1, 2])
        with btn_cols[0]:
            if st.session_state.compare_mode:
                st.button(S["demo_btn"], use_container_width=True, disabled=True)
            elif st.session_state.demo_running:
                if st.button(S["demo_stop_btn"], use_container_width=True):
                    st.session_state.demo_running = False
                    st.session_state.demo_step = 0
                    st.rerun()
            else:
                if st.button(S["demo_btn"], use_container_width=True):
                    reset_conversation()
                    st.session_state.demo_running = True
                    st.session_state.demo_step = 0
                    st.rerun()
        with btn_cols[1]:
            if st.button(S["about_btn"], use_container_width=True):
                st.session_state["show_about"] = not st.session_state.get("show_about", False)
                st.rerun()
        with btn_cols[2]:
            if st.session_state.messages:
                export_data = {
                    "session_id": st.session_state.session_id,
                    "user_id": st.session_state.user_id,
                    "language": lang,
                    "exported_at": datetime.now().isoformat(),
                    "messages": st.session_state.messages,
                    "alert_history": [
                        {
                            "level": a["level"],
                            "label": a.get("label", ""),
                            "confidence": a.get("confidence", 0),
                            "dominant_features": a.get("dominant_features", []),
                        }
                        for a in st.session_state.alert_history
                    ],
                    "peak_alert": max((a["level"] for a in st.session_state.alert_history), default=0),
                }
                st.download_button(
                    S["export_btn"],
                    data=json.dumps(export_data, indent=2, ensure_ascii=False),
                    file_name=f"cedd_transcript_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True,
                )

        with btn_cols[3]:
            compare_label = S["compare_btn_off"] if st.session_state.compare_mode else S["compare_btn"]
            if st.button(compare_label, use_container_width=True):
                new_mode = not st.session_state.compare_mode
                reset_conversation()
                st.session_state.compare_mode = new_mode
                st.rerun()

        # About CEDD panel / Panneau À propos de CEDD
        if st.session_state.get("show_about", False):
            with st.expander(S["about_title"], expanded=True):
                st.markdown(ABOUT_CEDD[lang])

        # Demo running banner / Bannière de démo en cours
        if st.session_state.demo_running:
            scenario = DEMO_SCENARIOS[lang]
            step = st.session_state.demo_step
            char_key = "demo_character_fr" if lang == "fr" else "demo_character_en"
            st.info(f"🎬 {S[char_key]} — {S['demo_running'].format(n=step + 1)}")

        # Withdrawal banner / Bannière de retour après absence
        if st.session_state.withdrawal_detected and not st.session_state.messages:
            st.info(S["withdrawal_banner"])

        # Compare mode: show two chat columns / Mode comparaison : deux colonnes de chat
        if st.session_state.compare_mode:
            cmp_left, cmp_right = st.columns(2, gap="small")
            with cmp_left:
                st.markdown(S["compare_left_header"])
                st.caption(S["compare_left_sub"])
                render_chat(S, theme, messages=st.session_state.compare_messages)
            with cmp_right:
                st.markdown(S["compare_right_header"])
                st.caption(S["compare_right_sub"])
                render_chat(S, theme)
        else:
            render_chat(S, theme)

        # Input form / Zone de saisie
        with st.form(key=f"chat_form_{st.session_state.input_key}", clear_on_submit=True):
            user_input = st.text_input(
                S["input_placeholder"],
                placeholder=S["input_placeholder"],
                label_visibility="collapsed",
            )
            submitted = st.form_submit_button(S["send_btn"], use_container_width=True)

        # Demo autopilot: inject next message / Démo automatique : injecter le prochain message
        if st.session_state.demo_running:
            scenario = DEMO_SCENARIOS[lang]
            step = st.session_state.demo_step
            if step < len(scenario):
                user_msg = scenario[step]
                submitted = True
                user_input = user_msg
                st.session_state.demo_step += 1
                if st.session_state.demo_step >= len(scenario):
                    st.session_state.demo_running = False

        if submitted and user_input.strip():
            user_msg = user_input.strip()

            # Add user message / Ajouter le message utilisateur
            now = datetime.now().strftime("%H:%M")
            st.session_state.messages.append({"role": "user", "content": user_msg, "timestamp": now})

            # Compare mode: also add user message to left side / Mode comparaison : ajouter aussi à gauche
            if st.session_state.compare_mode:
                st.session_state.compare_messages.append({"role": "user", "content": user_msg, "timestamp": now})

            # Track previous alert level for transition animation
            # Suivre le niveau précédent pour l'animation de transition
            prev_level = st.session_state.alert_history[-1]["level"] if st.session_state.alert_history else 0

            # Analyse with CEDD before generating the LLM response
            # Analyser avec CEDD avant de générer la réponse
            alert = clf.get_alert_level(st.session_state.messages, lang=lang)
            st.session_state.current_alert = alert
            st.session_state.alert_history.append(alert)

            # Alert transition toast / Toast de transition d'alerte
            if alert["level"] > prev_level:
                st.session_state["_alert_toast"] = alert["level"]

            # Log alert to cross-session tracker / Enregistrer dans le tracker
            tracker.log_alert(
                st.session_state.user_id,
                st.session_state.session_id,
                alert["level"],
                alert["confidence"],
                user_msg,
            )

            # Update last activity for withdrawal detection / MAJ activité pour détection d'abandon
            tracker.update_last_activity(st.session_state.user_id, st.session_state.session_id)

            # Warm handoff step management / Gestion des étapes de transfert accompagné
            if alert["level"] == 3:
                if st.session_state.handoff_step == 0:
                    # First Red detection — start handoff at step 1
                    # Première détection Rouge — démarrer le transfert à l'étape 1
                    st.session_state.handoff_step = 1
                elif st.session_state.handoff_step < 5:
                    # Advance to next step / Avancer à l'étape suivante
                    st.session_state.handoff_step += 1
                # If already at step 5, stay at step 5 (continued presence)
                # Si déjà à l'étape 5, rester à 5 (présence continue)

                # Log handoff step / Enregistrer l'étape de transfert
                tracker.log_handoff_step(
                    st.session_state.user_id,
                    st.session_state.session_id,
                    st.session_state.handoff_step,
                    alert["level"],
                )
            # If level drops below Red, keep handoff_step as-is (crisis may not be over)
            # Si le niveau descend sous Rouge, garder handoff_step tel quel

            # Generate assistant responses / Générer les réponses de l'assistant
            with st.spinner("..."):
                # Compare mode: generate "without CEDD" response first (always Green prompt)
                # Mode comparaison : générer d'abord la réponse "sans CEDD" (toujours prompt Vert)
                if st.session_state.compare_mode:
                    plain_result = get_llm_response(
                        st.session_state.compare_messages,
                        0,
                        force_model=st.session_state.selected_llm,
                        lang=lang,
                        handoff_step=0,
                        system_prompt_override="",  # No instructions — raw LLM response
                    )
                    if plain_result["content"]:
                        st.session_state.compare_messages.append({
                            "role": "assistant",
                            "content": plain_result["content"],
                            "timestamp": datetime.now().strftime("%H:%M"),
                        })

                # "With CEDD" response (adaptive prompt) / Réponse "avec CEDD" (prompt adaptatif)
                result = get_llm_response(
                    st.session_state.messages,
                    alert["level"],
                    force_model=st.session_state.selected_llm,
                    lang=lang,
                    handoff_step=st.session_state.handoff_step,
                )
            st.session_state.last_llm_source = result["source"]

            if result["content"]:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["content"],
                    "timestamp": datetime.now().strftime("%H:%M"),
                    "source": result["source"],
                    "alert_level": alert["level"],
                })

            st.rerun()

    # ── RIGHT: CEDD Dashboard / DROITE : Dashboard CEDD ───────────────────────
    with col_dash:
        st.markdown(S["dashboard_header"])

        alert = st.session_state.current_alert
        level = alert["level"]

        # Circular gauge / Jauge circulaire
        render_gauge(level, alert.get("confidence", 0.0), S, theme)

        st.divider()

        # Class probabilities / Probabilités par classe
        render_proba_bars(alert.get("probabilities", {}), S, level)

        st.divider()

        # Active signals / Signaux actifs
        st.markdown(S["signals_header"])
        if st.session_state.withdrawal_detected:
            st.markdown(
                f'<span class="feature-pill" style="background:#e74c3c22;border-color:#e74c3c;color:#e74c3c;">'
                f'⏰ {S["withdrawal_badge"]}</span>',
                unsafe_allow_html=True,
            )
        render_dominant_features(alert.get("dominant_features", []), S)

        # Feature importance chart (Yellow+ only, not on safety override)
        # Graphique d'importance des signaux (Jaune+ seulement, pas sur override)
        feature_scores = alert.get("feature_scores", [])
        if level >= 1 and feature_scores:
            with st.expander(S["feature_chart_title"]):
                render_feature_chart(feature_scores, S, theme)

        st.divider()

        # In-session history / Historique des niveaux
        st.markdown(S["history_header"])
        render_history_chart(S, theme)

        st.divider()

        # Cross-session longitudinal history / Historique longitudinal inter-sessions
        st.markdown(S["longitudinal_header"])
        render_longitudinal_section(tracker, st.session_state.user_id, S, theme)

        st.divider()

        # LLM selector / Sélecteur LLM
        st.markdown(S["llm_header"])
        llm_cols = st.columns(4)
        for col, (src, (emoji, _)) in zip(llm_cols, LLM_SOURCE_INDICATOR.items()):
            is_selected = st.session_state.selected_llm == src
            btn_label = S["llm_fallback"] if src == "fallback-statique" else LLM_DISPLAY_NAMES.get(src, src)
            if col.button(
                f"{emoji} {btn_label}",
                key=f"llm_btn_{src}",
                type="primary" if is_selected else "secondary",
                use_container_width=True,
            ):
                st.session_state.selected_llm = src
                st.rerun()

        # Last call source / Source du dernier appel
        last = st.session_state.get("last_llm_source")
        if last and last != st.session_state.selected_llm:
            _, last_color = LLM_SOURCE_INDICATOR.get(last, ("", "#6b7280"))
            st.caption(
                f'{S["llm_last_call"]} '
                f'<span style="color:{last_color}"><code>{last}</code></span>',
                unsafe_allow_html=True,
            )

        st.divider()

        # Active response mode / Mode de réponse actif
        st.markdown(S["mode_header"])
        mode_desc = get_level_description(level, lang=lang)
        color = LEVEL_COLORS[level]
        emoji = LEVEL_EMOJIS[level]
        text_color = "#000000" if theme == "light" else "#ffffff"
        st.markdown(
            f'<div style="background:{color}22;border-left:4px solid {color};'
            f'padding:8px 12px;border-radius:4px;margin:4px 0;color:{text_color};">'
            f'{emoji} <b>{mode_desc}</b></div>',
            unsafe_allow_html=True,
        )

        # Warm handoff progress indicator / Indicateur de transfert accompagné
        if st.session_state.handoff_step > 0:
            step = st.session_state.handoff_step
            step_desc = get_handoff_description(step, lang=lang)
            progress = step / 5

            step_colors = {1: "#e74c3c", 2: "#e67e22", 3: "#f1c40f", 4: "#27ae60", 5: "#2ecc71"}
            step_color = step_colors.get(step, "#e74c3c")

            handoff_title = S["handoff_title"]
            handoff_label = S["handoff_step_label"].format(step=step, desc=step_desc)

            st.markdown(f"**{handoff_title}**")
            st.markdown(
                f'<div style="background:{step_color}22;border-left:4px solid {step_color};'
                f'padding:8px 12px;border-radius:4px;margin:4px 0;color:{text_color};">'
                f'{handoff_label}</div>',
                unsafe_allow_html=True,
            )
            st.progress(progress)
            st.divider()

        # System prompt expander / Affichage du prompt
        with st.expander(S["prompt_expander"]):
            st.code(get_system_prompt(level, lang=lang, handoff_step=st.session_state.handoff_step), language=None)

        st.divider()

        # Session statistics / Statistiques de session
        st.markdown(S["stats_header"])
        n_user    = sum(1 for m in st.session_state.messages if m["role"] == "user")
        n_total   = len(st.session_state.messages)
        max_level = max((h["level"] for h in st.session_state.alert_history), default=0)
        c1, c2, c3 = st.columns(3)
        c1.metric(S["stat_messages"], n_user)
        c2.metric(S["stat_exchanges"], n_total // 2)
        c3.metric(S["stat_peak"], LEVEL_EMOJIS[max_level])


if __name__ == "__main__":
    main()
