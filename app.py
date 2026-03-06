"""
CEDD — Streamlit Application
Bilingual demo interface: chat + real-time monitoring dashboard.
Interface de démonstration bilingue : chat + dashboard de surveillance en temps réel.
"""

import os
import sys
import streamlit as st
import plotly.graph_objects as go

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cedd.classifier import CEDDClassifier, LEVEL_LABELS
from cedd.response_modulator import (
    get_llm_response,
    get_level_description,
    get_system_prompt,
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
</style>
"""

LLM_SOURCE_INDICATOR = {
    "claude-haiku":      ("🟣", "#7c3aed"),
    "mistral":           ("🔵", "#2563eb"),
    "llama3.2:1b":       ("⚪", "#6b7280"),
    "fallback-statique": ("⚠️", "#f59e0b"),
}
LEVEL_EMOJIS = {0: "🟢", 1: "🟡", 2: "🟠", 3: "🔴"}

# ─── Bilingual UI strings / Chaînes d'interface bilingues ─────────────────────
STRINGS = {
    "fr": {
        "lang_btn":            "🇬🇧 English",
        "page_title":          "CEDD - Détection de dérive émotionnelle",
        "app_title":           "🧠 CEDD — Détection de dérive émotionnelle conversationnelle",
        "app_subtitle":        "Hackathon Mila · Sécurité IA en santé mentale des jeunes · POC",
        "theme_btn":           "🌙 Sombre",
        "reset_btn":           "🔄 Réinitialiser",
        "chat_header":         "### 💬 Conversation",
        "chat_empty":          "Commence la conversation...",
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
            "verte": "Verte", "jaune": "Jaune", "orange": "Orange", "rouge": "Rouge"
        },
        # Internal recommendation keys from session_tracker (always French)
        # → mapped here for display / Clés internes → affichage traduit
        "rec_normal":       "Suivi normal",
        "rec_attention":    "Attention soutenue recommandée",
        "rec_consultation": "Consultation professionnelle suggérée",
        "rec_intervention": "Intervention prioritaire recommandée",
        "llm_fallback":     "Sans LLM",
    },
    "en": {
        "lang_btn":            "🇫🇷 Français",
        "page_title":          "CEDD - Conversational Emotional Drift Detection",
        "app_title":           "🧠 CEDD — Conversational Emotional Drift Detection",
        "app_subtitle":        "Mila Hackathon · AI Safety in Youth Mental Health · POC",
        "theme_btn":           "🌙 Dark",
        "reset_btn":           "🔄 Reset",
        "chat_header":         "### 💬 Conversation",
        "chat_empty":          "Start the conversation...",
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
            "verte": "Green", "jaune": "Yellow", "orange": "Orange", "rouge": "Red"
        },
        "rec_normal":       "Normal monitoring",
        "rec_attention":    "Sustained attention recommended",
        "rec_consultation": "Professional consultation suggested",
        "rec_intervention": "Priority intervention recommended",
        "llm_fallback":     "Without LLM",
    },
}

# Maps French recommendation strings (from session_tracker) to STRINGS keys
# Mappe les recommandations françaises (session_tracker) vers les clés STRINGS
_REC_KEY_MAP = {
    "Suivi normal":                          "rec_normal",
    "Attention soutenue recommandée":        "rec_attention",
    "Consultation professionnelle suggérée": "rec_consultation",
    "Intervention prioritaire recommandée":  "rec_intervention",
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
            "level": 0, "label": "verte", "confidence": 0.0,
            "dominant_features": [], "probabilities": {},
        },
        "selected_llm":    "claude-haiku",
        "last_llm_source": None,
        "input_key":       0,
        "user_id":         "demo_user",
        "session_id":      None,
        "lang":            "en",   # default language / langue par défaut
        "theme":           "light",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def reset_conversation():
    st.session_state.messages      = []
    st.session_state.alert_history = []
    st.session_state.current_alert = {
        "level": 0, "label": "verte", "confidence": 0.0,
        "dominant_features": [], "probabilities": {},
    }
    st.session_state.input_key += 1


# ─── UI components / Composants UI ──────────────────────────────────────────────

def render_chat(S: dict):
    """Display chat bubbles. / Affiche les bulles de conversation."""
    msgs_html = '<div class="chat-container"><div class="clearfix">'
    if not st.session_state.messages:
        msgs_html += (
            f'<p style="color:#aaa;text-align:center;margin-top:40px;">'
            f'{S["chat_empty"]}</p>'
        )
    else:
        for msg in st.session_state.messages:
            role = msg["role"]
            content = (
                msg["content"]
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace("\n", "<br>")
            )
            css_class = "chat-bubble-user" if role == "user" else "chat-bubble-assistant"
            msgs_html += f'<div class="{css_class}">{content}</div>'
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
        level_num = {"verte": 0, "jaune": 1, "orange": 2, "rouge": 3}[label_name]
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


# ─── Main application / Application principale ──────────────────────────────────
def main():
    init_state()
    clf     = load_model()
    tracker = load_tracker()
    lang    = st.session_state.lang
    theme   = st.session_state.theme
    S       = STRINGS[lang]

    st.markdown(get_theme_css(theme), unsafe_allow_html=True)

    # Start a session if needed (first visit or after reset)
    # Démarrer une session si nécessaire (première visite ou après reset)
    if st.session_state.session_id is None:
        st.session_state.session_id = tracker.start_session(st.session_state.user_id)

    # ── Header ─────────────────────────────────────────────────────────────────
    col_title, col_lang, col_theme, col_reset = st.columns([4, 1, 1, 1])
    with col_title:
        st.markdown(f"# {S['app_title']}")
        st.caption(S["app_subtitle"])

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
            reset_conversation()
            st.session_state.session_id = None
            st.rerun()

    st.divider()

    # ── Main layout: Chat | Dashboard ──────────────────────────────────────────
    col_chat, col_dash = st.columns([3, 2], gap="medium")

    # ── LEFT: Chat interface / GAUCHE : Interface de chat ─────────────────────
    with col_chat:
        st.markdown(S["chat_header"])
        render_chat(S)

        # Input form / Zone de saisie
        with st.form(key=f"chat_form_{st.session_state.input_key}", clear_on_submit=True):
            user_input = st.text_input(
                S["input_placeholder"],
                placeholder=S["input_placeholder"],
                label_visibility="collapsed",
            )
            submitted = st.form_submit_button(S["send_btn"], use_container_width=True)

        if submitted and user_input.strip():
            user_msg = user_input.strip()

            # Add user message / Ajouter le message utilisateur
            st.session_state.messages.append({"role": "user", "content": user_msg})

            # Analyse with CEDD before generating the LLM response
            # Analyser avec CEDD avant de générer la réponse
            alert = clf.get_alert_level(st.session_state.messages, lang=lang)
            st.session_state.current_alert = alert
            st.session_state.alert_history.append(alert)

            # Log alert to cross-session tracker / Enregistrer dans le tracker
            tracker.log_alert(
                st.session_state.user_id,
                st.session_state.session_id,
                alert["level"],
                alert["confidence"],
                user_msg,
            )

            # Generate assistant response / Générer la réponse de l'assistant
            with st.spinner("..."):
                result = get_llm_response(
                    st.session_state.messages,
                    alert["level"],
                    force_model=st.session_state.selected_llm,
                    lang=lang,
                )
            st.session_state.last_llm_source = result["source"]

            if result["content"]:
                st.session_state.messages.append(
                    {"role": "assistant", "content": result["content"]}
                )

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
        render_dominant_features(alert.get("dominant_features", []), S)

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
            btn_label = S["llm_fallback"] if src == "fallback-statique" else src
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

        # System prompt expander / Affichage du prompt
        with st.expander(S["prompt_expander"]):
            st.code(get_system_prompt(level, lang=lang), language=None)

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
