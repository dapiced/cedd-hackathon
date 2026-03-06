"""
DDEC — Streamlit Application
Bilingual demo interface: chat + real-time monitoring dashboard.
Interface de démonstration bilingue : chat + dashboard de surveillance en temps réel.
"""

import os
import sys
import streamlit as st
import plotly.graph_objects as go

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ddec.classifier import DDECClassifier, LEVEL_LABELS
from ddec.response_modulator import (
    get_llm_response,
    get_level_description,
    get_system_prompt,
)
from ddec.session_tracker import SessionTracker

# ─── Configuration ─────────────────────────────────────────────────────────────
MODEL_PATH = "models/ddec_model.joblib"

LEVEL_COLORS = {0: "#2ecc71", 1: "#f1c40f", 2: "#e67e22", 3: "#e74c3c"}

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
        "page_title":          "DDEC - Détection de dérive émotionnelle",
        "app_title":           "🧠 DDEC — Détection de dérive émotionnelle conversationnelle",
        "app_subtitle":        "Hackathon Mila · Sécurité IA en santé mentale des jeunes · POC",
        "reset_btn":           "🔄 Réinitialiser",
        "chat_header":         "### 💬 Conversation",
        "chat_empty":          "Commence la conversation...",
        "input_placeholder":   "Écris ton message ici et appuie sur Entrée",
        "send_btn":            "Envoyer ➤",
        "dashboard_header":    "### 📊 Dashboard DDEC",
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
    },
    "en": {
        "lang_btn":            "🇫🇷 Français",
        "page_title":          "DDEC - Conversational Emotional Drift Detection",
        "app_title":           "🧠 DDEC — Conversational Emotional Drift Detection",
        "app_subtitle":        "Mila Hackathon · AI Safety in Youth Mental Health · POC",
        "reset_btn":           "🔄 Reset",
        "chat_header":         "### 💬 Conversation",
        "chat_empty":          "Start the conversation...",
        "input_placeholder":   "Type your message here and press Enter",
        "send_btn":            "Send ➤",
        "dashboard_header":    "### 📊 DDEC Dashboard",
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
    page_title="DDEC",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { padding-top: 1rem; }
    .chat-bubble-user {
        background-color: #DCF8C6;
        border-radius: 18px 18px 4px 18px;
        padding: 10px 14px;
        margin: 6px 0 6px 40px;
        max-width: 85%;
        float: right;
        clear: both;
        color: #1a1a1a;
        font-size: 0.95rem;
    }
    .chat-bubble-assistant {
        background-color: #F0F0F0;
        border-radius: 18px 18px 18px 4px;
        padding: 10px 14px;
        margin: 6px 40px 6px 0;
        max-width: 85%;
        float: left;
        clear: both;
        color: #1a1a1a;
        font-size: 0.95rem;
    }
    .chat-container {
        overflow-y: auto;
        max-height: 420px;
        padding: 10px;
        border: 1px solid #e0e0e0;
        border-radius: 12px;
        background: #fafafa;
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
        background: #fff;
        border: 1px solid #e8e8e8;
        border-radius: 10px;
        padding: 12px;
        margin: 6px 0;
    }
    .feature-pill {
        background: #eef2ff;
        border: 1px solid #c7d2fe;
        border-radius: 12px;
        padding: 4px 10px;
        font-size: 0.82rem;
        display: inline-block;
        margin: 2px;
        color: #3730a3;
    }
    h1 { font-size: 1.4rem !important; }
    h3 { font-size: 1.05rem !important; margin-bottom: 0.4rem !important; }
    .stTextInput > div > div > input { border-radius: 20px; }
</style>
""", unsafe_allow_html=True)


# ─── Model & tracker loading ────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model not found / Modèle introuvable : {MODEL_PATH}. Run `python train.py`.")
        st.stop()
    return DDECClassifier.load(MODEL_PATH)


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
        "lang":            "fr",   # default language / langue par défaut
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


def render_gauge(level: int, confidence: float, S: dict):
    """Circular alert-level gauge using Plotly. / Jauge circulaire du niveau d'alerte."""
    color = LEVEL_COLORS[level]
    label = S["level_labels"][level]
    emoji = LEVEL_EMOJIS[level]

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=level,
        number={"suffix": f" {emoji}", "font": {"size": 28}},
        title={"text": f'{S["gauge_title"]}<br><b>{label}</b>', "font": {"size": 14}},
        gauge={
            "axis": {
                "range": [0, 3],
                "tickvals": [0, 1, 2, 3],
                "ticktext": S["gauge_ticks"],
                "tickfont": {"size": 10},
            },
            "bar": {"color": color, "thickness": 0.3},
            "bgcolor": "white",
            "borderwidth": 2,
            "bordercolor": "#ccc",
            "steps": [
                {"range": [0, 1], "color": "#d5f5e3"},
                {"range": [1, 2], "color": "#fef9e7"},
                {"range": [2, 3], "color": "#fdebd0"},
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
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # Confidence bar / Barre de confiance
    st.markdown(f'{S["confidence"]} : {confidence:.0%}')
    st.progress(confidence)


def render_proba_bars(probabilities: dict, S: dict):
    """Class probability bars. / Barres de probabilité par classe."""
    if not probabilities:
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


def render_longitudinal_section(tracker: SessionTracker, user_id: str, S: dict):
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
            tickfont=dict(size=12),
        ),
        xaxis=dict(tickfont=dict(size=9)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="#f0f0f0")
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
    st.markdown(
        f'<div style="background:{rc}22;border-left:4px solid {rc};'
        f'padding:6px 10px;border-radius:4px;margin:4px 0;">'
        f'<b>{rec_display}</b></div>',
        unsafe_allow_html=True,
    )
    st.caption(S["sessions_caption"].format(n=n, score=risk["risk_score"]))


def render_history_chart(S: dict):
    """In-session alert level history chart. / Graphique historique de la session."""
    history = st.session_state.alert_history
    if len(history) < 2:
        st.info(S["history_waiting"])
        return

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
            tickfont=dict(size=14),
        ),
        xaxis=dict(title="Msg", tickfont=dict(size=10)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    fig.update_xaxes(showgrid=True, gridcolor="#f0f0f0")
    fig.update_yaxes(showgrid=True, gridcolor="#f0f0f0")
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
    S       = STRINGS[lang]

    # Start a session if needed (first visit or after reset)
    # Démarrer une session si nécessaire (première visite ou après reset)
    if st.session_state.session_id is None:
        st.session_state.session_id = tracker.start_session(st.session_state.user_id)

    # ── Header ─────────────────────────────────────────────────────────────────
    col_title, col_lang, col_reset = st.columns([4, 1, 1])
    with col_title:
        st.markdown(f"# {S['app_title']}")
        st.caption(S["app_subtitle"])

    with col_lang:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button(S["lang_btn"], use_container_width=True):
            st.session_state.lang = "en" if lang == "fr" else "fr"
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

            # Analyse with DDEC before generating the LLM response
            # Analyser avec DDEC avant de générer la réponse
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

    # ── RIGHT: DDEC Dashboard / DROITE : Dashboard DDEC ───────────────────────
    with col_dash:
        st.markdown(S["dashboard_header"])

        alert = st.session_state.current_alert
        level = alert["level"]

        # Circular gauge / Jauge circulaire
        render_gauge(level, alert.get("confidence", 0.0), S)

        st.divider()

        # Class probabilities / Probabilités par classe
        render_proba_bars(alert.get("probabilities", {}), S)

        st.divider()

        # Active signals / Signaux actifs
        st.markdown(S["signals_header"])
        render_dominant_features(alert.get("dominant_features", []), S)

        st.divider()

        # In-session history / Historique des niveaux
        st.markdown(S["history_header"])
        render_history_chart(S)

        st.divider()

        # Cross-session longitudinal history / Historique longitudinal inter-sessions
        st.markdown(S["longitudinal_header"])
        render_longitudinal_section(tracker, st.session_state.user_id, S)

        st.divider()

        # LLM selector / Sélecteur LLM
        st.markdown(S["llm_header"])
        llm_cols = st.columns(4)
        for col, (src, (emoji, _)) in zip(llm_cols, LLM_SOURCE_INDICATOR.items()):
            is_selected = st.session_state.selected_llm == src
            if col.button(
                f"{emoji} {src}",
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
        st.markdown(
            f'<div style="background:{color}22;border-left:4px solid {color};'
            f'padding:8px 12px;border-radius:4px;margin:4px 0;">'
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
