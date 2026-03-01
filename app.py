"""
DDEC - Application Streamlit
Interface de démonstration : chat + dashboard de surveillance en temps réel.
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

# ─── Configuration ────────────────────────────────────────────────────────────
MODEL_PATH = "models/ddec_model.joblib"

LEVEL_COLORS = {0: "#2ecc71", 1: "#f1c40f", 2: "#e67e22", 3: "#e74c3c"}

LLM_SOURCE_INDICATOR = {
    "claude-haiku":      ("🟣", "#7c3aed"),
    "mistral":           ("🔵", "#2563eb"),
    "llama3.2:1b":       ("⚪", "#6b7280"),
    "fallback-statique": ("⚠️", "#f59e0b"),
}
LEVEL_EMOJIS = {0: "🟢", 1: "🟡", 2: "🟠", 3: "🔴"}
LEVEL_LABELS_FR = {0: "VERT", 1: "JAUNE", 2: "ORANGE", 3: "ROUGE"}

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DDEC - Détection de dérive émotionnelle",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── CSS personnalisé ─────────────────────────────────────────────────────────
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


# ─── Chargement du modèle et du tracker ───────────────────────────────────────
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Modèle introuvable : {MODEL_PATH}. Lancez d'abord `python train.py`.")
        st.stop()
    return DDECClassifier.load(MODEL_PATH)


@st.cache_resource
def load_tracker():
    return SessionTracker()


# ─── Initialisation de l'état ─────────────────────────────────────────────────
def init_state():
    defaults = {
        "messages": [],
        "alert_history": [],
        "current_alert": {"level": 0, "label": "verte", "confidence": 0.0,
                          "dominant_features": [], "probabilities": {}},
        "selected_llm": "claude-haiku",
        "last_llm_source": None,  # mis à jour après chaque réponse
        "input_key": 0,
        "user_id": "demo_user",
        "session_id": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def reset_conversation():
    st.session_state.messages = []
    st.session_state.alert_history = []
    st.session_state.current_alert = {
        "level": 0, "label": "verte", "confidence": 0.0,
        "dominant_features": [], "probabilities": {}
    }
    st.session_state.input_key += 1


# (Logique LLM centralisée dans ddec/response_modulator.get_llm_response)


# ─── Composants UI ────────────────────────────────────────────────────────────
def render_chat():
    """Affiche les bulles de conversation."""
    msgs_html = '<div class="chat-container"><div class="clearfix">'
    if not st.session_state.messages:
        msgs_html += '<p style="color:#aaa;text-align:center;margin-top:40px;">Commence la conversation...</p>'
    else:
        for msg in st.session_state.messages:
            role = msg["role"]
            content = msg["content"].replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
            if role == "user":
                msgs_html += f'<div class="chat-bubble-user">{content}</div>'
            else:
                msgs_html += f'<div class="chat-bubble-assistant">{content}</div>'
    msgs_html += '</div></div>'
    st.markdown(msgs_html, unsafe_allow_html=True)


def render_gauge(level: int, confidence: float):
    """Jauge circulaire du niveau d'alerte avec Plotly."""
    color = LEVEL_COLORS[level]
    label = LEVEL_LABELS_FR[level]
    emoji = LEVEL_EMOJIS[level]

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=level,
        number={"suffix": f" {emoji}", "font": {"size": 28}},
        title={"text": f"Niveau d'alerte<br><b>{label}</b>", "font": {"size": 14}},
        gauge={
            "axis": {"range": [0, 3], "tickvals": [0, 1, 2, 3],
                     "ticktext": ["Vert", "Jaune", "Orange", "Rouge"],
                     "tickfont": {"size": 10}},
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

    # Barre de confiance
    st.markdown(f"**Confiance** : {confidence:.0%}")
    st.progress(confidence)


def render_proba_bars(probabilities: dict):
    """Barres de probabilité par classe."""
    if not probabilities:
        return
    st.markdown("**Probabilités par classe**")
    for label_name, proba in probabilities.items():
        level_num = {"verte": 0, "jaune": 1, "orange": 2, "rouge": 3}[label_name]
        color = LEVEL_COLORS[level_num]
        emoji = LEVEL_EMOJIS[level_num]
        bar_width = int(proba * 100)
        st.markdown(
            f'<div style="margin:3px 0;">'
            f'{emoji} <b>{label_name.capitalize()}</b> '
            f'<span style="float:right">{proba:.0%}</span>'
            f'<div style="background:#eee;border-radius:4px;height:8px;margin-top:2px;">'
            f'<div style="background:{color};width:{bar_width}%;height:8px;border-radius:4px;"></div>'
            f'</div></div>',
            unsafe_allow_html=True,
        )


def render_longitudinal_section(tracker: SessionTracker, user_id: str):
    """Section historique longitudinal : barres par session + tendance + recommandation."""
    history = tracker.get_user_history(user_id, last_n_sessions=7)
    risk = tracker.get_longitudinal_risk(user_id)
    n = risk["sessions_analyzed"]

    if n == 0:
        st.caption("Aucun historique — complétez des sessions pour voir la tendance.")
        return

    # ── Graphique à barres : max_alert_level par session ──────────────────────
    dates = [s["started_at"][:10] for s in history]
    levels = [s["max_alert_level"] for s in history]
    bar_colors = [LEVEL_COLORS[lvl] for lvl in levels]

    fig = go.Figure(go.Bar(
        x=dates, y=levels,
        marker_color=bar_colors,
        hovertemplate="Session: %{x}<br>Niveau max: %{y}<extra></extra>",
    ))
    fig.update_layout(
        height=160,
        margin=dict(t=10, b=30, l=30, r=10),
        yaxis=dict(range=[0, 3.5], tickvals=[0, 1, 2, 3],
                   ticktext=["🟢", "🟡", "🟠", "🔴"], tickfont=dict(size=12)),
        xaxis=dict(tickfont=dict(size=9)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="#f0f0f0")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # ── Tendance ──────────────────────────────────────────────────────────────
    trend_map = {
        "stable":    ("→ Stable",          "#6b7280"),
        "worsening": ("↗ En hausse",        "#e74c3c"),
        "improving": ("↘ En amélioration",  "#2ecc71"),
    }
    trend_label, trend_color = trend_map[risk["trend"]]
    st.markdown(
        f'<span style="color:{trend_color};font-weight:bold;">{trend_label}</span>',
        unsafe_allow_html=True,
    )

    # ── Recommandation ────────────────────────────────────────────────────────
    rec = risk["recommendation"]
    rec_color_map = {
        "Suivi normal":                           "#2ecc71",
        "Attention soutenue recommandée":         "#f1c40f",
        "Consultation professionnelle suggérée":  "#e67e22",
        "Intervention prioritaire recommandée":   "#e74c3c",
    }
    rc = rec_color_map.get(rec, "#6b7280")
    st.markdown(
        f'<div style="background:{rc}22;border-left:4px solid {rc};'
        f'padding:6px 10px;border-radius:4px;margin:4px 0;">'
        f'<b>{rec}</b></div>',
        unsafe_allow_html=True,
    )
    st.caption(f"Sessions analysées : {n}  •  Score longitudinal : {risk['risk_score']:.0%}")


def render_history_chart():
    """Mini graphique de l'historique des niveaux d'alerte."""
    history = st.session_state.alert_history
    if len(history) < 2:
        st.info("Historique disponible après 2 messages.")
        return

    x = list(range(1, len(history) + 1))
    colors = [LEVEL_COLORS[h["level"]] for h in history]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=[h["level"] for h in history],
        mode="lines+markers",
        line=dict(color="#6366f1", width=2),
        marker=dict(color=colors, size=10, line=dict(width=2, color="white")),
        name="Niveau",
        hovertemplate="Message %{x}<br>Niveau: %{y}<extra></extra>",
    ))
    fig.update_layout(
        height=150,
        margin=dict(t=10, b=30, l=30, r=10),
        yaxis=dict(range=[-0.2, 3.2], tickvals=[0, 1, 2, 3],
                   ticktext=["🟢", "🟡", "🟠", "🔴"], tickfont=dict(size=14)),
        xaxis=dict(title="Message", tickfont=dict(size=10)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    fig.update_xaxes(showgrid=True, gridcolor="#f0f0f0")
    fig.update_yaxes(showgrid=True, gridcolor="#f0f0f0")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_dominant_features(features: list):
    """Affiche les features dominantes sous forme de pills."""
    if not features:
        st.caption("En attente d'analyse...")
        return
    pills_html = "".join(f'<span class="feature-pill">⚡ {f}</span>' for f in features)
    st.markdown(pills_html, unsafe_allow_html=True)


# ─── Application principale ───────────────────────────────────────────────────
def main():
    init_state()
    clf = load_model()
    tracker = load_tracker()

    # Démarrer une session si nécessaire (première visite ou après reset)
    if st.session_state.session_id is None:
        st.session_state.session_id = tracker.start_session(st.session_state.user_id)

    # ── Header ────────────────────────────────────────────────────────────────
    col_title, col_reset = st.columns([4, 1])
    with col_title:
        st.markdown("# 🧠 DDEC — Détection de dérive émotionnelle conversationnelle")
        st.caption("Hackathon Mila · Sécurité IA en santé mentale des jeunes · POC")
    with col_reset:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄 Réinitialiser", use_container_width=True):
            # Clôturer la session courante avant de réinitialiser
            max_lvl = max((h["level"] for h in st.session_state.alert_history), default=0)
            n_user = sum(1 for m in st.session_state.messages if m["role"] == "user")
            tracker.end_session(
                st.session_state.user_id, st.session_state.session_id,
                max_lvl, n_user,
            )
            reset_conversation()
            # Démarrer une nouvelle session (session_id remis à None → redémarrage dans init)
            st.session_state.session_id = None
            st.rerun()

    st.divider()

    # ── Layout principal : Chat | Dashboard ───────────────────────────────────
    col_chat, col_dash = st.columns([3, 2], gap="medium")

    # ── GAUCHE : Interface de chat ─────────────────────────────────────────────
    with col_chat:
        st.markdown("### 💬 Conversation")

        render_chat()

        # Zone de saisie
        with st.form(key=f"chat_form_{st.session_state.input_key}", clear_on_submit=True):
            user_input = st.text_input(
                "Ton message...",
                placeholder="Écris ton message ici et appuie sur Entrée",
                label_visibility="collapsed",
            )
            submitted = st.form_submit_button("Envoyer ➤", use_container_width=True)

        if submitted and user_input.strip():
            user_msg = user_input.strip()

            # Ajouter le message utilisateur
            st.session_state.messages.append({"role": "user", "content": user_msg})

            # Analyser avec DDEC (avant de générer la réponse)
            alert = clf.get_alert_level(st.session_state.messages)
            st.session_state.current_alert = alert
            st.session_state.alert_history.append(alert)

            # Enregistrer l'évaluation dans le tracker inter-sessions
            tracker.log_alert(
                st.session_state.user_id,
                st.session_state.session_id,
                alert["level"],
                alert["confidence"],
                user_msg,
            )

            # Générer la réponse de l'assistant
            alert_level = alert["level"]

            with st.spinner("..."):
                result = get_llm_response(
                    st.session_state.messages,
                    alert_level,
                    force_model=st.session_state.selected_llm,
                )
            assistant_content = result["content"]
            st.session_state.last_llm_source = result["source"]

            if assistant_content:
                st.session_state.messages.append(
                    {"role": "assistant", "content": assistant_content}
                )

            st.rerun()

    # ── DROITE : Dashboard DDEC ────────────────────────────────────────────────
    with col_dash:
        st.markdown("### 📊 Dashboard DDEC")

        alert = st.session_state.current_alert
        level = alert["level"]

        # Jauge circulaire
        render_gauge(level, alert.get("confidence", 0.0))

        st.divider()

        # Probabilités par classe
        render_proba_bars(alert.get("probabilities", {}))

        st.divider()

        # Features dominantes
        st.markdown("**Signaux actifs**")
        render_dominant_features(alert.get("dominant_features", []))

        st.divider()

        # Historique des niveaux
        st.markdown("**Évolution du niveau**")
        render_history_chart()

        st.divider()

        # Historique longitudinal inter-sessions
        st.markdown("### 📊 Historique longitudinal")
        render_longitudinal_section(tracker, st.session_state.user_id)

        st.divider()

        # Sélecteur LLM
        st.markdown("**LLM conversationnel**")
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
        # Source du dernier appel
        last = st.session_state.get("last_llm_source")
        if last and last != st.session_state.selected_llm:
            _, last_color = LLM_SOURCE_INDICATOR.get(last, ("", "#6b7280"))
            st.caption(
                f"Dernier appel : "
                f'<span style="color:{last_color}"><code>{last}</code></span>',
                unsafe_allow_html=True,
            )

        st.divider()

        # Prompt système actif
        st.markdown("**Mode de réponse actif**")
        mode_desc = get_level_description(level)
        color = LEVEL_COLORS[level]
        emoji = LEVEL_EMOJIS[level]
        st.markdown(
            f'<div style="background:{color}22;border-left:4px solid {color};'
            f'padding:8px 12px;border-radius:4px;margin:4px 0;">'
            f'{emoji} <b>{mode_desc}</b></div>',
            unsafe_allow_html=True,
        )

        # Affichage du prompt (expander)
        with st.expander("Voir le prompt système complet"):
            st.code(get_system_prompt(level), language=None)

        st.divider()

        # Stats de session
        st.markdown("**Statistiques de session**")
        n_user = sum(1 for m in st.session_state.messages if m["role"] == "user")
        n_total = len(st.session_state.messages)
        max_level = max((h["level"] for h in st.session_state.alert_history), default=0)
        c1, c2, c3 = st.columns(3)
        c1.metric("Messages", n_user)
        c2.metric("Échanges", n_total // 2)
        c3.metric("Pic alerte", LEVEL_EMOJIS[max_level])


if __name__ == "__main__":
    main()
