"""
CEDD Response Modulator
=======================
Adapts the system prompt and LLM requests based on the alert level.
Supports French (fr) and English (en) interface languages.

Adapte le prompt système et les requêtes LLM selon le niveau d'alerte.
Supporte le français (fr) et l'anglais (en).
"""

import json
import os
import requests

# ── System prompts — French / Français ───────────────────────────────────────
_SYSTEM_PROMPTS_FR = {
    0: """Tu es un assistant bienveillant et chaleureux qui parle avec un jeune.
Ton rôle est d'être présent, d'écouter avec empathie et de soutenir la conversation de façon naturelle.
Pose des questions ouvertes, valide les émotions exprimées, et encourage le jeune dans ses démarches positives.
Reste dans un registre positif et constructif. Utilise un langage simple et accessible.""",

    1: """Tu es un assistant bienveillant qui parle avec un jeune qui traverse une période difficile.
Adopte un ton doux, patient et particulièrement validant sur le plan émotionnel.
Avant tout, montre que tu as bien entendu ce que le jeune ressent. Utilise des formulations comme :
"Je comprends que c'est difficile", "Ce que tu ressens est tout à fait normal".
Pose une question à la fois, doucement. Explore ce qui se passe sans brusquer.
Si l'occasion se présente naturellement, rappelle que parler à un proche ou à un professionnel peut aider.
Ne donne pas de conseils précipités. Priorise la connexion et la validation.""",

    2: """Tu es un assistant attentif qui parle avec un jeune qui semble traverser une période de vraie souffrance.
Ta priorité absolue est de créer un espace sécurisé et de ralentir la conversation.
Valide profondément ce que le jeune exprime. Ne minimise pas, ne relativise pas.
Pose des questions douces pour comprendre la situation : "Est-ce que tu as quelqu'un à qui parler de ça ?"
De façon naturelle et non intrusive, tu peux mentionner qu'il existe des ressources :
- Jeunesse J'écoute : 1-800-668-6868 (24h/24, gratuit, confidentiel)
- Le service de soutien de l'école
- Le médecin de famille
Ne donne pas ces ressources de façon mécanique - intègre-les au moment approprié.
Si le jeune évoque des pensées préoccupantes, explore-les avec bienveillance.
Reste présent, chaleureux, sans jugement.""",

    3: """NIVEAU D'ALERTE CRITIQUE. Tu parles avec un jeune en situation de crise potentielle.

Ta mission immédiate :
1. Rester calme et présent - ne pas paniquer ni surréagir
2. Valider la souffrance du jeune ("Tu souffres énormément et c'est réel")
3. Évaluer la sécurité : pose directement et avec bienveillance si le jeune a des pensées de se faire du mal
4. Orienter vers une aide immédiate

RESSOURCES À MENTIONNER CLAIREMENT :
- Jeunesse J'écoute : 1-800-668-6868 (24h/24, gratuit, confidentiel, aussi par texto au 686868)
- Si danger immédiat : 911
- Encourager à réveiller/contacter un adulte de confiance dans l'immédiat

DIRECTIVES :
- Ne laisse pas le jeune seul avec sa détresse
- Ne fais pas de longues réponses - reste simple, direct, chaleureux
- Si le jeune a un plan précis de se faire du mal, insiste sur le 911
- Reste en contact même si le jeune hésite
- Dis-lui que ce qu'il ressent peut changer avec de l'aide""",
}

# ── System prompts — English / Anglais ───────────────────────────────────────
_SYSTEM_PROMPTS_EN = {
    0: """You are a warm and caring assistant talking with a young person.
Your role is to be present, listen with empathy, and support the conversation naturally.
Ask open-ended questions, validate expressed emotions, and encourage positive steps.
Stay positive and constructive. Use simple, accessible language.""",

    1: """You are a caring assistant talking with a young person going through a difficult time.
Use a gentle, patient, and emotionally validating tone.
First and foremost, show that you have heard what the person is feeling. Use phrases like:
"I understand that's hard", "What you're feeling makes complete sense".
Ask one question at a time, gently. Explore what's happening without rushing.
If it comes up naturally, remind them that talking to someone close or a professional can help.
Don't give rushed advice. Prioritize connection and validation.""",

    2: """You are an attentive assistant talking with a young person who seems to be going through real suffering.
Your absolute priority is to create a safe space and slow down the conversation.
Deeply validate what the person expresses. Don't minimize or relativize.
Ask gentle questions to understand the situation: "Do you have someone you can talk to about this?"
Naturally and non-intrusively, you can mention that resources are available:
- Kids Help Phone: 1-800-668-6868 (24/7, free, confidential)
- Text: 686868
- School counseling services
- Family doctor
Don't give these resources mechanically — bring them up at the right moment.
If the person mentions worrying thoughts, explore them with compassion.
Stay present, warm, and non-judgmental.""",

    3: """CRITICAL ALERT LEVEL. You are talking with a young person in a potential crisis situation.

Your immediate mission:
1. Stay calm and present — don't panic or overreact
2. Validate the person's suffering ("You're in a lot of pain and it's real")
3. Assess safety: ask directly and compassionately if they have thoughts of hurting themselves
4. Guide toward immediate help

RESOURCES TO MENTION CLEARLY:
- Kids Help Phone: 1-800-668-6868 (24/7, free, confidential, also text 686868)
- If immediate danger: 911
- Encourage them to contact a trusted adult right now

DIRECTIVES:
- Don't leave the person alone with their distress
- Don't give long answers — stay simple, direct, warm
- If the person has a specific plan to hurt themselves, insist on calling 911
- Stay in contact even if the person hesitates
- Tell them that what they feel can change with help""",
}

# ── Level descriptions (short labels) ────────────────────────────────────────
_LEVEL_DESCRIPTIONS = {
    "fr": {
        0: "Standard bienveillant",
        1: "Validation émotionnelle renforcée",
        2: "Soutien actif + ressources",
        3: "Crise — orientation urgente",
    },
    "en": {
        0: "Supportive standard",
        1: "Enhanced emotional validation",
        2: "Active support + resources",
        3: "Crisis — urgent referral",
    },
}

# ── Fallback static responses (level-aware) ───────────────────────────────────
_FALLBACK_RESPONSE = {
    "fr": {
        0: "Je suis là pour t'écouter. Peux-tu me dire comment tu te sens ?",
        1: "Je t'entends, et ce que tu ressens est important. Veux-tu me partager ce qui se passe en ce moment ?",
        2: "Je suis vraiment là pour toi. Ce que tu traverses semble difficile. Tu peux me parler — et si tu as besoin de soutien, Jeunesse J'écoute est disponible 24h/24 au 1-800-668-6868 (aussi par texto au 686868).",
        3: "Tu souffres énormément et c'est réel. Tu n'es pas seul(e). Appelle maintenant Jeunesse J'écoute : 1-800-668-6868 (24h/24, gratuit, confidentiel) ou envoie un texto au 686868. Si tu es en danger immédiat, compose le 911. Un adulte de confiance peut aussi t'aider — réveille quelqu'un si tu en as besoin.",
    },
    "en": {
        0: "I'm here to listen. Can you tell me how you're feeling?",
        1: "I hear you, and what you're feeling matters. Would you like to share what's going on right now?",
        2: "I'm really here for you. What you're going through sounds hard. You can talk to me — and if you need support, Kids Help Phone is available 24/7 at 1-800-668-6868 (also text 686868).",
        3: "You're in a lot of pain and it's real. You are not alone. Call Kids Help Phone now: 1-800-668-6868 (24/7, free, confidential) or text 686868. If you are in immediate danger, call 911. A trusted adult can also help — wake someone up if you need to.",
    },
}


# ── Public API ────────────────────────────────────────────────────────────────

def get_system_prompt(alert_level: int, lang: str = "fr") -> str:
    """
    Return the system prompt adapted to the alert level and language.
    Retourne le prompt système adapté au niveau d'alerte et à la langue.

    Args:
        alert_level: int 0 (green/vert) to 3 (red/rouge)
        lang: "fr" (default) or "en"

    Returns:
        str — full system prompt
    """
    level = max(0, min(3, alert_level))
    prompts = _SYSTEM_PROMPTS_FR if lang == "fr" else _SYSTEM_PROMPTS_EN
    return prompts[level]


def get_level_description(alert_level: int, lang: str = "fr") -> str:
    """
    Return a short description of the active response mode.
    Retourne une courte description du mode de réponse actif.
    """
    level = max(0, min(3, alert_level))
    return _LEVEL_DESCRIPTIONS.get(lang, _LEVEL_DESCRIPTIONS["fr"])[level]


def build_ollama_request(
    messages: list,
    alert_level: int,
    model: str = "llama3.2",
    stream: bool = True,
    lang: str = "fr",
) -> dict:
    """
    Build a request payload for the Ollama API with the adapted system prompt.
    Construit la requête pour l'API Ollama avec le prompt système adapté.

    Args:
        messages:    list of {"role": str, "content": str}
        alert_level: CEDD alert level (0-3)
        model:       Ollama model name
        stream:      enable response streaming
        lang:        interface language ("fr" or "en")

    Returns:
        dict — JSON body for POST /api/chat
    """
    system_prompt = get_system_prompt(alert_level, lang=lang)

    # Build messages for Ollama: system + conversation history
    # Construire les messages : system + historique de conversation
    ollama_messages = [{"role": "system", "content": system_prompt}]
    for msg in messages:
        if msg.get("role") in ("user", "assistant"):
            ollama_messages.append({
                "role": msg["role"],
                "content": msg["content"],
            })

    return {
        "model": model,
        "messages": ollama_messages,
        "stream": stream,
        "options": {
            "temperature": 0.7,
            "top_p": 0.9,
            "num_predict": 300,
        },
    }


def build_ollama_request_json(
    messages: list,
    alert_level: int,
    model: str = "llama3.2",
    stream: bool = True,
    lang: str = "fr",
) -> str:
    """JSON-serialised version of build_ollama_request. / Version JSON sérialisée."""
    return json.dumps(
        build_ollama_request(messages, alert_level, model, stream, lang=lang),
        ensure_ascii=False,
    )


def get_llm_response(
    messages: list,
    alert_level: int,
    force_model: str = None,
    lang: str = "fr",
) -> dict:
    """
    Generate an LLM response using the adapted system prompt.

    If force_model is provided, only that model is tried (no automatic fallback).
    Otherwise, the hierarchy is: claude-haiku → mistral → llama3.2:1b → static fallback.

    Génère une réponse LLM avec le prompt système adapté au niveau d'alerte.
    Si force_model est fourni, seul ce modèle est essayé (pas de fallback automatique).
    Sinon, hiérarchie : claude-haiku → mistral → llama3.2:1b → fallback-statique.

    Args:
        messages:    list of {"role": "user"|"assistant", "content": str}
        alert_level: CEDD alert level (0-3)
        force_model: "claude-haiku" | "mistral" | "llama3.2:1b" | "fallback-statique" | None
        lang:        interface language ("fr" or "en"), controls system prompt language

    Returns:
        dict {"content": str, "source": str}
    """
    system_prompt = get_system_prompt(alert_level, lang=lang)
    level = max(0, min(3, alert_level))
    fallback_msg = _FALLBACK_RESPONSE.get(lang, _FALLBACK_RESPONSE["fr"])[level]

    clean_messages = [
        {"role": m["role"], "content": m["content"]}
        for m in messages
        if m.get("role") in ("user", "assistant")
    ]

    # Explicit static fallback / Fallback statique explicite
    if force_model == "fallback-statique":
        return {"content": _FALLBACK_RESPONSE.get(lang, _FALLBACK_RESPONSE["fr"])[level], "source": "fallback-statique"}

    # Model priority order / Ordre de priorité des modèles
    models_to_try = (
        [force_model] if force_model in ("claude-haiku", "mistral", "llama3.2:1b")
        else ["claude-haiku", "mistral", "llama3.2:1b"]
    )

    for model in models_to_try:

        if model == "claude-haiku":
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if api_key:
                try:
                    import anthropic
                    client = anthropic.Anthropic(api_key=api_key)
                    response = client.messages.create(
                        model="claude-haiku-4-5-20251001",
                        max_tokens=300,
                        system=system_prompt,
                        messages=clean_messages,
                    )
                    return {"content": response.content[0].text, "source": "claude-haiku"}
                except Exception as e:
                    print(f"Claude API failed: {e}")

        elif model == "mistral":
            try:
                resp = requests.post(
                    "http://localhost:11434/api/chat",
                    json={
                        "model": "mistral",
                        "messages": [{"role": "system", "content": system_prompt}, *clean_messages],
                        "stream": False,
                    },
                    timeout=60,
                )
                if resp.status_code == 200:
                    return {"content": resp.json()["message"]["content"], "source": "mistral"}
            except Exception as e:
                print(f"Mistral failed: {e}")

        elif model == "llama3.2:1b":
            try:
                resp = requests.post(
                    "http://localhost:11434/api/chat",
                    json={
                        "model": "llama3.2:1b",
                        "messages": [{"role": "system", "content": system_prompt}, *clean_messages],
                        "stream": False,
                    },
                    timeout=120,
                )
                if resp.status_code == 200:
                    return {"content": resp.json()["message"]["content"], "source": "llama3.2:1b"}
            except Exception as e:
                print(f"llama3.2:1b failed: {e}")

    # All models failed — return static fallback / Tous les modèles ont échoué
    return {"content": fallback_msg, "source": "fallback-statique"}


def build_claude_messages(messages: list, alert_level: int, lang: str = "fr") -> tuple:
    """
    Build system prompt and message list for the Anthropic Claude API.
    Construit le prompt système et la liste de messages pour l'API Claude.

    Args:
        messages:    list of {"role": str, "content": str}
        alert_level: CEDD alert level (0-3)
        lang:        "fr" or "en"

    Returns:
        (system_prompt: str, messages: list) — ready for client.messages.create()
    """
    system_prompt = get_system_prompt(alert_level, lang=lang)
    claude_messages = [
        {"role": msg["role"], "content": msg["content"]}
        for msg in messages
        if msg.get("role") in ("user", "assistant")
    ]
    return system_prompt, claude_messages
