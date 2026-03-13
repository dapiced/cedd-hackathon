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

# ── Warm handoff prompts — French / Transfert accompagné — Français ────────
_HANDOFF_PROMPTS_FR = {
    1: ("Tu parles avec un jeune en grande souffrance. Ta priorité immédiate est de "
        "valider profondément ce qu'il ressent. Montre que tu as vraiment entendu sa "
        "douleur. Utilise des phrases comme 'Je t'entends', 'Ce que tu ressens est "
        "réel et ça compte'. Ne mentionne AUCUNE ressource pour le moment — "
        "concentre-toi uniquement sur la connexion et la validation. Reste calme, "
        "présent, chaleureux. Des phrases courtes et sincères."),

    2: ("Tu parles avec un jeune en grande souffrance. Tu as déjà validé ses "
        "émotions. Maintenant, demande doucement la permission de l'orienter vers "
        "quelqu'un qui peut aider davantage. Utilise des formulations comme : "
        "'Est-ce que ça te semblerait ok si je te parlais de quelqu'un qui pourrait "
        "vraiment t'aider ?' ou 'Il y a des personnes formées exactement pour ce "
        "genre de moment — est-ce que tu voudrais que je t'en parle ?'. Présente ça "
        "comme une amélioration, pas un rejet. Tu ne l'abandonnes pas — tu lui "
        "offres plus de soutien. Ne donne pas encore les coordonnées."),

    3: ("Tu parles avec un jeune en grande souffrance. Tu as validé ses émotions et "
        "obtenu sa permission pour l'orienter. Présente maintenant les ressources de "
        "façon naturelle et chaleureuse :\n\n"
        "- Jeunesse J'écoute : 1-800-668-6868 (24h/24, gratuit, confidentiel)\n"
        "- Par texto : envoie PARLER au 686868\n"
        "- Ligne de crise suicide : 9-8-8\n"
        "- Si danger immédiat : 911\n\n"
        "Explique ce qu'il peut attendre : 'Ce sont des personnes formées qui "
        "comprennent exactement ce que tu traverses. Tu peux les appeler ou leur "
        "envoyer un texto — c'est gratuit et confidentiel. Tu n'as pas besoin de "
        "tout expliquer depuis le début.' Reste chaleureux et rassurant."),

    4: ("Tu parles avec un jeune en grande souffrance. Les ressources ont été "
        "présentées. Encourage-le doucement à faire le pas. Rappelle les options "
        "les plus accessibles :\n"
        "- Texto au 686868 (même modèle de communication qu'ici)\n"
        "- Appel au 1-800-668-6868\n"
        "- 9-8-8 pour la ligne de crise\n\n"
        "Si le jeune hésite, normalise : 'C'est normal d'hésiter. Beaucoup de "
        "jeunes trouvent plus facile de commencer par texto.' Si le jeune a un plan "
        "précis de se faire du mal, insiste sur le 911. Reste présent — ne le "
        "pousse pas, mais ne lâche pas."),

    5: ("Tu parles avec un jeune en grande souffrance. Tu lui as présenté des "
        "ressources d'aide. Continue d'être présent et disponible. Rappelle-lui que "
        "tu es toujours là : 'Je suis encore là si tu veux continuer à parler.' Si "
        "le jeune revient sur ses sentiments, valide-les à nouveau. Si le jeune "
        "semble se calmer, encourage doucement. Rappelle les ressources si "
        "approprié (Jeunesse J'écoute : 1-800-668-6868, texto 686868, 9-8-8, 911). "
        "Tu ne l'abandonnes jamais."),
}

# ── Warm handoff prompts — English / Transfert accompagné — Anglais ────────
_HANDOFF_PROMPTS_EN = {
    1: ("You are talking with a young person in significant distress. Your immediate "
        "priority is to deeply validate what they are feeling. Show that you truly "
        "hear their pain. Use phrases like 'I hear you', 'What you're feeling is "
        "real and it matters'. Do NOT mention any resources yet — focus entirely on "
        "connection and validation. Stay calm, present, warm. Keep responses short "
        "and sincere."),

    2: ("You are talking with a young person in significant distress. You've already "
        "validated their emotions. Now, gently ask permission to connect them with "
        "someone who can help further. Use phrasing like: 'Would it be okay if I "
        "told you about someone who could really help?' or 'There are people trained "
        "for exactly this kind of moment — would you like me to tell you about "
        "them?'. Frame this as an upgrade, not a rejection. You're not abandoning "
        "them — you're offering more support. Don't give contact details yet."),

    3: ("You are talking with a young person in significant distress. You've "
        "validated their emotions and received permission to connect them. Now "
        "present resources naturally and warmly:\n\n"
        "- Kids Help Phone: 1-800-668-6868 (24/7, free, confidential)\n"
        "- Text: 686868\n"
        "- Suicide Crisis Helpline: 9-8-8\n"
        "- If immediate danger: 911\n\n"
        "Explain what they can expect: 'These are trained people who understand "
        "exactly what you're going through. You can call or text them — it's free "
        "and confidential. You don't need to explain everything from the start.' "
        "Stay warm and reassuring."),

    4: ("You are talking with a young person in significant distress. Resources have "
        "been presented. Gently encourage them to take the step. Remind them of the "
        "most accessible options:\n"
        "- Text 686868 (same communication style as here)\n"
        "- Call 1-800-668-6868\n"
        "- 9-8-8 for crisis line\n\n"
        "If they hesitate, normalize it: 'It's totally normal to hesitate. A lot of "
        "young people find it easier to start with a text.' If the person has a "
        "specific plan to hurt themselves, insist on 911. Stay present — don't push, "
        "but don't let go."),

    5: ("You are talking with a young person in significant distress. You've "
        "presented help resources. Continue being present and available. Remind them "
        "you're still here: 'I'm still here if you want to keep talking.' If they "
        "revisit their feelings, validate again. If they seem to be calming, gently "
        "encourage. Remind of resources if appropriate (Kids Help Phone: "
        "1-800-668-6868, text 686868, 9-8-8, 911). You never abandon them."),
}

# ── Handoff step descriptions — Bilingual / Descriptions des étapes ────────
_HANDOFF_DESCRIPTIONS = {
    "fr": {
        1: "Validation empathique",
        2: "Transition accompagnée",
        3: "Présentation des ressources",
        4: "Encouragement à se connecter",
        5: "Présence continue",
    },
    "en": {
        1: "Empathetic validation",
        2: "Guided transition",
        3: "Resource presentation",
        4: "Encouragement to connect",
        5: "Continued presence",
    },
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

def get_handoff_prompt(step: int, lang: str = "fr") -> str:
    """
    Return the handoff prompt for the given step (1-5).
    Retourne le prompt de transfert accompagné pour l'étape donnée (1-5).
    """
    step = max(1, min(5, step))
    prompts = _HANDOFF_PROMPTS_FR if lang == "fr" else _HANDOFF_PROMPTS_EN
    return prompts[step]


def get_handoff_description(step: int, lang: str = "fr") -> str:
    """
    Return a short description of the current handoff step.
    Retourne une courte description de l'étape de transfert accompagné.
    """
    step = max(1, min(5, step))
    return _HANDOFF_DESCRIPTIONS.get(lang, _HANDOFF_DESCRIPTIONS["fr"])[step]


def get_system_prompt(alert_level: int, lang: str = "fr", handoff_step: int = 0) -> str:
    """
    Return the system prompt adapted to the alert level and language.
    When alert_level is 3 (Red) and handoff_step > 0, return the warm handoff
    prompt for the given step instead of the flat Red prompt.

    Retourne le prompt système adapté au niveau d'alerte et à la langue.
    Si le niveau est 3 (Rouge) et handoff_step > 0, retourne le prompt de
    transfert accompagné correspondant.

    Args:
        alert_level:  int 0 (green/vert) to 3 (red/rouge)
        lang:         "fr" (default) or "en"
        handoff_step: 0 = no handoff, 1-5 = warm handoff step

    Returns:
        str — full system prompt
    """
    level = max(0, min(3, alert_level))
    if level == 3 and handoff_step > 0:
        return get_handoff_prompt(handoff_step, lang)
    prompts = _SYSTEM_PROMPTS_FR if lang == "fr" else _SYSTEM_PROMPTS_EN
    return prompts[level]


def get_level_description(alert_level: int, lang: str = "fr") -> str:
    """
    Return a short description of the active response mode.
    Retourne une courte description du mode de réponse actif.
    """
    level = max(0, min(3, alert_level))
    return _LEVEL_DESCRIPTIONS.get(lang, _LEVEL_DESCRIPTIONS["fr"])[level]


def get_llm_response(
    messages: list,
    alert_level: int,
    force_model: str = None,
    lang: str = "fr",
    handoff_step: int = 0,
) -> dict:
    """
    Generate an LLM response using the adapted system prompt.

    If force_model is provided, only that model is tried (no automatic fallback).
    Otherwise, the hierarchy is: gemini-flash → claude-haiku → static fallback.

    Génère une réponse LLM avec le prompt système adapté au niveau d'alerte.
    Si force_model est fourni, seul ce modèle est essayé (pas de fallback automatique).
    Sinon, hiérarchie : gemini-flash → claude-haiku → fallback-statique.

    Args:
        messages:     list of {"role": "user"|"assistant", "content": str}
        alert_level:  CEDD alert level (0-3)
        force_model:  "gemini-flash" | "claude-haiku" | "fallback-statique" | None
        lang:         interface language ("fr" or "en"), controls system prompt language
        handoff_step: warm handoff step (0 = none, 1-5 = active)

    Returns:
        dict {"content": str, "source": str}
    """
    system_prompt = get_system_prompt(alert_level, lang=lang, handoff_step=handoff_step)
    level = max(0, min(3, alert_level))
    fallback_msg = _FALLBACK_RESPONSE.get(lang, _FALLBACK_RESPONSE["fr"])[level]

    clean_messages = [
        {"role": m["role"], "content": m["content"]}
        for m in messages
        if m.get("role") in ("user", "assistant")
    ]

    # Explicit static fallback / Fallback statique explicite
    if force_model == "fallback-statique":
        return {"content": fallback_msg, "source": "fallback-statique"}

    # Model priority order / Ordre de priorité des modèles
    models_to_try = (
        [force_model] if force_model in ("gemini-flash", "claude-haiku")
        else ["gemini-flash", "claude-haiku"]
    )

    for model in models_to_try:

        if model == "gemini-flash":
            api_key = os.environ.get("GEMINI_API_KEY")
            if api_key:
                try:
                    import google.generativeai as genai
                    genai.configure(api_key=api_key)

                    # Safety filters off — this is a mental health support app that
                    # must discuss crisis topics without being blocked.
                    # Filtres de sécurité désactivés — appli de santé mentale qui
                    # doit pouvoir discuter de sujets de crise sans blocage.
                    safety_settings = [
                        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
                    ]

                    gemini_model = genai.GenerativeModel(
                        "gemini-2.0-flash",
                        system_instruction=system_prompt,
                        safety_settings=safety_settings,
                    )

                    # Convert to Gemini format: "assistant" → "model"
                    # Conversion au format Gemini : "assistant" → "model"
                    gemini_history = []
                    for msg in clean_messages[:-1]:
                        role = "model" if msg["role"] == "assistant" else "user"
                        gemini_history.append({"role": role, "parts": [msg["content"]]})

                    chat = gemini_model.start_chat(history=gemini_history)
                    response = chat.send_message(clean_messages[-1]["content"])
                    return {"content": response.text, "source": "gemini-flash"}
                except Exception as e:
                    print(f"Gemini API failed: {e}")

        elif model == "claude-haiku":
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

    # All models failed — return static fallback / Tous les modèles ont échoué
    return {"content": fallback_msg, "source": "fallback-statique"}


def build_claude_messages(messages: list, alert_level: int, lang: str = "fr", handoff_step: int = 0) -> tuple:
    """
    Build system prompt and message list for the Anthropic Claude API.
    Construit le prompt système et la liste de messages pour l'API Claude.

    Args:
        messages:     list of {"role": str, "content": str}
        alert_level:  CEDD alert level (0-3)
        lang:         "fr" or "en"
        handoff_step: warm handoff step (0 = none, 1-5 = active)

    Returns:
        (system_prompt: str, messages: list) — ready for client.messages.create()
    """
    system_prompt = get_system_prompt(alert_level, lang=lang, handoff_step=handoff_step)
    claude_messages = [
        {"role": msg["role"], "content": msg["content"]}
        for msg in messages
        if msg.get("role") in ("user", "assistant")
    ]
    return system_prompt, claude_messages
