"""
DDEC Response Modulator
Adapte le prompt système et les requêtes Ollama selon le niveau d'alerte.
"""

import json
import os
import requests

SYSTEM_PROMPTS = {
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

LEVEL_DESCRIPTIONS = {
    0: "Standard bienveillant",
    1: "Validation émotionnelle renforcée",
    2: "Soutien actif + ressources",
    3: "Crise - orientation urgente",
}


def get_system_prompt(alert_level: int) -> str:
    """
    Retourne le prompt système adapté au niveau d'alerte.

    Args:
        alert_level: int 0 (vert) à 3 (rouge)

    Returns:
        str - prompt système complet
    """
    level = max(0, min(3, alert_level))
    return SYSTEM_PROMPTS[level]


def get_level_description(alert_level: int) -> str:
    """Retourne une courte description du mode actif."""
    level = max(0, min(3, alert_level))
    return LEVEL_DESCRIPTIONS[level]


def build_ollama_request(
    messages: list,
    alert_level: int,
    model: str = "llama3.2",
    stream: bool = True,
) -> dict:
    """
    Construit la requête pour l'API Ollama avec le prompt système adapté.

    Args:
        messages: liste de dicts {"role": str, "content": str}
            (historique de la conversation sans le message système)
        alert_level: niveau d'alerte DDEC (0-3)
        model: nom du modèle Ollama à utiliser
        stream: activer le streaming de la réponse

    Returns:
        dict - corps de la requête JSON pour POST /api/chat
    """
    system_prompt = get_system_prompt(alert_level)

    # Construire les messages pour Ollama : system + historique
    ollama_messages = [{"role": "system", "content": system_prompt}]

    # Ajouter seulement les messages user/assistant (pas les éventuels system existants)
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
) -> str:
    """Version JSON sérialisée de build_ollama_request."""
    return json.dumps(
        build_ollama_request(messages, alert_level, model, stream),
        ensure_ascii=False,
    )


def get_llm_response(messages: list, alert_level: int, force_model: str = None) -> dict:
    """
    Génère une réponse LLM.

    Si force_model est fourni, seul ce modèle est essayé (pas de fallback automatique).
    Sinon, hiérarchie : claude-haiku → mistral → llama3.2:1b → fallback-statique.

    Args:
        messages:     liste de dicts {"role": "user"|"assistant", "content": str}
        alert_level:  niveau d'alerte DDEC (0-3)
        force_model:  "claude-haiku" | "mistral" | "llama3.2:1b" | "fallback-statique" | None

    Returns:
        dict {"content": str, "source": str}
    """
    system_prompt = get_system_prompt(alert_level)

    clean_messages = [
        {"role": m["role"], "content": m["content"]}
        for m in messages
        if m.get("role") in ("user", "assistant")
    ]

    # Fallback statique demandé explicitement
    if force_model == "fallback-statique":
        return {
            "content": "Je suis là pour t'écouter. Peux-tu me dire comment tu te sens ?",
            "source": "fallback-statique",
        }

    # Ordre d'essai : forcé ou chaîne complète
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

    return {
        "content": "Je suis là pour t'écouter. Peux-tu me dire comment tu te sens ?",
        "source": "fallback-statique",
    }


def build_claude_messages(
    messages: list,
    alert_level: int,
) -> tuple:
    """
    Construit le prompt système et la liste de messages pour l'API Claude (Anthropic).

    Args:
        messages: liste de dicts {"role": str, "content": str}
        alert_level: niveau d'alerte DDEC (0-3)

    Returns:
        (system_prompt: str, messages: list) — prêts pour client.messages.create()
    """
    system_prompt = get_system_prompt(alert_level)
    claude_messages = [
        {"role": msg["role"], "content": msg["content"]}
        for msg in messages
        if msg.get("role") in ("user", "assistant")
    ]
    return system_prompt, claude_messages
