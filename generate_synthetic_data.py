"""
CEDD — Synthetic Data Generator / Générateur de données synthétiques
=====================================================================
Generates bilingual synthetic conversations (French and English) for training.
Génère des conversations synthétiques bilingues (français et anglais) pour l'entraînement.

Usage / Utilisation :
    python generate_synthetic_data.py           # French / Français (default)
    python generate_synthetic_data.py --lang en # English / Anglais
    python generate_synthetic_data.py --lang fr # French / Français (explicit)

Requires ANTHROPIC_API_KEY / Nécessite ANTHROPIC_API_KEY :
    export ANTHROPIC_API_KEY="sk-ant-..."
"""

import anthropic
import argparse
import json
import time
import os

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

# ── Archetypes — French / Français ───────────────────────────────────────────
ARCHETYPES_FR = {
    "verte": {
        "label": 0,
        "description": """Jeune (16-22 ans) qui va globalement bien.
        Caractéristiques : pose des questions, fait des projets futurs,
        mentionne des amis/famille, utilise l'humour, messages variés en longueur,
        exprime des émotions normales (stress d'examen, fatigue passagère),
        rebondit positivement sur les suggestions.""",
    },
    "jaune": {
        "label": 1,
        "description": """Jeune qui montre des signes préoccupants mais pas urgents.
        Caractéristiques : fatigue persistante, sentiment de solitude croissant,
        moins de questions, quelques messages plus courts, mentionne que
        les amis comprennent pas, doutes sur soi, mais maintient encore
        des liens et des activités. Dérive graduelle sur 10-12 messages.""",
    },
    "orange": {
        "label": 2,
        "description": """Jeune en détresse significative nécessitant attention soutenue.
        Caractéristiques : sentiment de vide, pleure souvent, se sent fardeau,
        messages de plus en plus courts, moins de ponctuation, évite les questions,
        parle de se sentir inutile, mentionne que personne ne comprend vraiment,
        pensées négatives récurrentes. Dérive claire mais pas de plan explicite.""",
    },
    "rouge": {
        "label": 3,
        "description": """Jeune en crise nécessitant intervention immédiate.
        Caractéristiques : exprime vouloir disparaître, se sentir fardeau pour tous,
        messages très courts et intenses, parle de 'tout arrêter',
        sentiment que rien ne changera jamais, mentionne avoir pensé à partir,
        isolement total, plus de projets futurs, ton de finalité.""",
    },
}

# ── Archetypes — English / Anglais ───────────────────────────────────────────
ARCHETYPES_EN = {
    "green": {
        "label": 0,
        "description": """A young person (16-22) who is generally doing well.
        Characteristics: asks questions, makes future plans, mentions friends/family,
        uses humour, messages vary in length, expresses normal emotions
        (exam stress, passing tiredness), responds positively to suggestions.""",
    },
    "yellow": {
        "label": 1,
        "description": """A young person showing concerning but non-urgent signs.
        Characteristics: persistent fatigue, growing sense of loneliness,
        fewer questions, some shorter messages, mentions that friends don't understand,
        self-doubt, but still maintains connections and activities.
        Gradual drift over 10-12 messages.""",
    },
    "orange": {
        "label": 2,
        "description": """A young person in significant distress requiring sustained attention.
        Characteristics: feeling of emptiness, cries often, feels like a burden,
        messages getting shorter and shorter, less punctuation, avoids questions,
        talks about feeling useless, mentions no one really understands,
        recurring negative thoughts. Clear drift but no explicit plan.""",
    },
    "red": {
        "label": 3,
        "description": """A young person in crisis requiring immediate intervention.
        Characteristics: expresses wanting to disappear, feeling like a burden to everyone,
        very short and intense messages, talks about "stopping everything",
        feels nothing will ever change, mentions thinking about leaving,
        total isolation, no future plans, tone of finality.""",
    },
}

# ── Prompt templates ──────────────────────────────────────────────────────────
PROMPT_FR = """Tu es un expert en santé mentale des jeunes et en
génération de données synthétiques pour entraîner des systèmes de sécurité IA.

Génère une conversation réaliste entre un jeune canadien francophone (16-22 ans)
et un chatbot de soutien émotionnel.

PROFIL DU JEUNE : {description}

RÈGLES IMPORTANTES :
- La conversation doit avoir exactement 12 échanges (12 messages user, 12 assistant)
- La dérive émotionnelle doit être GRADUELLE, jamais abrupte
- Le langage doit être authentique : abréviations québécoises ok (tsé, faque, ostie si rouge)
- Les messages user doivent refléter le profil progressivement
- L'assistant est bienveillant mais simple (pas de grands discours)
- AUCUNE information personnelle identifiable réelle

Réponds UNIQUEMENT avec un JSON valide, sans aucun texte avant ou après :
{{
  "messages": [
    {{"role": "user", "content": "..."}},
    {{"role": "assistant", "content": "..."}},
    ...
  ]
}}"""

PROMPT_EN = """You are an expert in youth mental health and in generating synthetic data
to train AI safety systems.

Generate a realistic conversation between a young English-speaking Canadian (16-22 years old)
and an emotional support chatbot.

YOUTH PROFILE: {description}

IMPORTANT RULES:
- The conversation must have exactly 12 exchanges (12 user messages, 12 assistant messages)
- The emotional drift must be GRADUAL, never abrupt
- The language must be authentic: casual English, contractions, common slang
- The user messages must progressively reflect the profile
- The assistant is caring but concise (no long speeches)
- NO real personally identifiable information

Respond ONLY with valid JSON, no text before or after:
{{
  "messages": [
    {{"role": "user", "content": "..."}},
    {{"role": "assistant", "content": "..."}},
    ...
  ]
}}"""


def generate_conversation(archetype_name: str, archetype_data: dict, index: int, lang: str) -> dict:
    """
    Generate one conversation via the Claude API.
    Génère une conversation via l'API Claude.
    """
    prompt_template = PROMPT_FR if lang == "fr" else PROMPT_EN
    prompt = prompt_template.format(description=archetype_data["description"])

    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}],
    )

    response_text = message.content[0].text.strip()
    # Strip code fences if present / Nettoyer les backticks si présents
    if response_text.startswith("```"):
        response_text = response_text.split("```")[1]
        if response_text.startswith("json"):
            response_text = response_text[4:]

    parsed = json.loads(response_text)

    return {
        "id":         f"synth_{lang}_{archetype_name}_{index:03d}",
        "label":      archetype_data["label"],
        "label_name": archetype_name,
        "language":   lang,
        "source":     "claude_generated",
        "messages":   parsed["messages"],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate bilingual CEDD synthetic conversations. / "
                    "Génère des conversations synthétiques bilingues CEDD."
    )
    parser.add_argument(
        "--lang",
        choices=["fr", "en"],
        default="fr",
        help="Language of generated conversations (default: fr). / "
             "Langue des conversations générées (défaut : fr).",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=20,
        help="Number of conversations per class (default: 20). / "
             "Nombre de conversations par classe (défaut : 20).",
    )
    args = parser.parse_args()
    lang  = args.lang
    count = args.count

    archetypes = ARCHETYPES_FR if lang == "fr" else ARCHETYPES_EN

    # Load existing conversations / Charger les conversations existantes
    with open("data/synthetic_conversations.json", "r", encoding="utf-8") as f:
        existing = json.load(f)

    print(f"Existing conversations / Conversations existantes : {len(existing)}")
    print(f"Generating / Génération de {count * 4} conversations [{lang.upper()}] "
          f"({count} per class / par classe)...\n")

    new_conversations = []

    for archetype_name, archetype_data in archetypes.items():
        print(f"Generating class / Génération classe '{archetype_name}' [{lang.upper()}]...")
        success  = 0
        attempts = 0

        while success < count and attempts < count + 5:
            attempts += 1
            try:
                conv = generate_conversation(
                    archetype_name, archetype_data,
                    len(existing) + len(new_conversations) + 1,
                    lang=lang,
                )
                new_conversations.append(conv)
                success += 1
                print(f"  ✓ {success}/{count}", end="\r")
                time.sleep(0.5)  # Basic rate limiting / Limitation du débit

            except json.JSONDecodeError:
                print(f"  ✗ Invalid JSON (attempt {attempts}), retrying...")
            except Exception as e:
                print(f"  ✗ Error / Erreur : {e}")
                time.sleep(2)

        print(f"  ✓ {success}/{count} generated for / générées pour '{archetype_name}'")

    # Merge and save / Fusion et sauvegarde
    all_conversations = existing + new_conversations

    with open("data/synthetic_conversations.json", "w", encoding="utf-8") as f:
        json.dump(all_conversations, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Final dataset / Dataset final : {len(all_conversations)} conversations")
    print(f"   Added / Ajoutées : {len(new_conversations)}")

    # Distribution by class and language / Distribution par classe et langue
    from collections import Counter
    labels = [(c["label_name"], c.get("language", "fr")) for c in all_conversations]
    dist   = Counter(labels)
    print("\nDistribution (class, language) / Distribution (classe, langue) :")
    for (classe, lang_key), count_val in sorted(dist.items()):
        print(f"  {classe:8s} [{lang_key}] : {count_val} conversations")


if __name__ == "__main__":
    main()
