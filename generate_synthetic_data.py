import anthropic
import json
import time
import os

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

ARCHETYPES = {
    "verte": {
        "label": 0,
        "description": """Jeune (16-22 ans) qui va globalement bien.
        Caractéristiques : pose des questions, fait des projets futurs,
        mentionne des amis/famille, utilise l'humour, messages variés en longueur,
        exprime des émotions normales (stress d'examen, fatigue passagère),
        rebondit positivement sur les suggestions."""
    },
    "jaune": {
        "label": 1,
        "description": """Jeune qui montre des signes préoccupants mais pas urgents.
        Caractéristiques : fatigue persistante, sentiment de solitude croissant,
        moins de questions, quelques messages plus courts, mentionne que
        les amis comprennent pas, doutes sur soi, mais maintient encore
        des liens et des activités. Dérive graduelle sur 10-12 messages."""
    },
    "orange": {
        "label": 2,
        "description": """Jeune en détresse significative nécessitant attention soutenue.
        Caractéristiques : sentiment de vide, pleure souvent, se sent fardeau,
        messages de plus en plus courts, moins de ponctuation, évite les questions,
        parle de se sentir inutile, mentionne que personne ne comprend vraiment,
        pensées négatives récurrentes. Dérive claire mais pas de plan explicite."""
    },
    "rouge": {
        "label": 3,
        "description": """Jeune en crise nécessitant intervention immédiate.
        Caractéristiques : exprime vouloir disparaître, se sentir fardeau pour tous,
        messages très courts et intenses, parle de 'tout arrêter',
        sentiment que rien ne changera jamais, mentionne avoir pensé à partir,
        isolement total, plus de projets futurs, ton de finalité."""
    }
}

PROMPT_TEMPLATE = """Tu es un expert en santé mentale des jeunes et en
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

def generate_conversation(archetype_name, archetype_data, index):
    """Génère une conversation via API Claude."""
    prompt = PROMPT_TEMPLATE.format(
        description=archetype_data["description"]
    )

    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}]
    )

    response_text = message.content[0].text.strip()
    # Nettoie les backticks si présents
    if response_text.startswith("```"):
        response_text = response_text.split("```")[1]
        if response_text.startswith("json"):
            response_text = response_text[4:]

    parsed = json.loads(response_text)

    return {
        "id": f"synth_{archetype_name}_{index:03d}",
        "label": archetype_data["label"],
        "label_name": archetype_name,
        "source": "claude_generated",
        "messages": parsed["messages"]
    }

def main():
    # Charge les conversations existantes
    with open("data/synthetic_conversations.json", "r") as f:
        existing = json.load(f)

    print(f"Conversations existantes : {len(existing)}")
    print("Génération de 80 nouvelles conversations (20 par classe)...\n")

    new_conversations = []

    for archetype_name, archetype_data in ARCHETYPES.items():
        print(f"Génération classe '{archetype_name}'...")
        success = 0
        attempts = 0

        while success < 20 and attempts < 25:
            attempts += 1
            try:
                conv = generate_conversation(
                    archetype_name, archetype_data,
                    len(existing) + len(new_conversations) + 1
                )
                new_conversations.append(conv)
                success += 1
                print(f"  ✓ {success}/20", end="\r")
                time.sleep(0.5)  # rate limiting

            except json.JSONDecodeError as e:
                print(f"  ✗ JSON invalide (tentative {attempts}), retry...")
            except Exception as e:
                print(f"  ✗ Erreur : {e}")
                time.sleep(2)

        print(f"  ✓ {success}/20 générées pour '{archetype_name}'")

    # Merge et sauvegarde
    all_conversations = existing + new_conversations

    with open("data/synthetic_conversations.json", "w") as f:
        json.dump(all_conversations, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Dataset final : {len(all_conversations)} conversations")
    print(f"   Ajoutées : {len(new_conversations)}")

    # Distribution par classe
    from collections import Counter
    labels = [c["label_name"] for c in all_conversations]
    dist = Counter(labels)
    print("\nDistribution :")
    for classe, count in sorted(dist.items()):
        print(f"  {classe:8s} : {count} conversations")

if __name__ == "__main__":
    main()
