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

# ── Adversarial Archetypes — French / Français ────────────────────────────────
ADVERSARIAL_ARCHETYPES_FR = {
    "physical_only": {
        "label": 0,
        "description": """Jeune (16-22 ans) qui a des plaintes physiques SANS aucune détresse émotionnelle.
        Caractéristiques : mal de dos, maux de tête, blessure sportive, problèmes de ventre.
        ZÉRO langage de détresse émotionnelle — la frustration face à la douleur est ok, mais
        AUCUNE tristesse, désespoir ou solitude.
        Mentionne ses projets, ses amis, ses activités — va clairement bien émotionnellement.
        IMPORTANT : certains messages doivent être COURTS (mises à jour physiques comme "ouais mal au dos encore")
        pour éviter d'entraîner le modèle que "court = détresse".
        Peut utiliser des expressions physiques colorées comme "mort de fatigue", "j'en peux plus" (en
        parlant de douleur physique), "ça me tue" (en parlant de douleur).""",
    },
    "sarcasm_distress": {
        "label": 1,
        "description": """Jeune qui utilise l'humour noir et le sarcasme pour masquer une détresse légère.
        Caractéristiques : "oh ouais ma vie est totalement parfaite", "je vis le rêve",
        "super contente d'être ignorée encore une fois".
        Révèle graduellement un vrai isolement, de la fatigue, de la solitude derrière le sarcasme.
        AUCUN mot-clé de crise — la détresse est masquée, pas explicite.
        Les messages restent relativement longs (le sarcasme est verbeux).
        Ton ironique constant, auto-dépréciation humoristique.""",
    },
    "adversarial_bypass": {
        "label": 1,
        "description": """Jeune qui commence avec une légère détresse puis minimise activement.
        Pattern : révèle → minimise → révèle plus profond → minimise encore plus fort.
        Exemples : "bah non laisse faire", "je suis correct finalement", "oublie ça c'est pas grave",
        "non non ça va, c'était juste une mauvaise journée".
        AUCUN mot-clé de crise. La trajectoire montre une oscillation entre ouverture et fermeture.
        Les messages varient en longueur selon l'ouverture/fermeture.""",
    },
    "identity_distress": {
        "label": 2,
        "description": """Jeune 2SLGBTQ+ ou en conflit culturel/identitaire en détresse significative.
        Caractéristiques : rejet familial lié à l'identité, cacher qui on est, conflit culturel.
        Utilise du langage spécifique à l'identité : "ma famille m'acceptera jamais",
        "je peux pas être moi-même chez nous", "ils disent que c'est un choix",
        "j'ai peur de faire mon coming out", "je me sens pris entre deux mondes".
        AUCUN mot de crise générique — la détresse est enracinée dans l'identité.
        Les messages raccourcissent progressivement à mesure que la personne se referme.""",
    },
    "neurodivergent_flat": {
        "label": 2,
        "description": """Jeune neurodivergent qui décrit des situations objectivement préoccupantes
        avec un ton plat et factuel, sans vocabulaire émotionnel.
        Caractéristiques : messages très courts, sauts de sujet, langage littéral, pas d'émotion exprimée.
        Exemple : "mes parents divorcent. aussi j'ai eu un B à mon test. le chat est orange."
        Faible ratio de ponctuation, peu de questions, faible engagement — ressemble à du retrait.
        Décrit l'isolement, les problèmes familiaux, les difficultés scolaires de manière factuelle.
        AUCUN mot émotionnel explicite mais la situation décrite est préoccupante.""",
    },
    "crisis_with_deflection": {
        "label": 3,
        "description": """Jeune en crise qui exprime des pensées suicidaires PUIS essaie de minimiser.
        Contient du langage de crise EXPLICITE : "je veux mourir", "j'en peux plus",
        "je veux tout arrêter", "plus rien a de sens", "je veux disparaître".
        Suivi de minimisation : "c'était une blague", "oublie ça", "je suis correct maintenant",
        "non non je disais ça de même", "je suis juste fatigué".
        La conversation dans son ensemble reste Rouge car les mots-clés de crise persistent.
        Enseigne au modèle que des mots positifs APRÈS une crise ne rendent pas la trajectoire verte.
        IMPORTANT : les mots de crise doivent apparaître dans les premiers 6-8 messages.""",
    },
}

# ── Adversarial Archetypes — English / Anglais ────────────────────────────────
ADVERSARIAL_ARCHETYPES_EN = {
    "physical_only": {
        "label": 0,
        "description": """A young person (16-22) with physical complaints and ZERO emotional distress.
        Characteristics: back pain, headaches, sports injury, stomach issues.
        ZERO emotional distress language — frustration with pain is ok, but
        NO sadness, hopelessness, or loneliness.
        Mentions plans, friends, activities — clearly doing well emotionally.
        IMPORTANT: some messages should be SHORT (physical updates like "yeah back still hurts")
        to avoid training the model that "short = distress".
        May use colorful physical expressions like "this pain is killing me",
        "dead tired" (from exercise), "I can't take this anymore" (about physical pain).""",
    },
    "sarcasm_distress": {
        "label": 1,
        "description": """A young person using dark humour and sarcasm to mask mild distress.
        Characteristics: "oh yeah my life is totally perfect", "living the dream",
        "so thrilled to be ghosted again".
        Gradually reveals real isolation, fatigue, loneliness through sarcastic framing.
        NO crisis keywords — distress is masked, not explicit.
        Messages stay relatively long (sarcasm is verbose).
        Constant ironic tone, humorous self-deprecation.""",
    },
    "adversarial_bypass": {
        "label": 1,
        "description": """A young person who starts with mild distress then actively deflects.
        Pattern: reveal → minimize → reveal deeper → minimize harder.
        Examples: "nah forget it", "I'm fine actually", "never mind it's nothing",
        "no no it's ok, just a bad day".
        NO crisis keywords. The trajectory features should show oscillation
        between opening up and shutting down.
        Message length varies with opening/closing pattern.""",
    },
    "identity_distress": {
        "label": 2,
        "description": """A 2SLGBTQ+ or culturally conflicted young person in significant distress.
        Characteristics: family rejection related to identity, hiding who they are, cultural conflict.
        Uses identity-specific language: "my family will never accept me",
        "I can't be myself at home", "they say it's a choice",
        "I'm scared to come out", "I feel trapped between two worlds".
        NO generic crisis words — distress is identity-rooted.
        Messages get shorter as the person closes off.""",
    },
    "neurodivergent_flat": {
        "label": 2,
        "description": """A neurodivergent young person describing objectively concerning situations
        with flat, matter-of-fact tone and no emotional vocabulary.
        Characteristics: very short messages, topic jumps, literal language, no emotions expressed.
        Example: "my parents are getting divorced. also I got a B on my test. the cat is orange."
        Low punctuation ratio, few questions, low engagement — looks like withdrawal.
        Describes isolation, family problems, school difficulties in a factual manner.
        NO explicit emotional words but the described situation IS concerning.""",
    },
    "crisis_with_deflection": {
        "label": 3,
        "description": """A young person in crisis who expresses suicidal thoughts THEN tries to minimize.
        Contains EXPLICIT crisis language: "I want to die", "I can't do this anymore",
        "I want to end it all", "nothing matters anymore", "I want to disappear".
        Followed by deflection: "I was joking", "forget it", "I'm fine now",
        "no no I didn't mean it like that", "I'm just tired".
        The conversation as a whole remains Red because crisis keywords persist.
        Teaches the model that positive words AFTER crisis don't make the trajectory green.
        IMPORTANT: crisis words must appear in the first 6-8 messages.""",
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
    parser.add_argument(
        "--adversarial",
        action="store_true",
        help="Generate adversarial archetypes instead of standard ones. / "
             "Générer les archétypes adversariaux au lieu des standards.",
    )
    args = parser.parse_args()
    lang  = args.lang
    count = args.count

    if args.adversarial:
        archetypes = ADVERSARIAL_ARCHETYPES_FR if lang == "fr" else ADVERSARIAL_ARCHETYPES_EN
    else:
        archetypes = ARCHETYPES_FR if lang == "fr" else ARCHETYPES_EN

    # Load existing conversations / Charger les conversations existantes
    with open("data/synthetic_conversations.json", "r", encoding="utf-8") as f:
        existing = json.load(f)

    n_archetypes = len(archetypes)
    mode_label = "adversarial" if args.adversarial else "standard"
    print(f"Existing conversations / Conversations existantes : {len(existing)}")
    print(f"Mode: {mode_label} ({n_archetypes} archetypes)")
    print(f"Generating / Génération de {count * n_archetypes} conversations [{lang.upper()}] "
          f"({count} per archetype / par archétype)...\n")

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
