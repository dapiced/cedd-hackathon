"""
CEDD — Claude Quality Annotator / Annotateur de qualité Claude
================================================================
Evaluates synthetic training conversations using Claude Haiku API
for quality, realism, and label agreement. Filters out ambiguous
or mislabeled examples to improve training data quality.

Évalue les conversations synthétiques d'entraînement via l'API Claude Haiku
pour la qualité, le réalisme et la concordance des labels. Filtre les
exemples ambigus ou mal étiquetés pour améliorer la qualité des données.

Usage / Utilisation :
    python annotate_data.py                     # Annotate only / Annoter seulement
    python annotate_data.py --filter            # Annotate + filter / Annoter + filtrer
    python annotate_data.py --filter --generate-edge  # Also replace dropped convos
"""

import argparse
import json
import os
import sys
import time

# Add root directory to path / Ajouter le répertoire racine au path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DATA_PATH       = "data/synthetic_conversations.json"
ANNOTATED_PATH  = "data/annotated_conversations.json"
FILTERED_PATH   = "data/filtered_conversations.json"
LABEL_NAMES     = {0: "green", 1: "yellow", 2: "orange", 3: "red"}

# ── Annotation prompt ─────────────────────────────────────────────────────────

ANNOTATION_PROMPT = """You are a mental health conversation evaluator. Analyze this conversation between a youth (age 16-22) and a chatbot.

The conversation is labeled as: {label_name} (level {label}: {label_description})

Level descriptions:
- 0 (green): Normal conversation, no distress
- 1 (yellow): Concerning signs (fatigue, loneliness, sadness)
- 2 (orange): Significant distress (hopelessness, despair)
- 3 (red): Potential crisis (suicidal ideation, self-harm)

CONVERSATION:
{conversation_text}

Evaluate this conversation on three dimensions. Respond in JSON only:
{{
  "distress_level": <0-3, your independent assessment of distress>,
  "realism": <1-5, how natural and realistic the conversation feels>,
  "ambiguity": <1-5, how clear the distress signals are (1=very clear, 5=very ambiguous)>,
  "justification": "<2 sentences explaining your assessment>"
}}"""


def format_conversation(messages: list) -> str:
    """Format messages into readable text for the prompt."""
    lines = []
    for m in messages:
        role = "Youth" if m["role"] == "user" else "Chatbot"
        lines.append(f"{role}: {m['content']}")
    return "\n".join(lines)


def annotate_conversation(client, conv: dict) -> dict:
    """
    Send a conversation to Claude Haiku for quality annotation.
    Envoie une conversation à Claude Haiku pour annotation de qualité.
    """
    label = conv["label"]
    label_name = conv["label_name"]
    label_descriptions = {
        "green": "Normal conversation, no distress",
        "yellow": "Concerning signs (fatigue, loneliness)",
        "orange": "Significant distress (hopelessness, despair)",
        "red": "Potential crisis (suicidal ideation, self-harm)",
    }

    prompt = ANNOTATION_PROMPT.format(
        label_name=label_name,
        label=label,
        label_description=label_descriptions.get(label_name, "Unknown"),
        conversation_text=format_conversation(conv["messages"]),
    )

    try:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text.strip()

        # Parse JSON from response (handle potential markdown wrapping)
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        annotation = json.loads(text)

        return {
            "distress_level": int(annotation["distress_level"]),
            "realism": int(annotation["realism"]),
            "ambiguity": int(annotation["ambiguity"]),
            "justification": annotation["justification"],
            "agreement": int(annotation["distress_level"]) == label,
        }

    except Exception as e:
        print(f"  ⚠️  Annotation failed for {conv['id']}: {e}")
        return {
            "distress_level": -1,
            "realism": -1,
            "ambiguity": -1,
            "justification": f"Error: {e}",
            "agreement": False,
        }


def print_separator(char="=", length=60):
    print(char * length)


def main():
    parser = argparse.ArgumentParser(
        description="Annotate and filter CEDD training data. / "
                    "Annoter et filtrer les données d'entraînement CEDD."
    )
    parser.add_argument(
        "--filter", "-f",
        action="store_true",
        help="Produce filtered dataset (quality-filtered subset). / "
             "Produire un jeu de données filtré (sous-ensemble de qualité).",
    )
    parser.add_argument(
        "--generate-edge", "-g",
        action="store_true",
        help="Generate edge-case replacements for dropped conversations. / "
             "Générer des cas limites de remplacement pour les conversations supprimées.",
    )
    parser.add_argument(
        "--data", "-d",
        type=str,
        default=DATA_PATH,
        help=f"Path to input data JSON. Default: {DATA_PATH}",
    )
    parser.add_argument(
        "--resume", "-r",
        action="store_true",
        help="Resume from existing annotated file (skip already annotated). / "
             "Reprendre depuis le fichier annoté existant.",
    )
    args = parser.parse_args()

    # ── Check API key / Vérifier la clé API ───────────────────────────────────
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("❌ ANTHROPIC_API_KEY not set. / ANTHROPIC_API_KEY non définie.")
        print("   export ANTHROPIC_API_KEY=your_key")
        sys.exit(1)

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
    except ImportError:
        print("❌ anthropic package not installed. pip install anthropic")
        sys.exit(1)

    # ── Load data / Charger les données ───────────────────────────────────────
    if not os.path.exists(args.data):
        print(f"❌ Data file not found: {args.data}")
        sys.exit(1)

    with open(args.data, "r", encoding="utf-8") as f:
        conversations = json.load(f)

    print_separator()
    print("  CEDD — Claude Quality Annotator / Annotateur de qualité Claude")
    print_separator()
    print(f"  Data source : {args.data}")
    print(f"  Conversations : {len(conversations)}")
    print(f"  Filter mode : {'ON' if args.filter else 'OFF'}")

    # ── Load existing annotations if resuming ─────────────────────────────────
    existing_annotations = {}
    if args.resume and os.path.exists(ANNOTATED_PATH):
        with open(ANNOTATED_PATH, "r", encoding="utf-8") as f:
            existing = json.load(f)
        for conv in existing:
            if "annotation" in conv:
                existing_annotations[conv["id"]] = conv["annotation"]
        print(f"  Resuming: {len(existing_annotations)} existing annotations found")

    # ── Annotate each conversation / Annoter chaque conversation ──────────────
    print(f"\n  Annotating {len(conversations)} conversations...")
    print()

    annotated = []
    for i, conv in enumerate(conversations):
        conv_id = conv["id"]

        # Skip if already annotated and resuming
        if conv_id in existing_annotations:
            conv["annotation"] = existing_annotations[conv_id]
            annotated.append(conv)
            continue

        annotation = annotate_conversation(client, conv)
        conv["annotation"] = annotation

        # Progress display
        status = "✅" if annotation["agreement"] else "⚠️"
        print(f"  [{i+1:3d}/{len(conversations)}] {status} {conv_id:20s} "
              f"label={conv['label_name']:6s} "
              f"claude={LABEL_NAMES.get(annotation['distress_level'], '?'):6s} "
              f"realism={annotation['realism']} "
              f"ambiguity={annotation['ambiguity']}")

        annotated.append(conv)

        # Rate limiting: ~1 req/sec to stay under API limits
        time.sleep(0.3)

    # ── Save annotated data / Sauvegarder les données annotées ────────────────
    with open(ANNOTATED_PATH, "w", encoding="utf-8") as f:
        json.dump(annotated, f, ensure_ascii=False, indent=2)
    print(f"\n  📄 Annotated data saved to / Données annotées sauvegardées : {ANNOTATED_PATH}")

    # ── Statistics / Statistiques ─────────────────────────────────────────────
    print()
    print_separator("─")
    print("  ANNOTATION STATISTICS / STATISTIQUES D'ANNOTATION")
    print_separator("─")

    valid = [c for c in annotated if c["annotation"]["distress_level"] >= 0]
    agreed = [c for c in valid if c["annotation"]["agreement"]]

    print(f"  Total annotated        : {len(valid)}/{len(annotated)}")
    print(f"  Label agreement        : {len(agreed)}/{len(valid)} "
          f"({len(agreed)/max(len(valid),1):.0%})")

    # Per-class agreement
    for level in range(4):
        class_convs = [c for c in valid if c["label"] == level]
        class_agreed = [c for c in class_convs if c["annotation"]["agreement"]]
        if class_convs:
            print(f"    {LABEL_NAMES[level]:8s}: {len(class_agreed)}/{len(class_convs)} "
                  f"({len(class_agreed)/len(class_convs):.0%})")

    # Realism and ambiguity stats
    realism_scores = [c["annotation"]["realism"] for c in valid]
    ambiguity_scores = [c["annotation"]["ambiguity"] for c in valid]
    print(f"\n  Mean realism           : {sum(realism_scores)/max(len(realism_scores),1):.1f}/5")
    print(f"  Mean ambiguity         : {sum(ambiguity_scores)/max(len(ambiguity_scores),1):.1f}/5")

    # Common issues
    low_realism = [c for c in valid if c["annotation"]["realism"] < 3]
    high_ambiguity = [c for c in valid if c["annotation"]["ambiguity"] > 3]
    disagreements = [c for c in valid if not c["annotation"]["agreement"]]

    print(f"\n  Low realism (< 3)      : {len(low_realism)}")
    print(f"  High ambiguity (> 3)   : {len(high_ambiguity)}")
    print(f"  Label disagreements    : {len(disagreements)}")

    if disagreements:
        print(f"\n  Disagreement details / Détails des désaccords :")
        for c in disagreements[:10]:
            ann = c["annotation"]
            print(f"    {c['id']:20s} label={c['label_name']:6s} "
                  f"claude={LABEL_NAMES.get(ann['distress_level'], '?'):6s} "
                  f"— {ann['justification'][:80]}")

    # ── Filter if requested / Filtrer si demandé ──────────────────────────────
    if args.filter:
        print()
        print_separator("─")
        print("  FILTERING / FILTRAGE")
        print_separator("─")

        filtered = []
        dropped = []
        for c in annotated:
            ann = c["annotation"]
            claude_level = ann.get("distress_level", -1)
            original_level = c["label"]
            # Allow exact match OR 1-level adjacent disagreement (safety-aligned)
            # e.g., green labeled as yellow is OK (over-alerting is safe)
            level_close = abs(claude_level - original_level) <= 1
            # Keep if: close agreement + realism >= 3 + ambiguity <= 3
            keep = (
                level_close
                and ann.get("realism", 0) >= 3
                and ann.get("ambiguity", 5) <= 3
            )
            if keep:
                filtered.append(c)
            else:
                dropped.append(c)

        print(f"  Kept     : {len(filtered)}/{len(annotated)}")
        print(f"  Dropped  : {len(dropped)}/{len(annotated)}")

        # Per-class filtered counts
        print(f"\n  Per-class distribution after filtering :")
        for level in range(4):
            count = sum(1 for c in filtered if c["label"] == level)
            original = sum(1 for c in annotated if c["label"] == level)
            print(f"    {LABEL_NAMES[level]:8s}: {count}/{original}")

        # Save filtered dataset (without annotation metadata for clean training)
        filtered_clean = []
        for c in filtered:
            clean = {k: v for k, v in c.items() if k != "annotation"}
            filtered_clean.append(clean)

        with open(FILTERED_PATH, "w", encoding="utf-8") as f:
            json.dump(filtered_clean, f, ensure_ascii=False, indent=2)
        print(f"\n  📄 Filtered data saved to / Données filtrées sauvegardées : {FILTERED_PATH}")

        if dropped:
            print(f"\n  Dropped conversations / Conversations supprimées :")
            for c in dropped[:15]:
                ann = c["annotation"]
                reasons = []
                if not ann.get("agreement", False):
                    reasons.append("disagree")
                if ann.get("realism", 0) < 3:
                    reasons.append(f"realism={ann.get('realism', '?')}")
                if ann.get("ambiguity", 5) > 3:
                    reasons.append(f"ambiguity={ann.get('ambiguity', '?')}")
                print(f"    {c['id']:20s} [{', '.join(reasons)}]")

    print()
    print_separator()
    print("  Annotation complete! / Annotation terminée !")
    print_separator()


if __name__ == "__main__":
    main()
