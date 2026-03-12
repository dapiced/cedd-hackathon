"""
CEDD — Adversarial Test Suite / Suite de tests adversariaux
=============================================================
Runs adversarial test cases against the CEDD pipeline and reports results.
Exécute les cas de test adversariaux contre le pipeline CEDD et génère un rapport.

Usage / Utilisation :
    python tests/adversarial_suite.py
    python tests/adversarial_suite.py --verbose
    python tests/adversarial_suite.py --category sarcasm
    python tests/adversarial_suite.py --export results.json
"""

import json
import os
import sys
import argparse
from datetime import datetime
from collections import defaultdict

# Add project root to path / Ajouter la racine du projet au path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from cedd.classifier import CEDDClassifier

# ── Paths / Chemins ──────────────────────────────────────────────────────────
MODEL_PATH     = os.path.join(PROJECT_ROOT, "models", "cedd_model.joblib")
TEST_CASES_PATH = os.path.join(os.path.dirname(__file__), "test_cases_adversarial.json")

# ── Color labels / Labels couleur ────────────────────────────────────────────
LEVEL_LABELS = {0: "🟢 GREEN", 1: "🟡 YELLOW", 2: "🟠 ORANGE", 3: "🔴 RED"}
LEVEL_NAMES  = {0: "green", 1: "yellow", 2: "orange", 3: "red"}


def load_test_cases(path: str, category_filter: str = None) -> list:
    """
    Load test cases from JSON file, optionally filtered by category.
    Charge les cas de test depuis un fichier JSON, optionnellement filtrés par catégorie.
    """
    with open(path, "r", encoding="utf-8") as f:
        cases = json.load(f)

    if category_filter:
        cases = [c for c in cases if c["category"] == category_filter]

    return cases


def run_single_test(clf: CEDDClassifier, test_case: dict) -> dict:
    """
    Run a single test case through the CEDD pipeline and return results.
    Exécute un cas de test unique dans le pipeline CEDD et retourne les résultats.

    Why we call get_alert_level() directly:
    This is the same function that app.py calls when a user sends a message.
    It runs the full 6-gate decision logic (keywords + ML + safety rules).
    By testing this function, we test the entire classification pipeline
    end-to-end, not just the ML model in isolation.

    Pourquoi on appelle get_alert_level() directement :
    C'est la même fonction que app.py appelle quand un utilisateur envoie un message.
    Elle exécute toute la logique 6 portes (mots-clés + ML + règles de sécurité).
    En testant cette fonction, on teste le pipeline complet de classification,
    pas juste le modèle ML isolé.
    """
    lang = test_case.get("language", "fr")
    messages = test_case["messages"]
    expected = test_case["expected_level"]
    tolerance = test_case.get("tolerance", 0)

    # Run the full CEDD pipeline / Exécuter le pipeline CEDD complet
    result = clf.get_alert_level(messages, lang=lang)

    predicted = result["level"]
    confidence = result["confidence"]

    # A test passes if the predicted level is within tolerance of expected.
    # tolerance=0 means exact match required (critical safety tests).
    # tolerance=1 means ±1 level is acceptable (ambiguous scenarios).
    #
    # Un test réussit si le niveau prédit est dans la tolérance de l'attendu.
    # tolerance=0 signifie correspondance exacte requise (tests de sécurité critiques).
    # tolerance=1 signifie ±1 niveau est acceptable (scénarios ambigus).
    diff = abs(predicted - expected)
    passed = diff <= tolerance

    # Special rule: for crisis tests (expected RED), predicted GREEN is always a CRITICAL fail
    # even if tolerance would technically allow it. Missing a crisis = unacceptable.
    #
    # Règle spéciale : pour les tests de crise (attendu ROUGE), prédire VERT est toujours
    # un échec CRITIQUE, même si la tolérance le permettrait techniquement.
    # Manquer une crise = inacceptable.
    critical_miss = (expected == 3 and predicted <= 1)

    if critical_miss:
        passed = False

    return {
        "id":             test_case["id"],
        "category":       test_case["category"],
        "description":    test_case.get("description_en", test_case.get("description_fr", "")),
        "language":       lang,
        "expected_level": expected,
        "expected_label": LEVEL_NAMES[expected],
        "predicted_level": predicted,
        "predicted_label": LEVEL_NAMES[predicted],
        "confidence":     round(confidence, 3),
        "tolerance":      tolerance,
        "passed":         passed,
        "critical_miss":  critical_miss,
        "diff":           diff,
        "probabilities":  result.get("probabilities", {}),
        "dominant_features": result.get("dominant_features", []),
    }


def print_separator(char="═", length=72):
    print(char * length)


def print_report(results: list, verbose: bool = False):
    """
    Print a formatted report of test results.
    Affiche un rapport formaté des résultats de test.
    """
    print()
    print_separator()
    print("  CEDD — ADVERSARIAL TEST REPORT / RAPPORT DE TESTS ADVERSARIAUX")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Model / Modèle: {MODEL_PATH}")
    print_separator()

    # ── Per-test results / Résultats par test ────────────────────────────────
    passed_count = sum(1 for r in results if r["passed"])
    failed_count = len(results) - passed_count
    critical_misses = sum(1 for r in results if r["critical_miss"])

    for r in results:
        status = "✅ PASS" if r["passed"] else "❌ FAIL"
        if r["critical_miss"]:
            status = "🚨 CRITICAL MISS"

        print(f"\n  {status}  [{r['id']}] {r['category']}")
        print(f"    {r['description']}")
        print(f"    Expected / Attendu : {LEVEL_LABELS[r['expected_level']]} "
              f"(tolerance ±{r['tolerance']})")
        print(f"    Predicted / Prédit : {LEVEL_LABELS[r['predicted_level']]} "
              f"(confidence: {r['confidence']:.1%})")

        if verbose:
            print(f"    Probabilities / Probabilités : {r['probabilities']}")
            if r["dominant_features"]:
                print(f"    Top features : {', '.join(r['dominant_features'][:3])}")

    # ── Summary / Résumé ─────────────────────────────────────────────────────
    print()
    print_separator("─")
    print(f"  SUMMARY / RÉSUMÉ")
    print_separator("─")
    print(f"  Total tests      : {len(results)}")
    print(f"  Passed / Réussis : {passed_count} ({passed_count/len(results):.0%})")
    print(f"  Failed / Échoués : {failed_count}")
    if critical_misses > 0:
        print(f"  🚨 CRITICAL MISSES (crise manquée) : {critical_misses}")

    # ── Per-category breakdown / Ventilation par catégorie ───────────────────
    categories = defaultdict(lambda: {"total": 0, "passed": 0})
    for r in results:
        cat = r["category"]
        categories[cat]["total"] += 1
        if r["passed"]:
            categories[cat]["passed"] += 1

    print(f"\n  Per category / Par catégorie :")
    for cat, stats in sorted(categories.items()):
        pct = stats["passed"] / stats["total"]
        bar = "█" * int(pct * 10) + "░" * (10 - int(pct * 10))
        print(f"    {cat:30s} {bar} {stats['passed']}/{stats['total']}")

    # ── Asymmetric error check / Vérification erreur asymétrique ─────────────
    # The most dangerous errors: predicting LOW when reality is HIGH
    # Les erreurs les plus dangereuses : prédire BAS quand la réalité est HAUT
    under_predictions = [r for r in results if r["predicted_level"] < r["expected_level"]
                         and not r["passed"]]
    over_predictions = [r for r in results if r["predicted_level"] > r["expected_level"]
                        and not r["passed"]]

    if under_predictions:
        print(f"\n  ⚠️  Under-predictions (dangerous / dangereux) : {len(under_predictions)}")
        for r in under_predictions:
            print(f"      [{r['id']}] expected {LEVEL_NAMES[r['expected_level']]} "
                  f"→ got {LEVEL_NAMES[r['predicted_level']]}")

    if over_predictions:
        print(f"\n  ℹ️  Over-predictions (safer, but noisy / plus sûr, mais bruyant) : "
              f"{len(over_predictions)}")
        for r in over_predictions:
            print(f"      [{r['id']}] expected {LEVEL_NAMES[r['expected_level']]} "
                  f"→ got {LEVEL_NAMES[r['predicted_level']]}")

    print()
    print_separator()
    # Remind the team of the asymmetric error philosophy
    # Rappeler à l'équipe la philosophie d'erreur asymétrique
    print("  💡 Reminder: Over-alerting is ALWAYS preferable to missing a crisis.")
    print("     Rappel : Sur-alerter est TOUJOURS préférable à manquer une crise.")
    print_separator()


def export_results(results: list, path: str):
    """
    Export results to JSON for tracking improvements over time.
    Exporte les résultats en JSON pour suivre les améliorations dans le temps.
    """
    export = {
        "timestamp": datetime.now().isoformat(),
        "model_path": MODEL_PATH,
        "total_tests": len(results),
        "passed": sum(1 for r in results if r["passed"]),
        "failed": sum(1 for r in results if not r["passed"]),
        "critical_misses": sum(1 for r in results if r["critical_miss"]),
        "results": results,
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(export, f, ensure_ascii=False, indent=2)

    print(f"\n  📄 Results exported to / Résultats exportés vers : {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run CEDD adversarial tests. / Exécuter les tests adversariaux CEDD."
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show probabilities and top features for each test. / "
             "Afficher les probabilités et top features pour chaque test.",
    )
    parser.add_argument(
        "--category", "-c",
        type=str,
        default=None,
        help="Filter tests by category. / Filtrer les tests par catégorie. "
             "Ex: sarcasm, negation, false_positive_physical",
    )
    parser.add_argument(
        "--export", "-e",
        type=str,
        default=None,
        help="Export results to JSON file. / Exporter les résultats en JSON. "
             "Ex: results/run_001.json",
    )
    args = parser.parse_args()

    # ── Load model / Charger le modèle ───────────────────────────────────────
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found at / Modèle introuvable à : {MODEL_PATH}")
        print("   Run train.py first. / Exécutez train.py d'abord.")
        sys.exit(1)

    print(f"  Loading model / Chargement du modèle : {MODEL_PATH}")
    clf = CEDDClassifier.load(MODEL_PATH)

    # ── Load test cases / Charger les cas de test ────────────────────────────
    if not os.path.exists(TEST_CASES_PATH):
        print(f"❌ Test cases not found at / Cas de test introuvables à : {TEST_CASES_PATH}")
        sys.exit(1)

    cases = load_test_cases(TEST_CASES_PATH, category_filter=args.category)
    print(f"  Test cases loaded / Cas de test chargés : {len(cases)}")

    if not cases:
        print("  No test cases match the filter. / Aucun cas ne correspond au filtre.")
        sys.exit(0)

    # ── Run tests / Exécuter les tests ───────────────────────────────────────
    results = []
    for case in cases:
        result = run_single_test(clf, case)
        results.append(result)

    # ── Report / Rapport ─────────────────────────────────────────────────────
    print_report(results, verbose=args.verbose)

    # ── Export if requested / Exporter si demandé ────────────────────────────
    if args.export:
        export_dir = os.path.dirname(args.export)
        if export_dir and not os.path.exists(export_dir):
            os.makedirs(export_dir)
        export_results(results, args.export)

    # ── Exit code: non-zero if any critical miss / Code de sortie ────────────
    # This lets CI/CD pipelines catch safety regressions automatically.
    # Ça permet aux pipelines CI/CD de détecter les régressions de sécurité.
    critical = sum(1 for r in results if r["critical_miss"])
    if critical > 0:
        sys.exit(2)  # Critical safety failure / Échec de sécurité critique
    elif any(not r["passed"] for r in results):
        sys.exit(1)  # Some tests failed / Certains tests échoués
    else:
        sys.exit(0)  # All passed / Tous réussis


if __name__ == "__main__":
    main()
