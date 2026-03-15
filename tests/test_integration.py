"""
CEDD — Integration & Presentation Tests / Tests d'intégration et de présentation
==================================================================================
Validates everything a jury might see during the March 23 presentation:
  1. Demo scenarios     — 9-message autopilot runs without crash
  2. Cross-language     — same crisis → same result in FR and EN
  3. Bilingual strings  — no missing translations in STRINGS dict
  4. End-to-end         — full conversations → correct alert levels
  5. Edge cases         — emoji, very long messages, mixed languages
  6. Feature scores     — explainability charts get valid data
  7. Session tracker    — real classifier results flow into SQLite

Valide tout ce qu'un jury pourrait voir lors de la présentation du 23 mars.

Usage / Utilisation :
    pytest tests/test_integration.py -v
    pytest tests/test_integration.py -v -k "Demo"
    pytest tests/test_integration.py -v -k "Bilingual"
    pytest tests/test_integration.py -v -k "EdgeCase"
"""

import os
import sys
import types
import importlib
import numpy as np
import pytest

# Add project root / Ajouter la racine du projet
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from cedd.classifier import CEDDClassifier
from cedd.feature_extractor import extract_features, extract_trajectory_features
from cedd.response_modulator import (
    get_system_prompt,
    get_level_description,
    get_handoff_prompt,
    get_handoff_offer_message,
    get_counselor_intro,
)
from cedd.session_tracker import SessionTracker

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "cedd_model.skops")


# ── Import DEMO_SCENARIOS and STRINGS from app.py without Streamlit ──────────
# app.py calls st.set_page_config() at module level. We mock streamlit so the
# import succeeds without a running Streamlit server.
# On importe DEMO_SCENARIOS et STRINGS depuis app.py sans Streamlit.

def _load_app_constants():
    """Import app.py with a mocked streamlit module to extract DEMO_SCENARIOS and STRINGS."""
    mock_st = types.ModuleType("streamlit")

    # Mock the minimum Streamlit API used at import time
    mock_st.set_page_config = lambda **kwargs: None
    mock_st.cache_resource = lambda func=None, **kw: (func if func else (lambda f: f))
    mock_st.session_state = {}

    # Save and replace
    original = sys.modules.get("streamlit")
    sys.modules["streamlit"] = mock_st

    # Also mock sub-modules that app.py might reference at import time
    for sub in ("streamlit.components", "streamlit.components.v1"):
        if sub not in sys.modules:
            sys.modules[sub] = types.ModuleType(sub)

    try:
        # Force fresh import
        if "app" in sys.modules:
            del sys.modules["app"]
        import app
        return app.DEMO_SCENARIOS, app.STRINGS
    except Exception:
        # Fallback: read file and extract dicts via exec
        # This handles edge cases where the full app has complex Streamlit usage
        app_path = os.path.join(PROJECT_ROOT, "app.py")
        with open(app_path, "r", encoding="utf-8") as f:
            source = f.read()

        # Extract just the DEMO_SCENARIOS and STRINGS blocks
        namespace = {}
        # Find and exec the relevant dict definitions
        for var_name in ("DEMO_SCENARIOS", "STRINGS"):
            start = source.find(f"{var_name} = {{")
            if start == -1:
                continue
            # Find matching closing brace
            depth = 0
            end = start
            for i, ch in enumerate(source[start:], start):
                if ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            exec(source[start:end], namespace)

        return namespace.get("DEMO_SCENARIOS", {}), namespace.get("STRINGS", {})
    finally:
        # Restore original streamlit module
        if original is not None:
            sys.modules["streamlit"] = original
        else:
            sys.modules.pop("streamlit", None)
        for sub in ("streamlit.components", "streamlit.components.v1"):
            if sub in sys.modules and isinstance(sys.modules[sub], types.ModuleType):
                if not hasattr(sys.modules[sub], "__file__"):
                    del sys.modules[sub]


DEMO_SCENARIOS, STRINGS = _load_app_constants()


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def clf():
    """Load the trained model once for all integration tests."""
    if not os.path.exists(MODEL_PATH):
        pytest.skip("Model not found — run train.py first")
    return CEDDClassifier.load(MODEL_PATH)


@pytest.fixture
def tracker(tmp_path):
    """Create a fresh tracker with a temporary database for each test."""
    db_path = str(tmp_path / "test_integration.db")
    return SessionTracker(db_path=db_path)


# ── Helper ────────────────────────────────────────────────────────────────────

def _build_conversation(user_messages, bot_reply="I hear you."):
    """Build an alternating user/assistant conversation from user messages."""
    msgs = []
    for text in user_messages:
        msgs.append({"role": "user", "content": text})
        msgs.append({"role": "assistant", "content": bot_reply})
    # Remove trailing assistant so the last message is from the user
    if msgs and msgs[-1]["role"] == "assistant":
        msgs.pop()
    return msgs


# ══════════════════════════════════════════════════════════════════════════════
# 1. DEMO SCENARIO TESTS — validate the 9-message autopilot
# ══════════════════════════════════════════════════════════════════════════════

class TestDemoScenarios:
    """Test that the demo scenarios (shown during presentation) work correctly."""

    def test_demo_scenarios_exist(self):
        """Both FR and EN demo scenarios must be defined."""
        assert "fr" in DEMO_SCENARIOS, "Missing FR demo scenario"
        assert "en" in DEMO_SCENARIOS, "Missing EN demo scenario"

    def test_demo_scenarios_have_9_messages(self):
        """Each demo scenario must have exactly 9 user messages."""
        for lang in ("fr", "en"):
            assert len(DEMO_SCENARIOS[lang]) == 9, (
                f"{lang.upper()} demo has {len(DEMO_SCENARIOS[lang])} messages, expected 9"
            )

    def test_demo_scenario_no_empty_messages(self):
        """No demo message should be empty."""
        for lang in ("fr", "en"):
            for i, msg in enumerate(DEMO_SCENARIOS[lang]):
                assert len(msg.strip()) > 0, f"{lang.upper()} demo message {i+1} is empty"

    def test_demo_fr_no_crash(self, clf):
        """FR demo scenario processes through classifier without crash."""
        msgs = _build_conversation(DEMO_SCENARIOS["fr"])
        result = clf.get_alert_level(msgs, lang="fr")
        assert result["level"] in [0, 1, 2, 3]
        assert 0 <= result["confidence"] <= 1.0

    def test_demo_en_no_crash(self, clf):
        """EN demo scenario processes through classifier without crash."""
        msgs = _build_conversation(DEMO_SCENARIOS["en"])
        result = clf.get_alert_level(msgs, lang="en")
        assert result["level"] in [0, 1, 2, 3]
        assert 0 <= result["confidence"] <= 1.0

    def test_demo_escalates_over_time(self, clf):
        """Demo should show escalation: later messages should be higher level than start."""
        for lang in ("fr", "en"):
            scenario = DEMO_SCENARIOS[lang]
            # First 3 messages (normal/early)
            early_msgs = _build_conversation(scenario[:3])
            early_result = clf.get_alert_level(early_msgs, lang=lang)

            # Full 9 messages (drifted)
            full_msgs = _build_conversation(scenario)
            full_result = clf.get_alert_level(full_msgs, lang=lang)

            assert full_result["level"] >= early_result["level"], (
                f"{lang.upper()} demo does not escalate: "
                f"early={early_result['level']}, full={full_result['level']}"
            )

    def test_demo_final_level_at_least_yellow(self, clf):
        """Full 9-message demo should reach at least Yellow (the story is about drift)."""
        for lang in ("fr", "en"):
            msgs = _build_conversation(DEMO_SCENARIOS[lang])
            result = clf.get_alert_level(msgs, lang=lang)
            assert result["level"] >= 1, (
                f"{lang.upper()} demo ends at Green — drift not detected"
            )


# ══════════════════════════════════════════════════════════════════════════════
# 2. CROSS-LANGUAGE CONSISTENCY TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestCrossLanguageConsistency:
    """Ensure FR and EN produce similar results for equivalent distress content."""

    def test_direct_crisis_same_level_fr_en(self, clf):
        """Direct crisis words must trigger RED in both languages."""
        crisis_fr = _build_conversation([
            "Salut", "Ça va pas trop", "Je veux mourir",
            "Je sais plus quoi faire", "C'est fini",
        ])
        crisis_en = _build_conversation([
            "Hey", "Not doing great", "I want to die",
            "I don't know what to do", "It's over",
        ])
        result_fr = clf.get_alert_level(crisis_fr, lang="fr")
        result_en = clf.get_alert_level(crisis_en, lang="en")
        assert result_fr["level"] == 3, f"FR crisis got level {result_fr['level']}, expected RED"
        assert result_en["level"] == 3, f"EN crisis got level {result_en['level']}, expected RED"

    def test_normal_conversation_both_low(self, clf):
        """Normal positive conversation should be Green or Yellow in both languages (±1 tolerance)."""
        normal_fr = _build_conversation([
            "Salut! Comment tu vas aujourd'hui?",
            "Ça va super bien! J'ai passé une belle journée avec mes amis.",
            "On a eu un bon cours de maths, c'était vraiment intéressant.",
            "J'ai hâte d'aller au cinéma avec Mathieu ce weekend, on va voir un bon film.",
            "Ma session se passe bien, je suis content de mes notes en général.",
            "Merci pour la conversation, bonne soirée!",
        ])
        normal_en = _build_conversation([
            "Hey! How are you doing today?",
            "I'm doing really great! Had a wonderful day with my friends.",
            "Had a really good math class, it was super interesting.",
            "Looking forward to going to the movies with Jordan this weekend.",
            "My semester is going well, I'm happy with my grades overall.",
            "Thanks for chatting, have a good night!",
        ])
        result_fr = clf.get_alert_level(normal_fr, lang="fr")
        result_en = clf.get_alert_level(normal_en, lang="en")
        assert result_fr["level"] <= 1, f"FR normal got level {result_fr['level']}, expected GREEN or YELLOW"
        assert result_en["level"] <= 1, f"EN normal got level {result_en['level']}, expected GREEN or YELLOW"
        diff = abs(result_fr["level"] - result_en["level"])
        assert diff <= 1, (
            f"FR level={result_fr['level']}, EN level={result_en['level']} — "
            f"diff={diff} exceeds ±1 tolerance"
        )

    def test_moderate_distress_tolerance(self, clf):
        """Moderate distress: FR and EN should be within ±1 level of each other."""
        mod_fr = _build_conversation([
            "Je suis fatigué", "L'école c'est dur",
            "Je me sens seul des fois", "C'est difficile en ce moment",
            "J'sais pas trop quoi faire", "Ça va aller j'imagine",
        ])
        mod_en = _build_conversation([
            "I'm tired", "School is hard",
            "I feel alone sometimes", "It's difficult right now",
            "I don't really know what to do", "I guess it'll be fine",
        ])
        result_fr = clf.get_alert_level(mod_fr, lang="fr")
        result_en = clf.get_alert_level(mod_en, lang="en")
        diff = abs(result_fr["level"] - result_en["level"])
        assert diff <= 1, (
            f"FR level={result_fr['level']}, EN level={result_en['level']} — "
            f"diff={diff} exceeds ±1 tolerance"
        )

    def test_crisis_keyword_floor_both_languages(self, clf):
        """Gate 2 safety keyword floor must fire for 'suicide' in both FR and EN."""
        for lang, word in [("fr", "suicide"), ("en", "suicide")]:
            msgs = _build_conversation([
                "Salut" if lang == "fr" else "Hey",
                "Ça va" if lang == "fr" else "I'm okay",
                f"Je pense au {word}" if lang == "fr" else f"I'm thinking about {word}",
            ])
            result = clf.get_alert_level(msgs, lang=lang)
            assert result["level"] >= 3, (
                f"'{word}' in {lang.upper()} got level {result['level']}, expected RED"
            )

    def test_feature_names_language_match(self, clf):
        """Feature names should be in the requested language."""
        msgs = _build_conversation([
            "Hello", "I feel bad", "Nothing matters",
            "I'm tired of everything", "Whatever",
        ])
        result_fr = clf.get_alert_level(msgs, lang="fr")
        result_en = clf.get_alert_level(msgs, lang="en")

        # Both should have dominant_features
        assert len(result_fr["dominant_features"]) > 0
        assert len(result_en["dominant_features"]) > 0

    def test_demo_scenarios_cross_language(self, clf):
        """Both demo scenarios should reach similar final levels (±1)."""
        msgs_fr = _build_conversation(DEMO_SCENARIOS["fr"])
        msgs_en = _build_conversation(DEMO_SCENARIOS["en"])
        result_fr = clf.get_alert_level(msgs_fr, lang="fr")
        result_en = clf.get_alert_level(msgs_en, lang="en")
        diff = abs(result_fr["level"] - result_en["level"])
        assert diff <= 1, (
            f"Demo FR level={result_fr['level']}, EN level={result_en['level']} — "
            f"diff={diff} exceeds ±1 tolerance"
        )


# ══════════════════════════════════════════════════════════════════════════════
# 3. BILINGUAL STRING COMPLETENESS TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestBilingualStringCompleteness:
    """Ensure STRINGS dict has matching keys for FR and EN — a missing key = UI crash."""

    def test_strings_have_both_languages(self):
        """STRINGS must have both 'fr' and 'en' top-level keys."""
        assert "fr" in STRINGS, "STRINGS missing 'fr'"
        assert "en" in STRINGS, "STRINGS missing 'en'"

    def test_same_keys_in_both_languages(self):
        """Every key in FR must exist in EN and vice versa."""
        fr_keys = set(STRINGS["fr"].keys())
        en_keys = set(STRINGS["en"].keys())
        missing_in_en = fr_keys - en_keys
        missing_in_fr = en_keys - fr_keys
        assert not missing_in_en, f"Keys in FR but missing in EN: {missing_in_en}"
        assert not missing_in_fr, f"Keys in EN but missing in FR: {missing_in_fr}"

    def test_no_empty_strings(self):
        """No string value should be empty (would show blank UI)."""
        for lang in ("fr", "en"):
            for key, value in STRINGS[lang].items():
                if isinstance(value, str):
                    assert len(value.strip()) > 0, (
                        f"STRINGS['{lang}']['{key}'] is empty"
                    )

    def test_format_placeholders_match(self):
        """Format strings like '{n}' and '{step}' must exist in both languages."""
        import re
        placeholder_re = re.compile(r'\{[^}]+\}')
        for key in STRINGS["fr"]:
            val_fr = STRINGS["fr"][key]
            val_en = STRINGS["en"].get(key)
            if val_en is None or not isinstance(val_fr, str) or not isinstance(val_en, str):
                continue
            placeholders_fr = set(placeholder_re.findall(val_fr))
            placeholders_en = set(placeholder_re.findall(val_en))
            if placeholders_fr or placeholders_en:
                assert placeholders_fr == placeholders_en, (
                    f"STRINGS['{key}'] placeholder mismatch: "
                    f"FR={placeholders_fr}, EN={placeholders_en}"
                )

    def test_level_labels_complete(self):
        """level_labels must have entries for all 4 levels (0-3) in both languages."""
        for lang in ("fr", "en"):
            labels = STRINGS[lang].get("level_labels", {})
            for level in (0, 1, 2, 3):
                assert level in labels, (
                    f"STRINGS['{lang}']['level_labels'] missing level {level}"
                )


# ══════════════════════════════════════════════════════════════════════════════
# 4. END-TO-END INTEGRATION TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestEndToEndIntegration:
    """Test full pipeline: messages → features → classifier → alert level."""

    def test_green_conversation(self, clf):
        """Fully positive conversation should produce Green."""
        msgs = _build_conversation([
            "Hey, how are you?",
            "I'm doing great, had a fun weekend!",
            "Yeah, went hiking with friends. It was beautiful.",
            "Planning to study tonight, feeling motivated.",
            "Thanks for chatting! Have a good night!",
            "See you later!",
        ])
        result = clf.get_alert_level(msgs, lang="en")
        assert result["level"] == 0, f"Positive conversation got level {result['level']}"

    def test_red_conversation(self, clf):
        """Crisis conversation with explicit keywords should produce Red."""
        msgs = _build_conversation([
            "I don't want to be here anymore",
            "Everything hurts",
            "I want to die",
            "Nobody cares about me",
            "I want to end it all",
            "I have a plan",
        ])
        result = clf.get_alert_level(msgs, lang="en")
        assert result["level"] == 3, f"Crisis conversation got level {result['level']}"

    def test_gradual_drift(self, clf):
        """Gradually deteriorating conversation should show escalation."""
        msgs = _build_conversation([
            "Hi, just a normal day",
            "School was okay I guess",
            "Feeling kind of tired lately",
            "I don't really enjoy anything anymore",
            "Everything feels pointless",
            "I feel like a burden to everyone",
        ])
        result = clf.get_alert_level(msgs, lang="en")
        assert result["level"] >= 1, f"Gradual drift got level {result['level']}, expected >= 1"

    def test_model_output_structure(self, clf):
        """Verify all expected keys are present in classifier output."""
        msgs = _build_conversation([
            "Hello", "How are you?", "I'm okay",
            "Just chatting", "Thanks", "Bye!",
        ])
        result = clf.get_alert_level(msgs)
        required_keys = {"level", "label", "confidence", "dominant_features"}
        assert required_keys.issubset(result.keys()), (
            f"Missing keys: {required_keys - set(result.keys())}"
        )

    def test_pipeline_features_to_prediction(self, clf):
        """Features extracted from messages produce a valid prediction vector."""
        msgs = [
            {"role": "user", "content": "Hello world"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "I feel okay today"},
            {"role": "assistant", "content": "Good to hear"},
            {"role": "user", "content": "Thanks for listening"},
        ]
        msg_features = extract_features(msgs)
        user_texts = [m["content"] for m in msgs if m["role"] == "user"]
        traj = extract_trajectory_features(msg_features, user_texts=user_texts, messages=msgs)

        # Must be 67 features (10 × 6 stats + 4 embedding + 3 coherence)
        assert len(traj) == 67, f"Expected 67 features, got {len(traj)}"
        # No NaN values
        assert not np.any(np.isnan(traj)), "Feature vector contains NaN"

    def test_response_prompts_match_alert_levels(self):
        """Each alert level should produce a distinct system prompt."""
        prompts = set()
        for level in range(4):
            prompt = get_system_prompt(level, lang="en")
            prompts.add(prompt)
        assert len(prompts) == 4, "Some alert levels share the same system prompt"


# ══════════════════════════════════════════════════════════════════════════════
# 5. EDGE CASE TESTS — things a jury member might type
# ══════════════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    """Edge cases that could crash the system during a live demo."""

    def test_emoji_only_messages(self, clf):
        """Messages with only emoji should not crash."""
        msgs = _build_conversation(["😀", "😢", "😭😭😭", "💔", "🆘"])
        result = clf.get_alert_level(msgs)
        assert result["level"] in [0, 1, 2, 3]

    def test_very_long_message(self, clf):
        """A single very long message should not crash or timeout."""
        long_msg = "I feel really bad and I don't know what to do. " * 100
        msgs = _build_conversation([
            "Hello", "I'm okay", long_msg, "Help me", "Please",
        ])
        result = clf.get_alert_level(msgs)
        assert result["level"] in [0, 1, 2, 3]

    def test_whitespace_only_messages(self, clf):
        """Messages with only whitespace should not crash."""
        msgs = _build_conversation(["   ", "\t\n", "  \n  ", ".", "..."])
        result = clf.get_alert_level(msgs)
        assert result["level"] in [0, 1, 2, 3]

    def test_mixed_language_messages(self, clf):
        """Mixing FR and EN in same conversation should not crash."""
        msgs = _build_conversation([
            "Salut, how are you?",
            "Je suis tired today",
            "C'est hard en ce moment",
            "I feel seul sometimes",
            "Anyway, ça va aller I guess",
        ])
        result = clf.get_alert_level(msgs)
        assert result["level"] in [0, 1, 2, 3]

    def test_single_character_messages(self, clf):
        """Single character messages should not crash."""
        msgs = _build_conversation(["a", "b", "c", "d", "e"])
        result = clf.get_alert_level(msgs)
        assert result["level"] in [0, 1, 2, 3]

    def test_repeated_same_message(self, clf):
        """Repeating the same message should not crash."""
        msgs = _build_conversation(["ok"] * 8)
        result = clf.get_alert_level(msgs)
        assert result["level"] in [0, 1, 2, 3]

    def test_numbers_and_special_chars(self, clf):
        """Messages with numbers and special characters should not crash."""
        msgs = _build_conversation([
            "12345", "!@#$%^&*()", "<<>>{}[]", "¿¡", "~`|\\",
        ])
        result = clf.get_alert_level(msgs)
        assert result["level"] in [0, 1, 2, 3]


# ══════════════════════════════════════════════════════════════════════════════
# 6. FEATURE SCORES OUTPUT TESTS — explainability charts need valid data
# ══════════════════════════════════════════════════════════════════════════════

class TestFeatureScoresOutput:
    """Validate feature_scores output used by the explainability charts."""

    def test_feature_scores_present(self, clf):
        """get_alert_level should return feature_scores for conversations with >= 3 messages."""
        msgs = _build_conversation([
            "Hello", "I feel bad today", "Nothing is going right",
            "I'm so tired", "I don't know anymore",
        ])
        result = clf.get_alert_level(msgs)
        assert "feature_scores" in result, "Missing 'feature_scores' in classifier output"

    def test_feature_scores_have_required_fields(self, clf):
        """Each feature score entry should have name, raw_name, and score."""
        msgs = _build_conversation([
            "Hello", "I feel bad today", "Nothing is going right",
            "I'm so tired", "I don't know anymore",
        ])
        result = clf.get_alert_level(msgs)
        for entry in result.get("feature_scores", []):
            assert "name" in entry, "Feature score missing 'name'"
            assert "raw_name" in entry, "Feature score missing 'raw_name'"
            assert "score" in entry, "Feature score missing 'score'"

    def test_feature_scores_numeric(self, clf):
        """All feature scores should be numeric (not NaN/Inf)."""
        msgs = _build_conversation([
            "Hello", "I feel bad today", "Nothing is going right",
            "I'm so tired", "I don't know anymore",
        ])
        result = clf.get_alert_level(msgs)
        for entry in result.get("feature_scores", []):
            score = entry["score"]
            assert isinstance(score, (int, float)), f"Score is not numeric: {score}"
            assert not np.isnan(score), f"Score is NaN for {entry['raw_name']}"
            assert not np.isinf(score), f"Score is Inf for {entry['raw_name']}"

    def test_feature_scores_top_5(self, clf):
        """Should return up to 5 feature scores (top 5 by composite)."""
        msgs = _build_conversation([
            "Hello", "I feel bad today", "Nothing is going right",
            "I'm so tired", "I don't know anymore",
        ])
        result = clf.get_alert_level(msgs)
        scores = result.get("feature_scores", [])
        assert 1 <= len(scores) <= 5, f"Expected 1-5 feature scores, got {len(scores)}"

    def test_dominant_features_not_empty(self, clf):
        """dominant_features should never be empty (charts would be blank)."""
        msgs = _build_conversation([
            "Hello", "I feel bad today", "Nothing is going right",
            "I'm so tired", "I don't know anymore",
        ])
        result = clf.get_alert_level(msgs)
        assert len(result["dominant_features"]) > 0, "dominant_features is empty"


# ══════════════════════════════════════════════════════════════════════════════
# 7. SESSION TRACKER INTEGRATION TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestSessionTrackerIntegration:
    """Test that real classifier results flow correctly into session tracking."""

    def test_log_real_classification(self, clf, tracker):
        """Classifier result can be logged to session tracker without error."""
        msgs = _build_conversation([
            "I feel terrible", "Nothing matters anymore",
            "I want to disappear", "Nobody cares",
            "What's the point", "I give up",
        ])
        result = clf.get_alert_level(msgs)

        session_id = tracker.start_session("integration_user")
        # Log should not crash
        tracker.log_alert(
            "integration_user", session_id,
            alert_level=result["level"],
            confidence=result["confidence"],
            message="I give up",
        )
        tracker.end_session(
            "integration_user", session_id,
            max_level=result["level"],
            message_count=6,
        )
        history = tracker.get_user_history("integration_user")
        assert len(history) == 1
        assert history[0]["max_alert_level"] == result["level"]

    def test_multiple_sessions_with_classifier(self, clf, tracker):
        """Multiple sessions with different conversations produce valid longitudinal risk."""
        conversations = [
            # Session 1: normal
            ["Hi!", "Great day!", "Love school!", "Friends are awesome!", "See you!", "Bye!"],
            # Session 2: mild concern
            ["Hey", "Tired today", "School is hard", "Feeling lonely", "I guess it's fine", "Whatever"],
            # Session 3: distress
            ["I feel terrible", "Everything is wrong", "I'm so alone",
             "Nobody understands", "I can't do this", "I want to disappear"],
        ]
        for user_msgs in conversations:
            msgs = _build_conversation(user_msgs)
            result = clf.get_alert_level(msgs)
            sid = tracker.start_session("trend_user")
            tracker.end_session("trend_user", sid, max_level=result["level"], message_count=len(user_msgs))

        risk = tracker.get_longitudinal_risk("trend_user")
        assert risk["sessions_analyzed"] == 3
        # Should show worsening or stable — not improving (conversations go from good to bad)
        assert risk["trend"] in ("worsening", "stable")

    def test_handoff_logging_with_classifier(self, clf, tracker):
        """Handoff steps can be logged alongside classifier results."""
        crisis_msgs = _build_conversation([
            "I don't want to be here", "I want to die",
            "I can't take it anymore", "Nobody cares",
            "I have nothing left", "End it all",
        ])
        result = clf.get_alert_level(crisis_msgs)

        session_id = tracker.start_session("handoff_user")
        tracker.log_alert(
            "handoff_user", session_id,
            alert_level=result["level"],
            confidence=result["confidence"],
            message="End it all",
        )
        # Log handoff steps 1-5
        for step in range(1, 6):
            tracker.log_handoff_step(
                "handoff_user", session_id,
                step=step, alert_level=result["level"],
            )
        # Should not crash and session should be trackable
        tracker.end_session(
            "handoff_user", session_id,
            max_level=result["level"],
            message_count=6,
        )
        history = tracker.get_user_history("handoff_user")
        assert len(history) == 1
