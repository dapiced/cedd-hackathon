"""
Unit tests for pure utility functions in app.py
"""

import os
import sys

# Add project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from app import _user_id_from_display, get_theme_css, _feature_color, THEMES


class TestUserIdFromDisplay:
    def test_with_parentheses(self):
        assert _user_id_from_display("Shuchita (stable green)") == "Shuchita"
        assert _user_id_from_display("Dominic (escalating)") == "Dominic"

    def test_without_parentheses(self):
        assert _user_id_from_display("Guest") == "Guest"
        assert _user_id_from_display("Amanda") == "Amanda"

    def test_empty_string(self):
        assert _user_id_from_display("") == ""

    def test_with_extra_spaces(self):
        assert _user_id_from_display("Priyanka ( improving )") == "Priyanka"

class TestGetThemeCss:
    def test_light_theme(self):
        css = get_theme_css("light")
        # Check if basic light theme colors are present
        assert THEMES["light"]["bg_main"] in css
        assert THEMES["light"]["text_main"] in css
        assert THEMES["light"]["bg_card"] in css
        # Check if the generated css is a string containing expected CSS structure
        assert "<style>" in css
        assert "</style>" in css
        assert "background-color:" in css

    def test_dark_theme(self):
        css = get_theme_css("dark")
        # Check if basic dark theme colors are present
        assert THEMES["dark"]["bg_main"] in css
        assert THEMES["dark"]["text_main"] in css
        assert THEMES["dark"]["bg_card"] in css
        # Check if the generated css is a string containing expected CSS structure
        assert "<style>" in css
        assert "</style>" in css
        assert "background-color:" in css


class TestFeatureColor:
    def test_red_category(self):
        assert _feature_color("finality_score_mean") == "#e74c3c"
        assert _feature_color("crisis_similarity") == "#e74c3c"

    def test_orange_category(self):
        assert _feature_color("negative_score_slope") == "#e67e22"
        assert _feature_color("negation_score_max") == "#e67e22"

    def test_green_category(self):
        assert _feature_color("hope_score_last") == "#27ae60"

    def test_purple_category(self):
        assert _feature_color("identity_conflict_score_mean") == "#9b59b6"
        assert _feature_color("somatization_score_max") == "#9b59b6"

    def test_teal_category(self):
        assert _feature_color("short_response_ratio") == "#1abc9c"
        assert _feature_color("min_topic_coherence") == "#1abc9c"
        assert _feature_color("question_response_ratio") == "#1abc9c"
        assert _feature_color("embedding_drift") == "#1abc9c"
        assert _feature_color("embedding_slope") == "#1abc9c"
        assert _feature_color("embedding_variance") == "#1abc9c"

    def test_blue_structural_category(self):
        # The fallback for anything not explicitly matched above
        assert _feature_color("word_count_slope") == "#3498db"
        assert _feature_color("punctuation_ratio_mean") == "#3498db"
        assert _feature_color("length_delta_last") == "#3498db"
        assert _feature_color("unknown_feature") == "#3498db"
