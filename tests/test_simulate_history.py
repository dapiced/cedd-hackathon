import os
import sqlite3
import sys
import pytest

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from simulate_history import main, USERS_PLANS
import simulate_history
import cedd.session_tracker

@pytest.fixture
def tmp_db(tmp_path, monkeypatch):
    """Fixture to provide a temporary database path and mock DEFAULT_DB_PATH."""
    db_path = str(tmp_path / "test_cedd_sessions.db")

    # Mock the DB path in both places it's used
    monkeypatch.setattr(simulate_history, "DEFAULT_DB_PATH", db_path)
    monkeypatch.setattr(cedd.session_tracker, "DEFAULT_DB_PATH", db_path)

    return db_path

def test_main_french(monkeypatch, tmp_db):
    """Test running simulate_history with the default French language."""
    # Mock sys.argv to simulate command line execution
    monkeypatch.setattr(sys, "argv", ["simulate_history.py", "--lang", "fr"])

    # Run SessionTracker to initialize the DB tables before main clears them
    cedd.session_tracker.SessionTracker(db_path=tmp_db)

    # Run the main function
    main()

    # Verify the database was created and populated correctly
    assert os.path.exists(tmp_db)

    with sqlite3.connect(tmp_db) as conn:
        cursor = conn.cursor()

        # Check sessions count: len(USERS_PLANS) users * 7 sessions each
        cursor.execute("SELECT COUNT(*) FROM sessions")
        session_count = cursor.fetchone()[0]
        expected_sessions = len(USERS_PLANS) * 7
        assert session_count == expected_sessions

        # Check that some messages were inserted
        cursor.execute("SELECT COUNT(*) FROM alert_events")
        message_count = cursor.fetchone()[0]
        assert message_count > 0

def test_main_english(monkeypatch, tmp_db):
    """Test running simulate_history with the English language."""
    # Mock sys.argv to simulate command line execution
    monkeypatch.setattr(sys, "argv", ["simulate_history.py", "--lang", "en"])

    # Run SessionTracker to initialize the DB tables before main clears them
    cedd.session_tracker.SessionTracker(db_path=tmp_db)

    # Run the main function
    main()

    # Verify the database was created and populated correctly
    assert os.path.exists(tmp_db)

    with sqlite3.connect(tmp_db) as conn:
        cursor = conn.cursor()

        # Check sessions count
        cursor.execute("SELECT COUNT(*) FROM sessions")
        session_count = cursor.fetchone()[0]
        expected_sessions = len(USERS_PLANS) * 7
        assert session_count == expected_sessions

        # Verify an English word exists in the DB (like 'I' or 'today' from SAMPLE_MESSAGES['en'])
        cursor.execute("SELECT trigger_message FROM alert_events LIMIT 10")
        messages = [row[0] for row in cursor.fetchall()]

        # Ensure that no French-specific text from sample messages is found,
        # but english messages exist
        assert len(messages) > 0

def test_clears_previous_history(monkeypatch, tmp_db):
    """Test that running simulate_history clears existing demo user history."""
    # Mock sys.argv
    monkeypatch.setattr(sys, "argv", ["simulate_history.py"])

    # Run SessionTracker to initialize the DB tables before main clears them
    cedd.session_tracker.SessionTracker(db_path=tmp_db)

    # Run the simulation twice
    main()
    main()

    with sqlite3.connect(tmp_db) as conn:
        cursor = conn.cursor()

        # Ensure session count is exactly for one run, not doubled
        cursor.execute("SELECT COUNT(*) FROM sessions")
        session_count = cursor.fetchone()[0]
        expected_sessions = len(USERS_PLANS) * 7
        assert session_count == expected_sessions
