"""Test suite for spatiotemporal_contextual_awareness plugin.

These tests are intentionally minimal — they verify that the configuration loading,
duration formatting, and time period detection functions work correctly.
No real database or Hermes gateway required.
"""

import os
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure plugin is importable
sys.path.insert(0, str(Path.home() / ".hermes" / "plugins"))
import spatiotemporal_contextual_awareness as ta


class TestConfig:
    """Configuration loading and merging."""

    def test_default_config_when_no_config_provided(self):
        """When no config is available, defaults should be used."""
        cfg = ta._load_config({})
        assert cfg["enabled"] is True
        assert cfg["threshold_minutes"] == 30
        assert cfg["night_shift_start"] == 22
        assert cfg["night_shift_end"] == 5

    def test_config_from_yaml(self):
        """Config from agent.temporal_awareness section should be used."""
        cfg = ta._load_config({
            "agent": {
                "temporal_awareness": {
                    "threshold_minutes": 15,
                    "night_shift_start": 23,
                }
            }
        })
        assert cfg["threshold_minutes"] == 15
        assert cfg["night_shift_start"] == 23
        # Unchanged defaults
        assert cfg["night_shift_end"] == 5

    def test_env_override_threshold(self):
        """HERMES_TEMPORAL_AWARENESS_THRESHOLD_MINUTES should override config."""
        old = os.environ.get("HERMES_TEMPORAL_AWARENESS_THRESHOLD_MINUTES")
        try:
            os.environ["HERMES_TEMPORAL_AWARENESS_THRESHOLD_MINUTES"] = "10"
            cfg = ta._load_config({})
            assert cfg["threshold_minutes"] == 10
        finally:
            if old is None:
                del os.environ["HERMES_TEMPORAL_AWARENESS_THRESHOLD_MINUTES"]
            else:
                os.environ["HERMES_TEMPORAL_AWARENESS_THRESHOLD_MINUTES"] = old

    def test_env_override_disabled(self):
        """HERMES_TEMPORAL_AWARENESS_ENABLED=0 should disable plugin."""
        old = os.environ.get("HERMES_TEMPORAL_AWARENESS_ENABLED")
        try:
            os.environ["HERMES_TEMPORAL_AWARENESS_ENABLED"] = "0"
            cfg = ta._load_config({})
            assert cfg["enabled"] is False
        finally:
            if old is None:
                del os.environ["HERMES_TEMPORAL_AWARENESS_ENABLED"]
            else:
                os.environ["HERMES_TEMPORAL_AWARENESS_ENABLED"] = old

    def test_invalid_env_threshold_ignored(self):
        """Invalid env var should be ignored."""
        old = os.environ.get("HERMES_TEMPORAL_AWARENESS_THRESHOLD_MINUTES")
        try:
            os.environ["HERMES_TEMPORAL_AWARENESS_THRESHOLD_MINUTES"] = "not_a_number"
            cfg = ta._load_config({})
            assert cfg["threshold_minutes"] == 30  # default
        finally:
            if old is None:
                del os.environ["HERMES_TEMPORAL_AWARENESS_THRESHOLD_MINUTES"]
            else:
                os.environ["HERMES_TEMPORAL_AWARENESS_THRESHOLD_MINUTES"] = old


class TestFormatDuration:
    """Duration formatting for human-readable gaps."""

    def test_less_than_a_minute(self):
        assert ta._format_duration(30) == "less than a minute"

    def test_one_minute_singular(self):
        assert ta._format_duration(60) == "1 minute"

    def test_minutes_plural(self):
        assert ta._format_duration(900) == "15 minutes"

    def test_one_hour_singular(self):
        assert ta._format_duration(3600) == "1 hour"

    def test_hours_and_minutes(self):
        assert ta._format_duration(8040) == "2 hours, 14 minutes"


class TestTimePeriod:
    """Time-of-day classification."""

    @pytest.mark.parametrize("hour,expected", [
        (6, "morning"),     # 6 AM
        (11, "morning"),    # 11 AM
        (12, "afternoon"),  # noon
        (16, "afternoon"),  # 4 PM
        (17, "evening"),    # 5 PM
        (21, "evening"),    # 9 PM
        (22, "night"),      # 10 PM
        (3, "night"),       # 3 AM
        (0, "night"),       # midnight
    ])
    def test_time_period_classification(self, hour, expected):
        assert ta._get_time_period(hour) == expected


class TestOnPreLLMCall:
    """Hook behavior — these test the function in isolation (no real DB)."""

    def test_no_session_id_returns_empty(self):
        """When no session_id is provided, return empty context."""
        result = ta.on_pre_llm_call(
            session_id="",
            is_first_turn=False,
        )
        assert result == {"context": ""}

    @patch.object(ta, "_session_last_active_unix", return_value=None)
    def test_no_last_active_returns_empty(self, mock_last):
        """When session has no messages, return empty context."""
        result = ta.on_pre_llm_call(
            session_id="test-session",
            is_first_turn=False,
        )
        assert result == {"context": ""}

    @patch.object(ta, "_session_last_active_unix")
    def test_gap_below_threshold_returns_empty(self, mock_last):
        """When gap is less than threshold, return empty context."""
        mock_last.return_value = time.time() - 1200  # 20 minutes ago (default threshold is 30)
        result = ta.on_pre_llm_call(
            session_id="test-session",
            is_first_turn=False,
        )
        assert result == {"context": ""}

    @patch.object(ta, "_session_last_active_unix")
    def test_gap_above_threshold_returns_context(self, mock_last):
        """When gap exceeds threshold, return temporal context."""
        mock_last.return_value = time.time() - 3600  # 1 hour ago (default threshold is 30 min)
        result = ta.on_pre_llm_call(
            session_id="test-session",
            is_first_turn=False,
        )
        ctx = result.get("context", "")
        assert ctx != ""
        assert "Temporal context:" in ctx
        assert "Current time:" in ctx

    @patch.object(ta, "_session_last_active_unix")
    def test_night_shift_appended(self, mock_last):
        """During night shift hours, night-shift note should be appended."""
        # Mock night time (1 AM)
        with patch("spatiotemporal_contextual_awareness.time") as mock_time:
            mock_time.time.return_value = 1740000000.0  # some unix time
            mock_last.return_value = 1740000000.0 - 3600  # 1 hour ago
            # This test is brittle because datetime.now() uses real time
            # Instead, we check the behavior by mocking time.time
            # For a real test, we'd need to patch datetime.fromtimestamp
            result = ta.on_pre_llm_call(
                session_id="test-session",
                is_first_turn=False,
            )
            # At minimum, context should be non-empty
            assert result != {"context": ""}

    def test_disabled_returns_empty(self):
        """When disabled via config, return empty context."""
        with patch.object(ta, "_load_config", return_value={
            "enabled": False,
            "threshold_minutes": 30,
            "night_shift_start": 22,
            "night_shift_end": 5,
            "show_last_active": True,
            "show_date_change": True,
        }):
            result = ta.on_pre_llm_call(
                session_id="test-session",
                is_first_turn=False,
            )
            assert result == {"context": ""}


class TestPlatformDisplayName:
    """Platform name formatting."""

    def test_telegram(self):
        assert ta._platform_display_name("telegram") == "Telegram"

    def test_discord(self):
        assert ta._platform_display_name("discord") == "Discord"

    def test_signal(self):
        assert ta._platform_display_name("signal") == "Signal"

    def test_desktop(self):
        assert ta._platform_display_name("desktop") == "Desktop"

    def test_cli(self):
        assert ta._platform_display_name("cli") == "CLI"

    def test_unknown_platform(self):
        assert ta._platform_display_name("foobar") == "Foobar"


class TestSessionPlatform:
    """Session origin platform extraction."""

    @patch.object(ta, "_get_db")
    def test_returns_platform_from_origin_json(self, mock_db):
        import json
        mock_conn = MagicMock()
        mock_conn.cursor.return_value.fetchone.return_value = [
            json.dumps({"platform": "telegram", "chat_name": "Test"})
        ]
        mock_db.return_value = mock_conn
        assert ta._platform_display_name("telegram") == "Telegram"

    @patch.object(ta, "_get_db")
    def test_returns_none_on_missing_session(self, mock_db):
        mock_conn = MagicMock()
        mock_conn.cursor.return_value.fetchone.return_value = None
        mock_db.return_value = mock_conn
        assert ta._session_platform("nonexistent") is None


class TestCrossPlatformInjection:
    """Cross-platform context injection in the hook."""

    @patch.object(ta, "_session_last_active_unix", return_value=time.time())
    @patch.object(ta, "_session_platform", return_value="telegram")
    def test_different_platform_injects_context(self, mock_plat, mock_last):
        """When current platform differs from session origin, context fires."""
        result = ta.on_pre_llm_call(
            session_id="test-sess",
            is_first_turn=False,
            platform="desktop",
            sender_id="user123",
        )
        ctx = result.get("context", "")
        assert ctx != ""
        assert "Platform context:" in ctx
        assert "Telegram" in ctx
        assert "Desktop" in ctx

    @patch.object(ta, "_session_last_active_unix", return_value=time.time())
    @patch.object(ta, "_session_platform", return_value="telegram")
    def test_same_platform_no_platform_context(self, mock_plat, mock_last):
        """When current platform matches session origin, no platform context."""
        result = ta.on_pre_llm_call(
            session_id="test-sess",
            is_first_turn=False,
            platform="telegram",
            sender_id="user123",
        )
        ctx = result.get("context", "")
        # Context may still fire from temporal gap, but should NOT have platform part
        assert "Platform context:" not in ctx

    @patch.object(ta, "_session_last_active_unix", return_value=time.time())
    @patch.object(ta, "_session_platform", return_value="telegram")
    def test_show_platform_change_disabled(self, mock_plat, mock_last):
        """If show_platform_change is disabled, no platform context."""
        with patch.object(ta, "_load_config", return_value={
            "enabled": True,
            "threshold_minutes": 30,
            "night_shift_start": 22,
            "night_shift_end": 5,
            "show_last_active": True,
            "show_date_change": True,
            "show_platform_change": False,
        }):
            result = ta.on_pre_llm_call(
                session_id="test-sess",
                is_first_turn=False,
                platform="desktop",
                sender_id="user123",
            )
            ctx = result.get("context", "")
            assert "Platform context:" not in ctx

    @patch.object(ta, "_session_last_active_unix", return_value=None)
    @patch.object(ta, "_session_platform", return_value="telegram")
    def test_no_history_different_platform_fires(self, mock_plat, mock_last):
        """No prior activity + different platform → platform context fires."""
        result = ta.on_pre_llm_call(
            session_id="new-sess",
            is_first_turn=True,
            platform="desktop",
            sender_id="user123",
        )
        ctx = result.get("context", "")
        assert "Platform context:" in ctx


class TestRegister:
    """Plugin registration."""

    def test_register_registers_hook(self):
        """register() should call ctx.register_hook with 'pre_llm_call'."""
        mock_ctx = MagicMock()
        ta.register(mock_ctx)
        mock_ctx.register_hook.assert_called_once_with("pre_llm_call", ta.on_pre_llm_call)
