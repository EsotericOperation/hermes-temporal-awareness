"""Spatiotemporal Contextual Awareness plugin — inject time-of-day, session recency,
and cross-platform context into the user message before each LLM call.

Cache-safe: rides the pre_llm_call hook (injected into user message, never the system prompt).
Only fires when the user has been away for longer than the configured threshold to avoid
injecting redundant context on rapid back-and-forth turns.

Config (in config.yaml under ``agent.spatiotemporal_contextual_awareness``):

    agent:
      spatiotemporal_contextual_awareness:
        enabled: true
        threshold_minutes: 30      # how long silence before we inject context
        night_shift_start: 22      # 24h hour for "night shift" tailoring
        night_shift_end: 5         # 24h hour for end of night shift
        show_last_active: true     # include "last message was on..." line
        show_date_change: true     # include "returned on a new day" line
        show_platform_change: true # include cross-platform handoff context

Environment:
    HERMES_TEMPORAL_AWARENESS_THRESHOLD_MINUTES  # overrides config (takes priority)
    HERMES_TEMPORAL_AWARENESS_ENABLED            # "0"|"false" to disable without editing config
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# --- Config ---

def _load_config(cfg: Any = None) -> dict:
    """Load plugin-specific config from agent.spatiotemporal_contextual_awareness section."""
    if cfg is None:
        try:
            from hermes_cli.config import load_config
            cfg = load_config() or {}
        except Exception:
            return {}
    
    agent = cfg.get("agent") or {}
    ta_cfg = agent.get("spatiotemporal_contextual_awareness") or agent.get("temporal_awareness") or {}
    
    # Config defaults
    out = {
        "enabled": ta_cfg.get("enabled", True),
        "threshold_minutes": ta_cfg.get("threshold_minutes", 30),
        "night_shift_start": ta_cfg.get("night_shift_start", 22),
        "night_shift_end": ta_cfg.get("night_shift_end", 5),
        "show_last_active": ta_cfg.get("show_last_active", True),
        "show_date_change": ta_cfg.get("show_date_change", True),
        "show_platform_change": ta_cfg.get("show_platform_change", True),
    }
    
    # Environment overrides (for quick tweaks without editing config.yaml)
    env_threshold = os.environ.get("HERMES_TEMPORAL_AWARENESS_THRESHOLD_MINUTES")
    if env_threshold is not None:
        try:
            out["threshold_minutes"] = int(env_threshold)
        except ValueError:
            pass
    
    env_enabled = os.environ.get("HERMES_TEMPORAL_AWARENESS_ENABLED")
    if env_enabled is not None:
        out["enabled"] = env_enabled.lower() not in ("0", "false", "no", "off")
    
    return out


# --- Database ---

_DB_PATH = Path.home() / ".hermes" / "state.db"


def _get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def _session_last_active_unix(session_id: str) -> float | None:
    """Return the unix timestamp of the last message in a session, or None."""
    if not session_id:
        return None
    try:
        conn = _get_db()
        cur = conn.cursor()
        cur.execute(
            "SELECT MAX(timestamp) FROM messages WHERE session_id = ?",
            (session_id,),
        )
        row = cur.fetchone()
        ts = row[0] if row else None
        conn.close()
        return ts
    except Exception:
        return None


def _session_started_unix(session_id: str) -> float | None:
    """Return the unix timestamp when the session started."""
    if not session_id:
        return None
    try:
        conn = _get_db()
        cur = conn.cursor()
        cur.execute("SELECT started_at FROM sessions WHERE id = ?", (session_id,))
        row = cur.fetchone()
        ts = row[0] if row else None
        conn.close()
        return ts
    except Exception:
        return None


def _session_platform(session_id: str) -> str | None:
    """Return the platform name for a session from origin_json, or None."""
    if not session_id:
        return None
    try:
        conn = _get_db()
        cur = conn.cursor()
        cur.execute("SELECT origin_json FROM sessions WHERE id = ?", (session_id,))
        row = cur.fetchone()
        conn.close()
        if row and row[0]:
            origin = json.loads(row[0])
            return origin.get("platform")
    except Exception:
        pass
    return None


# --- Formatting ---

def _format_duration(seconds: float) -> str:
    """Human-readable duration: '2 hours, 14 minutes' or '45 minutes'."""
    if seconds < 60:
        return "less than a minute"
    minutes = int(seconds // 60)
    if minutes < 60:
        return f"{minutes} minute{'s' if minutes != 1 else ''}"
    hours = minutes // 60
    remaining_minutes = minutes % 60
    if remaining_minutes == 0:
        return f"{hours} hour{'s' if hours != 1 else ''}"
    return f"{hours} hour{'s' if hours != 1 else ''}, {remaining_minutes} minute{'s' if remaining_minutes != 1 else ''}"


def _get_time_period(hour: int) -> str:
    """Classify hour into morning/afternoon/evening/night."""
    if 5 <= hour < 12:
        return "morning"
    elif 12 <= hour < 17:
        return "afternoon"
    elif 17 <= hour < 22:
        return "evening"
    else:
        return "night"


def _platform_display_name(platform: str) -> str:
    """Human-readable platform name."""
    names = {
        "telegram": "Telegram",
        "discord": "Discord",
        "signal": "Signal",
        "photon": "Photon",
        "desktop": "Desktop",
        "cli": "CLI",
        "tui": "TUI",
        "webui": "WebUI",
        "api_server": "API",
        "abyss": "Abyss iOS",
    }
    return names.get(platform, platform.title() if platform else "Unknown")


# --- Hook ---

def on_pre_llm_call(
    *,
    session_id: str = "",
    task_id: str = "",
    turn_id: str = "",
    user_message: Any = None,
    conversation_history: Any = None,
    is_first_turn: bool = False,
    model: str = "",
    platform: str = "",
    parent_session_id: str = "",
    sender_id: str = "",
    **_: Any,
) -> dict:
    """Inject temporal + platform context into the user message.

    Cache-safe: only adds to user message content, never touches the system prompt.
    Returns a dict with ``context`` key so the Hermes hook runner appends it.

    Fires when:
    - The user has been away longer than the configured threshold (temporal gap)
    - OR the user is accessing a session from a different platform than where it started
    """
    cfg = _load_config()
    
    if not cfg["enabled"]:
        return {"context": ""}
    
    now = time.time()
    last_active = _session_last_active_unix(session_id)
    gap_seconds = (now - last_active) if last_active else None
    
    # Standard temporal gap path — only when session has history
    threshold_seconds = cfg["threshold_minutes"] * 60
    temporal_fires = gap_seconds is not None and gap_seconds >= threshold_seconds
    
    # Platform awareness — compare session's origin platform with current platform
    # This fires ANY time they differ, regardless of turn count or gap
    platform_fires = False
    platform_part = ""
    if cfg["show_platform_change"] and session_id:
        session_platform = _session_platform(session_id)
        if session_platform and session_platform != platform:
            platform_fires = True
            origin_name = _platform_display_name(session_platform)
            current_name = _platform_display_name(platform)
            platform_part = (
                f"Platform context: This session was started on {origin_name}. "
                f"You are now resuming from {current_name}."
            )
    
    # Neither condition met — return empty
    if not temporal_fires and not platform_fires:
        return {"context": ""}
    
    # Temporal context (gap-based)
    parts = []
    if temporal_fires:
        # Format timestamps in local time
        now_dt = datetime.fromtimestamp(now)
        last_dt = datetime.fromtimestamp(last_active)
        
        gap_human = _format_duration(gap_seconds)
        hour = now_dt.hour
        time_period = _get_time_period(hour)
        
        day_name = now_dt.strftime("%A")
        date_str = now_dt.strftime("%B %d, %Y")
        time_str = now_dt.strftime("%I:%M %p").lstrip("0")
        
        parts.append(f"Temporal context: You are resuming this session after {gap_human} away.")
        parts.append(f"Current time: {day_name}, {date_str} — {time_str} ({time_period} shift).")
        
        # Last active info
        if cfg["show_last_active"]:
            last_time_str = last_dt.strftime("%I:%M %p").lstrip("0")
            last_day_str = last_dt.strftime("%A")
            parts.append(f"Last message was on {last_day_str} at {last_time_str}.")
        
        # Day change
        if cfg["show_date_change"] and now_dt.date() != last_dt.date():
            parts.append("The user has returned on a new day.")
        
        # Night shift awareness
        night_start = cfg["night_shift_start"]
        night_end = cfg["night_shift_end"]
        if night_start > night_end:  # wraps midnight
            in_night = hour >= night_start or hour < night_end
        else:
            in_night = night_start <= hour < night_end
        
        if in_night:
            parts.append(
                f"User is in night-shift hours ({night_start}:00–{night_end}:00). "
                "Energy may be lower, context-switching cost is higher — be direct and don't meander."
            )
    
    # Platform context — appended after temporal (or standalone if no temporal)
    if platform_part:
        parts.append(platform_part)
    
    return {"context": " ".join(parts)}


# --- Registration ---

def register(ctx: Any) -> None:
    """Register the pre_llm_call hook with Hermes."""
    ctx.register_hook("pre_llm_call", on_pre_llm_call)
