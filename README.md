# Spatiotemporal Contextual Awareness Plugin for Hermes Agent

> Know when you've been gone, where you were, and what you're bringing with you.

A Hermes plugin that injects time-of-day, session recency, and cross-platform context into your message before each LLM call. When you return to a session after a gap, Hermes greets you with awareness — how long you've been away, what time it is, whether you're in night-shift mode, and whether you're resuming from a different platform.

**Version:** 1.1.0 — renamed from "Temporal Awareness" to reflect spatiotemporal + contextual dimensions.

## How it works

The plugin rides the `pre_llm_call` hook — a Hermes feature that lets plugins inject text into the user message before it goes to the LLM.

**Cache-safe:** The text is added *after* the cached system prompt prefix, so prompt caching stays valid. No extra API cost on rapid turns.

**Threshold-gated:** Only fires when you've been away longer than a configurable threshold (default 30 minutes). Quick back-and-forth stays silent.

**Night-aware:** Between 10pm and 5am, it appends a note reminding the model that you're in night-shift hours — energy may be lower, context-switching cost is higher — so it should be direct.

**Platform-aware:** When you resume a session from a different platform than where it started (e.g., Telegram → Desktop), it injects cross-platform context so Hermes knows where you were.

## Example output

After 41 minutes away:

```
Temporal context: You are resuming this session after 41 minutes away.
Current time: Wednesday, September 19, 2026 — 12:47 PM (afternoon shift).
Last message was on Wednesday at 12:06 PM.
```

After 6 hours, with a day change:

```
Temporal context: You are resuming this session after 6 hours away.
Current time: Thursday, September 20, 2026 — 8:15 AM (morning shift).
Last message was on Wednesday at 2:14 PM.
The user has returned on a new day.
```

At 2am:

```
Temporal context: You are resuming this session after 3 hours away.
Current time: Saturday, September 22, 2026 — 2:33 AM (night shift).
Last message was on Friday at 11:47 PM.
The user has returned on a new day.
User is in night-shift hours (22:00–5:00). Energy may be lower,
context-switching cost is higher — be direct and don't meander.
```

Cross-platform (Telegram session resumed from Desktop):

```
Temporal context: You are resuming this session after 2 hours away.
Current time: Monday, September 21, 2026 — 10:33 PM (evening shift).
Last message was on Monday at 8:33 PM.
Platform context: This session was started on Telegram. You are now resuming from Desktop.
```

Cross-platform, below temporal threshold:

```
Platform context: This session was started on Telegram. You are now resuming from Desktop.
```

## Install

```bash
git clone https://github.com/EsotericOperation/hermes-spatiotemporal-contextual-awareness.git \
  ~/.hermes/plugins/spatiotemporal_contextual_awareness
hermes gateway restart
```

## Configuration

In `~/.hermes/config.yaml`:

```yaml
agent:
  spatiotemporal_contextual_awareness:
    enabled: true                # default: true
    threshold_minutes: 30        # silence before context injection
    night_shift_start: 22        # hour (24h) for night shift start
    night_shift_end: 5           # hour (24h) for night shift end
    show_last_active: true       # include "last message was on..." line
    show_date_change: true       # include "returned on a new day" line
    show_platform_change: true   # include cross-platform handoff context
```

## Environment

For quick config overrides:

- `HERMES_TEMPORAL_AWARENESS_THRESHOLD_MINUTES` — override threshold (takes priority over config)
- `HERMES_TEMPORAL_AWARENESS_ENABLED` — `"0"`|`"false"` to disable without editing config

> Note: Environment variable names retained from v1.x for backward compatibility.

## Migration from v1.x

If you had `temporal_awareness` in your config, the plugin will still find it via the `agent.temporal_awareness` fallback path. To migrate:

1. Rename the directory: `mv ~/.hermes/plugins/temporal_awareness ~/.hermes/plugins/spatiotemporal_contextual_awareness`
2. Add to `plugins.enabled`: `- spatiotemporal-contextual-awareness`
3. (Optional) Rename the config key from `temporal_awareness` to `spatiotemporal_contextual_awareness`

The plugin checks `spatiotemporal_contextual_awareness` first, then falls back to `temporal_awareness` for backward compatibility.

## Requirements

- Hermes Agent ≥ 0.4.0 (pre_llm_call hook support)
- SQLite session database (`~/.hermes/state.db`)

## License

MIT
