# Temporal Awareness Plugin for Hermes Agent

> Know when you've been gone. Speak to the moment.

A Hermes plugin that injects time-of-day and session recency context into your message before each LLM call. When you return to a session after a gap, Hermes greets you with awareness — how long you've been away, what time it is, whether you're in night-shift mode.

## How it works

The plugin rides the `pre_llm_call` hook — a Hermes feature that lets plugins inject text into the user message before it goes to the LLM.

**Cache-safe:** The text is added *after* the cached system prompt prefix, so prompt caching stays valid. No extra API cost on rapid turns.

**Threshold-gated:** Only fires when you've been away longer than a configurable threshold (default 30 minutes). Quick back-and-forth stays silent.

**Night-aware:** Between 10pm and 5am, it appends a note reminding the model that you're in night-shift hours — energy may be lower, context-switching cost is higher — so it should be direct.

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

## Install

```bash
git clone https://github.com/EsotericOperation/hermes-temporal-awareness.git ~/.hermes/plugins/temporal_awareness
hermes gateway restart
```

## Configuration

In `~/.hermes/config.yaml`:

```yaml
agent:
  temporal_awareness:
    enabled: true              # default: true
    threshold_minutes: 30      # silence before context injection
    night_shift_start: 22      # hour (24h) for night shift start
    night_shift_end: 5         # hour (24h) for night shift end
    show_last_active: true     # include "last message was on..." line
    show_date_change: true     # include "returned on a new day" line
```

Environment overrides (no config edit needed):

```bash
HERMES_TEMPORAL_AWARENESS_THRESHOLD_MINUTES=15   # override threshold
HERMES_TEMPORAL_AWARENESS_ENABLED=0               # disable without editing config
```

## Requirements

- Hermes Agent ≥ 0.4.0 (pre_llm_call hook support)
- SQLite session database (`~/.hermes/state.db`)

## Testing

```bash
python3 -m pytest tests.py -v
```

## Contributing

1. Fork the repo
2. Create a feature branch
3. Add tests for new behavior
4. Submit a PR

## License

MIT
