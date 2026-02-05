#!/usr/bin/env python3
"""
PostToolUse hook (filtered to Skill): log skill invocations.

Reads tool input from stdin (JSON with skill name) and appends a
JSONL entry to .claude/skill-use-log.jsonl.
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

LOG_FILE = Path(".claude") / "skill-use-log.jsonl"
DEBUG_FILE = Path(".claude") / "skill-use-debug.jsonl"


def main():
    # Read hook input from stdin
    raw = sys.stdin.read().strip()
    if not raw:
        return

    try:
        hook_input = json.loads(raw)
    except json.JSONDecodeError:
        return

    # Debug: log raw input structure
    DEBUG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(DEBUG_FILE, "a") as f:
        debug_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tool_name": hook_input.get("tool_name"),
            "keys": list(hook_input.keys()),
            "tool_input_keys": list(hook_input.get("tool_input", {}).keys()) if isinstance(hook_input.get("tool_input"), dict) else str(type(hook_input.get("tool_input"))),
        }
        f.write(json.dumps(debug_entry) + "\n")

    # Extract skill name from tool input
    tool_input = hook_input.get("tool_input", {})
    skill_name = tool_input.get("skill", "unknown")

    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "skill": skill_name,
        "session_id": os.environ.get("CLAUDE_SESSION_ID", "unknown"),
        "project": Path.cwd().name,
    }

    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        pass  # Never block the session
    sys.exit(0)
