#!/usr/bin/env python3
"""
PostToolUse hook that reminds Claude to run /retrospective
after commits on feature/research branches (not main/master).
"""

import json
import subprocess
import sys


def get_current_branch():
    """Get the current git branch name."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def is_protected_branch(branch: str) -> bool:
    """Check if branch is a protected branch (main, master, develop)."""
    protected = {"main", "master", "develop", "dev"}
    return branch.lower() in protected


def main():
    # Read hook input from stdin
    try:
        hook_input = json.load(sys.stdin)
    except Exception:
        sys.exit(0)  # No input, skip

    # Get the command that was executed
    command = hook_input.get("tool_input", {}).get("command", "")

    # Check if this was a git commit
    if not command.startswith("git commit"):
        sys.exit(0)

    # Get current branch
    branch = get_current_branch()
    if not branch:
        sys.exit(0)

    # Skip if on protected branches
    if is_protected_branch(branch):
        sys.exit(0)

    # Output reminder for Claude
    reminder = """
RESEARCH RETROSPECTIVE REMINDER

You just committed research work on a feature branch.

Consider running /retrospective to:
- Document what went well/badly/ugly
- Analyze skill utilization
- Generate actionable improvements
- Save learnings to output/{branch}/retrospective_{timestamp}.md

This compounds learnings for future sessions.
"""
    print(reminder)
    sys.exit(0)


if __name__ == "__main__":
    if sys.platform == "win32":
        import io

        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    main()
