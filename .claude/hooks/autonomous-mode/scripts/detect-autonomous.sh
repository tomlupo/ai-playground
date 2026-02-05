#!/bin/bash
# Detect autonomous mode markers in prompt and inject guidance

# Read prompt from stdin
PROMPT=$(cat)

# Check for autonomous mode markers (case-insensitive)
if echo "$PROMPT" | grep -qiE "(AUTONOMOUS|--autonomous|/auto)"; then
  cat << 'EOF'
╔══════════════════════════════════════════════════════════════╗
║  AUTONOMOUS MODE DETECTED                                     ║
╠══════════════════════════════════════════════════════════════╣
║  • Make reasonable assumptions - don't ask questions          ║
║  • Document assumptions in output                             ║
║  • Use defaults: RSI-14, SMA-20/50/200, data/samples/         ║
║  • Always save output to output/{branch}/                     ║
║  • Generate gist-report when done                             ║
╚══════════════════════════════════════════════════════════════╝
EOF
fi
