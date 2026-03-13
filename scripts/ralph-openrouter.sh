#!/usr/bin/env bash
# Launch Ralph card-improve loop with OpenRouter backend via opencode.
#
# Usage:
#   ./scripts/ralph-openrouter.sh [ralph flags...]
#   ./scripts/ralph-openrouter.sh --prompt "Focus on deck:Kotlin"
#   ./scripts/ralph-openrouter.sh --max-iterations 5
#
# Prerequisites:
#   1. Install opencode: curl -fsSL https://opencode.ai/install | bash
#   2. Configure auth: opencode auth login (select OpenRouter, paste API key)
#   3. Verify: opencode models | grep openrouter

set -euo pipefail

if ! command -v opencode &>/dev/null; then
  echo "Error: opencode not found. Install: curl -fsSL https://opencode.ai/install | bash" >&2
  exit 1
fi

if ! command -v ralph &>/dev/null; then
  echo "Error: ralph not found. Install: npm i -g ralph-orchestrator" >&2
  exit 1
fi

exec ralph run -c presets/card-improve-openrouter.yml "$@"
