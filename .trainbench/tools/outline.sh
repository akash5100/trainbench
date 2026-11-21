#!/usr/bin/env bash
# .trainbench/tools/outline.sh — Structural file skeleton extractor for Python & TypeScript/JavaScript.
# Extracts section boundaries (# ━━), classes, methods, and types with exact line numbers.

set -euo pipefail

usage() {
  cat << USAGE
Usage:
  .trainbench/tools/outline.sh <file_path>         # Print structural outline of classes, functions, constants, sections
  .trainbench/tools/outline.sh --help              # Show this help message

Examples:
  .trainbench/tools/outline.sh trainbench/models/jax/qwen3_5.py
USAGE
  exit 0
}

if [ $# -eq 0 ] || [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
  usage
fi

FILE="$1"

if [ ! -f "$FILE" ]; then
  echo "Error: File '$FILE' not found." >&2
  exit 1
fi

EXT="${FILE##*.}"

case "$EXT" in
  py)
    # Python: Section headers (# ━━), classes, async/sync defs, type aliases, uppercase constants, __dunders__
    grep -nE "^\s*(async )?def |^\s*class |^type [A-Z]|^([A-Z_][A-Z0-9_]+|__[a-z_]+__)\s*[:=]|# ━━" "$FILE" || true
    ;;
  ts|tsx|js|jsx)
    # TypeScript/JavaScript: Section headers (// ━━), export/classes/functions/interfaces/types, exported constants
    grep -nE "^\s*(export )?(async )?(function|class|interface|type)\b|^\s*(export )?const [A-Za-z0-9_]+\s*[:=]|// ━━" "$FILE" || true
    ;;
  *)
    # Generic fallback
    grep -nE "^\s*(async )?(def|function|class|interface|type)\b|# ━━|// ━━" "$FILE" || true
    ;;
esac
