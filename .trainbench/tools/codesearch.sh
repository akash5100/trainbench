#!/usr/bin/env bash
# .trainbench/tools/codesearch.sh — AST-aware semantic code search and relationship graph.
# Powered by colgrep with lightonai/LateOn-Code-edge (ColBERT late-interaction).

set -euo pipefail

# Ensure index and lock files live inside workspace cache by default to prevent
# macOS seatbelt sandbox EPERM errors (e.g. under Codex or restricted runners).
_tools_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=../paths.sh
source "${_tools_dir%/.trainbench/tools}/.trainbench/paths.sh"
SCRIPT_DIR="$TRAINBENCH_TOOLS_DIR"
export REPO_ROOT="${REPO_ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || echo "$TRAINBENCH_REPO_ROOT")}"
export XDG_DATA_HOME="${XDG_DATA_HOME:-$TRAINBENCH_CACHE_DIR}"
unset _tools_dir
mkdir -p "${XDG_DATA_HOME}"

# Auto-migrate existing index from macOS legacy path if not present in cache
LEGACY_DIR="${HOME}/Library/Application Support/colgrep/indices"
TARGET_DIR="${XDG_DATA_HOME}/colgrep/indices"
if [ ! -d "${TARGET_DIR}" ] && [ -d "${LEGACY_DIR}" ]; then
  mkdir -p "${TARGET_DIR}"
  cp -R "${LEGACY_DIR}"/* "${TARGET_DIR}/" 2>/dev/null || true
fi

if ! command -v colgrep >/dev/null 2>&1; then
  echo "Error: 'colgrep' binary not found. Install via: brew install lightonai/tap/colgrep" >&2
  exit 1
fi

usage() {
  cat << USAGE
Usage:
  .trainbench/tools/codesearch.sh "<query>"                    # Semantic AST graph search (default)
  .trainbench/tools/codesearch.sh -e "<pattern>" "<query>"     # Hybrid search: exact keyword + semantic intent
  .trainbench/tools/codesearch.sh -k <num> "<query>"           # Return top-k results (default: 8)
  .trainbench/tools/codesearch.sh --include="*.py" "<query>"   # Filter by file extension or glob
  .trainbench/tools/codesearch.sh --raw [flags...] "<query>"   # Direct colgrep raw output
  .trainbench/tools/codesearch.sh --help                       # Show this help message

Querying Strategies:
  • Natural Language Intent:
      .trainbench/tools/codesearch.sh "muon newton-schulz iteration"
  • Hybrid Exact Filter + Intent (Highest Precision):
      .trainbench/tools/codesearch.sh -e "stop_gradient" "XLA binary compilation"
      .trainbench/tools/codesearch.sh -e "Mesh" "TPU device sharding"
  • Scoped to Subdirectory:
      .trainbench/tools/codesearch.sh "learning rate warmup" ./trainbench/optim
USAGE
  exit 0
}

if [ $# -eq 0 ]; then
  usage
fi

case "$1" in
  -h|--help)
    usage
    ;;
  --raw)
    shift
    (cd "${REPO_ROOT:-.}" && colgrep "$@")
    ;;
  *)
    python3 -c '
import sys, json, os, subprocess

args = sys.argv[1:]
repo_root = os.environ.get("REPO_ROOT")
colgrep_cmd = ["colgrep", "--json"]

has_k = any(a in ("-k", "--results") for a in args)
if not has_k:
    colgrep_cmd.extend(["-k", "8"])

colgrep_cmd.extend(args)

try:
    proc = subprocess.run(colgrep_cmd, cwd=repo_root, capture_output=True, text=True, check=True)
    raw = proc.stdout.strip()
    data = json.loads(raw) if raw else []
except Exception:
    fallback_cmd = ["colgrep"] + [a for a in args if a != "--json"]
    subprocess.run(fallback_cmd, cwd=repo_root)
    sys.exit(0)

if not data:
    print("No matching code units found.")
    sys.exit(0)

print(f"━━━ CodeSearch AST Graph ({len(data)} hits) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
for i, item in enumerate(data, 1):
    u = item.get("unit", {})
    score = item.get("score", 0.0)
    utype = (u.get("unit_type") or "code").upper()
    name = u.get("name") or "unknown"
    filepath = u.get("file") or ""
    qname = u.get("qualified_name") or f"{filepath}::{name}"
    sig = (u.get("signature") or "").strip()
    line = u.get("line", 1)
    end_line = u.get("end_line", line)

    doc = (u.get("docstring") or "").strip().splitlines()
    doc_summary = doc[0].strip().strip("\"\"\"").strip("\x27\x27\x27").strip() if doc else ""

    called_by = u.get("called_by") or []
    calls = u.get("calls") or []
    ret_type = u.get("return_type")

    called_by_str = ", ".join(called_by[:6])
    calls_str = ", ".join(calls[:6])

    print(f"#{i} [{utype}] {qname} (score: {score:.2f})")
    print(f"   ├─ Location:  {filepath}:{line}-{end_line}")
    if sig:
        print(f"   ├─ Signature: {sig}")
    if ret_type:
        print(f"   ├─ Returns:   {ret_type}")
    if doc_summary:
        print(f"   ├─ Doc:       {doc_summary}")
    if called_by:
        print(f"   ├─ Called By: {called_by_str}")
    if calls:
        print(f"   ├─ Calls:     {calls_str}")

    code_snippet = (u.get("code") or "").strip()
    if code_snippet:
        lines = code_snippet.splitlines()
        preview = lines[:6]
        snippet_str = "\n".join(f"   │  {ln}" for ln in preview)
        print(f"   └─ Excerpt:\n{snippet_str}")
        if len(lines) > 6:
            print(f"   │  ... ({len(lines) - 6} more lines)")
    else:
        print("   └─")
    print()
' "$@"
    ;;
esac
