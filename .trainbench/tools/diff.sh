#!/usr/bin/env bash
# .trainbench/tools/diff.sh — Semantically-expanded diff tool.
#
# Unlike `git diff` which shows ±3-line hunks, this tool expands every changed
# hunk to the full enclosing symbol (function, method, inner function) so
# reviewers see complete context — including closure captures, full argument
# lists, decorator chains, and surrounding logic.
#
# Symbol boundary detection:
#   .py files  → .trainbench/tools/src/sym_context.py (Python ast module, exact)
#   .ts/.tsx/.js/.jsx → .trainbench/tools/outline.sh (grep-based, body only)
#   other      → ±30-line fallback with explicit warning
#
# Containment labels (CLASS / OUTER / INSIDE) are only emitted for .py files
# where ast gives verified data. No labels are shown for other languages.
#
# Usage:
#   .trainbench/tools/diff.sh                          # unstaged changes
#   .trainbench/tools/diff.sh --staged                 # staged changes
#   .trainbench/tools/diff.sh --cached                 # alias for --staged
#   .trainbench/tools/diff.sh --base <upstream>        # branch diff
#   .trainbench/tools/diff.sh -- <file> [<file>...]    # limit to specific files

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# REPO_ROOT resolves from the CALLER's cwd — works correctly from any worktree.
REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null)"
OUTLINE="$SCRIPT_DIR/outline.sh"
SYM_CONTEXT="$SCRIPT_DIR/src/sym_context.py"

# ━━━━━━━━ arg parsing ━━━━━━━━

DIFF_MODE="unstaged"
UPSTREAM=""
FILE_FILTER=()

while [ $# -gt 0 ]; do
  case "$1" in
    --staged|--cached) DIFF_MODE="staged"; shift ;;
    --base)
      DIFF_MODE="base"
      UPSTREAM="${2:?--base requires an upstream branch}"
      shift 2
      ;;
    --) shift; FILE_FILTER=("$@"); break ;;
    -h|--help) sed -n '2,25p' "$0" | sed 's/^# \?//'; exit 0 ;;
    -*) echo "Unknown option: $1" >&2; exit 1 ;;
    *) FILE_FILTER=("$@"); break ;;
  esac
done

# ━━━━━━━━ diff command & show ref ━━━━━━━━

case "$DIFF_MODE" in
  staged)
    DIFF_CMD="git diff --cached"
    SHOW_REF=":0"
    ;;
  base)
    MERGE_BASE=$(git merge-base HEAD "$UPSTREAM")
    DIFF_CMD="git diff $MERGE_BASE"
    SHOW_REF="$MERGE_BASE"
    ;;
  *)
    DIFF_CMD="git diff"
    SHOW_REF="HEAD"
    ;;
esac

# ━━━━━━━━ JS/TS boundary via outline.sh ━━━━━━━━

# find_js_symbol_range FILE HUNK_LINE
# Uses outline.sh to find the enclosing symbol start and end for JS/TS files.
# Outputs: "start end label" — no containment hierarchy (outline is grep-based).
find_js_symbol_range() {
  local file="$1"
  local hunk_line="$2"
  local total
  total=$(wc -l < "$file" | tr -d ' ')

  local best_start=0 best_label=""
  local lineno rest
  while IFS=: read -r lineno rest; do
    case "$lineno" in ''|*[!0-9]*) continue ;; esac
    if [ "$lineno" -le "$hunk_line" ] && [ "$lineno" -gt "$best_start" ]; then
      best_start="$lineno"
      best_label=$(echo "$rest" | sed 's/^ *//')
    fi
  done < <("$OUTLINE" "$file" 2>/dev/null)

  if [ "$best_start" -eq 0 ]; then
    echo "0 0"
    return
  fi

  # Find end: next outline entry after best_start, or EOF.
  local next=$(( total + 1 ))
  while IFS=: read -r lineno rest; do
    case "$lineno" in ''|*[!0-9]*) continue ;; esac
    if [ "$lineno" -gt "$best_start" ] && [ "$lineno" -lt "$next" ]; then
      next="$lineno"
    fi
  done < <("$OUTLINE" "$file" 2>/dev/null)

  echo "$best_start $(( next - 1 )) $best_label"
}

# ━━━━━━━━ per-file hunk expansion ━━━━━━━━

# expand_file_hunks NEW_FILE OLD_FILE HUNKS_FILE
expand_file_hunks() {
  local new_file="$1"
  local old_file="$2"
  local hunks_file="$3"

  local abs_new="$REPO_ROOT/$new_file"
  local ext="${new_file##*.}"

  local new_total=0
  [ -f "$abs_new" ] && new_total=$(wc -l < "$abs_new" | tr -d ' ')

  # Fetch old file into a temp dir with its extension intact for outline.sh.
  local old_tmp_dir old_tmp
  old_tmp_dir=$(mktemp -d /tmp/diff_sh_XXXXXX)
  old_tmp="${old_tmp_dir}/old.${ext}"
  local old_total=0
  if git show "${SHOW_REF}:${old_file}" > "$old_tmp" 2>/dev/null; then
    old_total=$(wc -l < "$old_tmp" | tr -d ' ')
  fi

  # Build added-lines and removed-lines indexes for this file.
  local added_idx removed_idx raw_diff_tmp
  added_idx=$(mktemp /tmp/diff_sh_added_XXXXXX)
  removed_idx=$(mktemp /tmp/diff_sh_removed_XXXXXX)
  raw_diff_tmp=$(mktemp /tmp/diff_sh_rawdiff_XXXXXX)

  eval "$DIFF_CMD" -- "$old_file" "$new_file" > "$raw_diff_tmp" 2>/dev/null || true

  # Map added/removed line numbers from the raw diff.
  awk '
    /^@@/ {
      n = split($0, parts, /[ \t]+/)
      for (i = 1; i <= n; i++) {
        if (parts[i] ~ /^\-[0-9]/) { split(substr(parts[i],2), a, ","); old_cur = a[1]+0 }
        if (parts[i] ~ /^\+[0-9]/) { split(substr(parts[i],2), a, ","); new_cur = a[1]+0 }
      }
      next
    }
    /^\+[^+]/ { print new_cur >> added_out; new_cur++; next }
    /^\-[^-]/ { print old_cur >> removed_out; old_cur++; next }
    /^ /       { new_cur++; old_cur++; next }
  ' added_out="$added_idx" removed_out="$removed_idx" "$raw_diff_tmp"

  rm -f "$raw_diff_tmp"

  # Track emitted symbol ranges to deduplicate (bash 3.2 compat — plain string).
  local emitted_symbols=""

  while IFS=, read -r hunk_start hunk_count; do
    case "$hunk_start" in ''|*[!0-9]*) continue ;; esac

    local hunk_end=$(( hunk_start + hunk_count - 1 ))
    [ "$hunk_count" -eq 0 ] && hunk_end=$hunk_start

    # ── resolve symbol boundaries ──────────────────────────────────────────

    local sym_start=0 sym_end=0 sym_label=""
    local ancestry=""   # only populated for .py via ast

    case "$ext" in
      py)
        if [ -f "$abs_new" ] && [ "$new_total" -gt 0 ]; then
          local ctx_out
          ctx_out=$(python3 "$SYM_CONTEXT" "$abs_new" "$hunk_start" 2>/dev/null || true)
          if [ -n "$ctx_out" ]; then
            # First line is always SYMBOL start end label.
            sym_start=$(echo "$ctx_out" | awk 'NR==1{print $2}')
            sym_end=$(echo   "$ctx_out" | awk 'NR==1{print $3}')
            sym_label=$(echo "$ctx_out" | awk 'NR==1{$1=$2=$3=""; sub(/^ +/,""); print}')
            # Remaining lines are CLASS / OUTER — preserved verbatim for header.
            ancestry=$(echo "$ctx_out" | tail -n +2)
          fi
        fi
        ;;
      ts|tsx|js|jsx)
        if [ -f "$abs_new" ] && [ "$new_total" -gt 0 ]; then
          local js_range
          js_range=$(find_js_symbol_range "$abs_new" "$hunk_start")
          sym_start=$(echo "$js_range" | awk '{print $1}')
          sym_end=$(  echo "$js_range" | awk '{print $2}')
          sym_label=$( echo "$js_range" | awk '{$1=$2=""; sub(/^ +/,""); print}')
        fi
        ;;
    esac

    # Fallback for unsupported languages or module-level code (sym_start == 0).
    local is_fallback=0
    if [ "$sym_start" -eq 0 ]; then
      is_fallback=1
      local ctx=30
      sym_start=$(( hunk_start - ctx > 1 ? hunk_start - ctx : 1 ))
      local fallback_end=$(( hunk_end + ctx ))
      if [ -f "$abs_new" ]; then
        sym_end=$(( fallback_end < new_total ? fallback_end : new_total ))
      else
        sym_end=$fallback_end
      fi
      sym_label="(module-level / fallback ±${ctx} lines)"
    fi

    # Deduplication.
    local sym_key="|${sym_start}-${sym_end}|"
    case "$emitted_symbols" in *"$sym_key"*) continue ;; esac
    emitted_symbols="${emitted_symbols}${sym_key}"

    # ── emit symbol header ──────────────────────────────────────────────────

    printf '\n'
    printf '── SYMBOL: %s  [lines %s\xe2\x80\x93%s]  (%s)\n' \
      "$sym_label" "$sym_start" "$sym_end" "$new_file"

    # Ancestry lines from ast (CLASS / OUTER) — only for .py, only when verified.
    if [ -n "$ancestry" ]; then
      echo "$ancestry" | while IFS='	' read -r kind astart aend alabel; do
        case "$kind" in
          CLASS) printf '    class context : %s [lines %s\xe2\x80\x93%s]\n' "$alabel" "$astart" "$aend" ;;
          OUTER) printf '    outer scope   : %s [lines %s\xe2\x80\x93%s]\n' "$alabel" "$astart" "$aend" ;;
        esac
      done
    fi

    if [ "$is_fallback" -eq 1 ] && [ "$ext" != "py" ] && [ "$ext" != "ts" ] && \
       [ "$ext" != "tsx" ] && [ "$ext" != "js" ] && [ "$ext" != "jsx" ]; then
      printf '    ⚠ language "%s" — no symbol parser available, showing ±30 lines\n' "$ext"
    fi

    printf '%.0s─' {1..80}; printf '\n'

    # ── emit new-file symbol body with + markers on added lines ────────────

    if [ -f "$abs_new" ] && [ "$new_total" -gt 0 ]; then
      local line_num=$sym_start
      while IFS= read -r line; do
        if grep -qxF "$line_num" "$added_idx" 2>/dev/null; then
          printf '+%s\n' "$line"
        else
          printf ' %s\n' "$line"
        fi
        line_num=$(( line_num + 1 ))
      done < <(sed -n "${sym_start},${sym_end}p" "$abs_new")
    fi

    # ── emit removed lines from old-file symbol ─────────────────────────────

    if [ "$old_total" -gt 0 ] && [ -f "$old_tmp" ]; then
      local old_sym_start=0 old_sym_end=0

      case "$ext" in
        py)
          local old_ctx
          old_ctx=$(python3 "$SYM_CONTEXT" "$old_tmp" "$hunk_start" 2>/dev/null || true)
          if [ -n "$old_ctx" ]; then
            old_sym_start=$(echo "$old_ctx" | awk 'NR==1{print $2}')
            old_sym_end=$(  echo "$old_ctx" | awk 'NR==1{print $3}')
          fi
          ;;
        ts|tsx|js|jsx)
          local old_js
          old_js=$(find_js_symbol_range "$old_tmp" "$hunk_start")
          old_sym_start=$(echo "$old_js" | awk '{print $1}')
          old_sym_end=$(  echo "$old_js" | awk '{print $2}')
          ;;
      esac

      if [ "$old_sym_start" -gt 0 ]; then
        local removed_in_sym
        removed_in_sym=$(awk -v s="$old_sym_start" -v e="$old_sym_end" \
          '$1+0 >= s && $1+0 <= e { print $1 }' "$removed_idx" || true)
        if [ -n "$removed_in_sym" ]; then
          printf '\n── removed lines (old file, same symbol [%s\xe2\x80\x93%s]):\n' \
            "$old_sym_start" "$old_sym_end"
          echo "$removed_in_sym" | while IFS= read -r old_lineno; do
            [ -z "$old_lineno" ] && continue
            local old_line
            old_line=$(sed -n "${old_lineno}p" "$old_tmp")
            printf -- '-%s\n' "$old_line"
          done
        fi
      fi
    fi

  done < "$hunks_file"

  rm -f "$added_idx" "$removed_idx"
  rm -rf "$old_tmp_dir"
}

# ━━━━━━━━ main: parse raw diff and dispatch per file ━━━━━━━━

cd "$REPO_ROOT"

RAW_DIFF_MASTER=$(mktemp /tmp/diff_sh_master_XXXXXX)
if [ "${#FILE_FILTER[@]}" -gt 0 ]; then
  eval "$DIFF_CMD" -- "${FILE_FILTER[@]}" > "$RAW_DIFF_MASTER" 2>/dev/null || true
else
  eval "$DIFF_CMD" > "$RAW_DIFF_MASTER" 2>/dev/null || true
fi

if [ ! -s "$RAW_DIFF_MASTER" ]; then
  echo "No changes found (diff is empty)."
  rm -f "$RAW_DIFF_MASTER"
  exit 0
fi

printf '%.0s═' {1..80}; printf '\n'
printf ' .trainbench/tools/diff.sh \xe2\x80\x94 Semantically Expanded Diff\n'
case "$DIFF_MODE" in
  staged) printf ' Mode: staged (git diff --cached)\n' ;;
  base)   printf ' Mode: branch diff from %s (upstream: %s)\n' "$MERGE_BASE" "$UPSTREAM" ;;
  *)      printf ' Mode: unstaged (git diff)\n' ;;
esac
printf '%.0s═' {1..80}; printf '\n'

current_new_file=""
current_old_file=""
hunks_tmp=$(mktemp /tmp/diff_sh_hunks_XXXXXX)

flush_file() {
  [ -z "$current_new_file" ] && return
  [ ! -s "$hunks_tmp" ] && return
  expand_file_hunks "$current_new_file" "$current_old_file" "$hunks_tmp"
  > "$hunks_tmp"
}

while IFS= read -r line; do
  case "$line" in
    "diff --git a/"*)
      flush_file
      current_old_file=$(echo "$line" | sed 's|^diff --git a/||; s| b/.*||')
      current_new_file=$(echo "$line" | sed 's|^diff --git a/.* b/||')
      > "$hunks_tmp"
      ;;
    "+++ b/"*)
      current_new_file=$(echo "$line" | sed 's|^+++ b/||')
      ;;
    "--- a/"*)
      current_old_file=$(echo "$line" | sed 's|^--- a/||')
      ;;
    "@@ -"*)
      local_new_start=$(echo "$line" | sed 's/^@@ -[^ ]* +\([0-9]*\).*/\1/')
      local_new_count=$(echo "$line" | sed 's/^@@ -[^ ]* +[0-9]*,\([0-9]*\).*/\1/')
      case "$local_new_count" in *[!0-9]*) local_new_count=1 ;; esac
      printf '%s,%s\n' "$local_new_start" "$local_new_count" >> "$hunks_tmp"
      ;;
  esac
done < "$RAW_DIFF_MASTER"

flush_file

rm -f "$RAW_DIFF_MASTER" "$hunks_tmp"

printf '\n'
printf '%.0s═' {1..80}; printf '\n'
printf ' End of expanded diff.\n'
printf '%.0s═' {1..80}; printf '\n'
