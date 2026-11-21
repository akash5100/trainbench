#!/usr/bin/env bash
# .trainbench/tools/worktree.sh — Create and checkpoint harness-agnostic agent worktrees.

set -euo pipefail

# ━━━━━━━━ usage ━━━━━━━━

usage() {
  cat <<'USAGE'
Usage:
  .trainbench/tools/worktree.sh create [--no-codebase] [--base <ref>] <slug>
                                                      Create numbered folder + sibling git worktree
  .trainbench/tools/worktree.sh codebase <folder> [base]    Create/attach sibling git worktree for <folder>
  .trainbench/tools/worktree.sh clean-codebase <folder> [--force]
                                                      Remove the sibling git worktree for <folder>
  .trainbench/tools/worktree.sh plan <folder>               Ensure plan directories in an existing work directory
  .trainbench/tools/worktree.sh checkpoint <target> <msg>   Record a git commit checkpoint for a file or folder
  .trainbench/tools/worktree.sh ledge [options] <target> [text]
                                                      Append text to a ledger without pre-reading tail lines
  .trainbench/tools/worktree.sh status [folder]             Print baseline checkout, worktree HEAD, and ledger state
  .trainbench/tools/worktree.sh --help                     Show this help

The slug must use lowercase letters, numbers, and single hyphens.
Commands print the absolute path or commit checkpoint of their target.

Ledge options:
  --from <harness> [id] Extract user prompt from harness history (antigravity or codex)
  --index <N>           Index of user message when using --from (default: -1)
  -n, --no-checkpoint   Skip auto-checkpointing (checkpoint is default)
USAGE
}

# ━━━━━━━━ command validation ━━━━━━━━

if [[ $# -eq 0 || "$1" == "-h" || "$1" == "--help" || "$1" == "help" ]]; then
  usage
  exit 0
fi

command_name="$1"
shift

if [[ "$command_name" != "create" && "$command_name" != "codebase" && "$command_name" != "clean-codebase" && "$command_name" != "plan" && "$command_name" != "checkpoint" && "$command_name" != "ledge" && "$command_name" != "ledger" && "$command_name" != "status" && "$command_name" != "baseline" ]]; then
  printf 'Error: unknown command: %s\n' "$command_name" >&2
  usage >&2
  exit 2
fi

# Load shared canonical constants (absolute paths, no ../ or user-home hardcoding).
_tools_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=../paths.sh
source "${_tools_dir%/.trainbench/tools}/.trainbench/paths.sh"
script_dir="$TRAINBENCH_TOOLS_DIR"
repo_root="$TRAINBENCH_REPO_ROOT"
main_repo_root="$TRAINBENCH_MAIN_REPO_ROOT"
workspace_root="$TRAINBENCH_WORKTREE_ROOT"
codebase_worktrees_root="${TRAINBENCH_MAIN_REPO_ROOT%/*}/${TRAINBENCH_MAIN_REPO_ROOT##*/}-worktree"
unset _tools_dir

# Helper to validate and resolve a target file or folder within a numbered work directory
resolve_target() {
  local input="$1"
  local allow_file="${2:-0}"
  local allow_create="${3:-0}"
  local resolved folder_name parent_dir

  if [[ "$input" == /* ]]; then
    resolved="$input"
  else
    resolved="$workspace_root/$input"
  fi

  if [[ -L "$resolved" ]]; then
    printf 'Error: target must not be a symlink: %s\n' "$resolved" >&2
    exit 2
  fi

  if [[ ! -e "$resolved" ]]; then
    if [[ "$allow_create" -ne 1 ]]; then
      printf 'Error: target must already exist: %s\n' "$resolved" >&2
      exit 2
    fi
    parent_dir="$(cd -- "$(dirname -- "$resolved")" 2>/dev/null && pwd -P || true)"
    if [[ -z "$parent_dir" || ! -d "$parent_dir" ]]; then
      printf 'Error: parent directory must exist for target: %s\n' "$resolved" >&2
      exit 2
    fi
    resolved="$parent_dir/${resolved##*/}"
  else
    if [[ -d "$resolved" ]]; then
      resolved="$(cd -- "$resolved" && pwd -P)"
    else
      resolved="$(cd -- "$(dirname -- "$resolved")" && pwd -P)/${resolved##*/}"
    fi
  fi

  if [[ -f "$resolved" && "$allow_file" -ne 1 ]]; then
    printf 'Error: target must be a directory: %s\n' "$resolved" >&2
    exit 2
  fi

  # Ensure the target resides inside $workspace_root
  if [[ "$resolved" != "$workspace_root"/* ]]; then
    printf 'Error: target must be inside %s.\n' "$workspace_root" >&2
    exit 2
  fi

  # Extract the top-level work directory name directly under $workspace_root
  local rel="${resolved#"$workspace_root/"}"
  folder_name="${rel%%/*}"

  if [[ ! "$folder_name" =~ ^[0-9]{4,}-[a-z0-9]+(-[a-z0-9]+)*$ ]]; then
    printf 'Error: target must reside in a numbered artifact folder: %s\n' "$folder_name" >&2
    exit 2
  fi

  printf '%s\t%s\n' "$rel" "$resolved"
}

# ━━━━━━━━ checkpoint recording ━━━━━━━━

do_checkpoint() {
  local target_rel="$1"
  local target_resolved="$2"
  local message="$3"

  if [[ ! -d "$workspace_root/.git" ]]; then
    printf 'Error: git repository not initialized in %s\n' "$workspace_root" >&2
    exit 1
  fi

  # Stage target (file or directory)
  if [[ -d "$target_resolved" ]]; then
    git -C "$workspace_root" add -A -- "$target_resolved"
  else
    git -C "$workspace_root" add -- "$target_resolved"
  fi

  # If nothing is staged for this target, report clean and succeed
  if git -C "$workspace_root" diff --cached --quiet -- "$target_resolved"; then
    printf 'Clean: no changes to checkpoint in %s\n' "$target_rel"
    return 0
  fi

  git -C "$workspace_root" commit -m "$message" --quiet -- "$target_resolved"
  local commit_hash
  commit_hash="$(git -C "$workspace_root" rev-parse --short HEAD)"
  printf 'Checkpoint %s: %s (%s)\n' "$commit_hash" "$message" "$target_rel"
}

if [[ "$command_name" == "checkpoint" ]]; then
  if [[ $# -lt 2 ]]; then
    printf 'Error: checkpoint requires a target path and a commit message.\n' >&2
    usage >&2
    exit 2
  fi

  target_input="$1"
  shift
  message="$*"

  parsed="$(resolve_target "$target_input" 1 0)"
  rel="${parsed%%	*}"
  resolved="${parsed#*	}"

  do_checkpoint "$rel" "$resolved" "$message"
  exit 0
fi

# ━━━━━━━━ ledger appending ━━━━━━━━

if [[ "$command_name" == "ledge" || "$command_name" == "ledger" ]]; then
  no_checkpoint=0
  from_harness=""
  from_conv_id=""
  from_index=""
  target_input=""
  text=""

  while [[ $# -gt 0 ]]; do
    case "$1" in
      -n|--no-checkpoint)
        no_checkpoint=1
        shift
        ;;
      --from)
        if [[ $# -lt 2 ]]; then
          printf 'Error: --from requires a harness name (e.g. antigravity)\n' >&2
          usage >&2
          exit 2
        fi
        from_harness="$2"
        shift 2
        if [[ $# -gt 0 && "$1" =~ ^[0-9a-fA-F-]{8,}$ ]]; then
          from_conv_id="$1"
          shift
        fi
        ;;
      --index)
        if [[ $# -lt 2 ]]; then
          printf 'Error: --index requires an integer index (e.g. -1)\n' >&2
          usage >&2
          exit 2
        fi
        from_index="$2"
        shift 2
        ;;
      --)
        shift
        while [[ $# -gt 0 ]]; do
          if [[ -z "$target_input" ]]; then
            target_input="$1"
          elif [[ -z "$text" ]]; then
            text="$1"
          else
            text="$text $1"
          fi
          shift
        done
        break
        ;;
      -*)
        printf 'Error: unknown option: %s\n' "$1" >&2
        usage >&2
        exit 2
        ;;
      *)
        if [[ -z "$target_input" ]]; then
          target_input="$1"
        elif [[ -z "$text" ]]; then
          text="$1"
        else
          text="$text $1"
        fi
        shift
        ;;
    esac
  done

  if [[ -z "$target_input" ]]; then
    printf 'Error: ledge requires a target directory or file.\n' >&2
    usage >&2
    exit 2
  fi

  custom_timestamp=""

  # Preserve the shorthand for selecting a harness when no explicit text is supplied.
  if [[ ( "$text" == "antigravity" || "$text" == "codex" ) && -z "$from_harness" ]]; then
    from_harness="$text"
    text=""
  fi

  # Extract prompt from harness transcript if --from was given
  if [[ -n "$from_harness" ]]; then
    harness_script="$script_dir/src/worktree__harness.py"
    if [[ ! -f "$harness_script" ]]; then
      printf 'Error: harness extractor not found at %s\n' "$harness_script" >&2
      exit 1
    fi
    harness_output="$(python3 "$harness_script" "$from_harness" "${from_conv_id:-}" "${from_index:- -1}")"
    text="$(python3 -c 'import json, sys; sys.stdout.write(json.loads(sys.argv[1]).get("text", ""))' "$harness_output"; printf '\001')"
    text="${text%$'\001'}"
    custom_timestamp="$(python3 -c 'import json, sys; print(json.loads(sys.argv[1]).get("timestamp", ""))' "$harness_output")"
    extracted_step="$(python3 -c 'import json, sys; print(json.loads(sys.argv[1]).get("step_index", ""))' "$harness_output")"
    printf 'Extracted user message[%s] from %s (step %s)\n' "${from_index:- -1}" "$from_harness" "$extracted_step"
  elif [[ -z "$text" ]]; then
    if [[ ! -t 0 ]]; then
      text="$(cat)"
    fi
  fi

  if [[ -z "$text" ]]; then
    printf 'Error: ledge requires text, piped stdin, or --from <harness>.\n' >&2
    usage >&2
    exit 2
  fi

  parsed="$(resolve_target "$target_input" 1 1)"
  rel="${parsed%%	*}"
  resolved="${parsed#*	}"

  # Default a folder target to request.md inside that folder
  if [[ -d "$resolved" ]]; then
    resolved="$resolved/request.md"
    rel="$rel/request.md"
  fi

  # Auto-initialize request ledger if target file does not yet exist
  if [[ ! -f "$resolved" ]]; then
    if [[ "$resolved" == */request.md ]]; then
      printf '# Request Ledger\n' > "$resolved"
    else
      touch "$resolved"
    fi
  fi

  max_entry=0
  while read -r num; do
    if [[ "$num" =~ ^[0-9]+$ ]] && (( num > max_entry )); then
      max_entry="$num"
    fi
  done < <(grep -oE '^## Entry [0-9]+' "$resolved" 2>/dev/null | awk '{print $3}' || true)
  next_entry=$((max_entry + 1))

  # Ensure the existing file content ends with a clean separation
  if [[ -s "$resolved" ]]; then
    last_char="$(tail -c 1 "$resolved" 2>/dev/null || true)"
    if [[ -n "$last_char" ]]; then
      printf '\n' >> "$resolved"
    fi
  fi

  if [[ -n "$custom_timestamp" ]]; then
    timestamp="$custom_timestamp"
  else
    timestamp="$(date +%Y-%m-%dT%H:%M:%S%z | sed 's/\([0-9][0-9]\)$/:\1/')"
  fi

  fence='```'
  while [[ "$text" == *"$fence"* ]]; do
    fence+='`'
  done

  {
    printf '\n## Entry %d [%s]\n\n' "$next_entry" "$timestamp"
    printf '%s text\n' "$fence"
    printf '%s' "$text"
    if [[ "$text" != *$'\n' ]]; then
      printf '\n'
    fi
    printf '%s\n' "$fence"
  } >> "$resolved"

  printf 'Ledged Entry %d -> %s\n' "$next_entry" "$rel"

  if [[ "$no_checkpoint" -ne 1 ]]; then
    do_checkpoint "$rel" "$resolved" "Ledger Entry $next_entry"
  fi

  exit 0
fi

# ━━━━━━━━ status inspection ━━━━━━━━

if [[ "$command_name" == "status" || "$command_name" == "baseline" ]]; then
  target_folder=""
  if [[ $# -gt 0 ]]; then
    target_folder="$1"
  else
    for entry in "$workspace_root"/*; do
      [[ -d "$entry" ]] || continue
      name="${entry##*/}"
      if [[ "$name" =~ ^[0-9]{4,}- ]]; then
        target_folder="$name"
      fi
    done
  fi

  if [[ -z "$target_folder" ]]; then
    printf 'Error: no worktree folder specified and none found under %s\n' "$workspace_root" >&2
    exit 2
  fi

  parsed="$(resolve_target "$target_folder" 0 0)"
  rel="${parsed%%	*}"
  resolved="${parsed#*	}"

  main_branch="$(git -C "$repo_root" branch --show-current 2>/dev/null || echo "detached")"
  main_head="$(git -C "$repo_root" rev-parse --short HEAD 2>/dev/null || echo "unknown")"
  dirty_count="$(git -C "$repo_root" status --porcelain 2>/dev/null | wc -l | tr -d ' ')"
  if [[ "$dirty_count" -eq 0 ]]; then
    main_status="clean"
  else
    main_status="dirty ($dirty_count modified files)"
  fi

  wt_head="$(git -C "$workspace_root" rev-parse --short HEAD 2>/dev/null || echo "uninitialized")"

  latest_entry="none"
  if [[ -f "$resolved/request.md" ]]; then
    latest_header="$(grep -E '^## Entry [0-9]+' "$resolved/request.md" 2>/dev/null | tail -n 1 || true)"
    if [[ -n "$latest_header" ]]; then
      latest_entry="${latest_header#\#\# }"
    fi
  fi

  phases=()
  if [[ -d "$resolved/plan/phases" ]]; then
    for p in "$resolved/plan/phases"/*.md; do
      [[ -e "$p" ]] || continue
      p_name="${p##*/}"
      p_name="${p_name%.md}"
      phases+=("$p_name")
    done
  fi

  printf 'Checkout HEAD:    %s (%s, %s)\n' "$main_head" "$main_branch" "$main_status"
  printf 'Worktree HEAD:    %s (%s)\n' "$wt_head" "$rel"
  printf 'Request Ledger:   %s\n' "$latest_entry"
  cb_path="$codebase_worktrees_root/$rel"
  if [[ -e "$cb_path/.git" ]]; then
    cb_branch="$(git -C "$cb_path" branch --show-current 2>/dev/null || echo "detached")"
    cb_dirty="$(git -C "$cb_path" status --porcelain 2>/dev/null | wc -l | tr -d ' ')"
    printf 'Codebase WT:      %s (%s, %s dirty)\n' "$cb_path" "$cb_branch" "$cb_dirty"
  fi
  if [[ ${#phases[@]} -gt 0 ]]; then
    printf 'Plan Phases:      %s\n' "${phases[*]}"
  fi

  exit 0
fi

# ━━━━━━━━ codebase git worktree management ━━━━━━━━

link_worktree_deps() {
  local wt_path="$1"

  # Link root .env if it exists in main repo
  if [[ -f "$main_repo_root/.env" && ! -e "$wt_path/.env" ]]; then
    ln -s "$main_repo_root/.env" "$wt_path/.env"
  fi

  # Symlink shared system cache (colgrep neural index, embeddings, etc.)
  mkdir -p "$main_repo_root/.trainbench/.cache"
  if [[ ! -e "$wt_path/.trainbench/.cache" ]]; then
    mkdir -p "$(dirname "$wt_path/.trainbench/.cache")"
    ln -s "$main_repo_root/.trainbench/.cache" "$wt_path/.trainbench/.cache"
  fi

  # Pre-seed colgrep semantic code search index for the worktree from main repo HEAD
  if command -v colgrep >/dev/null 2>&1 && [[ -d "$main_repo_root/.trainbench/.cache/colgrep/indices" ]]; then
    XDG_DATA_HOME="$main_repo_root/.trainbench/.cache" colgrep init -y "$wt_path" >/dev/null 2>&1 || true
  fi
}

create_codebase_worktree() {
  local rel="$1"
  local base_ref="$2"
  local quiet="${3:-0}"
  local wt_path="$codebase_worktrees_root/$rel"
  local branch_name="feat/$rel"

  mkdir -p -- "$codebase_worktrees_root"

  if [[ -e "$wt_path/.git" ]]; then
    link_worktree_deps "$wt_path"
    return 0
  fi

  if [[ -e "$wt_path" ]]; then
    printf 'Error: %s already exists and is not a git worktree.\n' "$wt_path" >&2
    return 1
  fi

  if git -C "$main_repo_root" show-ref --verify --quiet "refs/heads/$branch_name"; then
    if [[ "$quiet" -eq 1 ]]; then
      GIT_LFS_SKIP_SMUDGE=1 git -C "$main_repo_root" worktree add --quiet "$wt_path" "$branch_name" >/dev/null
    else
      GIT_LFS_SKIP_SMUDGE=1 git -C "$main_repo_root" worktree add "$wt_path" "$branch_name"
    fi
  else
    if [[ "$quiet" -eq 1 ]]; then
      GIT_LFS_SKIP_SMUDGE=1 git -C "$main_repo_root" worktree add --quiet -b "$branch_name" "$wt_path" "$base_ref" >/dev/null
    else
      GIT_LFS_SKIP_SMUDGE=1 git -C "$main_repo_root" worktree add -b "$branch_name" "$wt_path" "$base_ref"
    fi
  fi

  link_worktree_deps "$wt_path"
}

if [[ "$command_name" == "codebase" ]]; then
  if [[ $# -lt 1 || $# -gt 2 ]]; then
    printf 'Error: codebase requires <folder> and optional [base_ref].\n' >&2
    usage >&2
    exit 2
  fi

  parsed="$(resolve_target "$1" 0 0)"
  rel="${parsed%%	*}"
  if [[ "$rel" == */* ]]; then
    printf 'Error: codebase folder must be a direct child of %s.\n' "$workspace_root" >&2
    exit 2
  fi

  base_ref="${2:-HEAD}"
  create_codebase_worktree "$rel" "$base_ref" 0
  printf '%s\n' "$codebase_worktrees_root/$rel"
  exit 0
fi

if [[ "$command_name" == "clean-codebase" ]]; then
  force_flag=0
  target_arg=""
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --force)
        force_flag=1
        shift
        ;;
      *)
        if [[ -z "$target_arg" ]]; then
          target_arg="$1"
          shift
        else
          printf 'Error: unexpected argument: %s\n' "$1" >&2
          exit 2
        fi
        ;;
    esac
  done

  if [[ -z "$target_arg" ]]; then
    printf 'Error: clean-codebase requires <folder>.\n' >&2
    usage >&2
    exit 2
  fi

  parsed="$(resolve_target "$target_arg" 0 0)"
  rel="${parsed%%	*}"
  if [[ "$rel" == */* ]]; then
    printf 'Error: folder must be a direct child of %s.\n' "$workspace_root" >&2
    exit 2
  fi

  wt_path="$codebase_worktrees_root/$rel"
  if [[ ! -e "$wt_path/.git" ]]; then
    printf 'Clean: no codebase worktree at %s\n' "$wt_path"
    exit 0
  fi

  if command -v colgrep >/dev/null 2>&1; then
    XDG_DATA_HOME="$main_repo_root/.trainbench/.cache" colgrep clear "$wt_path" >/dev/null 2>&1 || true
    rm -rf "$main_repo_root/.trainbench/.cache/colgrep/indices/${rel}-"*
  fi

  if [[ "$force_flag" -eq 1 ]]; then
    git -C "$main_repo_root" worktree remove --force "$wt_path"
  else
    git -C "$main_repo_root" worktree remove "$wt_path"
  fi
  printf 'Removed codebase worktree: %s\n' "$wt_path"
  exit 0
fi

# ━━━━━━━━ plan directory initialization ━━━━━━━━

if [[ "$command_name" == "plan" ]]; then
  if [[ $# -ne 1 ]]; then
    printf 'Error: plan requires exactly one existing numbered folder.\n' >&2
    usage >&2
    exit 2
  fi

  parsed="$(resolve_target "$1" 0)"
  rel="${parsed%%	*}"
  resolved="${parsed#*	}"

  if [[ "$rel" == */* ]]; then
    printf 'Error: plan folder must be a direct child of %s.\n' "$workspace_root" >&2
    exit 2
  fi

  if [[ -L "$resolved/plan" ]]; then
    printf 'Error: plan directory must not be a symlink.\n' >&2
    exit 2
  fi

  mkdir -p -- "$resolved/plan"
  printf '%s\n' "$resolved/plan"
  exit 0
fi

# ━━━━━━━━ workspace allocation ━━━━━━━━

no_codebase=0
base_ref="HEAD"
slug=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-codebase)
      no_codebase=1
      shift
      ;;
    --base)
      if [[ $# -lt 2 ]]; then
        printf 'Error: --base requires a git ref.\n' >&2
        usage >&2
        exit 2
      fi
      base_ref="$2"
      shift 2
      ;;
    -*)
      printf 'Error: unknown option for create: %s\n' "$1" >&2
      usage >&2
      exit 2
      ;;
    *)
      if [[ -z "$slug" ]]; then
        slug="$1"
        shift
      else
        printf 'Error: create accepts only one slug.\n' >&2
        usage >&2
        exit 2
      fi
      ;;
  esac
done

if [[ -z "$slug" ]]; then
  printf 'Error: create requires a slug.\n' >&2
  usage >&2
  exit 2
fi

if [[ ! "$slug" =~ ^[a-z0-9]+(-[a-z0-9]+)*$ ]]; then
  printf 'Error: slug must contain lowercase letters or numbers separated by single hyphens.\n' >&2
  exit 2
fi

mkdir -p -- "$workspace_root"

next_id=1
for entry in "$workspace_root"/*; do
  [[ -d "$entry" ]] || continue
  name="${entry##*/}"
  if [[ "$name" =~ ^([0-9]{4,})- ]]; then
    existing_id=$((10#${BASH_REMATCH[1]}))
    if (( existing_id >= next_id )); then
      next_id=$((existing_id + 1))
    fi
  fi
done

# mkdir is the allocation point; advance on a concurrent creator's collision.
while :; do
  folder_name="$(printf '%04d' "$next_id")-$slug"
  target="$workspace_root/$folder_name"
  if mkdir -- "$target" 2>/dev/null; then
    if [[ "$no_codebase" -ne 1 ]]; then
      create_codebase_worktree "$folder_name" "$base_ref" 1
    fi
    printf '%s\n' "$target"
    exit 0
  fi
  if [[ -e "$target" ]]; then
    next_id=$((next_id + 1))
    continue
  fi
  printf 'Error: could not create work directory: %s\n' "$target" >&2
  exit 1
done
