#!/usr/bin/env bash
# .trainbench/install.sh — Sets up repo-local agent harness projections.
# Bridges .trainbench/skills into .agents/, .codex/, and .claude/ (all gitignored)

set -euo pipefail

_tb_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=paths.sh
source "$_tb_dir/paths.sh"
unset _tb_dir

echo "trainbench harness setup — Repo Root: $TRAINBENCH_REPO_ROOT"

# 1. Ensure local directories exist
mkdir -p "$TRAINBENCH_CACHE_DIR" "$TRAINBENCH_WORKTREE_ROOT" "$TRAINBENCH_SKILLS_DIR" "$TRAINBENCH_TOOLS_DIR"

# 2. Ensure executable permissions on any tools scripts
shopt -s nullglob
for script in "$TRAINBENCH_TOOLS_DIR"/*.sh "$TRAINBENCH_TOOLS_DIR"/*.py; do
  chmod +x "$script"
done
shopt -u nullglob

# 3. Project repo-local vendor symlinks (gitignored)
for vendor_dir in ".agents" ".codex" ".claude"; do
  mkdir -p "$TRAINBENCH_REPO_ROOT/$vendor_dir"
  ln -sfn "../.trainbench/skills" "$TRAINBENCH_REPO_ROOT/$vendor_dir/skills"
done

# 4. Bridge slash commands (Claude Code) and prompts (Codex) for any registered skills
mkdir -p "$TRAINBENCH_REPO_ROOT/.claude/commands" "$TRAINBENCH_REPO_ROOT/.codex/prompts"
shopt -s nullglob
skill_count=0
for skill_dir in "$TRAINBENCH_SKILLS_DIR"/*; do
  if [[ -d "$skill_dir" && -f "$skill_dir/SKILL.md" ]]; then
    skill_name="$(basename "$skill_dir")"
    ln -sfn "../../.trainbench/skills/$skill_name/SKILL.md" "$TRAINBENCH_REPO_ROOT/.claude/commands/$skill_name.md"
    ln -sfn "../../.trainbench/skills/$skill_name/SKILL.md" "$TRAINBENCH_REPO_ROOT/.codex/prompts/$skill_name.md"
    skill_count=$((skill_count + 1))
  fi
done
shopt -u nullglob

# 5. Verify .gitignore excludes vendor projections
GITIGNORE="$TRAINBENCH_REPO_ROOT/.gitignore"
if [[ -f "$GITIGNORE" ]]; then
  for entry in "/.agents/" "/.codex/" "/.claude/"; do
    if ! grep -qxF "$entry" "$GITIGNORE" 2>/dev/null; then
      echo "$entry" >> "$GITIGNORE"
    fi
  done
fi

# 6. Check and initialize colgrep semantic search
if command -v colgrep >/dev/null 2>&1; then
  colgrep set-model lightonai/LateOn-Code-edge >/dev/null 2>&1 || true
  colgrep init -y >/dev/null 2>&1 || true
fi

echo "Done. Linked .agents/, .codex/, and .claude/ -> .trainbench/skills ($skill_count skills)."
