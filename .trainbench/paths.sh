#!/usr/bin/env bash
# Single source of truth for canonical absolute .trainbench and repo paths.
# Source from .trainbench/tools/*.sh (or run directly to print resolved constants).

_tb_paths_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
export TRAINBENCH_DIR="$_tb_paths_dir"
export TRAINBENCH_REPO_ROOT="${_tb_paths_dir%/.trainbench}"
export TRAINBENCH_TOOLS_DIR="$TRAINBENCH_DIR/tools"
export TRAINBENCH_SKILLS_DIR="$TRAINBENCH_DIR/skills"
export TRAINBENCH_CACHE_DIR="$TRAINBENCH_DIR/.cache"

_tb_git_common_dir="$(git -C "$TRAINBENCH_REPO_ROOT" rev-parse --git-common-dir 2>/dev/null || true)"
if [[ -n "$_tb_git_common_dir" ]]; then
  if [[ "$_tb_git_common_dir" != /* ]]; then
    _tb_git_common_dir="$(cd -- "$TRAINBENCH_REPO_ROOT/$_tb_git_common_dir" && pwd -P)"
  else
    _tb_git_common_dir="$(cd -- "$_tb_git_common_dir" && pwd -P)"
  fi
  export TRAINBENCH_MAIN_REPO_ROOT="${_tb_git_common_dir%/*}"
else
  export TRAINBENCH_MAIN_REPO_ROOT="$TRAINBENCH_REPO_ROOT"
fi

export TRAINBENCH_WORKTREE_ROOT="$TRAINBENCH_MAIN_REPO_ROOT/.trainbench/worktree"

unset _tb_paths_dir _tb_git_common_dir

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  cat <<PATHS
TRAINBENCH_REPO_ROOT=$TRAINBENCH_REPO_ROOT
TRAINBENCH_MAIN_REPO_ROOT=$TRAINBENCH_MAIN_REPO_ROOT
TRAINBENCH_DIR=$TRAINBENCH_DIR
TRAINBENCH_TOOLS_DIR=$TRAINBENCH_TOOLS_DIR
TRAINBENCH_SKILLS_DIR=$TRAINBENCH_SKILLS_DIR
TRAINBENCH_WORKTREE_ROOT=$TRAINBENCH_WORKTREE_ROOT
PATHS
fi
