#!/usr/bin/env bash
# Run static AST contract and slop checks across changed or all Python source files.
set -euo pipefail

_tools_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=../paths.sh
source "${_tools_dir%/.trainbench/tools}/.trainbench/paths.sh"
tools_dir="$TRAINBENCH_TOOLS_DIR"
repo_root="$TRAINBENCH_REPO_ROOT"
src_dir="$tools_dir/src"
unset _tools_dir

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'USAGE'
Usage:
  .trainbench/tools/check_slop.sh [--changed | --all | <path>...]

Options:
  --changed   Check only modified/untracked Python source files (default)
  --all       Scan all Python source files under trainbench/ and experiments/
  <path>...   Check explicit file(s) or directory(ies)
USAGE
  exit 0
fi

python_bin="${WORKON_HOME:-$HOME/.virtualenvs}/trainbench/bin/python"
if [[ ! -x "$python_bin" ]]; then
  python_bin="python3"
fi

exec "$python_bin" "$src_dir/check_slop.py" --root "$repo_root" "$@"
