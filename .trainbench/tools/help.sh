#!/usr/bin/env bash
# List agent, skill, and tool metadata or resolve native model options without launching roles.
set -euo pipefail

# ━━━━━━━━ command validation ━━━━━━━━

_tools_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=../paths.sh
source "${_tools_dir%/.trainbench/tools}/.trainbench/paths.sh"
script_dir="$TRAINBENCH_TOOLS_DIR"
repo_root="$TRAINBENCH_REPO_ROOT"
unset _tools_dir

if [[ $# -eq 0 || "$1" == "--help" || "$1" == "-h" ]]; then
  cat << USAGE
Usage:
  .trainbench/tools/help.sh tools                     # List available operational tools
  .trainbench/tools/help.sh skills                    # List registered skills
  .trainbench/tools/help.sh agents                    # List available child agent roles
  .trainbench/tools/help.sh <profile|role.md> <harness> # Resolve model options as JSON
USAGE
  exit 0
fi

# ━━━━━━━━ tools listing ━━━━━━━━

if [[ $# -eq 1 && "$1" == "tools" ]]; then
  echo "Operational tools in .trainbench/tools/:"
  for tool in "$script_dir"/*.sh; do
    if [[ -f "$tool" && -x "$tool" ]]; then
      tool_name="$(basename "$tool")"
      desc="$(grep -E '^# \.trainbench/tools/' "$tool" | head -n 1 | sed 's/^# [^—]*— *//' || true)"
      if [[ -z "$desc" ]]; then
        desc="$(sed -n '2p' "$tool" | sed 's/^# *//')"
      fi
      printf "  %-18s %s\n" "$tool_name" "$desc"
    fi
  done
  exit 0
fi

# ━━━━━━━━ agent and skill listings ━━━━━━━━

if [[ $# -eq 1 && ( "$1" == "agents" || "$1" == "skills" ) ]]; then
  exec python3 - "$repo_root" "$1" <<'PY'
import json
from pathlib import Path
import sys


def get_metadata(path: Path) -> dict[str, str]:
  """
  Read the single-line frontmatter fields used by the listings.
  Handles both top-level and metadata-nested fields.
  """
  fields: dict[str, str] = {}
  with path.open(encoding="utf-8") as source:
    if source.readline().strip() != "---":
      raise ValueError(f"{path}: must begin with YAML frontmatter")
    for line in source:
      clean = line.strip()
      if clean == "---":
        break
      if not clean or clean.startswith("#") or clean == "metadata:":
        continue
      if ":" in clean:
        key, val = clean.split(":", 1)
        key = key.strip()
        val = val.strip().strip("\"'")
        if key in {"name", "description", "tools", "model_profile"} and val:
          fields[key] = val
    else:
      raise ValueError(f"{path}: frontmatter is not closed")
  for required in ("name", "description"):
    if not fields.get(required):
      raise ValueError(f"{path}: missing {required}")
  return fields


try:
  root = Path(sys.argv[1])
  kind = sys.argv[2]
  directory = root / ".trainbench" / kind
  if not directory.exists():
    print(f"No {kind} found in {directory.relative_to(root)}")
    sys.exit(0)
  paths = list(directory.rglob("*.md" if kind == "agents" else "SKILL.md"))
  if not paths:
    print(f"No {kind} found in {directory.relative_to(root)}")
    sys.exit(0)
  entries = [(get_metadata(path), path.relative_to(root)) for path in paths]
  if kind == "agents":
    print("To invoke: launch a native child agent with the task and absolute role file path.")
    print('  Instruct it: "Before starting, read <absolute_role_path> in full and follow it. '
          'If it cannot be read, report missing context and stop."')
    print("  Orchestrators: let the child read its role; do not load it into your own context.\n")
  else:
    print("To invoke: use the harness's skill mechanism, or read and follow the skill file.")
    print("  A child agent must be instructed to read the absolute skill file path in full.")
    print("  Orchestrators: let the child read a delegated skill; do not load it yourself.\n")
  for fields, path in sorted(entries, key=lambda entry: (entry[0]["name"], str(entry[1]))):
    print(f'{fields["name"]}\n  Description: {fields["description"]}')
    print(f'  Tools: {fields.get("tools", "not specified")}\n  File: {path}\n')
except (OSError, ValueError) as error:
  print(f"help.sh: {error}", file=sys.stderr)
  sys.exit(2)
PY
fi

if [[ $# -ne 2 ]]; then
  printf 'Usage: .trainbench/tools/help.sh tools|agents|skills\n       .trainbench/tools/help.sh <profile|role.md> <harness>\n' >&2
  exit 2
fi

# ━━━━━━━━ model resolution ━━━━━━━━

exec python3 - "$repo_root" "$1" "$2" <<'PY'
import json
from pathlib import Path
import re
import sys

try:
  root = Path(sys.argv[1])
  target = sys.argv[2]
  harness = sys.argv[3].lower()

  # Check if target is a role file or a profile name
  profile = target
  if target.endswith(".md") or "/" in target:
    role_file = Path(target)
    if not role_file.is_absolute():
      role_file = root / role_file
    if not role_file.is_file():
      raise ValueError(f"role file not found: {target}")

    found_profile = None
    with role_file.open(encoding="utf-8") as f:
      if f.readline().strip() != "---":
        raise ValueError(f"{target} missing frontmatter")
      for line in f:
        if line.strip() == "---":
          break
        m = re.match(r"^model_profile:\s*([a-zA-Z0-9_-]+)", line.strip())
        if m:
          found_profile = m.group(1)
          break
    if not found_profile:
      raise ValueError(f"no model_profile declared in {target}")
    profile = found_profile

  yaml_path = root / ".trainbench/models.yaml"
  if not yaml_path.is_file():
    raise ValueError(f"models configuration not found at {yaml_path}")

  active_profile = None
  matches = []
  for line in yaml_path.read_text(encoding="utf-8").splitlines():
    line_clean = line.strip()
    if not line_clean or line_clean.startswith("#"):
      continue

    header = re.match(r"^([a-zA-Z0-9_-]+):\s*$", line)
    if header:
      active_profile = header.group(1)
      continue

    if active_profile == profile:
      entry = re.match(r"^\s+([a-zA-Z0-9_-]+):\s*(\{.*\})\s*$", line)
      if entry and entry.group(1).lower() == harness:
        matches.append(json.loads(entry.group(2)))

  if not matches:
    raise ValueError(f"profile '{profile}' has no mapping for harness '{harness}' in {yaml_path.name}")

  print(json.dumps(matches[0], separators=(",", ":")))
except (OSError, ValueError) as error:
  print(f"help.sh: {error}", file=sys.stderr)
  sys.exit(2)
PY
