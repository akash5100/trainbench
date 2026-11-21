#!/usr/bin/env bash
# .trainbench/tools/check_env.sh — Zero-leakage environment variable inspector.
# Confirms presence, length, and format of environment variables WITHOUT ever printing secret values.

set -euo pipefail

usage() {
  cat << USAGE
Usage:
  .trainbench/tools/check_env.sh <VAR_NAME>...     # Check if one or more environment variables are set and non-empty
  .trainbench/tools/check_env.sh --help            # Show this help message

Examples:
  .trainbench/tools/check_env.sh WANDB_API_KEY
  .trainbench/tools/check_env.sh HF_TOKEN KAGGLE_KEY
USAGE
  exit 0
}

if [ $# -eq 0 ] || [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
  usage
fi

python3 - "$@" << 'PYEOF'
import os
import sys
from pathlib import Path

all_ok = True

def parse_env_file(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    res = {}
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("export "):
                    line = line[7:].strip()
                if "=" in line:
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip("\"'")
                    res[k] = v
    except Exception:
        pass
    return res

# Collect candidate .env files:
# 1. Current dir upwards to git root
cwd = Path.cwd().resolve()
candidate_paths = []

curr = cwd
while curr != curr.parent:
    candidate = curr / ".env"
    if candidate.is_file() and candidate not in candidate_paths:
        candidate_paths.append(candidate)
    if (curr / ".git").exists():
        break
    curr = curr.parent

env_cache: dict[Path, dict[str, str]] = {}
for p in candidate_paths:
    env_cache[p] = parse_env_file(p)

for var_name in sys.argv[1:]:
    val = os.environ.get(var_name)
    if val is not None:
        if len(val.strip()) == 0:
            print(f"[!] {var_name}: SET BUT EMPTY in shell environment")
            all_ok = False
        else:
            print(f"[✓] {var_name}: LOADED in shell environment (length: {len(val)} chars)")
        continue

    # Check candidate .env files
    found_in_env = False
    for p, env_dict in env_cache.items():
        if var_name in env_dict:
            found_in_env = True
            env_val = env_dict[var_name]
            try:
                rel_path = p.relative_to(cwd)
            except ValueError:
                rel_path = p
            if len(env_val.strip()) == 0:
                print(f"[!] {var_name}: SET BUT EMPTY in {rel_path}")
                all_ok = False
            else:
                print(f"[✓] {var_name}: LOADED from {rel_path} (length: {len(env_val)} chars)")
            break

    if found_in_env:
        continue

    # Check if defined in ~/.zshrc without printing the value
    zshrc_path = os.path.expanduser("~/.zshrc")
    found_in_zshrc = False
    if os.path.isfile(zshrc_path):
        with open(zshrc_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                stripped = line.strip()
                if stripped.startswith(f"export {var_name}=") or stripped.startswith(f"{var_name}="):
                    found_in_zshrc = True
                    break

    if found_in_zshrc:
        print(f"[-] {var_name}: NOT loaded in current subshell (but present in ~/.zshrc — source it)")
    else:
        print(f"[x] {var_name}: MISSING / NOT SET")
    all_ok = False

sys.exit(0 if all_ok else 1)
PYEOF
