"""
Static AST contract and slop linter for Indus Python services.

Enforces empirical invariants mined from repository history (canonical contract
purity, alpha-clean no-shim rules, absolute imports, section boundaries, and
infrastructure isolation) without relying on prompt memory.
"""

from __future__ import annotations

import argparse
import ast
import re
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# ━━━━━━━━ configuration & data models ━━━━━━━━

SECTION_HEADER_RE = re.compile(r"^# ━━━━━━━━ .+ ━━━━━━━━$")
SUPPRESSION_MARKER = "# slop-ok:"
SOURCE_ROOTS = ("trainbench", "experiments")
SKIP_DIRS = {".venv", "__pycache__", "tests", "evals", "build", "dist", ".trainbench", "node_modules"}
ENTRY_DECORATORS = frozenset(
  {
    "property",
    "abstractmethod",
    "override",
    "cached_property",
    "compact",
    "jit",
    "no_grad",
    "validator",
    "field_validator",
    "model_validator",
  }
)
ENTRY_DECORATOR_OWNERS = frozenset({"cli", "jax", "torch"})


@dataclass(frozen=True)
class Finding:
  """
  Single static contract or slop violation found in a Python source file.

  Attributes:
    path: Repository-relative file path.
    line: 1-based source line number.
    rule: Short machine-readable rule identifier.
    message: Actionable explanation and remediation hint.
  """

  path: Path
  line: int
  rule: str
  message: str


# ━━━━━━━━ file discovery ━━━━━━━━


def is_source_python_file(rel_path: Path) -> bool:
  """
  Return whether a repository-relative path is an active Python source file.

  Excludes tests, migrations, generated protobuf stubs, and caches so rules
  target production runtime modules cleanly.

  Args:
    rel_path: Path relative to the repository root.

  Returns:
    True when the path is a production `.py` module under `services/` or `libs/`.
  """
  if rel_path.suffix != ".py":
    return False
  parts = rel_path.parts
  if not parts or parts[0] not in SOURCE_ROOTS:
    return False
  if any(part in SKIP_DIRS for part in parts):
    return False
  return "src" in parts or parts[:2] == ("libs", "warp")


def get_base_ref(repo_root: Path) -> str:
  """
  Resolve the base commit for comparing branch and working-tree changes.

  Uses `merge-base HEAD develop` (or `origin/develop`) when on a feature branch
  so already-committed branch edits are still audited alongside uncommitted work.

  Args:
    repo_root: Repository root directory.

  Returns:
    Git ref SHA or `"HEAD"` fallback.
  """
  for cand in ("develop", "origin/develop", "main", "origin/main"):
    proc = subprocess.run(
      ["git", "-C", str(repo_root), "merge-base", "HEAD", cand],
      capture_output=True,
      text=True,
      check=False,
    )
    if proc.returncode == 0 and proc.stdout.strip():
      return proc.stdout.strip()
  return "HEAD"


def get_changed_files(repo_root: Path, base_ref: str) -> list[Path]:
  """
  Collect branch-modified, staged, unstaged, and untracked production Python files.

  Args:
    repo_root: Repository root directory.
    base_ref: Resolved base commit for branch comparison.

  Returns:
    Sorted list of existing repository-relative Python file paths.
  """
  candidates: set[str] = set()
  commands: list[list[str]] = [
    ["git", "-C", str(repo_root), "diff", "--name-only", "--diff-filter=ACMR", base_ref],
    ["git", "-C", str(repo_root), "ls-files", "--others", "--exclude-standard"],
  ]
  if base_ref != "HEAD":
    commands.append(["git", "-C", str(repo_root), "diff", "--name-only", "--diff-filter=ACMR", "HEAD"])
  for cmd in commands:
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode == 0:
      for line in proc.stdout.splitlines():
        if line.strip():
          candidates.add(line.strip())

  files: list[Path] = []
  for raw in sorted(candidates):
    rel = Path(raw)
    if (repo_root / rel).is_file() and is_source_python_file(rel):
      files.append(rel)
  return files


def get_all_files(repo_root: Path) -> list[Path]:
  """
  Discover all production Python source files under `services/` and `libs/`.

  Args:
    repo_root: Repository root directory.

  Returns:
    Sorted list of repository-relative Python file paths.
  """
  files: list[Path] = []
  for root_name in SOURCE_ROOTS:
    root_dir = repo_root / root_name
    if not root_dir.is_dir():
      continue
    for path in root_dir.rglob("*.py"):
      rel = path.relative_to(repo_root)
      if is_source_python_file(rel):
        files.append(rel)
  return sorted(files)


# ━━━━━━━━ AST rule checks ━━━━━━━━


def is_suppressed(lines: list[str], line_no: int) -> bool:
  """
  Check whether a line carries an explicit `# slop-ok: <reason>` suppression.

  Args:
    lines: File lines (0-indexed).
    line_no: 1-based line number of the candidate node.

  Returns:
    True when the target line has a `# slop-ok:` annotation.
  """
  if line_no < 1 or line_no > len(lines):
    return False
  return SUPPRESSION_MARKER in lines[line_no - 1]


def is_speculative_fallback_or(node: ast.BoolOp) -> bool:
  """
  Detect speculative `.get(...) or ...` / `getattr(...) or ...` fallback chains.

  Allows empty-container normalization (`x.get("k", []) or []`, `or ""`, `or {}`)
  and external HTTP/env boundary lookups (`headers.get(...)`, `environ.get(...)`),
  while flagging multi-key guesses, non-empty literal defaults, and expression fallbacks.

  Args:
    node: Boolean `or` AST node.

  Returns:
    True when the `or` chain encodes a speculative fallback.
  """
  if not isinstance(node.op, ast.Or) or len(node.values) < 2:
    return False
  head = node.values[0]
  if not isinstance(head, ast.Call):
    return False
  is_get = isinstance(head.func, ast.Attribute) and head.func.attr == "get"
  is_getattr = isinstance(head.func, ast.Name) and head.func.id == "getattr"
  if not (is_get or is_getattr):
    return False

  # Ignore explicit HTTP header / environment boundary lookups.
  if isinstance(head.func, ast.Attribute):
    owner = head.func.value
    if isinstance(owner, ast.Attribute) and owner.attr in {"headers", "environ", "query_params", "cookies"}:
      return False
    if isinstance(owner, ast.Name) and owner.id in {"headers", "environ"}:
      return False

  fallback = node.values[1]
  if isinstance(fallback, ast.Constant) and fallback.value in ("", 0, False, None):
    return False
  if isinstance(fallback, ast.List | ast.Tuple | ast.Set) and not fallback.elts:
    return False
  if isinstance(fallback, ast.Dict) and not fallback.keys:
    return False
  return True


def check_file(repo_root: Path, rel_path: Path) -> list[Finding]:
  """
  Run all static AST and structure checks on a single Python module.

  Args:
    repo_root: Repository root directory.
    rel_path: Repository-relative path to inspect.

  Returns:
    List of Findings ordered by source line number.
  """
  full_path = repo_root / rel_path
  source = full_path.read_text(encoding="utf-8", errors="replace")
  lines = source.splitlines()
  findings: list[Finding] = []

  try:
    tree = ast.parse(source, filename=str(rel_path))
  except SyntaxError as exc:
    return [Finding(rel_path, exc.lineno or 1, "syntax-error", str(exc))]

  # Require semantic `# ━━━━━━━━ <section> ━━━━━━━━` boundaries on files > 100 lines.
  if len(lines) > 100 and not any(SECTION_HEADER_RE.match(ln.strip()) for ln in lines):
    findings.append(
      Finding(
        rel_path,
        1,
        "missing-section-header",
        "File >100 lines lacks `# ━━━━━━━━ <section_name> ━━━━━━━━` markers.",
      )
    )

  for node in ast.walk(tree):
    line_no = getattr(node, "lineno", 1)
    if is_suppressed(lines, line_no):
      continue

    # 1. Parent relative imports (`from ..something import X`)
    if isinstance(node, ast.ImportFrom) and node.level >= 2:
      findings.append(
        Finding(
          rel_path,
          line_no,
          "parent-relative-import",
          "Use absolute package imports instead of `from ..` parent relative imports.",
        )
      )

    # 2. Legacy typing imports (`Optional`, `Union`)
    if isinstance(node, ast.ImportFrom) and node.module == "typing":
      legacy = [alias.name for alias in node.names if alias.name in {"Optional", "Union"}]
      if legacy:
        findings.append(
          Finding(
            rel_path,
            line_no,
            "old-typing-syntax",
            f"Replace `{', '.join(legacy)}` with PEP 604 `X | Y` union syntax.",
          )
        )

    # 3. Speculative fallback chains (`x.get(...) or ...` / `getattr(...) or ...`)
    if isinstance(node, ast.BoolOp) and is_speculative_fallback_or(node):
      findings.append(
        Finding(
          rel_path,
          line_no,
          "speculative-fallback-or",
          "Do not chain `.get(...) or ...` / `getattr(...) or ...` guesses; read canonical field directly (or `# slop-ok: <reason>`).",
        )
      )

    # 4. Legacy alpha `dict.pop("...", None)` compatibility shims
    if (
      isinstance(node, ast.Call)
      and isinstance(node.func, ast.Attribute)
      and node.func.attr == "pop"
      and len(node.args) == 2
      and isinstance(node.args[0], ast.Constant)
      and isinstance(node.args[0].value, str)
      and isinstance(node.args[1], ast.Constant)
      and node.args[1].value is None
    ):
      findings.append(
        Finding(
          rel_path,
          line_no,
          "legacy-pop-shim",
          f'Avoid defensive `.pop("{node.args[0].value}", None)` compatibility shims; fix producers/callers directly.',
        )
      )

    # 5. Normalized verb purity (`find_*` -> `get_*`)
    if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name.startswith("find_"):
      findings.append(
        Finding(
          rel_path,
          line_no,
          "non-standard-verb",
          f"Rename `{node.name}` to `get_*` per AGENTS.md verb conventions.",
        )
      )

  return sorted(set(findings), key=lambda f: (f.line, f.rule))


# ━━━━━━━━ new dead / 1-caller symbol checks ━━━━━━━━


def is_entry_decorator(node: ast.expr) -> bool:
  """
  Check whether a decorator marks a framework entry point or property.

  Args:
    node: Decorator expression AST node.

  Returns:
    True for properties, validators, routes, or overrides.
  """
  target = node.func if isinstance(node, ast.Call) else node
  if isinstance(target, ast.Name):
    return target.id in ENTRY_DECORATORS
  if isinstance(target, ast.Attribute):
    if target.attr in ENTRY_DECORATORS:
      return True
    if isinstance(target.value, ast.Name) and target.value.id in ENTRY_DECORATOR_OWNERS:
      return True
  return False


def collect_defined_symbols(tree: ast.AST) -> list[tuple[str, int, int, bool]]:
  """
  Collect top-level and class-level `def`/`async def`/`class` definitions.

  Args:
    tree: Parsed module AST.

  Returns:
    List of `(name, start_line, line_span, is_entry)` tuples.
  """
  symbols: list[tuple[str, int, int, bool]] = []

  def visit_body(body: list[ast.stmt]) -> None:
    for stmt in body:
      if isinstance(stmt, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
        end_line = stmt.end_lineno or stmt.lineno
        is_entry = any(is_entry_decorator(dec) for dec in stmt.decorator_list)
        if isinstance(stmt, ast.ClassDef):
          for base in stmt.bases:
            if isinstance(base, ast.Name) and base.id.endswith(("Error", "Exception")):
              is_entry = True
        symbols.append((stmt.name, stmt.lineno, end_line - stmt.lineno + 1, is_entry))
        if isinstance(stmt, ast.ClassDef):
          visit_body(stmt.body)

  if isinstance(tree, ast.Module):
    visit_body(tree.body)
  return symbols


def get_base_symbol_names(repo_root: Path, rel_path: Path, base_ref: str) -> set[str]:
  """
  Read symbol names that already existed in `base_ref` for `rel_path`.

  Args:
    repo_root: Repository root directory.
    rel_path: Repository-relative file path.
    base_ref: Git revision string.

  Returns:
    Set of symbol names present in the base version (empty for new files).
  """
  proc = subprocess.run(
    ["git", "-C", str(repo_root), "show", f"{base_ref}:{rel_path.as_posix()}"],
    capture_output=True,
    text=True,
    check=False,
  )
  if proc.returncode != 0:
    return set()
  try:
    base_tree = ast.parse(proc.stdout)
  except SyntaxError:
    return set()
  return {name for name, _, _, _ in collect_defined_symbols(base_tree)}


def count_symbol_references(tree: ast.AST) -> Counter[str]:
  """
  Count identifier and attribute references inside a module AST.

  Args:
    tree: Parsed module AST.

  Returns:
    Counter mapping symbol name strings to reference counts.
  """
  counts: Counter[str] = Counter()
  for node in ast.walk(tree):
    if isinstance(node, ast.Name):
      counts[node.id] += 1
    elif isinstance(node, ast.Attribute):
      counts[node.attr] += 1
  return counts


def check_new_symbols(repo_root: Path, targets: list[Path], base_ref: str) -> list[Finding]:
  """
  Flag newly introduced dead (0-caller) or short 1-caller helper symbols.

  Scans all production Python files across `services/` and `libs/` so cross-file
  references are counted accurately without false positives on shared helpers.

  Args:
    repo_root: Repository root directory.
    targets: Changed files to audit for newly added symbols.
    base_ref: Git base revision used to identify new definitions.

  Returns:
    List of `dead-new-symbol` and `single-use-new-symbol` Findings.
  """
  all_files = sorted(set(get_all_files(repo_root)) | set(targets))
  per_file_counts: dict[Path, Counter[str]] = {}
  total_counts: Counter[str] = Counter()

  for rel_path in all_files:
    try:
      tree = ast.parse((repo_root / rel_path).read_text(encoding="utf-8", errors="replace"))
    except SyntaxError:
      continue
    counts = count_symbol_references(tree)
    per_file_counts[rel_path] = counts
    total_counts.update(counts)

  findings: list[Finding] = []
  for rel_path in targets:
    full_path = repo_root / rel_path
    lines = full_path.read_text(encoding="utf-8", errors="replace").splitlines()
    try:
      tree = ast.parse("\n".join(lines), filename=str(rel_path))
    except SyntaxError:
      continue

    base_names = get_base_symbol_names(repo_root, rel_path, base_ref)
    for name, line_no, span, is_entry in collect_defined_symbols(tree):
      if name in base_names or is_entry or name.startswith("__") and name.endswith("__") or name == "main":
        continue
      if is_suppressed(lines, line_no):
        continue

      # Note: `ast.Name` / `ast.Attribute` only counts uses (not `def`/`class` headers).
      same_file_uses = per_file_counts.get(rel_path, Counter())[name]
      other_file_uses = total_counts[name] - same_file_uses

      if same_file_uses + other_file_uses == 0:
        findings.append(
          Finding(
            rel_path,
            line_no,
            "dead-new-symbol",
            f"New symbol `{name}` has 0 production references outside tests. Delete it or wire it in.",
          )
        )
      elif other_file_uses == 0 and same_file_uses == 1 and span <= 12:
        findings.append(
          Finding(
            rel_path,
            line_no,
            "single-use-new-symbol",
            f"New symbol `{name}` ({span} lines) has only 1 reference in this file; inline it or annotate `# slop-ok: <reason>`.",
          )
        )

  return findings


# ━━━━━━━━ CLI entry point ━━━━━━━━


def main(argv: list[str] | None = None) -> int:
  """
  Parse CLI arguments, scan target files, and report findings.

  Args:
    argv: Optional argument list overriding `sys.argv[1:]`.

  Returns:
    Exit code 0 when clean, 1 when violations are found.
  """
  parser = argparse.ArgumentParser(description="Static AST contract & slop checker")
  parser.add_argument("--root", type=Path, required=True, help="Repository root path")
  mode = parser.add_mutually_exclusive_group()
  mode.add_argument("--changed", action="store_true", help="Check changed/untracked files (default)")
  mode.add_argument("--all", action="store_true", help="Check all Python source files")
  parser.add_argument("-q", "--quiet", action="store_true", help="Suppress output on success")
  parser.add_argument("paths", nargs="*", help="Explicit file or directory paths to check")
  args: Any = parser.parse_args(argv)

  repo_root: Path = args.root.resolve()
  base_ref = get_base_ref(repo_root)
  if args.paths:
    targets: list[Path] = []
    for raw in args.paths:
      p = Path(raw)
      abs_p = (p if p.is_absolute() else (repo_root / p)).resolve()
      if not abs_p.exists():
        print(f"check_slop: path not found: {raw}", file=sys.stderr)
        return 2
      if abs_p.is_file():
        if abs_p.suffix == ".py":
          targets.append(abs_p.relative_to(repo_root))
      elif abs_p.is_dir():
        for py_file in sorted(abs_p.rglob("*.py")):
          rel = py_file.relative_to(repo_root)
          if is_source_python_file(rel):
            targets.append(rel)
  elif args.all:
    targets = get_all_files(repo_root)
  else:
    targets = get_changed_files(repo_root, base_ref)

  if not targets:
    if not args.quiet:
      print("check_slop: no matching Python source files to check.")
    return 0

  all_findings: list[Finding] = []
  for rel_path in targets:
    all_findings.extend(check_file(repo_root, rel_path))

  if not args.all:
    all_findings.extend(check_new_symbols(repo_root, targets, base_ref))

  all_findings.sort(key=lambda f: (str(f.path), f.line, f.rule))

  if not all_findings:
    if not args.quiet:
      print(f"check_slop: PASS ({len(targets)} file(s) checked)")
    return 0

  for finding in all_findings:
    print(f"FAIL {finding.path}:{finding.line} [{finding.rule}] {finding.message}")
  print(f"\ncheck_slop: {len(all_findings)} violation(s) across {len(targets)} file(s).")
  return 1


if __name__ == "__main__":
  sys.exit(main())
