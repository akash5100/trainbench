#!/usr/bin/env python3
"""
sym_context.py — Find the enclosing symbol for a line number using Python AST.

Requires Python 3.8+ (uses ast.end_lineno).

Usage:
  python sym_context.py <filepath> <lineno>

Output: tab-separated lines, one per enclosing scope (outermost first):
  SYMBOL  <start>  <end>  <label>   ← innermost enclosing function/method
  CLASS   <start>  <end>  <label>   ← if that symbol is inside a class
  OUTER   <start>  <end>  <label>   ← if nested inside another function

Exits 0 with no output when the line is at module level (no enclosing symbol).
Exits 1 on parse errors.
"""

from __future__ import annotations

import ast
import sys


def label_for(node: ast.AST) -> str:
  """
  Build a readable label string for a function or class AST node.
  """
  if isinstance(node, ast.AsyncFunctionDef):
    return f"async def {node.name}"
  if isinstance(node, ast.FunctionDef):
    return f"def {node.name}"
  if isinstance(node, ast.ClassDef):
    return f"class {node.name}"
  return repr(node)


def find_enclosing(filepath: str, lineno: int) -> None:
  """
  Parse filepath and print the enclosing symbol chain for lineno to stdout.

  Each line is tab-separated: KIND, start, end, label.
  """
  with open(filepath) as f:
    source = f.read()

  try:
    tree = ast.parse(source, filename=filepath)
  except SyntaxError as exc:
    print(f"ERROR\tSyntaxError: {exc}", file=sys.stderr)
    sys.exit(1)

  # Collect every function/class node whose range contains the target line.
  # ast.end_lineno is available from Python 3.8+.
  enclosing: list[ast.AST] = []
  for node in ast.walk(tree):
    if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
      continue
    if not hasattr(node, "end_lineno"):
      print("ERROR\tPython 3.8+ required (ast.end_lineno missing)", file=sys.stderr)
      sys.exit(1)
    if node.lineno <= lineno <= node.end_lineno:
      enclosing.append(node)

  if not enclosing:
    # Module-level — caller handles fallback.
    sys.exit(0)

  # Sort by start line ascending so innermost is last.
  enclosing.sort(key=lambda n: n.lineno)

  innermost = enclosing[-1]
  ancestors = enclosing[:-1]  # outermost → closest parent

  # Emit innermost symbol first.
  print(f"SYMBOL\t{innermost.lineno}\t{innermost.end_lineno}\t{label_for(innermost)}")

  # Emit ancestors with their kind (CLASS or OUTER function).
  for ancestor in reversed(ancestors):
    if isinstance(ancestor, ast.ClassDef):
      kind = "CLASS"
    else:
      kind = "OUTER"
    print(f"{kind}\t{ancestor.lineno}\t{ancestor.end_lineno}\t{label_for(ancestor)}")


if __name__ == "__main__":
  if len(sys.argv) != 3:
    print(f"Usage: {sys.argv[0]} <filepath> <lineno>", file=sys.stderr)
    sys.exit(1)
  find_enclosing(sys.argv[1], int(sys.argv[2]))
