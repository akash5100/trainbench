#!/usr/bin/env python3
"""
worktree__harness.py — Extract user prompt and metadata from ambient harness trajectories.

Currently supports:
- Antigravity: extracts verbatim <USER_REQUEST>, cleanly resolves @[Quote] references
  from <ADDITIONAL_METADATA> into markdown blockquotes, and extracts local timestamps.
- Codex: extracts the complete user message from the current thread history, including
  response annotations and comments exactly as recorded.
"""

import json
import os
import re
import sqlite3
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


# ━━━━━━━━ antigravity extraction ━━━━━━━━


def extract_antigravity(conversation_id: str | None = None, index: int = -1) -> dict[str, Any]:
  """
  Extract user input from Antigravity session trajectory logs.

  Resolves UI quotes (@[Quote]) into inline markdown blockquotes (> ...).

  Args:
    conversation_id: Explicit UUID of the conversation, or None to read ambient env.
    index: 0-based or negative index into USER_INPUT steps (defaults to -1, latest).

  Returns:
    Dict containing text, timestamp, step_index, and conversation_id.
  """
  conv_id = conversation_id or os.environ.get("ANTIGRAVITY_CONVERSATION_ID")
  if not conv_id:
    raise ValueError("Conversation ID is required. Pass conversation_id or set ANTIGRAVITY_CONVERSATION_ID.")

  app_data_dir = os.environ.get("ANTIGRAVITY_APP_DATA_DIR")
  if app_data_dir:
    base_dir = Path(app_data_dir)
  else:
    base_dir = Path.home() / ".gemini" / "antigravity"

  logs_dir = base_dir / "brain" / conv_id / ".system_generated" / "logs"
  compact_log = logs_dir / "transcript.jsonl"
  full_log = logs_dir / "transcript_full.jsonl"

  if not compact_log.is_file():
    raise FileNotFoundError(f"Transcript not found for conversation {conv_id} at {compact_log}")

  user_inputs: list[tuple[int, dict[str, Any]]] = []
  with compact_log.open(encoding="utf-8") as f:
    for line_idx, line in enumerate(f):
      line = line.strip()
      if not line:
        continue
      try:
        record = json.loads(line)
        if record.get("type") == "USER_INPUT":
          user_inputs.append((line_idx, record))
      except json.JSONDecodeError:
        continue

  if not user_inputs:
    raise ValueError(f"No USER_INPUT steps found in transcript for conversation {conv_id}")

  try:
    target_line_idx, target_rec = user_inputs[index]
  except IndexError as err:
    raise IndexError(f"Requested USER_INPUT index {index} out of range (found {len(user_inputs)} user inputs)") from err

  content = target_rec.get("content", "")
  # If content was truncated in compact log, retrieve the full record from transcript_full.jsonl
  if "content" in target_rec.get("truncated_fields", []) and full_log.is_file():
    with full_log.open(encoding="utf-8") as f:
      for idx, line in enumerate(f):
        if idx == target_line_idx:
          try:
            full_rec = json.loads(line.strip())
            content = full_rec.get("content", content)
          except json.JSONDecodeError:
            pass
          break

  # Extract metadata blocks to find referenced quotes/items
  metadata_blocks = re.findall(r"<ADDITIONAL_METADATA>\s*(.*?)\s*</ADDITIONAL_METADATA>", content, re.DOTALL)
  metadata = "\n".join(metadata_blocks)

  item_pattern = re.compile(
    r"^(@\[([^\]]+)\])\s+is\s+a\s+\[([^\]]+)\]:\s*\n(.*?)(?=\n@\[|\Z)",
    re.DOTALL | re.MULTILINE,
  )
  items: dict[str, str] = {}
  for m in item_pattern.finditer(metadata):
    tag = m.group(1)
    val = m.group(4).strip()
    items[tag] = val

  # Extract user requests and resolve @[Quote] references into clean markdown blockquotes
  requests = re.findall(r"<USER_REQUEST>\s*(.*?)\s*</USER_REQUEST>", content, re.DOTALL)
  if not requests:
    requests = [content.strip()]

  cleaned_parts: list[str] = []
  for req in requests:
    text = req.strip()
    for tag, val in items.items():
      if tag in text:
        quoted_block = "\n".join(f"> {line}" for line in val.splitlines())
        text = re.sub(re.escape(tag) + r"\s*", f"{quoted_block}\n\n", text).strip()
    cleaned_parts.append(text)

  user_text = "\n\n".join(cleaned_parts)

  # Extract local timestamp from <ADDITIONAL_METADATA> if present
  time_match = re.search(r"The current local time is:\s*([^\n\r<]+)", metadata or content)
  if time_match:
    timestamp = time_match.group(1).strip().rstrip(".")
  else:
    timestamp = target_rec.get("created_at", "")

  return {
    "text": user_text,
    "timestamp": timestamp,
    "step_index": target_rec.get("step_index", 0),
    "conversation_id": conv_id,
  }


# ━━━━━━━━ codex extraction ━━━━━━━━


def extract_codex(thread_id: str | None = None, index: int = -1) -> dict[str, Any]:
  """
  Extract a complete user message from Codex's local thread history.

  Codex stores response annotations and their comments in the user message text.
  Keep that text untouched so selections and comments remain together in the ledger.

  Args:
    thread_id: Explicit Codex thread ID, or None to read CODEX_THREAD_ID.
    index: 0-based or negative index into user messages (defaults to -1, latest).

  Returns:
    Dict containing the exact message text, local timestamp, message ordinal, and thread ID.
  """
  current_thread_id = thread_id or os.environ.get("CODEX_THREAD_ID")
  if not current_thread_id:
    raise ValueError("Codex thread ID is required. Pass thread_id or set CODEX_THREAD_ID.")

  codex_home = Path(os.environ.get("CODEX_HOME", Path.home() / ".codex"))
  history_db = codex_home / "thread_history_1.sqlite"
  if not history_db.is_file():
    raise FileNotFoundError(f"Codex thread history database not found at {history_db}")

  database_uri = f"{history_db.resolve().as_uri()}?mode=ro"
  with sqlite3.connect(database_uri, uri=True) as connection:
    rows = connection.execute(
      """
      SELECT item_json, created_at_ms, rollout_ordinal
      FROM thread_items
      WHERE thread_id = ? AND item_type = 'userMessage'
      ORDER BY rollout_ordinal
      """,
      (current_thread_id,),
    ).fetchall()

  if not rows:
    raise ValueError(f"No user messages found in Codex thread {current_thread_id}")

  try:
    raw_item, created_at_ms, rollout_ordinal = rows[index]
  except IndexError as error:
    raise IndexError(f"Requested user message index {index} out of range (found {len(rows)} messages)") from error

  item = json.loads(raw_item)
  content = item.get("content", [])
  text_parts = [part["text"] for part in content if isinstance(part, dict) and part.get("type") == "text"]
  if not text_parts:
    raise ValueError(f"Codex user message {item.get('id', rollout_ordinal)} contains no text")

  # The normal history entry has one text part; preserve it byte-for-byte. Multiple text
  # parts are joined with a blank line so no part or annotation comment is discarded.
  message_text = "\n\n".join(text_parts)
  timestamp = datetime.fromtimestamp(created_at_ms / 1000, UTC).astimezone().isoformat(timespec="seconds")
  return {
    "text": message_text,
    "timestamp": timestamp,
    "step_index": rollout_ordinal,
    "conversation_id": current_thread_id,
  }


# ━━━━━━━━ cli entrypoint ━━━━━━━━


def main() -> None:
  """
  Execute CLI extraction from harness trajectory.

  Usage: worktree__harness.py <harness> [conversation_id] [index].
  """
  if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
    print("Usage: worktree__harness.py <harness> [conversation_id] [index]", file=sys.stderr)
    sys.exit(2)

  harness = sys.argv[1].lower()
  conv_id = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2].strip() else None
  index = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3].strip() else -1

  try:
    if harness == "antigravity":
      res = extract_antigravity(conversation_id=conv_id, index=index)
    elif harness == "codex":
      res = extract_codex(thread_id=conv_id, index=index)
    else:
      raise ValueError(f"Unsupported harness: {harness} (supported: antigravity, codex)")

    print(json.dumps(res, ensure_ascii=False))
  except Exception as error:
    print(f"Error: {error}", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
  main()
