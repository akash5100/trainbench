#!/usr/bin/env bash
# .trainbench/tools/websearch.sh — Unified Exa search & URL content extractor.
# Supports semantic search, deep research (type: deep), and URL markdown content extraction.

set -euo pipefail

# ━━━━━━━━ env check (strictly relies on environment, never scrapes files) ━━━━━━━━
if [ -z "${EXA_API_KEY:-}" ]; then
  echo "Error: EXA_API_KEY environment variable is not set." >&2
  echo "Ensure EXA_API_KEY is exported in your environment." >&2
  exit 1
fi

usage() {
  cat << USAGE
Usage:
  # 1. Web Search & Deep Research
  .trainbench/tools/websearch.sh "<query>"                     # Fast semantic web search (default)
  .trainbench/tools/websearch.sh --deep "<query>"              # Deep research mode (exhaustive retrieval)
  .trainbench/tools/websearch.sh -n <num> "<query>"            # Return top-n results (default: 5)
  .trainbench/tools/websearch.sh --domain "<domain>" "<query>" # Restrict search to specific domain (e.g. github.com, arxiv.org)
  .trainbench/tools/websearch.sh --highlights "<query>"        # Include semantic highlights/excerpts

  # 2. URL Content & Page Reading
  .trainbench/tools/websearch.sh --contents "<url>"            # Extract markdown text from URL (default budget: 12000 chars)
  .trainbench/tools/websearch.sh --contents "<url>" --full     # Extract FULL uncapped text from URL
  .trainbench/tools/websearch.sh --contents "<url>" --max-chars <num> # Extract up to specified character limit
  .trainbench/tools/websearch.sh --summary "<url>"             # Extract structured LLM summary of a specific URL

  .trainbench/tools/websearch.sh --help                        # Show this help message

Examples:
  .trainbench/tools/websearch.sh "Muon optimizer Newton-Schulz iterations"
  .trainbench/tools/websearch.sh --deep "TPU v4 vs v5e bfloat16 matrix multiplication throughput"
  .trainbench/tools/websearch.sh --contents "https://arxiv.org/abs/2410.01201"
  .trainbench/tools/websearch.sh --contents "https://docs.astral.sh/uv/concepts/projects/dependencies/" --full
USAGE
  exit 0
}

if [ $# -eq 0 ]; then
  usage
fi

MODE="search"
SEARCH_TYPE="auto"
QUERY=""
URL=""
NUM_RESULTS=5
HIGHLIGHTS=false
INCLUDE_DOMAINS=""
SUMMARY_MODE=false
MAX_CHARS=12000
IS_FULL=false

while [ $# -gt 0 ]; do
  case "$1" in
    -h|--help)
      usage
      ;;
    --deep)
      SEARCH_TYPE="deep"
      shift
      ;;
    --contents)
      MODE="contents"
      shift
      URL="$1"
      shift
      ;;
    --summary)
      MODE="contents"
      SUMMARY_MODE=true
      shift
      URL="$1"
      shift
      ;;
    --full)
      IS_FULL=true
      shift
      ;;
    --max-chars)
      shift
      MAX_CHARS="$1"
      shift
      ;;
    -n|--num)
      shift
      NUM_RESULTS="$1"
      shift
      ;;
    --highlights)
      HIGHLIGHTS=true
      shift
      ;;
    -d|--domain)
      shift
      INCLUDE_DOMAINS="$1"
      shift
      ;;
    *)
      if [ -z "$QUERY" ]; then
        QUERY="$1"
      else
        QUERY="$QUERY $1"
      fi
      shift
      ;;
  esac
done

# ━━━━━━━━ URL Contents Extraction Mode ━━━━━━━━
if [ "$MODE" = "contents" ]; then
  if [ -z "$URL" ]; then
    echo "Error: No URL specified." >&2
    exit 1
  fi

  PAYLOAD=$(python3 -c '
import json, sys
url = sys.argv[1]
summary_mode = sys.argv[2] == "true"
is_full = sys.argv[3] == "true"
max_chars = int(sys.argv[4])

body = {"urls": [url]}
if summary_mode:
    body["summary"] = True
elif is_full:
    body["text"] = True  # Exa returns full uncapped page text
else:
    body["text"] = {"maxCharacters": max_chars}

print(json.dumps(body))
' "$URL" "$SUMMARY_MODE" "$IS_FULL" "$MAX_CHARS")

  RESPONSE=$(curl -sS -X POST "https://api.exa.ai/contents" \
    -H "Content-Type: application/json" \
    -H "x-api-key: $EXA_API_KEY" \
    -d "$PAYLOAD")

  python3 -c '
import json, sys

try:
    data = json.loads(sys.argv[1])
except Exception:
    print(f"Error parsing response: {sys.argv[1]}")
    sys.exit(1)

if "results" in data and len(data["results"]) > 0:
    item = data["results"][0]
    title = item.get("title", "No Title")
    url = item.get("url", "")
    print(f"# {title}\n")
    print(f"**Source**: {url}\n")
    if "summary" in item and item["summary"]:
        summary_val = item["summary"]
        print(f"### Summary:\n{summary_val}\n")
    if "text" in item and item["text"]:
        print(item["text"])
elif "error" in data:
    err = data["error"]
    print(f"Exa Error: {err}")
else:
    print(json.dumps(data, indent=2))
' "$RESPONSE"
  exit 0
fi

# ━━━━━━━━ Search & Deep Research Mode ━━━━━━━━
if [ -z "$QUERY" ]; then
  echo "Error: No search query specified." >&2
  exit 1
fi

PAYLOAD=$(python3 -c '
import json, sys

query = sys.argv[1]
num_results = int(sys.argv[2])
search_type = sys.argv[3]
highlights = sys.argv[4] == "true"
domain = sys.argv[5]

body = {
    "query": query,
    "type": search_type,
    "numResults": num_results,
    "contents": {
        "text": {"maxCharacters": 1000}
    }
}

if highlights:
    body["contents"]["highlights"] = True

if domain:
    body["includeDomains"] = [d.strip() for d in domain.split(",") if d.strip()]

print(json.dumps(body))
' "$QUERY" "$NUM_RESULTS" "$SEARCH_TYPE" "$HIGHLIGHTS" "$INCLUDE_DOMAINS")

RESPONSE=$(curl -sS -X POST "https://api.exa.ai/search" \
  -H "Content-Type: application/json" \
  -H "x-api-key: $EXA_API_KEY" \
  -d "$PAYLOAD")

python3 -c '
import json, sys

try:
    data = json.loads(sys.argv[1])
except Exception:
    print(f"Error parsing Exa response: {sys.argv[1]}")
    sys.exit(1)

if "results" in data:
    results = data["results"]
    if not results:
        print("No search results found.")
        sys.exit(0)

    for i, res in enumerate(results, 1):
        title = res.get("title") or "Untitled"
        url = res.get("url") or ""
        score = res.get("score")
        score_str = f" (score: {score:.3f})" if score is not None else ""
        print(f"### {i}. [{title}]({url}){score_str}")

        highlights = res.get("highlights")
        if highlights:
            for h in highlights:
                print(f"> {h.strip()}\n")
        else:
            text = res.get("text")
            if text:
                snippet = " ".join(text.split()[:80])
                print(f"{snippet}...\n")
elif "error" in data:
    err = data["error"]
    print(f"Exa Error: {err}")
else:
    print(json.dumps(data, indent=2))
' "$RESPONSE"
