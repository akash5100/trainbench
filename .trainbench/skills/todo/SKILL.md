---
name: todo
description: Manage, track, and triage repository tasks, issues, and experiment milestones using GitHub CLI (gh issue). Use when creating, viewing, listing, commenting, or closing project tasks, or when the user invokes /todo.
---

# Task & Issue Management (`/todo`)

Runbook for token-efficient issue tracking, task triage, and training milestone checkpoints using the GitHub CLI (`gh`).

---

## 1. Fast Listing & Triage

Avoid dumping large issue tables into context. Request only open issues or specific labels:

```bash
# Quick overview of open issues
gh issue list --limit 15 --state open

# Filter by label (e.g. bug, enhancement, experiment)
gh issue list --limit 10 --label "experiment"

# Token-compact JSON for programmatic parsing
gh issue list --limit 10 --json number,title,labels,updatedAt --jq '.[] | "#\(.number): \(.title) [\(.labels | map(.name) | join(","))]"'
```

---

## 2. Inspecting Issues

```bash
# View title, metadata, and description
gh issue view <id>

# View with conversation comments
gh issue view <id> --comments
```

---

## 3. Creating Tasks

When creating issues, use clear titles and appropriate labels:

```bash
gh issue create \
  --title "Implement Muon optimizer Newton-Schulz iteration in JAX" \
  --body "### Goal\nPort and verify Muon optimizer with Newton-Schulz orthogonalization in trainbench/optim/.\n\n### Acceptance Criteria\n- [ ] Aurora numerical parity in tests/optim/test_muon.py\n- [ ] Support both PyTorch and Optax/JAX transformations" \
  --label "experiment,enhancement"
```

---

## 4. Commenting & Progress Checkpoints

Append milestones, execution logs, or discussion updates:

```bash
# Short status update
gh issue comment <id> --body "Checkpoint: completed TPU mesh sharding helpers in worktree 0002."

# Body from file (for loss curves or benchmark tables)
gh issue comment <id> --body-file /path/to/summary.md
```

---

## 5. Resolving & Closing

Always link closing comments to the relevant commit hash or benchmark run:

```bash
gh issue close <id> --comment "Resolved in commit $(git rev-parse --short HEAD). Tests passing."
```

---

## 6. Experiment & Benchmark Integration

Whenever an issue is opened or worked on for a specific experiment:
1. Reference the issue in the experiment plan (`experiments/<name>/README.md`).
2. When closed, document the resulting metric (e.g. step loss, MFU, throughput) alongside the closing commit.
