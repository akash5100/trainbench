---
trigger: always_on
description: Operating protocols, harness tools, and code-quality contracts for trainbench.
---

## Build & Verification Gate

- **Local Dev Setup**: `make setup` (installs pre-commit hooks, editable package, and initializes harness & semantic search).
- **All-in-One Gate**: `make check` (runs format check, lint, mypy strict typing, AST slop audit, and tests quietly; exits 0).
- **Targeted Tests**: `make test k=<pattern>` or `make test target=test/<file>`.
- **Accelerator Setup**: `make setup-gpu` (`pip install -e ".[gpu]"`), `make setup-tpu` (`pip install -e ".[tpu]"`).

---

## Agent Harness & Operational Tools

Deterministic operational tools live in `.trainbench/`:
```text
.trainbench/
├── skills/
│   └── todo/            # Native GitHub issue tracker runbook
└── tools/
    ├── check_env.sh     # Zero-leakage environment & secret status checker
    ├── check_slop.sh    # AST contract & slop purity linter
    ├── codesearch.sh    # AST-aware semantic search & caller/callee graphs
    ├── diff.sh          # Enclosing-symbol AST diffs (run with --help to unfold)
    ├── help.sh          # Tool manual inspector & model profile resolver
    ├── outline.sh       # Fast structural skeleton & section extractor
    ├── websearch.sh     # Exa semantic web search & URL markdown extractor
    └── worktree.sh      # Isolated worktree management & ledgers (run with --help to unfold)
```

Run `.trainbench/tools/help.sh tools` to list tools. Unfold complex workflows on-demand via `--help` (e.g. `.trainbench/tools/worktree.sh --help`, `.trainbench/tools/diff.sh --help`).

---

## Retrieval & Inspection Discipline

<CRITICAL_ENFORCEMENT: TOOL_CALL_ORDERING>
1. CODESEARCH-FIRST EXPLORATION:
   - When researching architecture, caller/callee relationships, cross-module dependencies, or locating where logic lives, your VERY FIRST action MUST be to run `.trainbench/tools/codesearch.sh`.
   - **DO NOT waste a tool call running `--help`**. Pre-baked syntax and options:
     ```bash
     .trainbench/tools/codesearch.sh "<query>"                     # Semantic AST graph search (default)
     .trainbench/tools/codesearch.sh -e "<keyword>" "<query>"      # Hybrid: exact keyword match + semantic intent (Highest Precision)
     .trainbench/tools/codesearch.sh -k <num> "<query>"            # Return top-k results (default: 8)
     .trainbench/tools/codesearch.sh --include="*.py" "<query>"    # Filter by file glob (e.g. "*.py")
     .trainbench/tools/codesearch.sh "<query>" <path>              # Scoped search to directory or file
     .trainbench/tools/codesearch.sh --raw [flags...] "<query>"    # Raw colgrep terminal output
     ```

2. NO BLIND FILE READING (OUTLINE FIRST):
   - NEVER call file reading tools on files (>100 lines) without FIRST inspecting their structural skeleton.
   - **DO NOT waste a tool call running `--help`**. Syntax:
     ```bash
     .trainbench/tools/outline.sh <file_path>    # Print line-numbered outline of classes, functions, and sections
     ```
   - Pinpoint exact line ranges and section boundaries first, then read ONLY targeted sections in one call.

3. WEB SEARCH & URL CONTENT EXTRACTION (`websearch.sh`):
   - Route external retrieval through `.trainbench/tools/websearch.sh`.
   - **DO NOT waste a tool call running `--help`**. Syntax:
     ```bash
     .trainbench/tools/websearch.sh "<query>"                     # Semantic web search (titles, URLs, snippets)
     .trainbench/tools/websearch.sh --deep "<query>"              # Deep research mode
     .trainbench/tools/websearch.sh -n <num> "<query>"            # Return top-n results (default: 5)
     .trainbench/tools/websearch.sh --contents "<url>"            # Extract markdown text from URL (budget: 12000 chars)
     .trainbench/tools/websearch.sh --contents "<url>" --full     # Extract FULL uncapped text from URL
     ```

4. ZERO SECRET LEAKAGE (`check_env.sh`):
   - NEVER cat, grep, or print `.env`, credential files, or shell variables.
   - Inspect variable presence and length safely via `.trainbench/tools/check_env.sh <VAR1> <VAR2>...`.

5. CONTEXT CACHE DISCIPLINE:
   - NEVER re-read a file or line range that was already retrieved in the current or immediately preceding turns. Context is an authoritative cache.
</CRITICAL_ENFORCEMENT>

---

## Code Construction & Style Guidelines

- **Formatting & Types**: Python 3.11+ native syntax (PEP 604 `X | Y`, built-in `list`, `dict`), 2-space indentation, 150-char line length, double quotes (`ruff.toml`, `mypy.ini`).
- **Semantic Section Headers**: Delimit logical blocks using exact section comments with 8 heavy bars:
  `# ━━━━━━━━ <section_name> ━━━━━━━━` (enforced by `check_slop.py`).
- **Strict Contract Purity (Fail Fast)**: Read fields strictly from canonical schema locations. NEVER write speculative `.get()` or `getattr()` fallback chains (`obj.get("x") or obj.get("y")`).
- **Import Rules**:
  - Absolute imports only for external/cross-module imports (`from trainbench.models...`).
  - NEVER use parent relative imports (`from ..something import XYZ`).
  - Relative imports (`from .sibling import XYZ`) are ONLY permitted for siblings within the exact same directory.
- **Lazy Module Imports**:
  Use `trainbench.lazy.create_lazy_importer` (PEP 562) with `if TYPE_CHECKING:` in package `__init__.py`. Keeps `import trainbench` <5ms and prevents PyTorch and JAX accelerator driver conflicts.

---

## Codebase Map

```text
trainbench/
├── Makefile                  # Thin entrypoint: setup, check, test, run, push-gpu, push-tpu
├── pyproject.toml            # Core deps + optional [gpu], [tpu], [dev] extras
├── AGENTS.md                 # Repo rules, harness commands, and codebase map
├── .trainbench/              # Repo-local agent harness (skills, tools, paths, install.sh)
│
├── trainbench/               # Reusable, stateless library primitives
│   ├── __init__.py           # PEP 562 lazy exports via lazy.py (<5ms import)
│   ├── lazy.py               # Prevents eager torch/jax imports & HBM/CUDA conflicts
│   ├── losses/
│   │   ├── torch/            # PyTorch loss functions
│   │   └── jax/              # JAX loss functions
│   ├── models/
│   │   ├── torch/            # PyTorch backbones & heads (GPU)
│   │   └── jax/              # Flax NNX / JAX models (TPU)
│   ├── trainers/
│   │   ├── torch/            # Pure step helpers, AMP & DDP utilities
│   │   └── jax/              # Pure grad_step / apply_step, sharding & remat helpers
│   ├── optim/
│   │   ├── torch/            # PyTorch optimizers & schedules
│   │   └── jax/              # Optax transforms & schedules
│   └── data/                 # NumPy / PyArrow boundary loaders (zero-copy into torch or jax)
│
├── experiments/              # Runnable, self-contained recipes
│   └── <exp_name>/
│       ├── train.py          # Explicit, readable ~150-line training loop (argparse CLI)
│       ├── records/          # Plain-text .txt run logs for verified milestone runs
│       └── README.md         # Repro command & logged results
│
└── test/                     # Unit & numerical parity tests mirroring trainbench/
```

### Architectural Principles

- **Library vs. Experiments**: `trainbench/` holds stateless, reusable primitives; `experiments/<name>/train.py` holds explicit, self-contained training loops.
- **Experiment CLI**: Training scripts accept hyperparameters via `argparse` CLI flags (logged by W&B and the remote runner).
- **Run Records**: Verified milestone runs store plain-text stdout logs under `experiments/<name>/records/`.
- **Dual-Stack Isolation**: PyTorch (`torch/`) and JAX/Flax (`jax/`) modules never import each other, enforced by `lazy.py` and optional `pyproject.toml` extras.
- **Incremental Scaffolding**: Create submodules only when an implementation requires them.
