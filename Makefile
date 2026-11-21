# trainbench Makefile
# Enforces code standards, AST contracts, and test execution.

.DEFAULT_GOAL := help
.PHONY: help format lint slop typecheck test check clean setup setup-gpu setup-tpu

VENV ?= $(if $(VIRTUAL_ENV),$(VIRTUAL_ENV),$(if $(WORKON_HOME),$(WORKON_HOME)/trainbench,.venv))
PYTHON := $(if $(wildcard $(VENV)/bin/python),$(VENV)/bin/python,python3)
RUFF := $(if $(wildcard $(VENV)/bin/ruff),$(VENV)/bin/ruff,ruff)
PYTEST := $(if $(wildcard $(VENV)/bin/pytest),$(VENV)/bin/pytest,pytest)
MYPY := $(if $(wildcard $(VENV)/bin/mypy),$(VENV)/bin/mypy,mypy)

help: ## Show available Makefile targets
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

format: ## Format code with ruff and fix auto-fixable lint issues
	@$(RUFF) format -q .
	@$(RUFF) check --fix -q .
	@echo "✓ Formatted"

lint: ## Run ruff lint check without modifying files
	@$(RUFF) check -q .
	@echo "✓ Lint: ok"

slop: ## Run AST slop and contract purity linter
	@bash .trainbench/tools/check_slop.sh

typecheck: ## Run static type checking via mypy
	@$(MYPY) --config-file=mypy.ini trainbench test

K ?= $(k)
TARGET ?= $(target)

test: ## Run unit tests with pytest (optional: K=<pattern> or TARGET=<file>)
	@$(PYTEST) $(if $(TARGET),$(TARGET),test/) -q --no-header $(if $(K),-k "$(K)")

check: ## All-in-one check: formatting, lint, typecheck, slop audit, and tests
	@$(RUFF) format --check -q .
	@$(RUFF) check -q .
	@$(MYPY) --config-file=mypy.ini trainbench test
	@bash .trainbench/tools/check_slop.sh -q
	@$(PYTEST) test/ -q --no-header
	@echo "✓ All checks passed"

clean: ## Clean build artifacts, pycache, and temporary test directories
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true

INSTALL_CMD := $(if $(shell which uv 2>/dev/null),uv pip install -q --python $(PYTHON) -e,pip install -q -e)

setup: ## Initialize local dev environment (pre-commit, editable install, and harness setup)
	@pre-commit install
	@$(INSTALL_CMD) .
	@bash .trainbench/install.sh
	@echo "✓ trainbench setup complete"

setup-gpu: ## Install trainbench in editable mode with GPU stack
	@$(INSTALL_CMD) ".[gpu]"
	@echo "✓ GPU stack ready"

setup-tpu: ## Install trainbench in editable mode with TPU stack
	@$(INSTALL_CMD) ".[tpu]"
	@echo "✓ TPU stack ready"
