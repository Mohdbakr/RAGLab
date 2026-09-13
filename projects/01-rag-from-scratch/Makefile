#################################################################################
# GLOBALS
#################################################################################

PROJECT_NAME = RAG From Scratch
PYTHON_VERSION = 3.14
PYTHON_INTERPRETER = python

SRC_DIRS := src tests

RUFF := uv run ruff
MYPY := uv run mypy
PYTEST := uv run pytest

#################################################################################
# ENVIRONMENT
#################################################################################

## Install Python dependencies
.PHONY: install
install:
	uv sync

## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	uv venv --python $(PYTHON_VERSION)
	@echo ">>> New uv virtual environment created."
	@echo ">>> Windows: .\\.venv\\Scripts\\activate"
	@echo ">>> Unix/macOS: source .venv/bin/activate"

#################################################################################
# DEVELOPMENT
#################################################################################

## Delete compiled files and caches
.PHONY: clean
clean:
	@echo "Cleaning caches and compiled files..."
	rm -rf .pytest_cache .ruff_cache .mypy_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f \( -name "*.pyc" -o -name "*.pyo" \) -delete

## Check code style and linting
.PHONY: lint
lint:
	$(RUFF) format --check $(SRC_DIRS)
	$(RUFF) check $(SRC_DIRS)

## Format source code
.PHONY: format
format:
	$(RUFF) check --select I --fix $(SRC_DIRS)
	$(RUFF) format $(SRC_DIRS)

## Type-check with mypy
.PHONY: typecheck
typecheck:
	$(MYPY) src

## Run tests
.PHONY: test
test:
	$(PYTEST) -v

## Run tests with coverage report
.PHONY: test-cov
test-cov:
	$(PYTEST) -v --cov=plainrag --cov-report=term-missing

## Run full quality gate (lint + typecheck + tests)
.PHONY: check
check: lint typecheck test

## Run CI pipeline
.PHONY: ci
ci: clean check

#################################################################################
# APPLICATION
#################################################################################

## Ingest a text file into the index (make ingest FILE=path/to/some.txt)
.PHONY: ingest
ingest:
	uv run plainrag ingest $(FILE)

## Ask a question against the ingested index (make ask QUESTION="...")
.PHONY: ask
ask:
	uv run plainrag ask "$(QUESTION)"

#################################################################################
# DOCKER
#################################################################################

## Build the Docker image
.PHONY: docker-build
docker-build:
	docker compose build

## Run any plainrag command in Docker (make docker-run ARGS="ingest some.txt")
.PHONY: docker-run
docker-run:
	docker compose run --rm plainrag $(ARGS)

## Remove Docker containers, networks, and volumes
.PHONY: docker-down
docker-down:
	docker compose down

#################################################################################
# SELF DOCUMENTING COMMANDS
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "$${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
