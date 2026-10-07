.DEFAULT_GOAL := help

VENV ?= .venv
PYTHON := $(VENV)/bin/python

.PHONY: setup data test reproduce clean help

define require_venv
	@if [ ! -x "$(PYTHON)" ]; then \
		echo "Virtual environment not found at $(VENV); run 'make setup' first." >&2; \
		exit 1; \
	fi
endef

# Create an isolated environment and install the CLI and offline test tools.
setup:
	@if [ ! -x "$(PYTHON)" ]; then python3 -m venv --without-pip "$(VENV)"; fi
	@if ! $(PYTHON) -m pip --version >/dev/null 2>&1; then python3 -m pip --python "$(VENV)" install pip; fi
	$(PYTHON) -m pip install -e ".[test]"

# Explicitly networked: fetch and verify the pinned Kaggle archive.
data:
	$(require_venv)
	$(PYTHON) -m hk_rainfall.cli fetch

# Run offline tests; no dataset or network is needed.
test:
	$(require_venv)
	$(PYTHON) -m pytest -q tests

# Evaluate the existing local archive only; never fetches data.
reproduce:
	$(require_venv)
	$(PYTHON) -m hk_rainfall.cli reproduce

# Remove test/runtime caches only; preserve data and aggregate reports.
clean:
	rm -rf .pytest_cache tests/__pycache__ src/hk_rainfall/__pycache__

help:
	@awk 'BEGIN {FS = ":"} /^# / {description = substr($$0, 3); next} /^[a-zA-Z_-]+:/ {printf "%-14s %s\n", $$1, description; description = ""}' Makefile
