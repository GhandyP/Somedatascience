.DEFAULT_GOAL := help

VENV ?= .venv
PYTHON := $(VENV)/bin/python

.PHONY: setup data test reproduce clean clean-reports help

define require_venv
	@if [ ! -x "$(PYTHON)" ]; then \
		echo "Virtual environment not found at $(VENV); run 'make setup' first." >&2; \
		exit 1; \
	fi
endef

# Install the project and development tools in editable mode.
setup:
	@if [ ! -x "$(PYTHON)" ]; then python3 -m venv --without-pip "$(VENV)"; fi
	@if ! $(PYTHON) -m pip --version >/dev/null 2>&1; then python3 -m pip --python "$(VENV)" install pip; fi
	$(PYTHON) -m pip install -e ".[dev]"

# Download and verify the pinned raw dataset.
data:
	$(require_venv)
	$(PYTHON) -m namegender fetch

# Run the complete offline test suite.
test:
	$(require_venv)
	$(PYTHON) -m pytest -q

# Run the full pipeline and write reproducible reports.
reproduce:
	$(require_venv)
	$(PYTHON) -m namegender reproduce

# Remove generated caches and Python bytecode without touching reports.
clean:
	rm -rf .pytest_cache tests/__pycache__ src/namegender/__pycache__

# Remove reports before regenerating them with reproduce.
clean-reports:
	rm -rf reports

# Show the available project targets.
help:
	@awk 'BEGIN {FS = ":"} /^# / {description = substr($$0, 3); next} /^[a-zA-Z_-]+:/ {printf "%-12s %s\n", $$1, description; description = ""}' Makefile
