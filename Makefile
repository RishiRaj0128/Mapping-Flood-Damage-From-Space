.PHONY: setup test lint check run demo clean

PYTHON ?= .venv/bin/python
ifeq ($(OS),Windows_NT)
	PYTHON = .venv/Scripts/python.exe
endif

setup:
	uv venv .venv
	uv pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest tests/

lint:
	$(PYTHON) -m ruff check src/ tests/ evaluation/

check: lint test

run:
	$(PYTHON) -m floodmap.cli --help

demo:
	$(PYTHON) -m floodmap.cli demo --bbox "85.15,27.85,85.45,28.15" --date "2026-08-26"

clean:
	rm -rf .pytest_cache .ruff_cache dist build *.egg-info .cache data_cache
