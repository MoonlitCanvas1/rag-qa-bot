PYTHON ?= python

.PHONY: install ingest query chat test lint format

install:
	$(PYTHON) -m pip install -e ".[dev]"

ingest:
	rag-bot ingest

query:
	@test -n "$(Q)" || (echo "usage: make query Q='your question'" && exit 1)
	rag-bot query "$(Q)"

chat:
	rag-bot chat

test:
	$(PYTHON) -m pytest

lint:
	ruff check src tests
	ruff format --check src tests

format:
	ruff check --fix src tests
	ruff format src tests
