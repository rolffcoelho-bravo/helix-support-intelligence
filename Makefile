.PHONY: setup lint format typecheck test data-check retrieval-preflight assistance-preflight publication-audit quality

setup:
	uv sync --locked --group dev

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

typecheck:
	uv run mypy

test:
	uv run pytest

data-check:
	uv run python scripts/validate_phase1_data.py

retrieval-preflight:
	uv run python scripts/preflight_phase3_retrieval.py

assistance-preflight:
	uv run python scripts/preflight_phase4_assistance.py
	uv run python benchmarks/assistance/runtime_a41.py --preflight

publication-audit:
	uv run python scripts/audit_publication.py

quality: lint typecheck test data-check retrieval-preflight assistance-preflight publication-audit
