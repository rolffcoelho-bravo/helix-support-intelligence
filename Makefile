.PHONY: setup lint format typecheck test data-check publication-audit quality

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

publication-audit:
	uv run python scripts/audit_publication.py

quality: lint typecheck test data-check publication-audit
