.PHONY: setup lint format typecheck test data-check retrieval-preflight assistance-preflight assistance-a42-preflight assistance-a43a-preflight assistance-a44a-preflight assistance-a44b-preflight assistance-a44c-preflight assistance-a44d-preflight assistance-a44e-preflight assistance-a45a-preflight assistance-a45b-preflight assistance-a45b-recovery-preflight assistance-a45b-postresult-preflight assistance-a45bm1-preflight assistance-a45bm2-preflight publication-audit quality

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

assistance-a42-preflight:
	uv run python scripts/preflight_phase4_a42.py

assistance-a43a-preflight:
	uv run python scripts/preflight_phase4_a43a.py

assistance-a44a-preflight:
	uv run python scripts/preflight_phase4_a44a.py

assistance-a44b-preflight:
	uv run python scripts/preflight_phase4_a44b.py

assistance-a44c-preflight:
	uv run python scripts/preflight_phase4_a44c.py

assistance-a44d-preflight:
	uv run python scripts/preflight_phase4_a44d.py

assistance-a44e-preflight:
	uv run python scripts/preflight_phase4_a44e.py

assistance-a45a-preflight:
	uv run python scripts/preflight_phase4_a45a.py

assistance-a45b-preflight:
	uv run python scripts/preflight_phase4_a45b.py

assistance-a45b-recovery-preflight:
	uv run python scripts/preflight_phase4_a45b_recovery.py

assistance-a45b-postresult-preflight:
	uv run python scripts/preflight_phase4_a45b_postresult.py

assistance-a45bm1-preflight:
	uv run python scripts/preflight_phase4_a45bm1.py

assistance-a45bm2-preflight:
	uv run python scripts/preflight_phase4_a45bm2.py

publication-audit:
	uv run python scripts/audit_publication.py

quality: lint typecheck test data-check retrieval-preflight assistance-preflight assistance-a42-preflight assistance-a43a-preflight assistance-a44a-preflight assistance-a44b-preflight assistance-a44c-preflight assistance-a44d-preflight assistance-a44e-preflight assistance-a45a-preflight assistance-a45b-preflight assistance-a45b-recovery-preflight assistance-a45b-postresult-preflight assistance-a45bm1-preflight assistance-a45bm2-preflight publication-audit
