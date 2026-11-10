# INSPECT-RAG developer commands. Research prototype — not for clinical use.
PYTHON ?= python3.11
VENV   ?= .venv
PY     := $(abspath $(VENV))/bin/python
CLI    := cd backend && $(PY) -m app.cli
SUITES ?= experiments/smoke.yaml experiments/multimodal_ablation.yaml
NEURAL_SUITES := experiments/embedding_comparison.yaml experiments/chunking_comparison.yaml \
                 experiments/retrieval_comparison.yaml experiments/reranking_depth.yaml \
                 experiments/generation_comparison.yaml

.DEFAULT_GOAL := help
.PHONY: help setup setup-ml up down migrate test test-neural lint seed-synthetic index-baseline \
        run-baseline run-experiments run-neural-experiments select report serve frontend e2e-prepare e2e clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-24s %s\n", $$1, $$2}'

setup: ## Create the virtualenv, install pinned backend deps and frontend deps
	$(PYTHON) -m venv $(VENV)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r backend/requirements.lock -r backend/requirements-dev.lock
	$(PY) -m pip install --no-deps -e backend
	@command -v npm >/dev/null 2>&1 && (cd frontend && npm ci) || echo "npm not found: skipping frontend dependencies"
	@test -f .env || cp .env.example .env

setup-ml: ## Additionally install PyTorch / SentenceTransformers / Haystack / MLflow (optional)
	$(PY) -m pip install -r backend/requirements-ml.lock

up: ## Start PostgreSQL, Elasticsearch, backend and frontend with Docker Compose
	docker compose up -d --build

down: ## Stop the Docker Compose infrastructure
	docker compose down

migrate: ## Apply database migrations (PostgreSQL deployments)
	cd backend && $(PY) -m alembic upgrade head

test: ## Run backend unit + integration tests (hermetic, CPU-only, synthetic data)
	cd backend && $(PY) -m pytest -m "not neural"

test-neural: ## Run tests that need PyTorch and locally cached model weights
	cd backend && $(PY) -m pytest -m neural

lint: ## Static checks: ruff, mypy, eslint, tsc
	cd backend && $(PY) -m ruff check app tests && $(PY) -m mypy app
	@command -v npm >/dev/null 2>&1 && (cd frontend && npm run lint && npm run typecheck) || echo "npm not found: skipping frontend checks"

seed-synthetic: ## Generate and ingest the SYNTHETIC cohort (fabricated data)
	$(CLI) generate-synthetic
	$(CLI) ingest

index-baseline: ## Build the baseline search indexes (BM25 + CPU-only dense baseline + images)
	$(CLI) reindex

run-baseline: ## Execute the baseline evaluation (CPU-only smoke suite)
	$(CLI) run-experiment experiments/smoke.yaml

run-experiments: ## Execute the configured experiment suites (SUITES="...")
	$(CLI) evaluate $(SUITES)

run-neural-experiments: ## Embedding / chunking / retrieval / reranking / generation suites (needs setup-ml)
	$(CLI) evaluate $(NEURAL_SUITES)

select: ## Apply the model-selection policy to a validation suite (SUITE=smoke)
	$(CLI) select $(or $(SUITE),smoke)

report: ## Generate the Markdown + JSON evaluation report from completed runs
	$(CLI) aggregate-metrics

serve: ## Start the API on http://127.0.0.1:8000
	$(CLI) serve

frontend: ## Start the frontend dev server on http://localhost:5173
	cd frontend && VITE_API_KEY=dev-research-key npm run dev

e2e-prepare: seed-synthetic index-baseline run-baseline ## Prepare data for the Playwright tests

e2e: ## Run the Playwright end-to-end tests (after e2e-prepare; needs `npx playwright install chromium`)
	cd frontend && E2E_PYTHON=$(PY) npx playwright test

clean: ## Remove generated artifacts and synthetic data (never touches data/inspect)
	rm -rf artifacts data/synthetic/* backend/.pytest_cache backend/.mypy_cache backend/.ruff_cache
	touch data/synthetic/.gitkeep
