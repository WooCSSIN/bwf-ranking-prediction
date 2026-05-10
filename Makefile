# ══════════════════════════════════════════════════════════════
# BWF Ranking Prediction — Makefile
# Usage: make <target>
# ══════════════════════════════════════════════════════════════

.PHONY: help install install-dev lint format test train-all clean

# ── Default target ────────────────────────────────────────────
help:
	@echo ""
	@echo "  🏸 BWF Ranking Prediction — Available Commands"
	@echo "  ─────────────────────────────────────────────"
	@echo "  make install       Install core dependencies"
	@echo "  make install-dev   Install all deps including dev tools"
	@echo "  make lint          Run ruff linter"
	@echo "  make format        Run black formatter"
	@echo "  make test          Run pytest with coverage"
	@echo "  make train-all     Train models for all draw × region combos"
	@echo "  make train         Train one model (e.g. make train DRAW=MS REGION=Asia)"
	@echo "  make api           Start FastAPI dev server"
	@echo "  make clean         Remove cache files"
	@echo ""

# ── Setup ─────────────────────────────────────────────────────
install:
	pip install -e .

install-dev:
	pip install -e ".[all]"
	pre-commit install

# ── Code quality ──────────────────────────────────────────────
lint:
	ruff check src/ scripts/ tests/

format:
	black src/ scripts/ tests/ --line-length 100

format-check:
	black src/ scripts/ tests/ --check --line-length 100

# ── Testing ───────────────────────────────────────────────────
test:
	pytest tests/ -v --tb=short --cov=src --cov-report=term-missing

test-fast:
	pytest tests/ -v --tb=short -x

# ── Training (Phase 1) ────────────────────────────────────────
DRAW ?= MS
REGION ?= Global
TRIALS ?= 30

train:
	python scripts/train.py --draw $(DRAW) --region $(REGION)

train-all:
	python scripts/train.py --all

# ── Phase 2: MLOps ────────────────────────────────────────────
tune:
	python scripts/train.py --draw $(DRAW) --region $(REGION) --tune --trials $(TRIALS)

tune-all:
	python scripts/train.py --all --tune --trials $(TRIALS)

compare:
	python scripts/train.py --draw $(DRAW) --region $(REGION) --compare-models

mlflow-ui:
	mlflow ui --host 0.0.0.0 --port 5000

report:
	python scripts/evaluate.py --all

leaderboard:
	python scripts/evaluate.py --leaderboard

promote:
	@echo "Usage: python -c \"from src.models.registry import ModelRegistry; ModelRegistry().promote('$(DRAW)', '$(REGION)', '$(VERSION)')\""

# ── DVC ───────────────────────────────────────────────────────
dvc-init:
	dvc init
	dvc add data/raw/bwf_cleaned.csv
	@echo "✅ DVC initialized. Commit .dvc files to Git."

dvc-pull:
	dvc pull

dvc-push:
	dvc push

# ── API (Phase 3) ─────────────────────────────────────────────
api:
	uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# ── Cleanup ───────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@echo "✅ Clean complete"
