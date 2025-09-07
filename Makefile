# Makefile for SGFA-QMAP-PD project

.PHONY: help install install-dev test test-fast test-coverage test-integration clean lint format type-check

help:  ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install production dependencies
	pip install -r requirements.txt

install-dev:  ## Install development dependencies
	pip install -r requirements-dev.txt

test:  ## Run all tests
	pytest

test-fast:  ## Run only fast unit tests (skip slow/integration tests)
	pytest -m "not slow and not integration"

test-unit:  ## Run only unit tests
	pytest -m "unit"

test-integration:  ## Run only integration tests
	pytest -m "integration"

test-coverage:  ## Run tests with coverage report
	pytest --cov=. --cov-report=html --cov-report=term-missing

test-parallel:  ## Run tests in parallel
	pytest -n auto

lint:  ## Run linting
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

format:  ## Format code with black and isort
	black .
	isort .

format-check:  ## Check if code is properly formatted
	black --check .
	isort --check-only .

type-check:  ## Run type checking with mypy
	mypy --ignore-missing-imports .

clean:  ## Clean up generated files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf dist/
	rm -rf build/

# Development workflow commands
dev-setup: install-dev  ## Set up development environment
	@echo "Development environment set up successfully!"

dev-test: format lint type-check test-fast  ## Run full development test suite

ci-test: test-coverage lint type-check  ## Run CI test suite

# Model-specific commands
test-data:  ## Test data loading modules
	pytest tests/data/ -v

test-analysis:  ## Test analysis modules  
	pytest tests/analysis/ -v

test-utils:  ## Test utility functions
	pytest tests/utils/ -v

# Quick smoke test
smoke-test:  ## Run a quick smoke test to verify basic functionality
	python -c "from data.synthetic import generate_synthetic_data; print('Synthetic data generation works'); data = generate_synthetic_data(); print(f'Generated data with {len(data[\"X_list\"])} views, {data[\"X_list\"][0].shape[0]} subjects')"