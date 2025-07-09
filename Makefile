# Tektra AI Assistant - Test Commands
.PHONY: test test-all test-python test-frontend test-node test-coverage test-watch test-quick clean-test

# Default test command
test: test-all

# Run all tests across all components
test-all:
	@echo "🧪 Running all tests for Tektra..."
	@echo "\n📦 Frontend Tests (Vitest)..."
	npm test
	@echo "\n🐍 Python Backend Tests (pytest)..."
	uv run python -m pytest tests/ -v
	@echo "\n🟢 Node.js Extension Tests (Jest)..."
	cd dxt-extension/server && npm test
	@echo "\n✅ All tests completed!"

# Python tests only
test-python:
	@echo "🐍 Running Python tests (with mocked heavy models)..."
	uv run python -m pytest tests/ -v

test-python-unit:
	@echo "🐍 Running Python unit tests..."
	uv run python -m pytest tests/test_unit_*.py tests/test_simple_infrastructure.py -v

test-python-integration:
	@echo "🐍 Running Python integration tests..."
	uv run python -m pytest tests/test_integration_*.py -v

test-python-performance:
	@echo "🐍 Running Python performance tests..."
	uv run python -m pytest tests/test_performance_*.py -v -s

test-python-property:
	@echo "🐍 Running Python property-based tests..."
	uv run python -m pytest tests/test_property_*.py -v

test-python-heavy:
	@echo "🐍 Running Python tests with REAL AI models (requires 4GB+ RAM)..."
	@echo "⚠️  This will download and load large models - may take time!"
	uv run python -m pytest tests/ -v --no-heavy-models=false

test-python-heavy-only:
	@echo "🐍 Running ONLY heavy model tests..."
	@echo "⚠️  This will download and load large models - may take time!"
	uv run python -m pytest tests/ -v -m heavy --no-heavy-models=false

# Frontend tests only
test-frontend:
	@echo "📦 Running frontend tests..."
	npm test

test-frontend-unit:
	@echo "📦 Running frontend unit tests..."
	npm test src/test/unit/

test-frontend-property:
	@echo "📦 Running frontend property-based tests..."
	npm test src/test/property/

# Node.js tests only
test-node:
	@echo "🟢 Running Node.js tests..."
	cd dxt-extension/server && npm test

# Coverage reports for all components
test-coverage:
	@echo "📊 Running tests with coverage reports..."
	@echo "\n📦 Frontend coverage..."
	npm run test:coverage
	@echo "\n🐍 Python coverage..."
	uv run python -m pytest tests/ -v --cov=src --cov-report=html --cov-report=term
	@echo "\n🟢 Node.js coverage..."
	cd dxt-extension/server && npm run test:coverage
	@echo "\n📊 Coverage reports generated:"
	@echo "  - Frontend: coverage/"
	@echo "  - Python: htmlcov/"
	@echo "  - Node.js: dxt-extension/server/coverage/"

# Watch mode for development
test-watch:
	@echo "👀 Starting test watchers..."
	@echo "Run these in separate terminals:"
	@echo "  1. Frontend: npm run test:watch"
	@echo "  2. Python: uv run python -m pytest tests/ -v --watch"
	@echo "  3. Node.js: cd dxt-extension/server && npm run test:watch"

# Quick tests (skip slow/integration/heavy tests)
test-quick:
	@echo "⚡ Running quick tests only..."
	@echo "\n📦 Frontend quick tests..."
	npm test -- --run
	@echo "\n🐍 Python quick tests..."
	uv run python -m pytest tests/ -v -m "not slow and not heavy"
	@echo "\n🟢 Node.js quick tests..."
	cd dxt-extension/server && npm test -- --testTimeout=5000

# Clean test artifacts
clean-test:
	@echo "🧹 Cleaning test artifacts..."
	rm -rf coverage/ htmlcov/ .coverage .pytest_cache/
	rm -rf dxt-extension/server/coverage/
	rm -rf src/test/__snapshots__/
	@echo "✨ Test artifacts cleaned!"

# Individual test file runners
test-file:
	@if [ -z "$(FILE)" ]; then \
		echo "Usage: make test-file FILE=path/to/test.py"; \
		exit 1; \
	fi
	@echo "🧪 Running test file: $(FILE)"
	@if echo "$(FILE)" | grep -q ".py$$"; then \
		uv run python -m pytest "$(FILE)" -v; \
	elif echo "$(FILE)" | grep -q ".js$$\|.ts$$"; then \
		npm test "$(FILE)"; \
	else \
		echo "❌ Unknown file type: $(FILE)"; \
		exit 1; \
	fi

# Help command
help:
	@echo "Tektra Test Commands:"
	@echo "  make test                - Run all tests (skips heavy model tests)"
	@echo "  make test-python         - Run Python tests"
	@echo "  make test-frontend       - Run frontend tests"
	@echo "  make test-node          - Run Node.js tests"
	@echo "  make test-coverage      - Run tests with coverage"
	@echo "  make test-quick         - Run quick tests only (skips slow & heavy)"
	@echo "  make test-watch         - Show watch mode commands"
	@echo "  make test-file FILE=... - Run specific test file"
	@echo "  make clean-test         - Clean test artifacts"
	@echo ""
	@echo "Python-specific:"
	@echo "  make test-python-unit        - Unit tests only"
	@echo "  make test-python-integration - Integration tests only"
	@echo "  make test-python-performance - Performance tests only"
	@echo "  make test-python-property    - Property-based tests only"
	@echo "  make test-python-heavy       - All tests with REAL AI models (4GB+ RAM)"
	@echo "  make test-python-heavy-only  - Only heavy model tests"
	@echo ""
	@echo "Frontend-specific:"
	@echo "  make test-frontend-unit     - Unit tests only"
	@echo "  make test-frontend-property - Property-based tests only"
	@echo ""
	@echo "NOTE: Heavy AI model tests use mocked models by default to prevent"
	@echo "      system overheating. Use test-python-heavy to load real models."