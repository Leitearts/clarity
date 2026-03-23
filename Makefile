.PHONY: build run stop restart logs test lint clean demo health

build:
	docker compose build --no-cache

run:
	docker compose up -d
	@echo "CLARITY running at http://localhost:8000"
	@echo "Docs: http://localhost:8000/docs"

run-dev:
	WORKERS=1 CLARITY_DEBUG=true LOG_LEVEL=debug \
	uvicorn app.main:app --reload --port 8000

stop:
	docker compose down

restart:
	docker compose restart clarity

logs:
	docker compose logs -f clarity

health:
	@curl -s http://localhost:8000/api/v1/health | python -m json.tool

agent-card:
	@curl -s http://localhost:8000/api/v1/agent-card | python -m json.tool

status:
	@docker compose ps

test:
	pytest tests/ -v

test-unit:
	pytest tests/test_agents.py tests/test_risk_engine.py -v

test-e2e:
	pytest tests/test_e2e.py -v

test-cov:
	pytest --cov=app --cov-report=term-missing --cov-report=html

demo-critical:
	@curl -s -X POST http://localhost:8000/api/v1/a2a/invoke \
	  -H "Content-Type: application/json" \
	  -d @demo/case_critical.json | python -m json.tool

demo-medium:
	@curl -s -X POST http://localhost:8000/api/v1/a2a/invoke \
	  -H "Content-Type: application/json" \
	  -d @demo/case_medium.json | python -m json.tool

demo-low:
	@curl -s -X POST http://localhost:8000/api/v1/a2a/invoke \
	  -H "Content-Type: application/json" \
	  -d @demo/case_low.json | python -m json.tool

audit-tail:
	@tail -f logs/audit.jsonl | python -c \
	  "import sys,json; [print(json.dumps(json.loads(l),indent=2)) for l in sys.stdin]"

audit-critical:
	@curl -s "http://localhost:8000/api/v1/audit?level=critical" | python -m json.tool

lint:
	ruff check app/ tests/

clean:
	docker compose down --rmi local --volumes
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -delete
	rm -rf .pytest_cache htmlcov .coverage
