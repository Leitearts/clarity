# Contributing to CLARITY

## Development setup

```bash
git clone https://github.com/your-org/clarity.git
cd clarity
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
cp .env.example .env   # add CLARITY_ANTHROPIC_API_KEY
```

## Running tests

```bash
pytest tests/ -v                        # all 46 tests
pytest tests/test_agents.py -v          # unit tests only
pytest tests/test_risk_engine.py -v     # risk engine only
pytest tests/test_e2e.py -v             # end-to-end only
pytest --cov=app --cov-report=term-missing
```

All tests run without a network connection or API key (LLM is mocked).

## Adding a new agent

1. Create `app/agents/my_agent.py` — subclass `BaseAgent`, set `agent_type`, implement `_analyze`.
2. Create `app/agents/_my_agent_rules.py` — deterministic fallback logic.
3. Add `AgentType.MY_AGENT` to `app/models/agent.py`.
4. Register in `app/orchestrator/orchestrator.py` — add to `asyncio.gather`.
5. Add weight entry in `app/risk/engine.py` weight map.
6. Write tests in `tests/test_agents.py`.

## Adding a new A2A capability

1. Add a handler method to `app/a2a/adapter.py` (`_handle_my_capability`).
2. Register in `adapter._capability_router`.
3. Add a `A2ACapability` entry to `AgentCard.capabilities` in `app/a2a/models.py`.

## Code standards

- All public functions have docstrings.
- All inter-component boundaries use Pydantic models — no raw dicts.
- No `os.environ` calls outside `app/config.py`.
- No `print()` in production code — use `logging`.
- New agents must have at least 4 unit tests covering: normal case, edge case, empty input, response structure.

## Commit style

```
feat: add imaging analysis agent
fix: clamp context multiplier to [0.5, 1.5]
test: add edge case for empty medication list
docs: update weight preset table in README
```
