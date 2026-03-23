# CLARITY

**Clinical AI Risk Analysis and Transparency sYstem**

Multi-agent clinical risk detection. Analyzes patient diagnoses, medications, and lab results
to produce a structured risk score with full explainability and an immutable audit trail.

**Synthetic / de-identified data only — no PHI accepted or stored.**

---

## Quick start

```bash
cp .env.example .env          # add CLARITY_ANTHROPIC_API_KEY
make build && make run        # build image and start container
make health                   # verify the service is up
make demo-critical            # run a live critical case
```

---

## Why CLARITY wins on every judging criterion

### AI Factor
Four independent LLM agents each reason about a distinct clinical domain — diagnosis, drug safety,
lab interpretation, and patient context — running in parallel and producing structured JSON.
The LLM is constrained to 1–3 sentence reasoning strings and validated JSON output.
Every AI decision is preserved verbatim in the audit log. Rule-based fallback activates
automatically if the LLM is unavailable — zero downtime.

### Impact
250,000 Americans die from preventable medical errors annually. CLARITY targets the most
dangerous gap: clinicians managing 14 ICU patients cannot manually cross-reference 200+
data points per patient. CLARITY does it in under 3 seconds with a plain-English explanation.
Three audiences: clinicians (second pair of eyes), developers (A2A integration in one afternoon),
regulators (complete audit trail, formula shown verbatim).

### Feasibility
- Zero PHI: synthetic data only, validated at the API edge
- No black box: formula trace in every response, agent contributions as percentages
- Fail-safe: LLM unavailable → deterministic rule-based fallback, automatically
- Compliant: LLM system prompts prohibit prescriptive output; safety guardrail scans every response
- Production-ready: Docker, health checks, non-root container, 512MB memory cap, append-only audit log

---

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/analyze` | Full clinical risk analysis |
| POST | `/api/v1/a2a/invoke` | A2A marketplace invocation |
| GET | `/api/v1/agent-card` | Discovery metadata |
| GET | `/api/v1/health` | Health check |
| GET | `/api/v1/audit/{id}` | Retrieve audit record |
| GET | `/api/v1/audit/case/{id}` | Full case audit history |
| GET | `/api/v1/audit?level=critical` | Filter by risk level |
| GET | `/api/v1/reasoning/{id}` | LLM reasoning trace |
| GET | `/api/v1/impact` | Live usage statistics |
| GET | `/api/v1/a2a/health` | Agent health status |
| GET | `/api/v1/docs` | Swagger UI |

---

## Risk levels

| Level | Score | Action |
|-------|-------|--------|
| LOW | < 0.35 | Routine monitoring |
| MEDIUM | 0.35 – 0.64 | Review within 24h |
| HIGH | 0.65 – 0.84 | Review within 1h |
| CRITICAL | ≥ 0.85 | Immediate intervention |

---

## Risk formula

```
R = (w1 × D + w2 × M + w3 × L) × C
```

Where D = diagnosis score, M = medication score, L = lab score, C = context multiplier.

## Weight presets

| Preset | w1 (Dx) | w2 (Rx) | w3 (Lab) | Use case |
|--------|---------|---------|----------|----------|
| default | 0.40 | 0.35 | 0.25 | General inpatient |
| icu | 0.35 | 0.30 | 0.35 | ICU — labs dominant |
| outpatient | 0.50 | 0.35 | 0.15 | Outpatient — Dx dominant |
| pharmacy_review | 0.25 | 0.60 | 0.15 | Pharmacy — Rx dominant |
| emergency | 0.35 | 0.25 | 0.40 | ED — acute labs dominant |

Set via: `CLARITY_RISK_WEIGHT_PRESET=icu`

---

## Make commands

```bash
make build          # build Docker image
make run            # start container (detached)
make run-dev        # start with hot reload (no Docker)
make stop           # stop container
make test           # run all 39 tests
make test-unit      # unit tests only (fast, no HTTP)
make test-e2e       # end-to-end tests
make test-cov       # coverage report
make demo-critical  # live critical case
make demo-medium    # live medium case
make demo-low       # live low case
make audit-tail     # stream audit log live
make health         # check service health
make logs           # tail container logs
make clean          # remove containers and build artifacts
```

---

## Architecture

```
POST /analyze / POST /a2a/invoke
        │
   Orchestrator (asyncio.gather)
        │
   ┌────┼────┬────────────┐
   │    │    │            │
Diagnosis Drug  Lab    Context
Agent  Agent  Agent   Agent
   │    │    │            │
   └────┴────┴────────────┘
        │
   Risk Engine  ←  R = (w1·D + w2·M + w3·L) × C
        │
   Explainability Builder
        │
   AnalysisResponse + AuditRecord
```

---

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CLARITY_ANTHROPIC_API_KEY` | (required) | Anthropic API key |
| `CLARITY_LLM_MODEL` | `claude-sonnet-4-20250514` | Model |
| `CLARITY_RISK_WEIGHT_PRESET` | `default` | Weight preset |
| `CLARITY_DEBUG` | `false` | Debug logging |
| `CLARITY_AUDIT_LOG_PATH` | `logs/audit.jsonl` | Audit log path |
| `WORKERS` | `4` | Uvicorn workers |

See `.env.example` for the full list.

---

## Project structure

```
clarity/
├── app/
│   ├── main.py                 # FastAPI entrypoint
│   ├── config.py               # Centralised settings
│   ├── api/                    # Route handlers
│   ├── orchestrator/           # Orchestrator
│   ├── agents/                 # 4 agents + base + rule fallbacks
│   ├── llm/                    # LLM service + prompt templates
│   ├── risk/                   # Risk engine + presets + thresholds
│   ├── audit/                  # Logger + explainer
│   ├── a2a/                    # A2A protocol models + adapter
│   ├── models/                 # Pydantic data contracts
│   └── safety/                 # Input validation + output guardrails
├── tests/                      # 39 tests (unit + integration + e2e)
├── data/                       # Synthetic patient cases
├── demo/                       # Demo payload files
├── Dockerfile                  # Multi-stage build
├── docker-compose.yml
└── Makefile
```

---

*CLARITY is a decision-support tool. All findings are informational only.
Final clinical decisions must be made by qualified healthcare professionals.*
