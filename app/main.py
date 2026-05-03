from contextlib import asynccontextmanager
import uvicorn
from fastapi import FastAPI
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from app.api.limiter import limiter
from app.api.routes import router as core_router
from app.api.a2a_routes import router as a2a_router
from app.api.impact import router as impact_router
from app.a2a.adapter import A2AAdapter
from app.audit.logger import AuditLogger
from app.config import settings
from app.logging_utils import setup_logging
from app.orchestrator.orchestrator import Orchestrator

setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    audit_logger = AuditLogger()
    orchestrator = Orchestrator()
    orchestrator._audit_logger = audit_logger
    a2a_adapter = A2AAdapter(orchestrator)

    app.state.orchestrator = orchestrator
    app.state.audit_logger = audit_logger
    app.state.a2a_adapter = a2a_adapter

    yield


app = FastAPI(
    title="CLARITY",
    description=(
        "Clinical AI Risk Analysis and Transparency sYstem. "
        "Multi-agent clinical risk detection — synthetic data only. No PHI."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

PREFIX = f"/api/{settings.api_version}"
app.include_router(core_router, prefix=PREFIX)
app.include_router(a2a_router, prefix=PREFIX)
app.include_router(impact_router, prefix=PREFIX)


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
    )
