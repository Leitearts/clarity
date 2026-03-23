import logging
from contextlib import asynccontextmanager
import uvicorn
from fastapi import FastAPI

from app.api.routes import router as core_router
from app.api.a2a_routes import router as a2a_router
from app.api.impact import router as impact_router
from app.a2a.adapter import A2AAdapter
from app.audit.logger import AuditLogger
from app.config import settings
from app.orchestrator.orchestrator import Orchestrator

logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)


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

PREFIX = f"/api/{settings.api_version}"
app.include_router(core_router,   prefix=PREFIX)
app.include_router(a2a_router,    prefix=PREFIX)
app.include_router(impact_router, prefix=PREFIX)


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
    )
