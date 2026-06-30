import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from app.api import health_router, router
from app.config import Settings, get_settings
from app.service import RubricService


def create_app(
    *,
    settings: Settings | None = None,
    service: RubricService | None = None,
) -> FastAPI:
    settings = settings or get_settings()
    rubric_service = service or RubricService(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        logging.basicConfig(
            level=getattr(logging, settings.log_level.upper(), logging.INFO),
            format="%(asctime)s %(levelname)s %(name)s %(message)s",
        )
        app.state.settings = settings
        app.state.rubric_service = rubric_service
        await rubric_service.initialize()
        try:
            yield
        finally:
            await rubric_service.close()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="Ingest, store, and retrieve grading rubrics for a separate LLM service.",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )
    app.state.settings = settings
    app.state.rubric_service = rubric_service
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=settings.allowed_origins != ["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(health_router)
    app.include_router(router, prefix=settings.api_v1_prefix)

    @app.get("/", include_in_schema=False)
    async def root() -> RedirectResponse:
        return RedirectResponse(url="/docs")

    return app


app = create_app()
