import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from app.config import Settings, get_settings
from app.controller import course_material_router, health_router, search_router
from app.service import CourseMaterialService, SearchService

OPENAPI_TAGS = [
    {
        "name": "health",
        "description": "Service readiness and required infrastructure status.",
    },
    {
        "name": "course-material",
        "description": "Create direct-upload URLs and track course-material processing.",
    },
    {
        "name": "search",
        "description": "Retrieve semantically relevant course-material chunks.",
    },
]


def create_app(
    *,
    settings: Settings | None = None,
    service: CourseMaterialService | None = None,
) -> FastAPI:
    settings = settings or get_settings()
    course_material_service = service or CourseMaterialService(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        logging.basicConfig(
            level=getattr(logging, settings.log_level.upper(), logging.INFO),
            format="%(asctime)s %(levelname)s %(name)s %(message)s",
        )
        app.state.settings = settings
        app.state.course_material_service = course_material_service
        await course_material_service.initialize()
        try:
            yield
        finally:
            await course_material_service.close()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="Store, process, and search course materials uploaded directly to S3.",
        lifespan=lifespan,
        openapi_tags=OPENAPI_TAGS,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        swagger_ui_parameters={
            "displayRequestDuration": True,
            "filter": True,
        },
    )
    app.state.settings = settings
    app.state.course_material_service = course_material_service
    app.state.search_service = SearchService(course_material_service)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=settings.allowed_origins != ["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(health_router)
    app.include_router(course_material_router, prefix=settings.api_v1_prefix)
    app.include_router(search_router, prefix=settings.api_v1_prefix)

    @app.get("/", include_in_schema=False)
    async def root() -> RedirectResponse:
        return RedirectResponse(url="/docs")

    return app


app = create_app()
