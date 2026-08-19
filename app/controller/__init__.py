"""HTTP controllers exposed by the application."""

from app.controller.api import health_router, rubrics_router, search_router

__all__ = ["health_router", "rubrics_router", "search_router"]
