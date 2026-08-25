"""HTTP controllers exposed by the application."""

from app.controller.course_material import course_material_router
from app.controller.health import health_router
from app.controller.search import search_router

__all__ = ["course_material_router", "health_router", "search_router"]
