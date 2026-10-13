from fastapi import Request
from backend.app.core.model_manager import model_manager, ModelManager


def get_model_manager(request: Request) -> ModelManager:
    """Dependency provider returning singleton ModelManager from app state."""
    return getattr(request.app.state, "model_manager", model_manager)
