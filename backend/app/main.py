from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from backend.app.config import settings
from backend.app.core.exceptions import InterviewSenseException
from backend.app.core.model_manager import model_manager
from backend.app.api.routes import router as api_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager: sets up and tears down models."""
    model_manager.load_models()
    app.state.model_manager = model_manager
    yield
    model_manager.unload_models()


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Speech and delivery analysis engine for interview communication.",
    lifespan=lifespan,
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(InterviewSenseException)
async def interview_sense_exception_handler(request: Request, exc: InterviewSenseException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "detail": {
                "error_code": exc.error_code,
                "message": exc.message,
            }
        },
    )


# Attach API Routes
app.include_router(api_router, prefix=settings.API_V1_PREFIX)


@app.get("/", include_in_schema=False)
async def root():
    return {
        "message": f"Welcome to {settings.PROJECT_NAME} API. Visit /docs for documentation.",
        "version": settings.VERSION,
        "health_check": f"{settings.API_V1_PREFIX}/health",
    }
