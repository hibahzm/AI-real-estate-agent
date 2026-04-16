"""
main.py — FastAPI application entry point.

This file:
  1. Creates the FastAPI app
  2. Loads the ML model at startup (once, not on every request)
  3. Registers the route handlers
  4. Sets up CORS so the Streamlit frontend can call the API
  5. Provides a /health endpoint for monitoring

To run:
    uv run uvicorn app.main:app --reload --port 8000
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.ml.price_predictor import load_model_and_stats, is_loaded
from app.routes.prediction_route import router as prediction_router
from app.routes.insights_route import router as insights_router
from app.schemas.api_response_schema import HealthCheckResponse

# Configure logging — shows timestamps and log level in the console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager: runs once at startup and once at shutdown.
    We load the ML model here so it's ready before the first request arrives.
    """
    # ── Startup ──────────────────────────────────────────────
    logger.info("🚀 Starting AI Real Estate Agent API...")
    try:
        load_model_and_stats()
        logger.info("✅ ML model and training statistics loaded successfully")
    except FileNotFoundError as e:
        logger.error(f"❌ {e}")
        logger.error("The app will start but /predict will fail until model files are present.")

    yield   # Application runs here

    # ── Shutdown ─────────────────────────────────────────────
    logger.info("Shutting down AI Real Estate Agent API")


# Create the FastAPI app
app = FastAPI(
    title="AI Real Estate Agent API",
    description=(
        "LLM prompt chaining + ML price prediction for residential properties. "
        "Stage 1: extract features from natural language. "
        "Stage 2: interpret ML prediction with market context."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS ─────────────────────────────────────────────────────────────────────
# Allow the Streamlit frontend to call this API from any origin.
# In production, replace "*" with your actual Streamlit Cloud URL.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # TODO: tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Register Routes ───────────────────────────────────────────────────────────
app.include_router(prediction_router, prefix="/api/v1")
app.include_router(insights_router,   prefix="/api/v1")


# ── Health Check ─────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthCheckResponse, tags=["Health"])
async def health_check():
    """
    Simple health check endpoint.
    Used by Docker, Railway, and Streamlit to verify the service is up.

    Returns 200 OK if:
      - The app is running
      - The ML model is loaded
    """
    loaded = is_loaded()
    return HealthCheckResponse(
        status="ok" if loaded else "degraded",
        model_loaded=loaded,
        stats_loaded=loaded,
        message="All systems operational" if loaded else "Model not loaded — run Colab notebook first"
    )


@app.get("/", tags=["Health"])
async def root():
    """Root endpoint — confirms the API is running."""
    return {
        "message": "AI Real Estate Agent API is running",
        "docs": "/docs",
        "health": "/health"
    }