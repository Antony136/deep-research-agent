"""
FastAPI entry point for the Deep Research Agent.

The existing CLI remains available through app.main.
This module exposes the HTTP API for the React frontend.
"""

import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.observability.tracing import configure_observability


load_dotenv()

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown."""

    configure_observability()
    logger.info("Starting Deep Research Agent API.")

    yield

    logger.info("Stopping Deep Research Agent API.")


app = FastAPI(
    title="Deep Research Agent API",
    description=(
        "API for managing research sessions, reviewing "
        "research decisions, and retrieving final reports."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------
# CORS
# ---------------------------------------------------------

frontend_origins = [
    origin.strip()
    for origin in os.getenv(
        "FRONTEND_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=frontend_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

# ---------------------------------------------------------
# ROUTES
# ---------------------------------------------------------

app.include_router(router)


@app.get("/", tags=["Root"])
def root():
    """Return basic API information."""

    return {
        "service": "Deep Research Agent API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/health",
    }
