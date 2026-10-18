"""
FastAPI Main Application Entry Point.
Provides endpoints for health check, metadata, prediction, and lifestyle simulation.
"""

import sys
import os
from typing import Dict, Any
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

# Automatically ensure project root and backend dir are in sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
BACKEND_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

for path in [PROJECT_ROOT, BACKEND_ROOT]:
    if path not in sys.path:
        sys.path.insert(0, path)

try:
    from backend.app.config import settings
    from backend.app.schemas import (
        AssessmentInput,
        PredictionResponse,
        SimulationRequest,
        SimulationResponse
    )
    from backend.app.model_service import model_service
except ImportError:
    from app.config import settings
    from app.schemas import (
        AssessmentInput,
        PredictionResponse,
        SimulationRequest,
        SimulationResponse
    )
    from app.model_service import model_service


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Educational cross-sectional diabetes proxy classifier and association-based input scenarios."
)

# CORS middleware for Next.js frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["System"])
@app.get("/api/health", tags=["System"])
def health_check() -> Dict[str, Any]:
    """Health check endpoint confirming API status and model artifact loading."""
    return {
        "status": "healthy",
        "app": settings.app_name,
        "version": settings.app_version,
        "model_loaded": model_service.pipeline is not None
    }


@app.get("/api/metadata", tags=["Model"])
def get_model_metadata() -> Dict[str, Any]:
    """Retrieve model training metrics, feature list, and population statistics."""
    try:
        return model_service.get_metadata()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch metadata: {str(e)}"
        )


@app.post("/api/predict", response_model=PredictionResponse, tags=["Inference"])
def predict_risk(payload: AssessmentInput) -> PredictionResponse:
    """
    Calculate diabetes risk probability, classification category, and local SHAP feature drivers.
    """
    try:
        result = model_service.predict(payload)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Inference error: {str(e)}"
        )


@app.post("/api/simulate", response_model=SimulationResponse, tags=["Simulation"])
def simulate_lifestyle_impact(payload: SimulationRequest) -> SimulationResponse:
    """
    Simulate impact of proposed lifestyle modifications (weight loss, waist reduction, diet, exercise)
    on risk score.
    """
    try:
        result = model_service.simulate(payload)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Simulation error: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app" if os.path.exists("app/main.py") else "backend.app.main:app", host="0.0.0.0", port=8000, reload=True)
