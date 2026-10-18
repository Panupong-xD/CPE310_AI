"""
Backend configuration and artifact paths.
"""

import sys
import os
from pydantic import BaseModel

# Automatically ensure project root is in sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
BACKEND_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

for path in [PROJECT_ROOT, BACKEND_ROOT]:
    if path not in sys.path:
        sys.path.insert(0, path)


class Settings(BaseModel):
    app_name: str = "NHANES Diabetes Risk Assessment API"
    app_version: str = "1.0.0"
    debug: bool = False
    
    # Path resolution: find ml_pipeline/artifacts
    base_dir: str = PROJECT_ROOT
    artifacts_dir: str = os.path.join(PROJECT_ROOT, "ml_pipeline", "artifacts")
    
    pipeline_path: str = os.path.join(artifacts_dir, "model_pipeline.joblib")
    explainer_path: str = os.path.join(artifacts_dir, "shap_explainer.joblib")
    metadata_path: str = os.path.join(artifacts_dir, "metadata.json")
    
    # CORS Origins
    cors_origins: list[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://localhost:8000",
        "*"
    ]


settings = Settings()
