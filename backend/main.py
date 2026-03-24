from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from backend.models.inference import ImageModelService, ModelNotReadyError

app = FastAPI(title="Medical AI Backend", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_service = ImageModelService()
FRONTEND_INDEX_PATH = Path(__file__).resolve().parents[1] / "frontend" / "index.html"

try:
    model_service.load_model()
except FileNotFoundError:
    # API can still start; /health and /predict will indicate model readiness.
    pass
except Exception:
    # API can still start when model file is invalid or cannot be loaded.
    pass


@app.get("/")
def frontend() -> FileResponse:
    if not FRONTEND_INDEX_PATH.exists():
        raise HTTPException(status_code=404, detail="Frontend page not found.")
    return FileResponse(FRONTEND_INDEX_PATH)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model": model_service.status(),
    }


@app.post("/model/reload")
def reload_model() -> dict:
    try:
        model_service.load_model()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {exc}") from exc

    return {
        "message": "Model loaded successfully.",
        "model": model_service.status(),
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="File must be an image.")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        prediction = model_service.predict(image_bytes)
    except ModelNotReadyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    return {
        "filename": file.filename,
        "mask_base64": prediction.mask_base64,
        "overlay_base64": prediction.overlay_base64,
        "mask_mean": prediction.mask_mean,
        "mask_coverage": prediction.mask_coverage,
        "width": prediction.width,
        "height": prediction.height,
    }