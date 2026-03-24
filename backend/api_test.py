from __future__ import annotations

from fastapi.testclient import TestClient

from backend.main import app
from backend.models.inference import ModelNotReadyError, PredictionResult

client = TestClient(app)


class _StubService:
	def __init__(self, should_fail: bool = False) -> None:
		self.should_fail = should_fail

	def status(self) -> dict[str, str | bool]:
		return {
			"ready": not self.should_fail,
			"device": "cpu",
			"weights_path": "backend/models/model.h5",
			"error": "" if not self.should_fail else "not loaded",
		}

	def predict(self, image_bytes: bytes) -> PredictionResult:
		if self.should_fail:
			raise ModelNotReadyError("model not loaded")
		if not image_bytes:
			raise ValueError("empty")
		return PredictionResult(
			mask_base64="ZmFrZS1tYXNr",
			overlay_base64="ZmFrZS1vdmVybGF5",
			mask_mean=0.42,
			mask_coverage=0.17,
			width=224,
			height=224,
		)


def test_health_returns_status(monkeypatch) -> None:
	monkeypatch.setattr("backend.main.model_service", _StubService())

	response = client.get("/health")

	assert response.status_code == 200
	payload = response.json()
	assert payload["status"] == "ok"
	assert payload["model"]["ready"] is True


def test_predict_returns_segmentation_payload(monkeypatch) -> None:
	monkeypatch.setattr("backend.main.model_service", _StubService())

	files = {"file": ("sample.png", b"fake-image-content", "image/png")}
	response = client.post("/predict", files=files)

	assert response.status_code == 200
	payload = response.json()
	assert payload["mask_base64"] == "ZmFrZS1tYXNr"
	assert payload["overlay_base64"] == "ZmFrZS1vdmVybGF5"
	assert payload["mask_mean"] == 0.42
	assert payload["mask_coverage"] == 0.17


def test_predict_rejects_non_image_file() -> None:
	files = {"file": ("sample.txt", b"hello", "text/plain")}
	response = client.post("/predict", files=files)
	assert response.status_code == 415


def test_predict_returns_503_when_model_not_ready(monkeypatch) -> None:
	monkeypatch.setattr("backend.main.model_service", _StubService(should_fail=True))

	files = {"file": ("sample.png", b"fake-image-content", "image/png")}
	response = client.post("/predict", files=files)

	assert response.status_code == 503
