from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

DEFAULT_IMAGE_SIZE = (224, 224)
DEFAULT_WEIGHTS_PATH = Path(__file__).resolve().parent / "model.h5"


class ModelNotReadyError(RuntimeError):
    """Raised when inference is requested before the model is loaded."""


class PredictionResult(BaseModel):
    mask_base64: str
    overlay_base64: str
    mask_mean: float
    mask_coverage: float
    width: int
    height: int


class ImageModelService:
    def __init__(self, weights_path: Path | None = None) -> None:
        self.weights_path = weights_path or DEFAULT_WEIGHTS_PATH
        self.device = "gpu" if tf.config.list_physical_devices("GPU") else "cpu"
        self.model: tf.keras.Model | None = None
        self.load_error: str | None = None

    def load_model(self) -> None:
        if not self.weights_path.exists():
            self.model = None
            self.load_error = f"Model weights not found at: {self.weights_path}"
            raise FileNotFoundError(self.load_error)

        try:
            from backend.models.model_architecture import build_model

            model = build_model(input_shape=(DEFAULT_IMAGE_SIZE[0], DEFAULT_IMAGE_SIZE[1], 3))
            model.load_weights(str(self.weights_path))
            self.model = model
            self.load_error = None
        except Exception as exc:
            self.model = None
            self.load_error = (
                "Failed to build architecture and load .h5 weights. "
                "Update backend/models/model_architecture.py to match training model. "
                f"Details: {exc}"
            )
            raise RuntimeError(self.load_error) from exc

    def status(self) -> dict[str, str | bool]:
        if self.model is None:
            return {
                "ready": False,
                "device": self.device,
                "weights_path": str(self.weights_path),
                "error": self.load_error or "Model is not loaded.",
            }

        return {
            "ready": True,
            "device": self.device,
            "weights_path": str(self.weights_path),
            "error": "",
        }

    def _load_image(self, image_bytes: bytes) -> Image.Image:
        try:
            return Image.open(BytesIO(image_bytes)).convert("RGB")
        except UnidentifiedImageError as exc:
            raise ValueError("Invalid image file.") from exc

    def _preprocess(self, image: Image.Image) -> np.ndarray:
        image = image.resize(DEFAULT_IMAGE_SIZE)
        array = np.asarray(image, dtype=np.float32) / 255.0
        return np.expand_dims(array, axis=0)

    def _encode_png_base64(self, image: Image.Image) -> str:
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    def _build_overlay(self, image: Image.Image, binary_mask: np.ndarray) -> Image.Image:
        image_arr = np.asarray(image, dtype=np.float32)
        red_mask = np.zeros_like(image_arr)
        red_mask[..., 0] = 255.0

        alpha = np.expand_dims(binary_mask.astype(np.float32) * 0.45, axis=-1)
        overlay_arr = image_arr * (1.0 - alpha) + red_mask * alpha
        overlay_arr = np.clip(overlay_arr, 0, 255).astype(np.uint8)
        return Image.fromarray(overlay_arr, mode="RGB")

    def _extract_mask(self, outputs: np.ndarray) -> np.ndarray:
        mask = np.asarray(outputs, dtype=np.float32)

        if mask.ndim == 4:
            mask = mask[0]
        if mask.ndim == 3 and mask.shape[-1] > 1:
            mask = np.max(mask, axis=-1)
        elif mask.ndim == 3 and mask.shape[-1] == 1:
            mask = np.squeeze(mask, axis=-1)
        elif mask.ndim != 2:
            mask = np.squeeze(mask)
            if mask.ndim != 2:
                raise RuntimeError("Model output is not compatible with segmentation mask format.")

        if np.min(mask) < 0.0 or np.max(mask) > 1.0:
            mask = 1.0 / (1.0 + np.exp(-mask))

        return np.clip(mask, 0.0, 1.0)

    def predict(self, image_bytes: bytes) -> PredictionResult:
        if self.model is None:
            raise ModelNotReadyError(
                self.load_error
                or "Model is not loaded. Place weights in backend/models/model.h5."
            )

        original_image = self._load_image(image_bytes)
        inputs = self._preprocess(original_image)
        outputs = self.model.predict(inputs, verbose=0)

        if isinstance(outputs, (tuple, list)):
            outputs = outputs[0]

        mask = self._extract_mask(np.asarray(outputs, dtype=np.float32))
        if mask.size == 0:
            raise RuntimeError("Model returned an empty output tensor.")

        width, height = original_image.size
        mask_image = Image.fromarray((mask * 255.0).astype(np.uint8), mode="L")
        mask_image = mask_image.resize((width, height), Image.BILINEAR)

        mask_arr = np.asarray(mask_image, dtype=np.float32) / 255.0
        binary_mask = (mask_arr >= 0.5).astype(np.uint8)
        overlay_image = self._build_overlay(original_image, binary_mask)

        return PredictionResult(
            mask_base64=self._encode_png_base64(mask_image),
            overlay_base64=self._encode_png_base64(overlay_image),
            mask_mean=float(mask_arr.mean()),
            mask_coverage=float(binary_mask.mean()),
            width=width,
            height=height,
        )