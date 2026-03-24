# Medical AI Platform

A modular full-stack platform designed to support computer-aided medical diagnosis by integrating machine learning models, backend services, and an interactive web interface.

The system allows researchers and developers to deploy, manage, and evaluate AI models that analyze medical data such as medical images and clinical attributes to assist in diagnostic predictions.

---

## Objective:

Medical AI Platform aims to provide a flexible environment for integrating artificial intelligence into clinical decision support workflows.

The platform includes:

- AI model integration for diagnostic prediction
- A backend API for model management and inference
- A web-based frontend interface
- Support for computer vision models and structured clinical data
- A modular architecture that allows multiple models to be deployed and evaluated

This project is intended for experimentation, research, and development of intelligent diagnostic support systems.

---

## Backend Quick Start

### 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Add model weights

Place a Keras/TensorFlow `.h5` model file at:

- `backend/models/model.h5`

Implement the training architecture in:

- `backend/models/model_architecture.py` (`build_model` function)

### 3. Run API

```bash
uvicorn backend.main:app --reload
```

### 4. Open frontend upload page

Open in browser:

- `http://127.0.0.1:8000/`

### 5. Main endpoints

- `GET /health` -> backend and model readiness
- `POST /model/reload` -> reload model weights from disk
- `POST /predict` -> send image file using `multipart/form-data` key `file`

Example request:

```bash
curl -X POST http://127.0.0.1:8000/predict \
	-F "file=@/path/to/image.png"
```

Expected response (example):

```json
{
	"filename": "image.png",
	"mask_base64": "<base64_png>",
	"overlay_base64": "<base64_png>",
	"mask_mean": 0.23,
	"mask_coverage": 0.11,
	"width": 512,
	"height": 512
}
```

### 6. Run tests

```bash
pytest -q
```