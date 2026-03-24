# Backend Models

Place model weights in this folder.

## Default path
- `backend/models/model.h5`

## Expected format
- Weights-only `.h5` saved with `model.save_weights(...)`.

## Notes
- Do not commit large model binaries to Git.
- For local development, copy weights into this directory before starting the API.
- Implement your exact training architecture in `backend/models/model_architecture.py`
  in `build_model(...)`.
