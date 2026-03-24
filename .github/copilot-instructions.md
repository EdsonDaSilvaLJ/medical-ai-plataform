# Project Guidelines

## Scope
- This repository is an early-stage Python-first scaffold for a Medical AI Platform.
- Prefer changes in existing areas first: `backend/` and root docs.
- Use `README.md` as the source of truth for project intent and product scope.

## Architecture
- `backend/` is the current implementation area.
- `backend/models/` is reserved for model-related code/assets.
- `backend/api_test.py` exists as a placeholder and should evolve into real API tests.
- Frontend is described in `README.md` but is not present in this workspace yet.

## Build And Test
- There is no pinned Python environment or dependency manifest yet (`requirements.txt`/`pyproject.toml` are absent).
- Before introducing new dependencies, add or update a dependency manifest in the same change.
- When tests exist, prefer running `pytest` from the repository root.
- If tooling is missing, add only the minimum required setup and document it in `README.md`.

## Code Conventions
- Write Python using clear names, small functions, and type hints for new/changed code.
- Keep modules focused: API logic in backend API modules, model logic in `backend/models/`.
- Do not commit large datasets, model binaries, or checkpoint files.
- Preserve current style in touched files and avoid unrelated refactors.

## Documentation Rules
- Link to existing docs instead of duplicating long explanations.
- Keep implementation-facing guidance here; keep product/context narrative in `README.md`.
- If you add new commands, scripts, or required environment variables, update docs in the same PR.