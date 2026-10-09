# AGENTS.md

Guidance for coding agents working in this repository.

## Project

General-domain **text-to-image retrieval** (COMP5523). Python package `tsr` under
`src/`, FastAPI web UI in `web/`, OpenCLIP for embeddings, FAISS for vector search,
SQLite for metadata.

## Environment

- Docker (Python 3.11) is the reproducible target for deployment. Native installs also
  work where PyTorch/OpenCLIP wheels exist (verified on Python 3.14 + torch 2.11).
- Package source lives in `src/`; run modules with `PYTHONPATH=src` or `pip install -e .`.

## Common commands

```bash
# install (dev)
pip install -r requirements-dev.txt && pip install -e .

# build gallery + index (network required, ~1.5 GB)
python -m tsr build --max-images 2000

# run server
python -m tsr serve            # http://localhost:8000

# evaluate
python -m tsr eval --hybrid

# tests + lint (always run before finishing)
pytest
ruff check src tests
```

Docker equivalents:

```bash
docker compose --profile tools run --rm build python -m tsr build --max-images 10000
docker compose up -d
```

## Conventions

- Python 3.10+ syntax, `from __future__ import annotations`, 100-char lines.
- Config is a nested `AttrDict` loaded by `tsr.config.load_config`; add new options to
  `configs/default.yaml` and read them with `cfg.<section>.<key>` / `.get(default)`.
- Never commit `data/`, `.env` or generated indexes (see `.gitignore`).
- Tests that need FAISS/torch use `pytest.importorskip` so the suite degrades gracefully.
- Keep the API response schemas in `src/tsr/schemas.py` in sync with the web UI.

## Architecture notes

- `embedder.py` loads OpenCLIP lazily; do not force-load it on non-search paths.
- `index.py` stores L2-normalised vectors so inner product == cosine similarity; the
  FAISS row -> SQLite id map is `data/index_ids.npy`.
- `db.py` row id is the shared key between FAISS and SQLite; keep insertion order and
  vector order aligned in `pipeline.build_index`.
