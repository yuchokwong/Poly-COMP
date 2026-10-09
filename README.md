# Text-to-Image Retrieval (COMP5523)

A general-domain **text-to-image retrieval** system built with **OpenCLIP**, **FAISS**
and a **FastAPI web UI**. Type a natural-language query and get a ranked list of the
most relevant images from a gallery of ~10,000 images.

```
query ──► OpenCLIP text encoder ──► FAISS cosine search ──► ranked images (+captions)
                                            ▲
                    OpenCLIP image embeddings (pre-computed)
```

## Features

- **Cross-modal representation** – OpenCLIP (`ViT-B-32`, `laion2b_s34b_b79k`) maps text
  and images into a shared embedding space.
- **Similarity search & indexing** – L2-normalised embeddings in a FAISS index
  (`flat`, `ivf` or `hnsw`) for cosine similarity; approximate search scales to large
  galleries.
- **Hybrid retrieval** – optional fusion of dense CLIP scores with SQLite FTS5 BM25
  keyword scores (`hybrid_alpha`).
- **Prompt templates / query expansion** – optional prompt ensembling for text queries.
- **Metadata store** – SQLite (`gallery.db`) with captions, splits and an FTS5 index.
- **Web UI** – clean, responsive interface to type queries and browse ranked results
  with scores, captions and a lightbox.
- **Evaluation** – built-in `Precision@K`, `Recall@K`, `mAP`, `MRR` and median rank,
  with an automatically generated or custom evaluation set.
- **Deployable** – Docker + docker-compose (CPU or GPU), config via `.env`, persistent
  volumes for data and model cache.

## Quick start (Docker, recommended)

Docker gives a reproducible environment for deployment (Python 3.11) regardless of the
host Python version.

```bash
# 1. configure
cp .env.example .env          # optional; edit as needed

# 2. build the gallery (~10k images) and the index (downloads ~1.5 GB once)
docker compose --profile tools run --rm build python -m tsr build --max-images 10000

# 3. run the web server
docker compose up -d
# open http://localhost:8000
```

The first search loads the OpenCLIP weights into the `openclip_cache` volume, so later
starts are fast. All gallery/index artifacts live in `./data` on the host.

### GPU

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml --profile tools \
  run --rm build python -m tsr build
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
```

## Native installation

Works with Python 3.10–3.14 where PyTorch/OpenCLIP wheels are available (verified on
this machine with Python 3.14 + torch 2.11). If your platform lacks wheels, use Docker.

```bash
python -m venv .venv && . .venv/Scripts/activate     # Windows
pip install -r requirements-dev.txt
pip install -e .
python -m tsr build --max-images 10000
python -m tsr serve
```

## CLI

| Command | Description |
| --- | --- |
| `tsr download` | Download/collect images, normalise them and write `manifest.jsonl` |
| `tsr build-index` | Embed images and build the FAISS index + SQLite database |
| `tsr build` | `download` + `build-index` + `make-eval` |
| `tsr make-eval` | Generate evaluation queries from gallery captions |
| `tsr eval` | Evaluate retrieval (P@K, R@K, mAP, MRR) and write `eval_report.json` |
| `tsr serve` | Run the FastAPI web server |
| `tsr stats` | Print gallery/index statistics |

Useful flags: `--max-images`, `--source coco|coco_hf|local`, `--config`, `--hybrid`,
`--queries`, `--top-k`, `--log-level`.

## Data

The default source is **MS-COCO 2017** (captions + images), which suits a general
domain and provides human captions that double as ground truth. Only the annotation
archive is downloaded in full; images are fetched individually and resized to
`image_max_side` (default 512 px) at JPEG quality 85, keeping the whole gallery under
2 GB.

The official COCO image server is often very slow. For a much faster download use the
HuggingFace webdataset mirror:

```bash
python -m tsr download --source coco_hf --max-images 10000
```

`coco_hf` streams COCO (Karpathy split) tars from
`undefined443/coco-karpathy-wds` (override with `gallery.hf_repo`). It is typically
~50x faster than the official server and each sample carries its captions, so no
annotation archive is downloaded.

To use your own images, set `gallery.source: local` and put files in
`data/local_images`. Captions are read from `captions.json`, sidecar `.txt` files, or
the filename.

> Respect dataset licenses and avoid collecting or exposing personal data.

## Evaluation

Relevance criterion for the general domain: *an image is relevant to a query if it is
the image the query caption was written for* (single-relevant ground truth, the
standard COCO retrieval setup). `tsr make-eval` samples captions from the gallery; you
can also hand-label queries in `eval/queries.json` (see `eval/queries.example.json`).

```bash
python -m tsr eval                       # dense CLIP
python -m tsr eval --hybrid              # CLIP + BM25
```

Results are printed and saved to `data/eval_report.json`, including per-query ranks and
per-split breakdowns for failure analysis. Metrics: `Precision@K`, `Recall@K`,
`nDCG@K`, `Recall@R`, `mAP`, `MRR` and median rank.

## API

| Endpoint | Description |
| --- | --- |
| `GET /` | Web UI |
| `GET /api/search?q=...&k=12&hybrid=false` | Ranked results |
| `GET /api/similar/{id}?k=12` | Image-to-image (visually similar) search |
| `GET /api/image/{id}` | Full image file |
| `GET /api/thumbnail/{id}?size=256` | Cached thumbnail (fast grid loading) |
| `GET /api/random?n=12` | Random gallery sample |
| `GET /api/stats` | Gallery/model/index info |
| `GET /api/health` | Health probe |

## Configuration

`configs/default.yaml` holds defaults; environment variables prefixed `TSR_` override
them (nested keys use `__`, e.g. `TSR_MODEL__NAME=ViT-B-16`). See `.env.example`.

Key options: `model.name/pretrained`, `index.type` (`auto|flat|ivf|hnsw`),
`search.hybrid`, `search.use_prompt_ensemble`, `search.prompt_templates`,
`gallery.max_images`, `gallery.image_max_side`, `gallery.dedup`
(`sha1|phash`), `gallery.cache_tars`.

Swapping to a stronger backbone is a one-line change, e.g.
`TSR_MODEL__NAME=ViT-L-14 TSR_MODEL__PRETRAINED=laion2b_s32b_b82k` (rebuild the index
afterwards).

## Project layout

```
configs/default.yaml     configuration defaults
src/tsr/                 Python package
  config.py              config loading + path helpers
  embedder.py            OpenCLIP wrapper
  index.py               FAISS vector index
  db.py                  SQLite metadata + FTS5
  sources.py             COCO / local gallery sources
  gallery.py             download + resize + manifest
  evaluate.py            retrieval metrics
  pipeline.py            build-index / eval pipelines
  api.py                 FastAPI app + retrieval service
  cli.py                 command-line interface
web/                     web UI (HTML/CSS/JS)
eval/                    example evaluation queries
tests/                   unit tests
data/                    runtime artifacts (gitignored)
```

## Testing & linting

```bash
pip install -r requirements-dev.txt
pytest
ruff check src tests
```

## Deployment notes

- Build the data once (`data/` volume) and ship it with the image or mount it on the
  target host.
- Scale-up path: keep SQLite/FAISS for ≤ ~1M vectors, or swap `index.py`/`db.py` for
  Qdrant/pgvector and object storage (S3/MinIO) without touching the API/UI.
- Set `TSR_APP__CORS_ORIGINS` and a reverse proxy (nginx/Caddy) for production.
