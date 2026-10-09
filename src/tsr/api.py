"""FastAPI application: retrieval service + web UI."""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from tsr.config import (
    AttrDict,
    cache_dir,
    db_path,
    ids_path,
    images_dir,
    index_path,
    load_config,
    web_dir,
)
from tsr.db import GalleryDB, ImageRecord
from tsr.embedder import Embedder
from tsr.index import VectorIndex
from tsr.logging_utils import get_logger, setup_logging
from tsr.schemas import (
    HealthResponse,
    SearchResponse,
    SearchResult,
    StatsResponse,
)

logger = get_logger(__name__)


class RetrievalService:
    """Loads the gallery and exposes text->image search."""

    def __init__(self, cfg: AttrDict) -> None:
        self.cfg = cfg
        self.db: GalleryDB | None = None
        self.index: VectorIndex | None = None
        self.embedder: Embedder | None = None
        self.ready = False
        self.load_error: str | None = None
        self._query_cache: dict[str, np.ndarray] = {}
        self._translator = None
        self._cross_encoder = None
        self._load()

    def _load(self) -> None:
        idx_file = index_path(self.cfg)
        ids_file = ids_path(self.cfg)
        db_file = db_path(self.cfg)
        if not idx_file.exists() or not db_file.exists():
            self.load_error = (
                "No index found. Build the gallery first: `tsr build` "
                "(or mount a prepared data/ directory)."
            )
            logger.warning(self.load_error)
            return
        try:
            self.db = GalleryDB(db_file)
            self.index = VectorIndex.load(
                idx_file,
                ids_file,
                index_type=self.cfg.index.type,
                nprobe=int(self.cfg.index.get("nprobe", 32)),
                ef_search=int(self.cfg.index.get("ef_search", 64)),
            )
            self.embedder = Embedder(self.cfg)
            if self.embedder.dim != self.index.dim:
                raise ValueError(
                    f"Model embedding dim ({self.embedder.dim}) does not match the "
                    f"index dim ({self.index.dim}). Rebuild the index: `tsr build-index`."
                )
            self.ready = True
            logger.info(
                "Retrieval service ready: %d vectors, index=%s",
                self.index.size,
                self.cfg.index.type,
            )
        except Exception as exc:  # noqa: BLE001
            self.load_error = f"Failed to load index: {exc}"
            logger.exception(self.load_error)

    # ------------------------------------------------------------------ #
    def _require_ready(self) -> None:
        if not self.ready or self.db is None or self.index is None:
            raise HTTPException(status_code=503, detail=self.load_error or "Service not ready")

    def _maybe_translate(self, query: str) -> str:
        if not bool(self.cfg.search.get("translate", False)):
            return query
        from tsr.translate import Translator, has_cjk

        if not has_cjk(query):
            return query
        if self._translator is None:
            self._translator = Translator(self.cfg.search.get("translator_model"))
        translated = self._translator.translate(query)
        logger.info("Translated query %r -> %r", query, translated)
        return translated or query

    def _encode_query(self, query: str) -> np.ndarray:
        cached = self._query_cache.get(query)
        if cached is not None:
            return cached
        assert self.embedder is not None
        text = self._maybe_translate(query)
        templates = self.cfg.search.prompt_templates or ["{}"]
        use_ensemble = (
            bool(self.cfg.search.get("use_prompt_ensemble", False)) and len(templates) > 1
        )
        if use_ensemble:
            vec = self.embedder.encode_text_with_templates(text, templates)
        else:
            vec = self.embedder.encode_texts([text])
        if len(self._query_cache) >= 512:
            self._query_cache.clear()
        self._query_cache[query] = vec
        return vec

    def _rerank(
        self,
        query: str,
        combined: dict[int, float],
        vec_map: dict[int, float],
        qvec: np.ndarray,
        mode: str,
    ) -> dict[int, float]:
        assert self.db is not None and self.embedder is not None
        n = int(self.cfg.search.get("rerank_top_n", 60))
        alpha = float(self.cfg.search.get("rerank_alpha", 0.5))
        top_ids = [p for p, _ in sorted(combined.items(), key=lambda kv: kv[1], reverse=True)[:n]]
        records = self.db.get_many(top_ids)
        caption_lists = [records[p].captions if p in records else [] for p in top_ids]

        if mode == "caption":
            from tsr.rerank import caption_match_scores

            scores = caption_match_scores(self.embedder, qvec, caption_lists)
        else:  # cross-encoder
            if self._cross_encoder is None:
                from tsr.rerank import CrossEncoderReranker

                self._cross_encoder = CrossEncoderReranker(self.cfg.search.get("rerank_model"))
            docs = [" ".join(caps) for caps in caption_lists]
            scores = self._cross_encoder.score(query, docs)

        out = dict(combined)
        for row_id, rerank_score in zip(top_ids, scores, strict=False):
            img = vec_map.get(row_id, 0.0)
            out[row_id] = (1 - alpha) * img + alpha * float(rerank_score)
        return out

    def search(
        self, query: str, top_k: int, hybrid: bool | None = None
    ) -> SearchResponse:
        self._require_ready()
        assert self.index is not None and self.db is not None and self.embedder is not None
        started = time.perf_counter()

        templates = self.cfg.search.prompt_templates or ["{}"]
        use_ensemble = bool(self.cfg.search.get("use_prompt_ensemble", False)) and len(templates) > 1
        qvec = self._encode_query(query)

        # over-fetch when merging with keyword scores
        hybrid = self.cfg.search.hybrid if hybrid is None else hybrid
        fetch_k = max(top_k * 3, top_k) if hybrid else top_k
        vec_scores, ids = self.index.search(qvec, fetch_k)
        vec_map = {int(i): float(s) for s, i in zip(vec_scores, ids, strict=False)}

        keyword_map: dict[int, float] = {}
        if hybrid:
            for pk, score in self.db.bm25_search(query, fetch_k):
                keyword_map[pk] = score

        combined = dict(vec_map)
        if hybrid and keyword_map:
            vmax = max(vec_map.values()) or 1.0
            kmax = max(keyword_map.values()) or 1.0
            alpha = float(self.cfg.search.get("hybrid_alpha", 0.7))
            all_ids = set(vec_map) | set(keyword_map)
            for pk in all_ids:
                v = vec_map.get(pk, 0.0) / vmax
                k = keyword_map.get(pk, 0.0) / kmax
                combined[pk] = alpha * v + (1 - alpha) * k

        rerank_mode = str(self.cfg.search.get("rerank", "none")).lower()
        if rerank_mode != "none" and combined:
            combined = self._rerank(query, combined, vec_map, qvec, rerank_mode)

        ranked = sorted(combined.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        records = self.db.get_many([pk for pk, _ in ranked])

        results: list[SearchResult] = []
        for rank, (pk, score) in enumerate(ranked, start=1):
            rec = records.get(pk)
            if rec is None:
                continue
            results.append(self._to_result(rank, pk, score, rec, vec_map, keyword_map))

        latency = (time.perf_counter() - started) * 1000.0
        return SearchResponse(
            query=query,
            expanded_queries=[t.format(query) for t in templates] if use_ensemble else [query],
            hybrid=bool(hybrid),
            latency_ms=round(latency, 2),
            count=len(results),
            results=results,
        )

    def _to_result(
        self,
        rank: int,
        pk: int,
        score: float,
        rec: ImageRecord,
        vec_map: dict[int, float],
        keyword_map: dict[int, float],
    ) -> SearchResult:
        return SearchResult(
            rank=rank,
            id=pk,
            image_id=rec.image_id,
            score=round(float(score), 6),
            vector_score=round(vec_map[pk], 6) if pk in vec_map else None,
            keyword_score=round(keyword_map[pk], 6) if pk in keyword_map else None,
            url=f"/api/image/{pk}",
            captions=rec.captions[:5],
            category=rec.category,
            split=rec.split,
            width=rec.width,
            height=rec.height,
        )

    def get_record(self, pk: int) -> ImageRecord | None:
        if self.db is None:
            return None
        return self.db.get(pk)

    def similar(self, pk: int, top_k: int) -> SearchResponse:
        """Image-to-image search: nearest neighbours of a gallery image."""
        self._require_ready()
        assert self.index is not None and self.db is not None
        try:
            qvec = self.index.reconstruct([pk])
        except RuntimeError as exc:
            raise HTTPException(status_code=501, detail=str(exc)) from exc
        if qvec.shape[0] == 0:
            raise HTTPException(status_code=404, detail="Image not in index")
        started = time.perf_counter()
        scores, ids = self.index.search(qvec, top_k + 1)
        vec_map = {int(i): float(s) for s, i in zip(scores, ids, strict=False)}
        ranked = [
            (int(i), float(s))
            for s, i in zip(scores, ids, strict=False)
            if int(i) != pk
        ][:top_k]
        records = self.db.get_many([p for p, _ in ranked])
        base = self.db.get(pk)
        results: list[SearchResult] = []
        for rank, (row_id, score) in enumerate(ranked, start=1):
            rec = records.get(row_id)
            if rec is None:
                continue
            results.append(self._to_result(rank, row_id, score, rec, vec_map, {}))
        latency = (time.perf_counter() - started) * 1000.0
        return SearchResponse(
            query=base.image_id if base else str(pk),
            expanded_queries=[],
            hybrid=False,
            latency_ms=round(latency, 2),
            count=len(results),
            results=results,
        )

    def thumbnail(self, pk: int, size: int) -> Path | None:
        """Return a cached JPEG thumbnail path for an image, creating it on demand."""
        if self.db is None:
            return None
        rec = self.db.get(pk)
        if rec is None:
            return None
        src = images_dir(self.cfg) / rec.path
        if not src.exists():
            return None
        dest = cache_dir(self.cfg) / "thumbs" / str(size) / f"{pk}.jpg"
        if dest.exists():
            return dest
        from PIL import Image

        dest.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(src) as img:
            img = img.convert("RGB")
            img.thumbnail((size, size), Image.LANCZOS)
            img.save(dest, "JPEG", quality=80, optimize=True)
        return dest

    def stats(self) -> StatsResponse:
        num_images = self.db.count() if self.db else 0
        index_type = self.index.index_type if self.index else str(self.cfg.index.type)
        return StatsResponse(
            num_images=num_images,
            index_size=self.index.size if self.index else 0,
            index_type=str(index_type),
            model=str(self.cfg.model.name),
            pretrained=str(self.cfg.model.pretrained),
            embedding_dim=self.index.dim if self.index else 0,
            device=self.embedder.device if self.embedder else "unknown",
            ready=self.ready,
            data_dir=str(self.cfg.paths.data_dir),
        )


def create_app(cfg: AttrDict | None = None) -> FastAPI:
    cfg = cfg or load_config()
    setup_logging()

    from tsr.settings import apply_persisted

    apply_persisted(cfg)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.cfg = cfg
        app.state.service = RetrievalService(cfg)
        yield

    app = FastAPI(title=cfg.app.title, version="0.1.0", lifespan=lifespan)

    origins = list(cfg.app.get("cors_origins", ["*"]))
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def service() -> RetrievalService:
        return app.state.service

    @app.get("/api/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        svc = service()
        return HealthResponse(status="ok", ready=svc.ready)

    @app.get("/api/stats", response_model=StatsResponse)
    def stats() -> StatsResponse:
        return service().stats()

    @app.get("/api/settings")
    def settings_get() -> dict:
        from tsr.settings import get_settings

        return {"fields": get_settings(cfg)}

    @app.put("/api/settings")
    def settings_put(updates: dict) -> dict:
        from tsr.settings import apply_settings, get_settings, save_settings

        try:
            apply_settings(cfg, updates)
        except (ValueError, TypeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        save_settings(cfg)
        return {"fields": get_settings(cfg)}

    @app.get("/api/search", response_model=SearchResponse)
    def search(
        q: str = Query(..., min_length=1, description="Natural-language query"),
        k: int = Query(default=12, ge=1, le=100),
        hybrid: bool | None = Query(default=None),
    ) -> SearchResponse:
        cfg_k = int(cfg.app.max_top_k)
        return service().search(q.strip(), min(k, cfg_k), hybrid=hybrid)

    @app.get("/api/image/{pk}")
    def image(pk: int):
        rec = service().get_record(pk)
        if rec is None:
            raise HTTPException(status_code=404, detail="Image not found")
        root = images_dir(cfg).resolve()
        file_path = (root / rec.path).resolve()
        if root not in file_path.parents and file_path != root:
            raise HTTPException(status_code=400, detail="Invalid image path")
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Image file missing")
        return FileResponse(file_path, media_type="image/jpeg")

    @app.get("/api/thumbnail/{pk}")
    def thumbnail(pk: int, size: int = Query(default=256, ge=32, le=1024)):
        thumb = service().thumbnail(pk, size)
        if thumb is None:
            raise HTTPException(status_code=404, detail="Image not found")
        return FileResponse(thumb, media_type="image/jpeg")

    @app.get("/api/similar/{pk}", response_model=SearchResponse)
    def similar(pk: int, k: int = Query(default=12, ge=1, le=100)) -> SearchResponse:
        return service().similar(pk, min(k, int(cfg.app.max_top_k)))

    @app.get("/api/random")
    def random_sample(n: int = Query(default=12, ge=1, le=50)) -> dict:
        svc = service()
        svc._require_ready()
        assert svc.db is not None
        import random

        records = svc.db.iter_all()
        sample = random.sample(records, min(n, len(records))) if records else []
        return {
            "results": [
                {
                    "id": r.id,
                    "image_id": r.image_id,
                    "url": f"/api/image/{r.id}",
                    "captions": r.captions[:3],
                }
                for r in sample
            ]
        }

    @app.exception_handler(FileNotFoundError)
    def _not_found(_request, exc: FileNotFoundError):
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    static_dir = web_dir(cfg)
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

        @app.get("/")
        def index():
            return FileResponse(static_dir / "index.html")
    else:  # pragma: no cover
        logger.warning("Web directory not found: %s", static_dir)

    return app


# Module-level app for `uvicorn tsr.api:app`
app = create_app()
