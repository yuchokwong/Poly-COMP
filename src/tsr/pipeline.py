"""Indexing and evaluation pipelines (used by the CLI)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from tsr.config import (
    AttrDict,
    cache_dir,
    db_path,
    eval_queries_path,
    ids_path,
    images_dir,
    index_path,
    report_path,
)
from tsr.db import GalleryDB
from tsr.embedder import Embedder
from tsr.evaluate import (
    bootstrap_ci,
    build_eval_queries_from_manifest,
    evaluate_queries,
    load_queries,
    paired_bootstrap_test,
    per_query_metrics,
    save_queries,
)
from tsr.gallery import read_manifest
from tsr.index import VectorIndex
from tsr.logging_utils import get_logger

logger = get_logger(__name__)


def build_index(cfg: AttrDict) -> int:
    """Embed every gallery image and build the FAISS index + metadata DB."""
    manifest = read_manifest(cfg)
    if not manifest:
        raise RuntimeError("Manifest is empty. Run `tsr download` first.")

    images_root = images_dir(cfg)
    embedder = Embedder(cfg)
    batch_size = int(cfg.model.batch_size)

    db = GalleryDB(db_path(cfg))
    db.clear()

    all_vectors: list[np.ndarray] = []
    all_ids: list[int] = []

    for start in tqdm(range(0, len(manifest), batch_size), desc="embedding"):
        batch = manifest[start : start + batch_size]
        paths = [images_root / rec["path"] for rec in batch]
        vectors = embedder.encode_images(paths)
        rows = []
        kept_vectors = []
        for rec, vec in zip(batch, vectors, strict=False):
            if not np.any(vec):  # failed image produced a zero vector
                continue
            rows.append(rec)
            kept_vectors.append(vec)
        if not rows:
            continue
        ids = db.insert_many(rows)
        all_vectors.append(np.asarray(kept_vectors, dtype=np.float32))
        all_ids.extend(ids)

    if not all_vectors:
        raise RuntimeError("No images were embedded successfully.")

    matrix = np.concatenate(all_vectors, axis=0)
    ids_arr = np.asarray(all_ids, dtype=np.int64)
    logger.info("Embedded %d images (dim=%d).", matrix.shape[0], matrix.shape[1])

    index_type = str(cfg.index.type)
    if index_type == "auto":
        # Exact flat search stays cheap in RAM (512B/vector) and latency well
        # under 100 ms up to a few hundred thousand images, so only switch to an
        # approximate index once it actually pays off.
        n = matrix.shape[0]
        flat_max = int(cfg.index.get("auto_flat_max", 300_000))
        ivf_max = int(cfg.index.get("auto_ivf_max", 1_000_000))
        if n <= flat_max:
            index_type = "flat"
        elif n <= ivf_max:
            index_type = "ivf"
        else:
            index_type = "pq"
        logger.info("index.type=auto -> '%s' for %d vectors.", index_type, n)

    index = VectorIndex(
        dim=matrix.shape[1],
        index_type=index_type,
        nlist=int(cfg.index.get("nlist", 1024)),
        nprobe=int(cfg.index.get("nprobe", 32)),
        m=int(cfg.index.get("m", 32)),
        ef_search=int(cfg.index.get("ef_search", 64)),
    )
    index.add(matrix, ids_arr)
    index.save(index_path(cfg), ids_path(cfg))
    logger.info("Index saved to %s (%d vectors).", index_path(cfg), index.size)
    db.close()
    return index.size


def pregenerate_thumbnails(cfg: AttrDict, size: int = 480) -> int:
    """Pre-render cached thumbnails so the first UI request is not blocked."""
    from PIL import Image

    db = GalleryDB(db_path(cfg))
    images_root = images_dir(cfg)
    out_root = cache_dir(cfg) / "thumbs" / str(size)
    out_root.mkdir(parents=True, exist_ok=True)
    made = 0
    records = db.iter_all()
    for rec in tqdm(records, desc=f"thumbs@{size}"):
        dest = out_root / f"{rec.id}.jpg"
        if dest.exists():
            continue
        src = images_root / rec.path
        if not src.exists():
            continue
        try:
            with Image.open(src) as img:
                img = img.convert("RGB")
                img.thumbnail((size, size), Image.LANCZOS)
                img.save(dest, "JPEG", quality=80, optimize=True)
            made += 1
        except Exception as exc:  # noqa: BLE001
            logger.warning("Thumbnail failed for %s: %s", rec.image_id, exc)
    db.close()
    logger.info("Generated %d thumbnails at %s", made, out_root)
    return made


def make_eval_set(cfg: AttrDict, num_queries: int | None = None) -> Path:
    manifest = read_manifest(cfg)
    num_queries = num_queries or int(cfg.eval.num_queries)
    queries = build_eval_queries_from_manifest(
        manifest, num_queries=num_queries, seed=int(cfg.eval.get("seed", 42))
    )
    out = eval_queries_path(cfg)
    save_queries(queries, out)
    logger.info("Wrote %d evaluation queries to %s", len(queries), out)
    return out


def run_evaluation(cfg: AttrDict, queries_file: str | Path | None = None) -> dict:
    from tsr.api import RetrievalService

    queries_path = Path(queries_file) if queries_file else eval_queries_path(cfg)
    if not queries_path.exists():
        logger.info("Evaluation queries not found; generating them now.")
        make_eval_set(cfg)
    queries = load_queries(queries_path)

    service = RetrievalService(cfg)
    if not service.ready:
        raise RuntimeError(service.load_error or "Retrieval service is not ready.")

    top_k = list(cfg.eval.top_k)
    limit = max(top_k)

    def retriever(query: str, k: int) -> list[str]:
        resp = service.search(query, k)
        return [r.image_id for r in resp.results]

    report = evaluate_queries(retriever, queries, top_k=top_k, max_rank=limit)
    report["config"] = {
        "model": str(cfg.model.name),
        "pretrained": str(cfg.model.pretrained),
        "index_type": str(service.index.index_type if service.index else cfg.index.type),
        "hybrid": bool(cfg.search.hybrid),
        "num_queries": len(queries),
    }

    out = report_path(cfg)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info("Evaluation results (top_k=%s):", top_k)
    for name, value in report["metrics"].items():
        if isinstance(value, float):
            logger.info("  %-16s %.4f", name, value)
        else:
            logger.info("  %-16s %s", name, value)
    logger.info("Full report written to %s", out)
    return report


DEFAULT_SEARCH_VARIANTS: list[tuple[str, dict]] = [
    ("dense", {}),
    ("hybrid", {"hybrid": True}),
    ("dense+ensemble", {"use_prompt_ensemble": True}),
    ("dense+caption-rerank", {"rerank": "caption"}),
]


def compare_search_configs(
    cfg: AttrDict,
    queries_file: str | Path | None = None,
    n_boot: int = 1000,
    variants: list[tuple[str, dict]] | None = None,
) -> list[dict]:
    """Evaluate several search configurations with bootstrap confidence intervals."""
    from tsr.api import RetrievalService

    queries_path = Path(queries_file) if queries_file else eval_queries_path(cfg)
    if not queries_path.exists():
        make_eval_set(cfg)
    queries = load_queries(queries_path)

    service = RetrievalService(cfg)
    if not service.ready:
        raise RuntimeError(service.load_error or "Retrieval service is not ready.")

    top_k = list(cfg.eval.top_k)
    variants = variants or DEFAULT_SEARCH_VARIANTS
    logger.info("Comparing %d search configs on %d queries (n_boot=%d) ...",
                len(variants), len(queries), n_boot)

    rows: list[dict] = []
    baseline_values: dict[str, list[float]] | None = None
    for name, overrides in variants:
        saved = {k: (k in cfg.search, cfg.search.get(k)) for k in overrides}
        for key, value in overrides.items():
            cfg.search[key] = value
        service._query_cache.clear()

        def retriever(query: str, k: int) -> list[str]:
            return [r.image_id for r in service.search(query, k).results]

        values = per_query_metrics(retriever, queries, top_k=top_k)
        ci = bootstrap_ci(values, n_boot=n_boot, seed=int(cfg.eval.get("seed", 42)))
        if baseline_values is None:
            baseline_values = values
            delta, dlo, dhi, p = (0.0, 0.0, 0.0, 1.0)
        else:
            delta, dlo, dhi, p = paired_bootstrap_test(
                baseline_values["mAP"], values["mAP"], n_boot=n_boot
            )
        rows.append(
            {
                "name": name,
                "metrics": {k: {"mean": v[0], "lo": v[1], "hi": v[2]} for k, v in ci.items()},
                "vs_baseline": {"delta_mAP": delta, "lo": dlo, "hi": dhi, "p_value": p},
            }
        )
        for key, (present, value) in saved.items():
            if present:
                cfg.search[key] = value
            else:
                cfg.search.pop(key, None)

    logger.info("%-22s %-8s %-20s %s", "config", "mAP", "95%% CI", "d(mAP) vs dense (p)")
    for row in rows:
        m = row["metrics"]["mAP"]["mean"]
        lo = row["metrics"]["mAP"]["lo"]
        hi = row["metrics"]["mAP"]["hi"]
        sig = row["vs_baseline"]
        logger.info(
            "%-22s %-8.3f [%.3f, %.3f]    %+.3f (p=%.3f)",
            row["name"], m, lo, hi, sig["delta_mAP"], sig["p_value"],
        )
    return rows
