"""Gallery preparation: download/collect images, normalise them and write a
manifest.

The manifest (``data/manifest.jsonl``) is the source of truth for indexing: one
JSON object per line with ``image_id``, ``split``, ``path`` (relative to the
images directory), ``captions`` and ``source``.
"""

from __future__ import annotations

import hashlib
import io
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from PIL import Image
from tqdm import tqdm

from tsr.config import AttrDict, images_dir, manifest_path
from tsr.logging_utils import get_logger
from tsr.sources import Candidate, get_source

logger = get_logger(__name__)

Image.MAX_IMAGE_PIXELS = None  # allow large source images


def _dhash(img: Image.Image, size: int = 8) -> str:
    """Perceptual difference hash (hex) used for near-duplicate detection."""
    gray = img.convert("L").resize((size + 1, size), Image.LANCZOS)
    px = gray.tobytes()
    bits = 0
    for row in range(size):
        base = row * (size + 1)
        for col in range(size):
            bits = (bits << 1) | (1 if px[base + col] > px[base + col + 1] else 0)
    return f"{bits:0{size * size // 4}x}"


def _resize_save(raw: bytes, dest: Path, max_side: int, quality: int) -> tuple[tuple[int, int], str]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(io.BytesIO(raw)) as img:
        img = img.convert("RGB")
        phash = _dhash(img)
        w, h = img.size
        scale = min(1.0, max_side / max(w, h))
        if scale < 1.0:
            img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
        img.save(dest, "JPEG", quality=quality, optimize=True, progressive=True)
        return img.size, phash


def _process_one(
    candidate: Candidate,
    images_root: Path,
    max_side: int,
    quality: int,
    timeout: int,
    retries: int,
) -> dict | None:
    dest = images_root / candidate.out_name
    dest = dest.with_suffix(".jpg")
    raw: bytes | None = None

    if candidate.local_path:
        try:
            raw = Path(candidate.local_path).read_bytes()
        except OSError as exc:
            logger.warning("Cannot read %s: %s", candidate.local_path, exc)
            return None
    else:
        last_err: Exception | None = None
        for _ in range(1, retries + 1):
            try:
                resp = requests.get(candidate.url, timeout=timeout)
                resp.raise_for_status()
                raw = resp.content
                break
            except Exception as exc:  # noqa: BLE001
                last_err = exc
        if raw is None:
            logger.warning("Download failed for %s: %s", candidate.url, last_err)
            return None

    try:
        (width, height), phash = _resize_save(raw, dest, max_side, quality)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Cannot process %s: %s", candidate.image_id, exc)
        return None

    rel = dest.relative_to(images_root).as_posix()
    return {
        "image_id": candidate.image_id,
        "split": candidate.split,
        "path": rel,
        "width": width,
        "height": height,
        "captions": candidate.captions,
        "category": candidate.category,
        "source": candidate.source,
        "sha1": hashlib.sha1(raw).hexdigest(),
        "phash": phash,
    }


def prepare_gallery(cfg: AttrDict, max_images: int | None = None) -> int:
    """Build the gallery and write the manifest. Returns the number of images."""
    images_root = images_dir(cfg)
    images_root.mkdir(parents=True, exist_ok=True)
    max_images = max_images or int(cfg.gallery.max_images)
    max_side = int(cfg.gallery.image_max_side)
    quality = int(cfg.gallery.image_quality)
    timeout = int(cfg.gallery.download_timeout)
    retries = int(cfg.gallery.download_retries)
    workers = int(cfg.gallery.download_workers)
    dedup = str(cfg.gallery.get("dedup", "sha1")).lower()
    dedup_key = "phash" if dedup == "phash" else "sha1"

    source = get_source(cfg)
    candidates = []
    for cand in source.iter_candidates():
        candidates.append(cand)
        if len(candidates) >= max_images:
            break
    logger.info("Preparing %d images from source '%s'.", len(candidates), cfg.gallery.source)

    records: list[dict] = []
    seen_hashes: set[str] = set()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_process_one, c, images_root, max_side, quality, timeout, retries): c
            for c in candidates
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="preparing"):
            rec = fut.result()
            if rec is None:
                continue
            if rec[dedup_key] in seen_hashes:
                continue  # duplicate content
            seen_hashes.add(rec[dedup_key])
            records.append(rec)

    records.sort(key=lambda r: r["image_id"])
    out = manifest_path(cfg)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    total_bytes = sum(
        (images_root / r["path"]).stat().st_size
        for r in records
        if (images_root / r["path"]).exists()
    )
    logger.info(
        "Gallery ready: %d images (%.1f MB) -> %s",
        len(records),
        total_bytes / 1e6,
        out,
    )
    return len(records)


def read_manifest(cfg: AttrDict) -> list[dict]:
    path = manifest_path(cfg)
    if not path.exists():
        raise FileNotFoundError(
            f"Manifest not found at {path}. Run `tsr download` (or `tsr build`) first."
        )
    records: list[dict] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
