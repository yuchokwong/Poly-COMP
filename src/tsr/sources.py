"""Gallery data sources.

A source yields :class:`Candidate` objects describing images to place in the
gallery. Candidates either point to a remote ``url`` (downloaded by
:mod:`tsr.gallery`) or to a ``local_path`` already on disk.

Three sources are provided:

* ``COCOSource`` - general-domain images with human captions from MS-COCO 2017,
  downloaded one-by-one from the official (often slow) server.
* ``COCOHFSource`` - the same COCO data (Karpathy split) streamed from a
  HuggingFace webdataset mirror, which is typically ~50x faster.
* ``LocalSource`` - your own image folder, with optional caption sidecars.
"""

from __future__ import annotations

import json
import random
import tarfile
import time
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import requests
from tqdm import tqdm

from tsr.config import AttrDict, cache_dir, local_root
from tsr.logging_utils import get_logger

logger = get_logger(__name__)

COCO_ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
COCO_IMAGE_BASE = {
    "train": "http://images.cocodataset.org/train2017/{:012d}.jpg",
    "val": "http://images.cocodataset.org/val2017/{:012d}.jpg",
}
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}

# HuggingFace webdataset mirror of the MS-COCO Karpathy split. Each tar holds
# ``<stem>.jpg`` + ``<stem>.json`` (with ``captions`` and ``cocoid``).
COCO_HF_REPO = "undefined443/coco-karpathy-wds"
COCO_HF_TAR_URL = (
    "https://huggingface.co/datasets/{repo}/resolve/main/{split}/{split}-{index:05d}.tar"
)
_COCO_HF_SPLIT_TARS = {"val": 5, "train": 114}
_COCO_HF_SPLITS = ("val", "train")


@dataclass
class Candidate:
    image_id: str
    split: str
    captions: list[str] = field(default_factory=list)
    source: str = "local"
    url: str | None = None
    local_path: str | None = None
    category: str | None = None
    out_name: str = ""


# --------------------------------------------------------------------------- #
# Local folder source
# --------------------------------------------------------------------------- #
class LocalSource:
    """Read images from a folder.

    Captions are resolved, in order of precedence:
      1. ``captions.json`` mapping ``relative/path.jpg`` -> list[str]
      2. a sidecar ``<image>.txt`` file (one or more lines, Flickr30k style)
      3. the file stem as a single caption
    """

    def __init__(self, cfg: AttrDict) -> None:
        self.root = local_root(cfg)
        self.captions: dict[str, list[str]] = {}
        cap_file = self.root / "captions.json"
        if cap_file.exists():
            self.captions = json.loads(cap_file.read_text(encoding="utf-8"))

    def iter_candidates(self) -> Iterator[Candidate]:
        if not self.root.exists():
            logger.warning("Local image root does not exist: %s", self.root)
            return
        files = sorted(
            p for p in self.root.rglob("*") if p.is_file() and p.suffix.lower() in _IMAGE_EXTS
        )
        for path in files:
            rel = path.relative_to(self.root).as_posix()
            captions = self.captions.get(rel)
            if captions is None:
                sidecar = path.with_suffix(".txt")
                if sidecar.exists():
                    captions = [
                        line.strip()
                        for line in sidecar.read_text(encoding="utf-8").splitlines()
                        if line.strip()
                    ]
                else:
                    captions = [path.stem.replace("_", " ").replace("-", " ")]
            yield Candidate(
                image_id=f"local_{rel}",
                split="local",
                captions=list(captions),
                source="local",
                local_path=str(path),
                out_name=rel,
            )


# --------------------------------------------------------------------------- #
# MS-COCO source
# --------------------------------------------------------------------------- #
class COCOSource:
    def __init__(self, cfg: AttrDict) -> None:
        self.cfg = cfg
        self.cache = cache_dir(cfg)

    def _ensure_annotations(self) -> Path:
        self.cache.mkdir(parents=True, exist_ok=True)
        ann_dir = self.cache / "coco_annotations"
        val_file = ann_dir / "captions_val2017.json"
        train_file = ann_dir / "captions_train2017.json"
        if val_file.exists() and train_file.exists():
            return ann_dir

        zip_path = self.cache / "annotations_trainval2017.zip"
        if not zip_path.exists():
            logger.info("Downloading COCO annotations (~241 MB) ...")
            _download(COCO_ANNOTATIONS_URL, zip_path)
        logger.info("Extracting captions ...")
        ann_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as zf:
            for name in ("annotations/captions_val2017.json", "annotations/captions_train2017.json"):
                member = zf.getinfo(name)
                target = ann_dir / Path(name).name
                with zf.open(member) as src, open(target, "wb") as dst:
                    dst.write(src.read())
        return ann_dir

    def _load_split(self, path: Path) -> dict[int, list[str]]:
        data = json.loads(path.read_text(encoding="utf-8"))
        by_image: dict[int, list[str]] = {}
        for ann in data["annotations"]:
            by_image.setdefault(int(ann["image_id"]), []).append(ann["caption"].strip())
        return by_image

    def iter_candidates(self) -> Iterator[Candidate]:
        ann_dir = self._ensure_annotations()
        val = self._load_split(ann_dir / "captions_val2017.json")
        train = self._load_split(ann_dir / "captions_train2017.json")

        rng = random.Random(int(self.cfg.gallery.get("seed", 42)))
        max_images = int(self.cfg.gallery.max_images)

        val_ids = sorted(val)
        rng.shuffle(val_ids)
        selected: list[tuple[str, int, dict[int, list[str]]]] = [
            ("val", i, val) for i in val_ids
        ]

        if len(selected) < max_images:
            train_ids = sorted(train)
            rng.shuffle(train_ids)
            need = max_images - len(selected)
            selected.extend(("train", i, train) for i in train_ids[:need])

        selected = selected[:max_images]
        logger.info("Selected %d COCO images for the gallery.", len(selected))

        for split, image_id, table in selected:
            yield Candidate(
                image_id=f"coco_{split}_{image_id}",
                split=split,
                captions=table[image_id],
                source="coco",
                url=COCO_IMAGE_BASE[split].format(image_id),
                out_name=f"{split}/{image_id:012d}.jpg",
            )


# --------------------------------------------------------------------------- #
# MS-COCO via HuggingFace webdataset mirror (fast)
# --------------------------------------------------------------------------- #
class COCOHFSource:
    """Stream COCO (Karpathy split) webdataset tars from the HuggingFace Hub.

    By default each tar is cached under ``.cache/coco_hf_tars`` so re-runs are
    instant; set ``gallery.cache_tars: false`` to stream them without touching
    disk. Decoded JPEGs are written to ``.cache/coco_hf_raw`` for the shared
    gallery pipeline. Each sample carries its captions, so no separate
    annotation download is needed.
    """

    def __init__(self, cfg: AttrDict) -> None:
        self.cfg = cfg
        self.repo = str(cfg.gallery.get("hf_repo", COCO_HF_REPO))
        self.cache_tars = bool(cfg.gallery.get("cache_tars", True))
        self.cache = cache_dir(cfg)
        self.tar_dir = self.cache / "coco_hf_tars"
        self.raw_dir = self.cache / "coco_hf_raw"

    def iter_candidates(self) -> Iterator[Candidate]:
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        for split in _COCO_HF_SPLITS:
            for index in range(_COCO_HF_SPLIT_TARS[split]):
                yield from self._iter_tar(split, index)

    def _iter_tar(self, split: str, index: int) -> Iterator[Candidate]:
        url = COCO_HF_TAR_URL.format(repo=self.repo, split=split, index=index)
        if self.cache_tars:
            path = self.tar_dir / f"{split}-{index:05d}.tar"
            if not path.exists() or path.stat().st_size == 0:
                logger.info("Downloading COCO tar %s ...", url)
                _download(url, path)
            with tarfile.open(path, mode="r") as tar:
                for stem, raw, caps, cocoid in _iter_webdataset(tar):
                    yield self._make_candidate(split, index, stem, raw, caps, cocoid)
        else:
            logger.info("Streaming COCO tar %s ...", url)
            with requests.get(url, stream=True, timeout=120) as resp:
                resp.raise_for_status()
                resp.raw.decode_content = True
                with tarfile.open(fileobj=resp.raw, mode="r|") as tar:
                    for stem, raw, caps, cocoid in _iter_webdataset(tar):
                        yield self._make_candidate(split, index, stem, raw, caps, cocoid)

    def _make_candidate(
        self,
        split: str,
        index: int,
        stem: str,
        raw: bytes,
        caps: list[str],
        cocoid: int | None,
    ) -> Candidate:
        name = f"{split}_{index:05d}_{stem}"
        dest = self.raw_dir / f"{name}.jpg"
        dest.write_bytes(raw)
        image_id = f"coco_{split}_{cocoid}" if cocoid is not None else f"coco_{name}"
        return Candidate(
            image_id=image_id,
            split=split,
            captions=caps,
            source="coco_hf",
            local_path=str(dest),
            out_name=f"{split}/{name}.jpg",
        )


def get_source(cfg: AttrDict):
    name = str(cfg.gallery.get("source", "coco")).lower()
    if name == "coco":
        return COCOSource(cfg)
    if name in {"coco_hf", "coco-hf"}:
        return COCOHFSource(cfg)
    if name == "local":
        return LocalSource(cfg)
    raise ValueError(
        f"Unknown gallery source: {name!r} (expected 'coco', 'coco_hf' or 'local')"
    )


def _iter_webdataset(tar: tarfile.TarFile) -> Iterator[tuple[str, bytes, list[str], int | None]]:
    """Yield ``(stem, image_bytes, captions, cocoid)`` from an open webdataset tar."""
    images: dict[str, bytes] = {}
    meta: dict[str, tuple[list[str], int | None]] = {}
    for member in tar:
        if not member.isfile():
            continue
        fileobj = tar.extractfile(member)
        if fileobj is None:
            continue
        stem = Path(member.name).stem
        suffix = Path(member.name).suffix.lower()
        if suffix in _IMAGE_EXTS:
            images[stem] = fileobj.read()
        elif suffix == ".json":
            obj = json.loads(fileobj.read().decode("utf-8"))
            caps = [c.strip() for c in obj.get("captions", []) if c.strip()]
            cocoid = int(obj["cocoid"]) if obj.get("cocoid") is not None else None
            meta[stem] = (caps, cocoid)
        if stem in images and stem in meta:
            caps, cocoid = meta.pop(stem)
            yield stem, images.pop(stem), caps, cocoid


def _download(url: str, dest: Path, retries: int = 3, timeout: int = 60) -> None:
    """Download ``url`` to ``dest`` with resume support and exponential backoff."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            existing = tmp.stat().st_size if tmp.exists() else 0
            headers = {"Range": f"bytes={existing}-"} if existing else {}
            with requests.get(url, stream=True, timeout=timeout, headers=headers) as resp:
                if existing and resp.status_code == 206:
                    mode = "ab"
                else:
                    existing = 0  # server ignored the range request; start over
                    mode = "wb"
                resp.raise_for_status()
                total = existing + int(resp.headers.get("content-length", 0))
                with open(tmp, mode) as fh, tqdm(
                    total=total, initial=existing, unit="B", unit_scale=True, desc=dest.name
                ) as bar:
                    for chunk in resp.iter_content(chunk_size=1 << 20):
                        fh.write(chunk)
                        bar.update(len(chunk))
            if total and tmp.stat().st_size < total:
                raise OSError("incomplete download")
            tmp.replace(dest)
            return
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            logger.warning("download attempt %d/%d failed for %s: %s", attempt, retries, url, exc)
            time.sleep(min(2**attempt, 30))
    raise RuntimeError(f"Failed to download {url}: {last_err}")
