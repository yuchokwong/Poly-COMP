"""Offline image captioning (BLIP) for local image folders.

The ``local`` gallery source reads captions from ``captions.json``. If your
images have none, ``tsr caption`` generates them with BLIP and writes that file,
so an arbitrary image folder becomes a searchable, evaluatable gallery.
"""

from __future__ import annotations

import json
from pathlib import Path

from tqdm import tqdm

from tsr.config import AttrDict, local_root
from tsr.logging_utils import get_logger
from tsr.sources import _IMAGE_EXTS

logger = get_logger(__name__)

DEFAULT_CAPTION_MODEL = "Salesforce/blip-image-captioning-base"


def _load_captioner(model_name: str | None = None):
    try:
        import torch
        from transformers import BlipForConditionalGeneration, BlipProcessor
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Auto-captioning needs `transformers` and `torch`. "
            "Install with: pip install transformers"
        ) from exc

    name = model_name or DEFAULT_CAPTION_MODEL
    logger.info("Loading captioning model %s ...", name)
    processor = BlipProcessor.from_pretrained(name)
    model = BlipForConditionalGeneration.from_pretrained(name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    return processor, model, device


def caption_image(processor, model, device, image, max_new_tokens: int = 30) -> str:
    import torch

    inputs = processor(image, return_tensors="pt").to(device)
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return processor.decode(output[0], skip_special_tokens=True).strip()


def caption_local_images(
    cfg: AttrDict,
    root: str | Path | None = None,
    out: str | Path | None = None,
    model: str | None = None,
) -> int:
    """Caption every image under ``root`` and write ``captions.json``."""
    from PIL import Image

    root_path = Path(root) if root else local_root(cfg)
    out_path = Path(out) if out else root_path / "captions.json"
    if not root_path.exists():
        raise FileNotFoundError(f"Image folder not found: {root_path}")

    files = sorted(
        p for p in root_path.rglob("*") if p.is_file() and p.suffix.lower() in _IMAGE_EXTS
    )
    if not files:
        logger.warning("No images found under %s", root_path)
        return 0

    processor, model_obj, device = _load_captioner(model)
    captions: dict[str, list[str]] = {}
    if out_path.exists():
        try:
            captions = json.loads(out_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            captions = {}

    made = 0
    for path in tqdm(files, desc="captioning"):
        rel = path.relative_to(root_path).as_posix()
        if captions.get(rel):
            continue
        try:
            with Image.open(path) as img:
                text = caption_image(processor, model_obj, device, img.convert("RGB"))
            if text:
                captions[rel] = [text]
                made += 1
        except Exception as exc:  # noqa: BLE001
            logger.warning("Captioning failed for %s: %s", path, exc)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(captions, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Wrote %d captions to %s", len(captions), out_path)
    return made
