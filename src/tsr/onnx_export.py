"""Export an OpenCLIP model to ONNX for fast CPU inference.

ONNX Runtime is typically 2-4x faster than PyTorch on CPU, which matters when
serving or (especially) building an index on a machine without a GPU. The
exported files are consumed by :class:`tsr.embedder.Embedder` when
``model.backend: onnx`` is set.
"""

from __future__ import annotations

from pathlib import Path

from tsr.config import AttrDict
from tsr.logging_utils import get_logger

logger = get_logger(__name__)

IMAGE_ONNX_NAME = "image_encoder.onnx"
TEXT_ONNX_NAME = "text_encoder.onnx"
DEFAULT_OPSET = 17


def default_onnx_dir(cfg: AttrDict) -> Path:
    configured = cfg.model.get("onnx_dir")
    if configured:
        return Path(configured)
    return Path(str(cfg.model.get("cache_dir", ".cache/open_clip"))) / "onnx"


def export_clip_to_onnx(cfg: AttrDict, out_dir: str | Path | None = None) -> tuple[Path, Path]:
    """Export the configured OpenCLIP model to an image + text ONNX pair."""
    import open_clip
    import torch

    out_dir = Path(out_dir) if out_dir else default_onnx_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)

    name = str(cfg.model.name)
    pretrained = str(cfg.model.pretrained)
    logger.info("Loading %s (%s) for ONNX export ...", name, pretrained)
    model, _, _ = open_clip.create_model_and_transforms(
        name, pretrained=pretrained, device="cpu"
    )
    model.eval()

    # The fused multi-head attention fast path is not exportable to ONNX; fall
    # back to the reference implementation.
    try:
        torch.backends.mha.set_fastpath_enabled(False)
    except Exception:  # pragma: no cover - older torch
        pass

    class _ImageEncoder(torch.nn.Module):
        def __init__(self, m: torch.nn.Module) -> None:
            super().__init__()
            self.m = m

        def forward(self, pixel_values):  # noqa: ANN001
            return self.m.encode_image(pixel_values)

    class _TextEncoder(torch.nn.Module):
        def __init__(self, m: torch.nn.Module) -> None:
            super().__init__()
            self.m = m

        def forward(self, input_ids):  # noqa: ANN001
            return self.m.encode_text(input_ids)

    image_path = out_dir / IMAGE_ONNX_NAME
    text_path = out_dir / TEXT_ONNX_NAME

    image_size = int(cfg.model.get("image_size", 224))
    dummy_image = torch.zeros(1, 3, image_size, image_size, dtype=torch.float32)
    logger.info("Exporting image encoder -> %s", image_path)
    with torch.no_grad():
        torch.onnx.export(
            _ImageEncoder(model),
            (dummy_image,),
            str(image_path),
            input_names=["pixel_values"],
            output_names=["features"],
            dynamic_axes={"pixel_values": {0: "batch"}, "features": {0: "batch"}},
            opset_version=DEFAULT_OPSET,
            do_constant_folding=True,
            dynamo=False,
        )

    tokenizer = open_clip.get_tokenizer(name)
    dummy_tokens = tokenizer(["a photo"]).to(torch.int64)
    logger.info("Exporting text encoder -> %s", text_path)
    with torch.no_grad():
        torch.onnx.export(
            _TextEncoder(model),
            (dummy_tokens,),
            str(text_path),
            input_names=["input_ids"],
            output_names=["features"],
            dynamic_axes={"input_ids": {0: "batch", 1: "tokens"}, "features": {0: "batch"}},
            opset_version=DEFAULT_OPSET,
            do_constant_folding=True,
            dynamo=False,
        )

    logger.info("ONNX export complete: %s, %s", image_path, text_path)
    return image_path, text_path
