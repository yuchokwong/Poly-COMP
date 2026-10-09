"""OpenCLIP embedding wrapper for images and text."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np

from tsr.config import AttrDict
from tsr.logging_utils import get_logger

logger = get_logger(__name__)


def resolve_device(device: str = "auto") -> str:
    import torch

    if device and device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Embedder:
    """Thin wrapper around an OpenCLIP model.

    The model is loaded lazily on first use so that commands which do not need
    it (e.g. ``stats``) stay fast and dependency-light.
    """

    def __init__(self, cfg: AttrDict) -> None:
        self.cfg = cfg
        self.model_name: str = cfg.model.name
        self.pretrained: str = cfg.model.pretrained
        self.batch_size: int = int(cfg.model.batch_size)
        self.normalize: bool = bool(cfg.model.get("normalize", True))
        self.cache_dir: str | None = cfg.model.get("cache_dir")
        self.backend: str = str(cfg.model.get("backend", "torch")).lower()
        self.onnx_dir: str | None = cfg.model.get("onnx_dir")
        self.device: str = "cpu" if self.backend == "onnx" else resolve_device(
            cfg.model.get("device", "auto")
        )
        self._model = None
        self._sessions: dict | None = None
        self._preprocess = None
        self._tokenizer = None
        self._dim: int | None = None

    # ------------------------------------------------------------------ #
    def load(self) -> None:
        if self._model is not None or self._sessions is not None:
            return
        if self.backend == "onnx":
            self._load_onnx()
        else:
            self._load_torch()

    def _load_torch(self) -> None:
        import open_clip
        import torch

        logger.info(
            "Loading OpenCLIP model %s (%s) on %s ...",
            self.model_name,
            self.pretrained,
            self.device,
        )
        kwargs = {}
        if self.cache_dir:
            Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
            kwargs["cache_dir"] = self.cache_dir
        model, _, preprocess = open_clip.create_model_and_transforms(
            self.model_name, pretrained=self.pretrained, device=self.device, **kwargs
        )
        model.eval()
        self._model = model
        self._preprocess = preprocess
        self._tokenizer = open_clip.get_tokenizer(self.model_name)
        with torch.inference_mode():
            dummy = model.encode_text(self._tokenizer(["x"]).to(self.device))
        self._dim = int(dummy.shape[-1])
        logger.info("Model ready. Embedding dimension = %d", self._dim)

    def _load_onnx(self) -> None:
        import onnxruntime as ort
        import open_clip

        from tsr.onnx_export import IMAGE_ONNX_NAME, TEXT_ONNX_NAME, default_onnx_dir

        base = Path(self.onnx_dir) if self.onnx_dir else default_onnx_dir(self.cfg)
        image_path = base / IMAGE_ONNX_NAME
        text_path = base / TEXT_ONNX_NAME
        if not image_path.exists() or not text_path.exists():
            raise FileNotFoundError(
                f"ONNX models not found in {base}. Run `tsr export-onnx` first."
            )
        logger.info("Loading ONNX encoders from %s ...", base)
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        threads = int(self.cfg.model.get("threads", 0))
        if threads > 0:
            opts.intra_op_num_threads = threads
        providers = ["CPUExecutionProvider"]
        self._sessions = {
            "image": ort.InferenceSession(str(image_path), opts, providers=providers),
            "text": ort.InferenceSession(str(text_path), opts, providers=providers),
        }
        _, _, preprocess = open_clip.create_model_and_transforms(
            self.model_name, pretrained=None, device="cpu"
        )
        self._preprocess = preprocess
        self._tokenizer = open_clip.get_tokenizer(self.model_name)
        self._dim = int(self._sessions["text"].get_outputs()[0].shape[-1])
        logger.info("ONNX model ready. Embedding dimension = %d", self._dim)

    @property
    def dim(self) -> int:
        if self._dim is None:
            self.load()
        assert self._dim is not None
        return self._dim

    @property
    def model(self):
        self.load()
        return self._model

    # ------------------------------------------------------------------ #
    def _finalize(self, features) -> np.ndarray:
        import torch

        if self.normalize:
            features = torch.nn.functional.normalize(features, dim=-1)
        return features.detach().to("cpu", dtype=torch.float32).numpy()

    def _finalize_np(self, features: np.ndarray) -> np.ndarray:
        features = np.asarray(features, dtype=np.float32)
        if self.normalize:
            norms = np.linalg.norm(features, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            features = features / norms
        return features.astype(np.float32)

    def _forward_images(self, batch) -> np.ndarray:
        if self.backend == "onnx":
            arr = batch.detach().cpu().numpy().astype(np.float32)
            out = self._sessions["image"].run(None, {"pixel_values": arr})[0]
            return self._finalize_np(out)
        import torch

        with torch.inference_mode():
            feats = self.model.encode_image(batch.to(self.device))
        return self._finalize(feats)

    def _forward_text(self, tokens) -> np.ndarray:
        if self.backend == "onnx":
            arr = tokens.detach().cpu().numpy().astype(np.int64)
            out = self._sessions["text"].run(None, {"input_ids": arr})[0]
            return self._finalize_np(out)
        import torch

        with torch.inference_mode():
            feats = self.model.encode_text(tokens.to(self.device))
        return self._finalize(feats)

    def encode_images(self, paths: Sequence[str | Path]) -> np.ndarray:
        from PIL import Image

        self.load()
        out: list[np.ndarray] = []
        batch_tensors = []
        for path in paths:
            try:
                img = Image.open(path).convert("RGB")
                batch_tensors.append(self._preprocess(img))
            except Exception as exc:  # pragma: no cover - corrupted file
                logger.warning("Failed to read image %s: %s", path, exc)
                batch_tensors.append(None)
            if len(batch_tensors) >= self.batch_size:
                out.append(self._encode_batch(batch_tensors))
                batch_tensors = []
        if batch_tensors:
            out.append(self._encode_batch(batch_tensors))
        if not out:
            return np.empty((0, self.dim), dtype=np.float32)
        return np.concatenate(out, axis=0)

    def _encode_batch(self, batch_tensors) -> np.ndarray:
        import torch

        valid = [t for t in batch_tensors if t is not None]
        if not valid:
            return np.empty((0, self.dim), dtype=np.float32)
        feats = self._forward_images(torch.stack(valid))
        # Re-insert zero rows for failed images to preserve alignment.
        if len(valid) != len(batch_tensors):
            full = np.zeros((len(batch_tensors), feats.shape[1]), dtype=np.float32)
            it = iter(feats)
            for i, t in enumerate(batch_tensors):
                if t is not None:
                    full[i] = next(it)
            return full
        return feats

    def encode_texts(self, texts: Iterable[str]) -> np.ndarray:
        self.load()
        texts = list(texts)
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)
        out: list[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            chunk = texts[start : start + self.batch_size]
            tokens = self._tokenizer(chunk)
            out.append(self._forward_text(tokens))
        return np.concatenate(out, axis=0)

    def encode_text_with_templates(self, text: str, templates: Sequence[str]) -> np.ndarray:
        """Average embeddings over prompt templates (prompt ensembling)."""
        prompts = [t.format(text) if "{}" in t else f"{t} {text}" for t in templates]
        feats = self.encode_texts(prompts)
        if feats.shape[0] == 0:
            return feats
        mean = feats.mean(axis=0, keepdims=True)
        norm = np.linalg.norm(mean, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        return (mean / norm).astype(np.float32)
