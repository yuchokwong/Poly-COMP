"""Optional query translation for multilingual search.

CLIP is English-only. When ``search.translate`` is enabled, CJK (or otherwise
non-Latin) queries are translated to English with a HuggingFace MarianMT model
before being encoded, so users can search in Chinese without rebuilding the
index.
"""

from __future__ import annotations

import re

from tsr.logging_utils import get_logger

logger = get_logger(__name__)

DEFAULT_TRANSLATOR = "Helsinki-NLP/opus-mt-zh-en"

_CJK_RE = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uac00-\ud7af]")


def has_cjk(text: str) -> bool:
    return bool(_CJK_RE.search(text))


class Translator:
    """Lazily-loaded MarianMT translator (query -> English)."""

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or DEFAULT_TRANSLATOR
        self._tokenizer = None
        self._model = None
        self._device = "cpu"

    def _load(self) -> None:
        try:
            import torch
            from transformers import MarianMTModel, MarianTokenizer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "Query translation needs `transformers` and `sentencepiece`. "
                "Install with: pip install transformers sentencepiece"
            ) from exc
        logger.info("Loading translator %s ...", self.model_name)
        self._tokenizer = MarianTokenizer.from_pretrained(self.model_name)
        self._model = MarianMTModel.from_pretrained(self.model_name)
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model.to(self._device).eval()

    def translate(self, text: str) -> str:
        import torch

        if self._model is None:
            self._load()
        batch = self._tokenizer([text], return_tensors="pt", padding=True).to(self._device)
        with torch.inference_mode():
            generated = self._model.generate(**batch)
        return self._tokenizer.decode(generated[0], skip_special_tokens=True).strip()
