"""Optional query rerankers.

Two rerankers are available:

* ``caption`` - re-score the top-N candidates by the best CLIP text-text
  similarity between the query and each candidate's captions. Uses the already
  loaded CLIP model, so it adds no dependencies and works on CPU.
* ``cross-encoder`` - score (query, caption) pairs with a HuggingFace
  cross-encoder. Stronger but needs ``transformers`` and a downloaded model.
"""

from __future__ import annotations

import numpy as np

from tsr.logging_utils import get_logger

logger = get_logger(__name__)

DEFAULT_CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def caption_match_scores(embedder, query_vec: np.ndarray, caption_lists: list[list[str]]) -> np.ndarray:
    """Best cosine similarity between the query and each candidate's captions."""
    texts: list[str] = []
    owner: list[int] = []
    for i, caps in enumerate(caption_lists):
        for cap in caps:
            texts.append(cap)
            owner.append(i)
    best = np.full(len(caption_lists), -1.0, dtype=np.float32)
    if not texts:
        return best
    feats = embedder.encode_texts(texts)
    q = query_vec.reshape(-1)
    sims = feats @ q
    for i, sim in zip(owner, sims, strict=False):
        if sim > best[i]:
            best[i] = float(sim)
    return best


class CrossEncoderReranker:
    """Lazily-loaded HuggingFace cross-encoder for (query, caption) scoring."""

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or DEFAULT_CROSS_ENCODER
        self._tokenizer = None
        self._model = None
        self._device = "cpu"

    def _load(self) -> None:
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "cross-encoder reranking needs `transformers`. "
                "Install with: pip install transformers"
            ) from exc
        logger.info("Loading cross-encoder %s ...", self.model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model.to(self._device).eval()

    def score(self, query: str, docs: list[str]) -> np.ndarray:
        if not docs:
            return np.empty((0,), dtype=np.float32)
        import torch

        if self._model is None:
            self._load()
        inputs = self._tokenizer(
            [query] * len(docs), docs, padding=True, truncation=True, return_tensors="pt"
        ).to(self._device)
        with torch.inference_mode():
            logits = self._model(**inputs).logits.view(-1)
        return logits.detach().cpu().numpy().astype(np.float32)
