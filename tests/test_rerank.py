import numpy as np

from tsr.rerank import caption_match_scores


class _DummyEmbedder:
    """Maps captions containing 'cat' to [1,0] and everything else to [0,1]."""

    def encode_texts(self, texts):
        rows = []
        for text in texts:
            rows.append([1.0, 0.0] if "cat" in text else [0.0, 1.0])
        return np.asarray(rows, dtype=np.float32)


def test_caption_match_scores_takes_best_per_candidate():
    embedder = _DummyEmbedder()
    query = np.asarray([[1.0, 0.0]], dtype=np.float32)
    scores = caption_match_scores(
        embedder, query, [["a cat"], ["a dog", "another cat"], []]
    )
    assert scores[0] == 1.0
    assert scores[1] == 1.0  # best over both captions
    assert scores[2] == -1.0  # no captions -> sentinel


def test_caption_match_scores_empty():
    embedder = _DummyEmbedder()
    query = np.asarray([[1.0, 0.0]], dtype=np.float32)
    scores = caption_match_scores(embedder, query, [[], []])
    assert list(scores) == [-1.0, -1.0]
