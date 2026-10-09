import numpy as np
import pytest

faiss = pytest.importorskip("faiss")

from tsr.index import VectorIndex, normalize  # noqa: E402


def _random_matrix(n, dim, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, dim)).astype(np.float32)


def test_normalize_unit_norm():
    vecs = _random_matrix(5, 8)
    out = normalize(vecs)
    norms = np.linalg.norm(out, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_flat_index_search_returns_exact_neighbour():
    dim = 16
    matrix = _random_matrix(50, dim, seed=1)
    ids = np.arange(1000, 1050, dtype=np.int64)
    index = VectorIndex(dim=dim, index_type="flat")
    index.add(matrix, ids)

    query = matrix[7]
    scores, found = index.search(query[None, :], k=1)
    assert found[0] == ids[7]
    assert scores[0] > 0.99


def test_index_save_load_roundtrip(tmp_path):
    dim = 8
    matrix = _random_matrix(20, dim, seed=2)
    ids = np.arange(20, dtype=np.int64)
    index = VectorIndex(dim=dim, index_type="flat")
    index.add(matrix, ids)

    idx_file = tmp_path / "index.faiss"
    ids_file = tmp_path / "index_ids.npy"
    index.save(idx_file, ids_file)

    loaded = VectorIndex.load(idx_file, ids_file)
    assert loaded.size == 20
    scores, found = loaded.search(matrix[3][None, :], k=1)
    assert found[0] == 3


def test_hnsw_index_returns_reasonable_neighbour():
    dim = 12
    matrix = _random_matrix(200, dim, seed=3)
    ids = np.arange(200, dtype=np.int64)
    index = VectorIndex(dim=dim, index_type="hnsw", m=16, ef_search=64)
    index.add(matrix, ids)
    scores, found = index.search(matrix[10][None, :], k=5)
    assert ids[10] in found


def test_reconstruct_returns_normalised_vectors():
    dim = 8
    matrix = _random_matrix(10, dim, seed=5)
    ids = np.arange(10, dtype=np.int64)
    index = VectorIndex(dim=dim, index_type="flat")
    index.add(matrix, ids)

    rec = index.reconstruct([3, 7])
    assert rec.shape == (2, dim)
    expected = normalize(matrix[[3, 7]])
    assert np.allclose(rec, expected, atol=1e-5)

    assert index.reconstruct([999]).shape == (0, dim)


def test_ivfpq_index_returns_neighbour():
    dim = 16
    rng = np.random.default_rng(0)
    centers = rng.standard_normal((8, dim)).astype(np.float32)
    assign = rng.integers(0, 8, size=600)
    matrix = centers[assign] + 0.01 * rng.standard_normal((600, dim)).astype(np.float32)
    ids = np.arange(600, dtype=np.int64)

    index = VectorIndex(dim=dim, index_type="pq", nlist=4, nprobe=4, m=8)
    index.add(matrix, ids)
    assert index.size == 600

    # IVF-PQ is lossy; just check it returns a full, ranked result set.
    scores, found = index.search(matrix[10][None, :], k=10)
    assert len(found) == 10
    assert np.all(np.diff(scores) <= 1e-6)


