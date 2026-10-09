import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from tsr.api import create_app  # noqa: E402
from tsr.config import load_config  # noqa: E402


def _empty_cfg(tmp_path):
    cfg = load_config()
    cfg.paths["data_dir"] = str(tmp_path)
    return cfg


def test_health_and_stats_without_index(tmp_path):
    cfg = _empty_cfg(tmp_path)
    with TestClient(create_app(cfg)) as client:
        health = client.get("/api/health")
        assert health.status_code == 200
        assert health.json() == {"status": "ok", "ready": False}

        stats = client.get("/api/stats")
        assert stats.status_code == 200
        assert stats.json()["num_images"] == 0
        assert stats.json()["ready"] is False


def test_search_without_index_returns_503(tmp_path):
    cfg = _empty_cfg(tmp_path)
    with TestClient(create_app(cfg)) as client:
        resp = client.get("/api/search", params={"q": "a cat"})
        assert resp.status_code == 503


def test_thumbnail_without_index_returns_404(tmp_path):
    cfg = _empty_cfg(tmp_path)
    with TestClient(create_app(cfg)) as client:
        resp = client.get("/api/thumbnail/1")
        assert resp.status_code == 404


def test_settings_get_and_put(tmp_path):
    cfg = _empty_cfg(tmp_path)
    with TestClient(create_app(cfg)) as client:
        listing = client.get("/api/settings")
        assert listing.status_code == 200
        fields = {f["key"]: f for f in listing.json()["fields"]}
        assert "search.hybrid" in fields
        assert fields["search.hybrid"]["editable"] is True

        ok = client.put(
            "/api/settings", json={"search.hybrid": True, "search.hybrid_alpha": 0.3}
        )
        assert ok.status_code == 200
        assert cfg.search.hybrid is True
        assert (tmp_path / "settings.json").exists()

        bad = client.put("/api/settings", json={"model.name": "nope"})
        assert bad.status_code == 400
