"""Tests for the SQLite metadata store."""

from __future__ import annotations

from tsr.db import GalleryDB


def _rows(n: int) -> list[dict]:
    return [
        {
            "image_id": f"img_{i}",
            "source": "test",
            "split": "val",
            "path": f"val/{i}.jpg",
            "width": 256,
            "height": 256,
            "captions": [f"a photo of object number {i}"],
            "category": None,
        }
        for i in range(n)
    ]


def test_insert_and_get(tmp_path):
    db = GalleryDB(tmp_path / "gallery.db")
    try:
        ids = db.insert_many(_rows(3))
        assert ids == [1, 2, 3]
        assert db.count() == 3
        rec = db.get(ids[0])
        assert rec is not None
        assert rec.image_id == "img_0"
        assert rec.captions == ["a photo of object number 0"]
    finally:
        db.close()


def test_bm25_search_finds_caption_terms(tmp_path):
    db = GalleryDB(tmp_path / "gallery.db")
    try:
        db.insert_many(_rows(5))
        hits = db.bm25_search("object", 10)
        assert len(hits) == 5
    finally:
        db.close()


def test_clear_removes_rows_and_fts(tmp_path):
    db = GalleryDB(tmp_path / "gallery.db")
    try:
        db.insert_many(_rows(5))
        assert db.count() == 5
        assert db.bm25_search("object", 10)

        db.clear()

        assert db.count() == 0
        assert db.bm25_search("object", 10) == []
        # AUTOINCREMENT sequence is reset, so ids restart at 1.
        assert db.insert_many(_rows(1)) == [1]
    finally:
        db.close()
