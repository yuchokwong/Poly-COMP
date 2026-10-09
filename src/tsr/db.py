"""SQLite metadata store for the image gallery.

The database keeps one row per indexed image. Row ids are stable and are used as
the identifier shared with the FAISS index (see ``index_ids.npy``). An FTS5
virtual table enables optional keyword (BM25) retrieval for hybrid search.
"""

from __future__ import annotations

import json
import re
import sqlite3
import threading
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS images (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    image_id    TEXT UNIQUE NOT NULL,
    source      TEXT,
    split       TEXT,
    path        TEXT NOT NULL,
    width       INTEGER,
    height      INTEGER,
    captions    TEXT,
    category    TEXT,
    search_text TEXT,
    created_at  TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_images_image_id ON images(image_id);
CREATE INDEX IF NOT EXISTS idx_images_split ON images(split);
CREATE VIRTUAL TABLE IF NOT EXISTS images_fts
    USING fts5(search_text, content='images', content_rowid='id');
CREATE TRIGGER IF NOT EXISTS images_ai AFTER INSERT ON images BEGIN
    INSERT INTO images_fts(rowid, search_text) VALUES (new.id, new.search_text);
END;
CREATE TRIGGER IF NOT EXISTS images_ad AFTER DELETE ON images BEGIN
    INSERT INTO images_fts(images_fts, rowid, search_text) VALUES ('delete', old.id, old.search_text);
END;
CREATE TRIGGER IF NOT EXISTS images_au AFTER UPDATE ON images BEGIN
    INSERT INTO images_fts(images_fts, rowid, search_text) VALUES ('delete', old.id, old.search_text);
    INSERT INTO images_fts(rowid, search_text) VALUES (new.id, new.search_text);
END;
"""

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


@dataclass
class ImageRecord:
    id: int
    image_id: str
    source: str | None
    split: str | None
    path: str
    width: int | None
    height: int | None
    captions: list[str]
    category: str | None

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> ImageRecord:
        try:
            captions = json.loads(row["captions"]) if row["captions"] else []
        except (json.JSONDecodeError, TypeError):
            captions = []
        return cls(
            id=row["id"],
            image_id=row["image_id"],
            source=row["source"],
            split=row["split"],
            path=row["path"],
            width=row["width"],
            height=row["height"],
            captions=captions,
            category=row["category"],
        )


class GalleryDB:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    # ------------------------------------------------------------------ #
    def clear(self) -> None:
        # ``images_fts`` is an external-content table: deleting from it directly
        # corrupts the index ("database disk image is malformed"). Removing rows
        # from the content table fires the images_ad trigger, which syncs FTS.
        with self._lock:
            self.conn.executescript(
                "DELETE FROM images; "
                "DELETE FROM sqlite_sequence WHERE name='images';"
            )
            self.conn.commit()

    def insert_many(self, rows: Iterable[dict]) -> list[int]:
        """Insert rows in order and return the assigned ids (same order)."""
        ids: list[int] = []
        with self._lock:
            cur = self.conn.cursor()
            for row in rows:
                captions = row.get("captions") or []
                search_text = " ".join(captions)
                if row.get("category"):
                    search_text = f"{search_text} {row['category']}".strip()
                cur.execute(
                    """
                    INSERT INTO images
                        (image_id, source, split, path, width, height, captions, category, search_text)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        row["image_id"],
                        row.get("source"),
                        row.get("split"),
                        row["path"],
                        row.get("width"),
                        row.get("height"),
                        json.dumps(captions, ensure_ascii=False),
                        row.get("category"),
                        search_text,
                    ),
                )
                ids.append(int(cur.lastrowid))
            self.conn.commit()
        return ids

    def get(self, image_pk: int) -> ImageRecord | None:
        with self._lock:
            row = self.conn.execute(
                "SELECT * FROM images WHERE id = ?", (image_pk,)
            ).fetchone()
        return ImageRecord.from_row(row) if row else None

    def get_many(self, ids: Sequence[int]) -> dict[int, ImageRecord]:
        if not ids:
            return {}
        placeholders = ",".join("?" for _ in ids)
        with self._lock:
            rows = self.conn.execute(
                f"SELECT * FROM images WHERE id IN ({placeholders})", tuple(ids)
            ).fetchall()
        return {int(r["id"]): ImageRecord.from_row(r) for r in rows}

    def count(self) -> int:
        with self._lock:
            return int(self.conn.execute("SELECT COUNT(*) FROM images").fetchone()[0])

    def iter_all(self) -> list[ImageRecord]:
        with self._lock:
            rows = self.conn.execute("SELECT * FROM images ORDER BY id").fetchall()
        return [ImageRecord.from_row(r) for r in rows]

    def bm25_search(self, query: str, k: int) -> list[tuple[int, float]]:
        """Return (image_pk, score) ranked by BM25 (higher is better)."""
        terms = _TOKEN_RE.findall(query.lower())
        if not terms:
            return []
        match = " OR ".join(f'"{t}"' for t in terms)
        try:
            with self._lock:
                rows = self.conn.execute(
                    """
                    SELECT rowid AS id, bm25(images_fts) AS score
                    FROM images_fts
                    WHERE images_fts MATCH ?
                    ORDER BY score
                    LIMIT ?
                    """,
                    (match, k),
                ).fetchall()
        except sqlite3.OperationalError:
            return []
        # bm25 returns a negative score where more negative = better.
        return [(int(r["id"]), -float(r["score"])) for r in rows]

    def close(self) -> None:
        with self._lock:
            self.conn.close()
