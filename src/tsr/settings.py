"""Runtime-editable settings, surfaced through the web UI.

Only settings that are safe to change while the server is running (search
behaviour and UI defaults) are editable. Build-time settings (model, index,
gallery) are exposed read-only so the UI can show them. Editable values are
persisted to ``data/settings.json`` and re-applied on startup.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tsr.config import AttrDict
from tsr.logging_utils import get_logger

logger = get_logger(__name__)

SEARCH = "Search"
INTERFACE = "Interface"
MODEL = "Model (read-only)"
INDEX = "Index (read-only)"
GALLERY = "Gallery (read-only)"

FIELDS: list[dict[str, Any]] = [
    {
        "key": "search.hybrid",
        "type": "bool",
        "label": "Hybrid (CLIP + keyword)",
        "group": SEARCH,
        "editable": True,
        "help": "Fuse dense CLIP scores with SQLite BM25 keyword scores.",
    },
    {
        "key": "search.hybrid_alpha",
        "type": "float",
        "label": "Hybrid weight (alpha)",
        "group": SEARCH,
        "editable": True,
        "min": 0.0,
        "max": 1.0,
        "step": 0.05,
        "help": "Weight of the vector score; keyword gets (1 - alpha).",
    },
    {
        "key": "search.use_prompt_ensemble",
        "type": "bool",
        "label": "Prompt ensemble",
        "group": SEARCH,
        "editable": True,
        "help": "Average the query embedding over search.prompt_templates.",
    },
    {
        "key": "search.prompt_templates",
        "type": "list",
        "label": "Prompt templates (one per line)",
        "group": SEARCH,
        "editable": True,
        "help": "Used for prompt ensembling; use {} as the query placeholder.",
    },
    {
        "key": "search.rerank",
        "type": "enum",
        "label": "Reranker",
        "group": SEARCH,
        "editable": True,
        "choices": ["none", "caption", "cross-encoder"],
        "help": "Re-score the top candidates. 'caption' uses CLIP on captions.",
    },
    {
        "key": "search.rerank_top_n",
        "type": "int",
        "label": "Rerank top-N",
        "group": SEARCH,
        "editable": True,
        "min": 1,
        "max": 200,
        "help": "How many first-stage candidates the reranker sees.",
    },
    {
        "key": "search.rerank_alpha",
        "type": "float",
        "label": "Rerank weight",
        "group": SEARCH,
        "editable": True,
        "min": 0.0,
        "max": 1.0,
        "step": 0.05,
    },
    {
        "key": "search.translate",
        "type": "bool",
        "label": "Translate queries (CJK -> EN)",
        "group": SEARCH,
        "editable": True,
        "help": "Translate non-Latin queries to English before encoding.",
    },
    {
        "key": "app.default_top_k",
        "type": "int",
        "label": "Default results",
        "group": INTERFACE,
        "editable": True,
        "min": 1,
        "max": 100,
    },
    {"key": "model.name", "type": "str", "label": "Model", "group": MODEL, "editable": False},
    {
        "key": "model.pretrained",
        "type": "str",
        "label": "Pretrained",
        "group": MODEL,
        "editable": False,
    },
    {"key": "model.backend", "type": "str", "label": "Backend", "group": MODEL, "editable": False},
    {"key": "model.device", "type": "str", "label": "Device", "group": MODEL, "editable": False},
    {"key": "index.type", "type": "str", "label": "Index type", "group": INDEX, "editable": False},
    {"key": "gallery.source", "type": "str", "label": "Source", "group": GALLERY, "editable": False},
    {
        "key": "gallery.max_images",
        "type": "int",
        "label": "Max images",
        "group": GALLERY,
        "editable": False,
    },
    {
        "key": "gallery.image_max_side",
        "type": "int",
        "label": "Image max side",
        "group": GALLERY,
        "editable": False,
    },
]

_SPEC_BY_KEY = {field["key"]: field for field in FIELDS}


def _get(cfg: AttrDict, key: str) -> Any:
    node: Any = cfg
    for part in key.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def _set(cfg: AttrDict, key: str, value: Any) -> None:
    parts = key.split(".")
    node: Any = cfg
    for part in parts[:-1]:
        node = node[part]
    node[parts[-1]] = value


def coerce(field: dict[str, Any], value: Any) -> Any:
    kind = field["type"]
    if kind == "bool":
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)
    if kind == "int":
        return int(value)
    if kind == "float":
        return float(value)
    if kind == "enum":
        value = str(value)
        if value not in field.get("choices", []):
            raise ValueError(f"{field['key']}: must be one of {field['choices']}")
        return value
    if kind == "list":
        if isinstance(value, str):
            return [line.strip() for line in value.splitlines() if line.strip()]
        if isinstance(value, (list, tuple)):
            return [str(item) for item in value]
        raise ValueError(f"{field['key']}: expected a list")
    return str(value)


def validate(field: dict[str, Any], value: Any) -> Any:
    value = coerce(field, value)
    lo = field.get("min")
    hi = field.get("max")
    if lo is not None and value < lo or hi is not None and value > hi:
        raise ValueError(f"{field['key']}: must be between {lo} and {hi}")
    return value


def get_settings(cfg: AttrDict) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for field in FIELDS:
        item = dict(field)
        item["value"] = _get(cfg, field["key"])
        out.append(item)
    return out


def apply_settings(cfg: AttrDict, updates: dict[str, Any]) -> dict[str, Any]:
    """Validate all edits first, then apply them in-place atomically."""
    resolved: dict[str, Any] = {}
    for key, raw in updates.items():
        field = _SPEC_BY_KEY.get(key)
        if field is None or not field.get("editable", False):
            raise ValueError(f"{key}: not an editable setting")
        resolved[key] = validate(field, raw)
    for key, value in resolved.items():
        _set(cfg, key, value)
    return resolved


def settings_path(cfg: AttrDict) -> Path:
    return Path(cfg.paths.data_dir) / "settings.json"


def save_settings(cfg: AttrDict) -> None:
    data = {field["key"]: _get(cfg, field["key"]) for field in FIELDS if field.get("editable")}
    path = settings_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def apply_persisted(cfg: AttrDict) -> None:
    path = settings_path(cfg)
    if not path.exists():
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Ignoring bad settings file %s: %s", path, exc)
        return
    for key, value in data.items():
        field = _SPEC_BY_KEY.get(key)
        if field is None or not field.get("editable"):
            continue
        try:
            _set(cfg, key, validate(field, value))
        except (ValueError, TypeError) as exc:
            logger.warning("Ignoring invalid persisted setting %s=%r: %s", key, value, exc)
