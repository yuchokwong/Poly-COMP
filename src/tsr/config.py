"""Configuration loading.

Layering (lowest to highest precedence):
    1. configs/default.yaml
    2. optional file from $TSR_CONFIG
    3. environment variables prefixed with ``TSR_`` (nested keys use ``__``)

Example: ``TSR_APP__PORT=9000`` sets ``cfg.app.port = 9000``.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"


class AttrDict(dict):
    """dict with attribute access, recursively applied."""

    def __getattr__(self, item: str) -> Any:
        try:
            return self[item]
        except KeyError as exc:  # pragma: no cover - defensive
            raise AttributeError(item) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


def _coerce(value: str) -> Any:
    v = value.strip()
    low = v.lower()
    if low in {"true", "false"}:
        return low == "true"
    if low in {"null", "none", ""}:
        return None
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    if (v.startswith("{") and v.endswith("}")) or (v.startswith("[") and v.endswith("]")):
        try:
            return json.loads(v)
        except json.JSONDecodeError:
            return value
    return value


def _env_overrides(prefix: str = "TSR_") -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for key, value in os.environ.items():
        if not key.startswith(prefix):
            continue
        parts = key[len(prefix):].lower().split("__")
        node = overrides
        ok = True
        for part in parts[:-1]:
            nxt = node.get(part)
            if nxt is None:
                nxt = {}
                node[part] = nxt
            if not isinstance(nxt, dict):
                ok = False
                break
            node = nxt
        if ok:
            node[parts[-1]] = _coerce(value)
    return overrides


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _to_attr(obj: Any) -> Any:
    if isinstance(obj, dict):
        return AttrDict({k: _to_attr(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_attr(v) for v in obj]
    return obj


def load_config(path: str | Path | None = None) -> AttrDict:
    """Load and validate configuration into an attribute-accessible dict.

    ``configs/default.yaml`` is always the base; an optional override file (given
    by ``path`` or ``$TSR_CONFIG``) is deep-merged on top of it, then ``TSR_*``
    environment variables on top of that. Override files therefore only need the
    keys they change (see ``configs/lite.yaml``).
    """
    raw: dict[str, Any] = {}
    if DEFAULT_CONFIG_PATH.exists():
        raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8")) or {}

    if path is None:
        path = os.environ.get("TSR_CONFIG")
    if path is not None:
        override_path = Path(path)
        if override_path.exists() and override_path.resolve() != DEFAULT_CONFIG_PATH.resolve():
            raw = _deep_merge(
                raw, yaml.safe_load(override_path.read_text(encoding="utf-8")) or {}
            )

    raw = _deep_merge(raw, _env_overrides())
    cfg = _to_attr(raw)
    cfg["root"] = str(PROJECT_ROOT)

    data_dir = Path(cfg.paths.data_dir)
    if not data_dir.is_absolute():
        data_dir = (PROJECT_ROOT / data_dir).resolve()
    cfg.paths["data_dir"] = str(data_dir)

    cache_dir = Path(cfg.paths.cache_dir)
    if not cache_dir.is_absolute():
        cache_dir = (data_dir / cache_dir).resolve()
    cfg.paths["cache_dir"] = str(cache_dir)

    return cfg


# --------------------------------------------------------------------------- #
# Path helpers
# --------------------------------------------------------------------------- #
def data_dir(cfg: AttrDict) -> Path:
    return Path(cfg.paths.data_dir)


def images_dir(cfg: AttrDict) -> Path:
    return data_dir(cfg) / cfg.paths.images_subdir


def db_path(cfg: AttrDict) -> Path:
    return data_dir(cfg) / cfg.paths.db_name


def index_path(cfg: AttrDict) -> Path:
    return data_dir(cfg) / cfg.paths.index_name


def ids_path(cfg: AttrDict) -> Path:
    return data_dir(cfg) / cfg.paths.ids_name


def manifest_path(cfg: AttrDict) -> Path:
    return data_dir(cfg) / cfg.paths.manifest_name


def eval_queries_path(cfg: AttrDict) -> Path:
    return data_dir(cfg) / cfg.paths.eval_name


def report_path(cfg: AttrDict) -> Path:
    return data_dir(cfg) / cfg.paths.report_name


def cache_dir(cfg: AttrDict) -> Path:
    return Path(cfg.paths.cache_dir)


def web_dir(cfg: AttrDict) -> Path:
    return Path(cfg.root) / cfg.paths.web_dir


def local_root(cfg: AttrDict) -> Path:
    root = Path(cfg.gallery.local_root)
    if not root.is_absolute():
        root = Path(cfg.root) / root
    return root
