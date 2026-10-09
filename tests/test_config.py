import yaml

from tsr.config import load_config


def test_override_file_is_merged_over_default(tmp_path):
    override = tmp_path / "override.yaml"
    override.write_text(
        yaml.safe_dump({"gallery": {"max_images": 7}, "model": {"device": "cpu"}}),
        encoding="utf-8",
    )
    cfg = load_config(override)
    assert cfg.gallery.max_images == 7
    assert cfg.model.device == "cpu"
    # keys not present in the override still come from default.yaml
    assert cfg.gallery.source in {"coco", "coco_hf", "local"}
    assert int(cfg.app.port) > 0


def test_load_config_without_path_uses_defaults():
    cfg = load_config()
    assert "gallery" in cfg
    assert "model" in cfg
