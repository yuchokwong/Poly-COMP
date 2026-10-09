import pytest

from tsr.config import load_config
from tsr.settings import apply_settings, get_settings


def test_get_settings_marks_editable_fields():
    cfg = load_config()
    fields = {f["key"]: f for f in get_settings(cfg)}
    assert fields["search.hybrid"]["value"] == cfg.search.hybrid
    assert fields["search.hybrid"]["editable"] is True
    assert fields["model.name"]["editable"] is False


def test_apply_settings_validates_and_sets():
    cfg = load_config()
    apply_settings(cfg, {"search.hybrid": True, "search.hybrid_alpha": 0.5})
    assert cfg.search.hybrid is True
    assert abs(cfg.search.hybrid_alpha - 0.5) < 1e-9


def test_apply_settings_rejects_readonly_and_bad_values():
    cfg = load_config()
    with pytest.raises(ValueError):
        apply_settings(cfg, {"model.name": "whatever"})
    with pytest.raises(ValueError):
        apply_settings(cfg, {"search.rerank": "bogus"})
    with pytest.raises(ValueError):
        apply_settings(cfg, {"search.hybrid_alpha": 5.0})


def test_apply_settings_is_atomic_on_error():
    cfg = load_config()
    before = cfg.search.hybrid
    with pytest.raises(ValueError):
        apply_settings(cfg, {"search.hybrid": True, "search.rerank": "bogus"})
    assert cfg.search.hybrid == before  # nothing was applied


def test_list_value_coercion():
    cfg = load_config()
    apply_settings(cfg, {"search.prompt_templates": "a {}\nb {}"})
    assert cfg.search.prompt_templates == ["a {}", "b {}"]
