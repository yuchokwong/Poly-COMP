from tsr.translate import has_cjk


def test_has_cjk_detects_chinese_japanese_korean():
    assert has_cjk("一隻貓在廁所上")
    assert has_cjk("猫")
    assert has_cjk("ねこ")
    assert has_cjk("고양이")


def test_has_cjk_false_for_latin():
    assert not has_cjk("a cat on a toilet")
    assert not has_cjk("")
