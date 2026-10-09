import io

from PIL import Image

from tsr.gallery import _dhash, _resize_save


def _jpeg_bytes(size=(80, 60), color=(10, 120, 200)):
    img = Image.new("RGB", size, color)
    # add a gradient so the difference hash is non-trivial
    px = img.load()
    for x in range(size[0]):
        for y in range(size[1]):
            px[x, y] = ((x * 3) % 256, (y * 5) % 256, color[2])
    buf = io.BytesIO()
    img.save(buf, "JPEG")
    return buf.getvalue()


def test_dhash_is_deterministic_and_hex():
    img = Image.open(io.BytesIO(_jpeg_bytes())).convert("RGB")
    a = _dhash(img)
    b = _dhash(img)
    assert a == b
    assert len(a) == 16
    int(a, 16)  # valid hex


def test_resize_save_returns_size_and_phash(tmp_path):
    dest = tmp_path / "out.jpg"
    (width, height), phash = _resize_save(_jpeg_bytes(), dest, max_side=32, quality=80)
    assert dest.exists()
    assert max(width, height) <= 32
    assert len(phash) == 16
