import io
import json
import tarfile

from PIL import Image

from tsr.sources import _iter_webdataset


def _make_tar(path):
    img = Image.new("RGB", (4, 4), (200, 30, 30))
    buf = io.BytesIO()
    img.save(buf, "JPEG")
    img_bytes = buf.getvalue()
    with tarfile.open(path, "w") as tar:
        for i in range(3):
            info = tarfile.TarInfo(f"{i:09d}.jpg")
            info.size = len(img_bytes)
            tar.addfile(info, io.BytesIO(img_bytes))
            meta = json.dumps({"captions": [f"cap {i}", "  extra  "], "cocoid": 100 + i})
            data = meta.encode("utf-8")
            info = tarfile.TarInfo(f"{i:09d}.json")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))


def test_iter_webdataset_pairs_images_and_metadata(tmp_path):
    path = tmp_path / "sample.tar"
    _make_tar(path)
    with tarfile.open(path) as tar:
        samples = list(_iter_webdataset(tar))
    assert len(samples) == 3
    stems = [s[0] for s in samples]
    assert stems == ["000000000", "000000001", "000000002"]
    stem, raw, caps, cocoid = samples[0]
    assert raw[:2] == b"\xff\xd8"  # JPEG magic
    assert caps == ["cap 0", "extra"]  # whitespace stripped
    assert cocoid == 100


def test_iter_webdataset_skips_metadata_without_image(tmp_path):
    path = tmp_path / "lone.tar"
    data = json.dumps({"captions": ["x"], "cocoid": 1}).encode("utf-8")
    with tarfile.open(path, "w") as tar:
        info = tarfile.TarInfo("000000000.json")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    with tarfile.open(path) as tar:
        assert list(_iter_webdataset(tar)) == []
