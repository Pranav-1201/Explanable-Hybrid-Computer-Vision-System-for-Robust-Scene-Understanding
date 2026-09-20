"""Upload validation, importable without the model (app.py loads it at import)."""
import io
import os
import re

import pytest
from PIL import Image
from werkzeug.datastructures import FileStorage

from serving import uploads
from serving.uploads import UploadError, load_validated_image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _image(fmt="JPEG", size=(64, 48)):
    buf = io.BytesIO()
    Image.new("RGB", size, (120, 80, 40)).save(buf, format=fmt)
    return buf.getvalue()


def _fs(data, name="x.jpg"):
    return FileStorage(stream=io.BytesIO(data), filename=name)


def test_valid_jpeg_is_returned():
    assert load_validated_image(_fs(_image())).size == (64, 48)


def test_empty_file_is_400():
    with pytest.raises(UploadError) as e:
        load_validated_image(_fs(b""))
    assert e.value.status == 400


def test_file_over_per_file_limit_is_413():
    with pytest.raises(UploadError) as e:
        load_validated_image(_fs(b"\0" * (uploads.MAX_UPLOAD_BYTES + 1)))
    assert e.value.status == 413


def test_non_image_is_400():
    with pytest.raises(UploadError) as e:
        load_validated_image(_fs(b"not an image at all"))
    assert e.value.status == 400


def test_too_small_is_400():
    with pytest.raises(UploadError):
        load_validated_image(_fs(_image(size=(16, 16))))


def test_request_cap_fits_a_full_chunk_of_max_size_files():
    assert uploads.MAX_REQUEST_BYTES >= uploads.FILES_PER_REQUEST * uploads.MAX_UPLOAD_BYTES


def test_frontend_chunk_size_matches_server():
    with open(os.path.join(ROOT, "frontend", "index.html"), encoding="utf-8") as f:
        m = re.search(r"const CHUNK_SIZE = (\d+);", f.read())
    assert m, "frontend must declare const CHUNK_SIZE"
    assert int(m.group(1)) == uploads.FILES_PER_REQUEST


@pytest.mark.parametrize("fmt", ["AVIF", "HEIF"])
def test_phone_formats_decode(fmt):
    img = load_validated_image(_fs(_image(fmt), name=f"x.{fmt.lower()}"))
    assert img.convert("RGB").size == (64, 48)


def test_unsupported_format_message_is_actionable():
    with pytest.raises(UploadError) as e:
        load_validated_image(_fs(_image("GIF"), name="x.gif"))
    assert "HEIF" in e.value.message and "convert" in e.value.message.lower()


def _noisy(fmt):
    buf = io.BytesIO()
    Image.effect_noise((320, 240), 60).convert("RGB").save(buf, format=fmt)
    return buf.getvalue()


@pytest.mark.parametrize("fmt", ["JPEG", "HEIF"])
@pytest.mark.parametrize("keep", [0.5, 0.9])
def test_truncated_image_is_a_clean_400(fmt, keep):
    data = _noisy(fmt)
    with pytest.raises(UploadError) as e:
        load_validated_image(_fs(data[: int(len(data) * keep)]))
    assert e.value.status == 400


def test_intact_noisy_image_still_loads():
    assert load_validated_image(_fs(_noisy("JPEG"))).size == (320, 240)
