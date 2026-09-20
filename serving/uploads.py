"""Upload validation for /predict and /predict_batch (B-12, B3).

Every upload is decoded to prove it is an image of an allowed format; the
filename and content-type are never trusted. Limits are per file; the Flask
request cap (MAX_REQUEST_BYTES) only has to admit one frontend chunk.
"""
import os

import pillow_heif
from PIL import Image

# iPhone default capture format. AVIF is decoded natively by Pillow 12.
pillow_heif.register_heif_opener()

MAX_UPLOAD_BYTES  = 10 * 1024 * 1024        # per file
FILES_PER_REQUEST = 8                       # frontend CHUNK_SIZE must match
MAX_REQUEST_BYTES = FILES_PER_REQUEST * MAX_UPLOAD_BYTES + 10 * 1024 * 1024  # + multipart headroom
ALLOWED_FORMATS   = {"JPEG", "PNG", "WEBP", "BMP", "AVIF", "HEIF"}
MIN_SIDE_PX       = 32                      # below this the CNN input is meaningless
MAX_SIDE_PX       = 10_000
MAX_TOTAL_PIXELS  = 40_000_000              # ~40 MP decompression-bomb guard

# Pillow raises DecompressionBombError past this instead of allocating the pixels.
Image.MAX_IMAGE_PIXELS = MAX_TOTAL_PIXELS


class UploadError(ValueError):
    """A rejected upload. Carries the HTTP status the caller should receive.

    Separate from unexpected server faults so that bad input yields a clean 4xx
    with a short, user-facing reason instead of a 500.
    """

    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.message = message
        self.status = status


def load_validated_image(file_storage) -> Image.Image:
    """Validate an uploaded file and return a decodable PIL image.

    Checks, in order: non-empty and within the byte limit; decodes as a real
    image of an allowed format (the declared filename/content-type is never
    trusted); and has sane dimensions. Raises UploadError with an appropriate
    4xx status; the caller turns that into a JSON error response.
    """
    stream = file_storage.stream
    stream.seek(0, os.SEEK_END)
    size = stream.tell()
    stream.seek(0)

    if size == 0:
        raise UploadError("Uploaded file is empty.")
    if size > MAX_UPLOAD_BYTES:
        raise UploadError(
            f"Image exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB per-file limit.", 413)

    # Decode-verify: the only trustworthy signal that this is really an image.
    try:
        probe = Image.open(stream)
        fmt = (probe.format or "").upper()
        probe.verify()                      # catches truncated / corrupt payloads
    except Image.DecompressionBombError:
        raise UploadError(
            "Image is too large to decode safely (decompression-bomb guard).", 413) from None
    except UploadError:
        raise
    except Exception:
        raise UploadError("File is not a readable image.") from None

    if fmt not in ALLOWED_FORMATS:
        raise UploadError(
            f"Unsupported image format {fmt or 'unknown'!r}. "
            f"Accepted: {', '.join(sorted(ALLOWED_FORMATS))}. "
            f"Convert the photo to JPEG and try again.")

    # verify() consumes the file object, so reopen for actual use.
    stream.seek(0)
    img = Image.open(stream)

    w, h = img.size
    if w < MIN_SIDE_PX or h < MIN_SIDE_PX:
        raise UploadError(
            f"Image is too small ({w}x{h}); minimum is {MIN_SIDE_PX}x{MIN_SIDE_PX}.")
    if w > MAX_SIDE_PX or h > MAX_SIDE_PX:
        raise UploadError(
            f"Image is too large ({w}x{h}); maximum side is {MAX_SIDE_PX}px.")
    if w * h > MAX_TOTAL_PIXELS:
        raise UploadError(
            f"Image has too many pixels ({w * h}); maximum is {MAX_TOTAL_PIXELS}.")

    # Force full decode to catch truncated pixel data (Pillow verify() does not
    # detect it for JPEG or HEIF; the error would surface later at convert(),
    # inside /predict_batch's batch-wide error handler, 500'ing the whole request).
    try:
        img.load()
    except Exception:
        raise UploadError("File is not a readable image (it may be truncated or corrupt).") from None

    return img
