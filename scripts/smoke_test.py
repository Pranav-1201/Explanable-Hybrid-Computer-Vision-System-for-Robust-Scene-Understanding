"""Container smoke test: health, single predictions for JPEG/AVIF/HEIF, and a
20-photo batch sent in chunks of 8. Stdlib HTTP plus Pillow for fixtures.

    python scripts/smoke_test.py --base http://127.0.0.1:7860
"""
import argparse
import io
import json
import sys
import time
import urllib.error
import urllib.request
import uuid

import pillow_heif
from PIL import Image

pillow_heif.register_heif_opener()
CHUNK = 8


def image_bytes(fmt: str, seed: int) -> bytes:
    colour = ((seed * 53) % 256, (seed * 97) % 256, (seed * 29) % 256)
    buf = io.BytesIO()
    Image.new("RGB", (320, 240), colour).save(buf, format=fmt)
    return buf.getvalue()


def multipart(files, fields=None):
    """files: iterable of (field, filename, bytes). Returns (body, content_type)."""
    boundary = uuid.uuid4().hex
    parts = []
    for name, value in (fields or {}).items():
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n'
                     f"{value}\r\n".encode())
    for field, filename, data in files:
        head = (f'--{boundary}\r\nContent-Disposition: form-data; name="{field}"; '
                f'filename="{filename}"\r\nContent-Type: application/octet-stream\r\n\r\n')
        parts.append(head.encode() + data + b"\r\n")
    body = b"".join(parts) + f"--{boundary}--\r\n".encode()
    return body, f"multipart/form-data; boundary={boundary}"


def post(url, body, content_type, timeout):
    req = urllib.request.Request(url, data=body, method="POST",
                                 headers={"Content-Type": content_type})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, json.load(e)


def wait_healthy(base, timeout_s):
    deadline, last = time.monotonic() + timeout_s, None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(base + "/health", timeout=5) as r:
                health = json.load(r)
            if health.get("baseline_loaded"):
                return health
            last = health
        except (urllib.error.URLError, OSError) as e:
            last = e
        time.sleep(3)
    raise SystemExit(f"FAIL: not healthy after {timeout_s}s (last: {last})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:7860")
    ap.add_argument("--health-timeout", type=int, default=600)
    ap.add_argument("--request-timeout", type=int, default=300)
    args = ap.parse_args()

    started = time.monotonic()
    health = wait_healthy(args.base, args.health_timeout)
    classes = set(health["classes"])
    print(f"health ok after {time.monotonic() - started:.0f}s: {len(classes)} classes")
    failures = []

    for fmt in ("JPEG", "AVIF", "HEIF"):
        body, ctype = multipart([("image", f"probe.{fmt.lower()}", image_bytes(fmt, 1))])
        t0 = time.monotonic()
        status, data = post(args.base + "/predict", body, ctype, args.request_timeout)
        label = data.get("original_prediction") or data.get("prediction")
        ok = status == 200 and label in classes
        print(f"/predict {fmt}: status={status} label={label} {time.monotonic() - t0:.1f}s {'OK' if ok else 'FAIL'}")
        if not ok:
            failures.append(f"/predict {fmt}: {status} {data}")

    rows = []
    for start in range(0, 20, CHUNK):
        chunk = [("images", f"b{i}.jpg", image_bytes("JPEG", i)) for i in range(start, min(start + CHUNK, 20))]
        body, ctype = multipart(chunk)
        status, data = post(args.base + "/predict_batch", body, ctype, args.request_timeout)
        print(f"/predict_batch chunk@{start}: status={status} rows={len(data.get('results', []))}")
        if status != 200:
            failures.append(f"/predict_batch chunk@{start}: {status} {data}")
        rows.extend(data.get("results", []))
    errored = [r for r in rows if r.get("error")]
    if len(rows) != 20 or errored:
        failures.append(f"batch: {len(rows)} rows, {len(errored)} errored")

    print(f"SUMMARY: {len(failures)} failure(s)")
    for f in failures:
        print("  -", f)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
