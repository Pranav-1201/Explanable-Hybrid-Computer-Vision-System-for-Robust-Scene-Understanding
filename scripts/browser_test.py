"""Browser test (roadmap CI1): upload photos, open a card, ask for the Grad-CAM
explanation and require that the image really renders.

Why this exists: the page's Content-Security-Policy once blocked the Grad-CAM
`data:` image while the button still read "Grad-CAM shown" (audit X1). No test
opened a browser, so nothing noticed. The assertion that matters is
`naturalWidth > 0` on #cam-img: a blocked image reports 0 even when its src is
set and the button text has changed.

    python scripts/browser_test.py --base http://127.0.0.1:7860

Needs a running server (docker-smoke boots one) and `playwright install chromium`.
Exit code 0 only if every check passed; each check prints PASS or FAIL.
"""
import argparse
import io
import sys
import time

from PIL import Image
from playwright.sync_api import sync_playwright

N_PHOTOS = 3
FAILURES = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILURES.append(name)
    return ok


def poll(predicate, timeout_ms: int, every_s: float = 0.25) -> None:
    """Wait in Python instead of page.wait_for_function: Playwright evaluates
    that string with eval(), which the page's own CSP forbids (and bypassing
    the CSP for the test would hide exactly the bug this test is for)."""
    deadline = time.monotonic() + timeout_ms / 1000
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(every_s)
    raise TimeoutError(f"condition not met within {timeout_ms} ms")


def photo(seed: int) -> dict:
    """A small valid JPEG as a Playwright in-memory upload. Content is
    irrelevant: the test is about the page, not the model's answer."""
    colour = ((seed * 53) % 256, (seed * 97) % 256, (seed * 29) % 256)
    buf = io.BytesIO()
    Image.new("RGB", (320, 240), colour).save(buf, format="JPEG")
    return {"name": f"browser_test_{seed}.jpg", "mimeType": "image/jpeg", "buffer": buf.getvalue()}


def run(base: str, timeout_ms: int) -> None:
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page()
        console_errors = []
        page.on("console", lambda m: console_errors.append(m.text) if m.type == "error" else None)

        page.goto(base, wait_until="domcontentloaded")
        # Audit X2: the page content had 0 px of top and bottom padding.
        pad = page.locator("main").evaluate(
            "el => parseFloat(getComputedStyle(el).paddingTop)")
        check("main has vertical padding", pad >= 24, f"paddingTop={pad}px")
        # Audit X9: the hint must name every format the server accepts.
        hint = page.inner_text(".dz-sub")
        check("format hint names HEIC and AVIF", "HEIC" in hint and "AVIF" in hint, hint)

        page.set_input_files("#file-input", [photo(i) for i in range(1, N_PHOTOS + 1)])

        # The results section appears once the first chunk is classified.
        page.wait_for_selector("#results:not(.hidden)", timeout=timeout_ms)
        poll(lambda: page.locator(".card-open").count() == N_PHOTOS, timeout_ms)
        check(f"{N_PHOTOS} cards rendered", page.locator(".card-open").count() == N_PHOTOS)

        page.locator(".card-open").first.click()
        page.wait_for_selector("#detail[open]", timeout=timeout_ms)
        check("detail dialog opened", True)

        page.click("#explain")
        # Either outcome of the request ends in one of these button states.
        done = ("Grad-CAM shown", "No explanation available", "Explanation failed — retry")
        poll(lambda: page.inner_text("#explain") in done, timeout_ms)
        button = page.inner_text("#explain")
        check("button reports the explanation was produced", button == "Grad-CAM shown", button)

        # Give the browser a moment to decode (or refuse) the data: URL.
        poll(lambda: page.locator("#cam-img").evaluate("el => el.complete"), timeout_ms)
        width = page.locator("#cam-img").evaluate("el => el.naturalWidth")
        check("Grad-CAM image actually decoded (naturalWidth > 0)", width > 0, f"naturalWidth={width}")
        check("Grad-CAM wrapper visible", page.is_visible("#cam-wrap"))

        csp = [e for e in console_errors if "Content Security Policy" in e]
        check("no Content-Security-Policy violations in the console", not csp, "; ".join(csp)[:300])
        browser.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:7860")
    ap.add_argument("--timeout", type=int, default=120, help="seconds per wait")
    args = ap.parse_args()
    try:
        run(args.base.rstrip("/"), args.timeout * 1000)
    except Exception as e:  # a timeout or missing element is itself a failed check
        check("run completed without error", False, f"{type(e).__name__}: {str(e)[:200]}")
    print(f"\n{len(FAILURES)} failed")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
