# bug: the Content-Security-Policy blanked the whole frontend
- Status: done   - Opened: 2026-09-24   - Model/session: not recorded (session 6137fae3)
- Reconstructed 2026-09-29 from commit `dff2c9b`, the `app.py` comment and the project memory note.

## Found / scoped
Phase C (B7) added security headers including `Content-Security-Policy: default-src 'self'`.
All 97 tests passed. Loading the page in a real browser during the Phase D
acceptance run showed the page not working: the inline `<script>` and `<style>`
were blocked, and every uploaded-photo thumbnail (a `blob:` URL) was refused.

## Tried
- Why no test caught it: the contract tests drive Flask's `test_client`, which
  executes no JavaScript and renders no images, so no HTTP-level test could see a
  page that no longer runs. `test_security_headers_present` now pins the three
  directives that keep it running (`tests/test_api_contract.py`), but a pin only
  stops this exact regression; the browser run is what finds the next one.

## Worked
Allow exactly what the page needs and nothing external:
`default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' blob:`.
Root cause: the page is deliberately one file with inline script and style (no
build step), and `'self'` matches same-origin http(s) only, not the `blob:` scheme.

## Verified
Real Playwright browser session: uploaded mixed photos, thumbnails rendered, the
correction dropdown worked through to a downloaded CSV, and a patched `fetch`
exercised the retry path. `ci` and `docker-smoke` green; 98 tests passing at the time.

## Left open
`'unsafe-inline'` weakens the XSS protection the header exists for. Removing it
means moving the script and style to external files, which conflicts with the
one-file constraint (docs/CONSTRAINTS.md #12). That trade-off is accepted, not solved.
