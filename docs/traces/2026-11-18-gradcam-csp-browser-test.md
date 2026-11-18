# bug: the Grad-CAM image was blocked by the CSP while the button said "shown"
- Status: fixed locally, not yet in CI   - Opened: 2026-11-15 (audit X1, X2, X9)   - Model/session: Claude Sonnet 5.5 (`claude-sonnet-5-5`), session 62c83a86, 2026-11-18
- Roadmap: Phase 0 rows CI1 and FE10 (`CV_DL_Elite_Roadmap_2026-11-15.docx`, kept outside the repo).

## Found / scoped
The 2026-09-24 fix (`2026-09-24-csp-blanked-frontend.md`) allowed `img-src 'self' blob:`
and said a browser run "is what finds the next one". Nobody then clicked "Explain".
`explain()` set `#cam-img.src` to a `data:image/png;base64,...` URL, which that policy
does not allow, and then changed the button to "Grad-CAM shown" without checking the image.
Two smaller defects came from the same audit: `.wrap { padding: 0 24px }` outranked the
bare `main { padding: 40px 0 80px }`, so the page had 0 px top and bottom padding (X2);
the drop zone said "JPEG, PNG or WEBP" while the server also accepts HEIC, AVIF and BMP (X9).

## Tried
Allowing `data:` in `img-src` would also work, but widens a policy whose whole job is to
allow nothing it does not need. Decoded the base64 in the page into a `Blob` instead:
`blob:` is already allowed.
First draft of the test used Playwright's `wait_for_function`; it evaluates a string with
`eval`, which the page's own CSP forbids, so the test died on its tooling. Bypassing the
CSP in the test would have hidden this exact bug, so the test polls from Python instead.

## Worked
- `scripts/browser_test.py`: opens the real page, uploads 3 photos, opens a card, clicks
  Explain and requires `#cam-img.naturalWidth > 0`, no CSP console error, vertical
  padding >= 24 px and a format hint that names HEIC and AVIF.
- `frontend/index.html`: base64 -> `Blob` -> `URL.createObjectURL` (previous URL revoked),
  `main.wrap` padding, new hint text.
- `requirements-browser.txt` (dev only) and a new step in `.github/workflows/docker-smoke.yml`
  that runs the test against the booted container.

## Verified (local, real model on the dev server)
- Original `index.html`: **4 failed** (padding 0 px, hint text, naturalWidth 0, CSP violation) of 8 checks.
- Fixed `index.html`: **0 failed**, 8 checks, `naturalWidth=224`.
- `pytest tests/ -q`: 132 passed. `ruff check scripts/browser_test.py --select F,E9`: clean.
- Not verified: the new CI step has not run on GitHub; the commit before the fix has not
  been run in CI to show it red. Nothing has been committed or pushed.

## Left open
The rest of Phase 0: BK2, DC3, CI3 (Dependabot, branch protection needs Pranav's GitHub admin).
