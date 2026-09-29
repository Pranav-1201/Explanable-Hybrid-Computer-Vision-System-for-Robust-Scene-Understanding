# Test checklist

A change is "done" when the commands below have been run **after** it and their
output read. A claim of success without this output is not a claim. Run from the
repo root with the project venv (`venv\Scripts\python.exe` on Windows); the global
Python lacks the dependencies and fails at collection.

## Every change

| # | Command | Expect |
|---|---|---|
| 1 | `venv\Scripts\python.exe -m pytest tests/ -q -p no:cacheprovider` | `N passed`, no failures, no errors. **Read N**: it must not fall from the last recorded count (see HANDOVER). 0 collected means the run proved nothing. |
| 2 | `venv\Scripts\python.exe -m ruff check app.py serve.py serving/ inference/ models/*.py utils/ tests/ scripts/smoke_test.py scripts/field_rerun.py scripts/export_serving_artifacts.py evaluation/field_metrics.py evaluation/field_eval.py evaluation/collect_probs.py --select F` | `All checks passed!` (this is CI's exact lint step; add new modules to it) |
| 3 | `git status --short` | only the files you meant to change; delete stray zero-byte files (names taken from `->` in text) |

## Change touches serving, routing or the model

| # | Command | Expect |
|---|---|---|
| 4 | `venv\Scripts\python.exe serve.py`, then `curl http://127.0.0.1:5000/health` | `"status":"ok"`, `"baseline_loaded":true`, `"num_classes":67` |
| 5 | `venv\Scripts\python.exe scripts/smoke_test.py --base http://127.0.0.1:5000` | exits 0 |
| 6 | `venv\Scripts\python.exe scripts/field_rerun.py --out <dir> "<photo root>\Testing dataset" "<photo root>\testing"` | `TOTAL format rejects: 0`; per folder `invalid=0` (72 files, 4 of them AVIF) |
| 7 | `venv\Scripts\python.exe evaluation/field_eval.py --field-root "<photo root>"` | prints the metrics table and writes `reports/field_eval.md`; compare with the recorded baseline in `reports/field_eval_baseline.json` |

## Change touches the frontend

Unit tests never run JS or render images; a CSP regression that blanked the whole
page passed all of them (DECISIONS D4). Load the page in a real browser
(Playwright), upload mixed photos, and confirm: thumbnails render, no console
errors, the correction dropdown works, the CSV downloads with `corrected_tag`.

## CI (what runs on push and PR to `main`)

- `ci`: install CPU torch + `requirements-test.txt`, ruff (F rules), pytest.
- `docker-smoke`: build `serve` and `test` images, assert the image has no dataset
  or weights, run the suite inside the image, start the container, run
  `scripts/smoke_test.py` against it.

Both must be green: `gh run list --limit 4`. CI cannot run the field evaluation
(the photos are private and outside the repo); item 7 is a local gate.

## Reading a result honestly

- Compare any difference to its spread. The field set has 72 photos and 26
  home-class ones; a one-photo change is 1.4 to 3.8 points and means nothing.
- A metric measured on the set a threshold was tuned on is not evidence.
- When a number matters, name what it counted (photos, tagged photos, home photos).
