# Constraints

What an AI session (or a human) must not do in this repo. "Allow" means "allow
within these lines". Each rule names its source, so a rule that no longer holds
can be found and removed rather than argued with.

## Never

| # | Rule | Source |
|---|---|---|
| 1 | Add AI or assistant attribution (`Co-Authored-By`, "Generated with...") to any commit, PR, tag or release. Pranav is the sole developer of record. | Pranav's global `CLAUDE.md` |
| 2 | Commit field photos. The 72 real photos stay outside the repo; only `evaluation/field_labels.csv` (filename -> class) is tracked. | Pranav, 2026-09-29 |
| 3 | Publish anything outward (GitHub release, Hugging Face repo or Space, registry push, tag) without asking first. Phase F needs Pranav's accounts. | Global `CLAUDE.md` (confirm hard-to-reverse actions); roadmap Phase F |
| 4 | Loosen a guard test to make a change pass. If an approved change breaks one, the test is the finding: revert the change. | Pranav's operating constitution, R3 |
| 5 | Tune a threshold on the data it is then scored on. Thresholds come from the validation split; the test set and the field set are for reporting. | R6; used for M2 (see DECISIONS D5) |
| 6 | Return a traceback to a client. Log it server-side against an `error_id`. | Audit N14, `app.py` |
| 7 | Re-enable the HOG-SVM arm or serve the fusion arm. Fusion is a documented negative result (-1.2 pts vs CNN-only). | README, AUDIT_REPORT B-8 |

## Always

| # | Rule | Source |
|---|---|---|
| 8 | `git fetch` before trusting the local checkout; `git pull --ff-only` only on a clean tree. | Global `CLAUDE.md` |
| 9 | Change which rooms count as "home" in `serving/scope.py` only. The frontend reads it from `GET /classes`; never copy the list. | `app.py` comment (F2), this repo |
| 10 | Pin torch as `+cpu` builds and install with `--extra-index-url`, not `--index-url`. | DECISIONS D2 |
| 11 | Check `requirements-*.txt` before importing a new package in shipped code, and diff the resolved package set after changing dependencies. | Pranav's constitution, R3 |
| 12 | Keep the frontend one file with no build step and no external scripts, styles, fonts or images (the CSP allows none). | `app.py` CSP comment, README |
| 13 | Read the count, not just the exit code: a green run that collected zero tests proved nothing. | Constitution, R3 |
| 14 | Run `git status --short` after every commit and delete stray zero-byte files before staging (explicit paths only). | Constitution, section 5 |

## Ask first

- Adding a runtime dependency (it changes the Docker image and CI resolution).
- Anything that needs the weights re-published (`model-v*`), including a new
  scoped model: see [ROLLBACK.md](ROLLBACK.md).
- Moving rate limiting or metrics to a shared store (currently single-process by design).
