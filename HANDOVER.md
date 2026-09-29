# Handover

Read this first, then trust none of it until you have run `git fetch`, `git status`
and the test suite (see [docs/TEST_CHECKLIST.md](docs/TEST_CHECKLIST.md)). It records
where things stood at the end of the last session, not what is true now.

**Last updated:** 2026-09-29, session 59024edd, Claude Sonnet 5.5 (`claude-sonnet-5-5`)
**Branch:** `main`. Base was `origin/main` at `dff2c9b`; four local commits follow it (M2, M5/T3, M3, docs).
Check `git log origin/main..HEAD` to see whether they were pushed.
**Tests at last run:** `122 passed`, 122 collected (`pytest tests/ -q`); CI lint command clean.

## Roadmap status (`FAANG_AUDIT_ROADMAP.md`)

| Phase | State |
|---|---|
| A Deployability | done, CI green. Field acceptance re-run 2026-09-29: 72 files, 0 invalid, 0 format rejects, 4 AVIF decoded |
| C Backend hardening | done (B6, B7, B8, T2) |
| D Product UX | done (F1, F2, F4, F5) |
| E CI / regression | T1, T4 done. T3 is a local gate: `evaluation/field_eval.py --check` (CI cannot see the private photos) |
| B Accuracy and trust | M5 done, M2 built and OFF, M3 measured and not shipped. **Acceptance not met**: cloister and nursery still auto-tagged; the 26-photo home subset cannot judge ">= 90%" |
| F Deployment | not started; needs Pranav's Hugging Face account (CONSTRAINTS #3) |

## What Phase B produced

- `serving/scope.py`, `serving/routing.py`: shared home-class list and the review decision (D7, D9). Margin/entropy rules exist, off unless `REVIEW_MARGIN_MIN` / `REVIEW_ENTROPY_MAX` are set. Frontend shows "close call" / "uncertain".
- `evaluation/field_labels.csv` + `field_metrics.py`, `field_eval.py`, `collect_probs.py`: field evaluation with 95% intervals and a `--check` gate. Baseline `reports/field_eval_baseline.json`: 24 auto-tagged, 21 right, 3 wrong, 2 non-home photos tagged (D8).
- `training/train_scoped.py`, `evaluation/scoped_compare.py`, `reports/scoped_compare.md`: the scoped 24 + other model. Test tag precision 0.898 -> 0.913, home recall 0.799 -> 0.708; field 0.875 -> 0.900 and 0.808 -> 0.692. Not shipped (D10). `models/scoped_best.pth` exists locally only (git-ignored).
- `reports/review_routing.md`: why margin/entropy and five other scores do not help.
- Guide documents: `CLAUDE.md`, `docs/` ARCHITECTURE, FLOW, DECISIONS (D1 to D10), CONSTRAINTS, TEST_CHECKLIST, ROLLBACK, `traces/`.

## Where the field photos are

`D:\Projects and Research papers\CV + DL Project - Explainable Hybrid Computer Vision System for Robust Scene Understanding\`
(`Testing dataset\` 62 files, `testing\` 10). They are not in the repo and must not be committed.
The old `E:\` drive is gone; nothing depends on it any more.

## Next steps, in order

1. If the local commits are not on `origin/main`: push, then `gh run list --limit 4` and confirm `ci` and `docker-smoke` are green.
2. Decide with Pranav whether precision or recall matters more for the product. That decides whether M3 is ever revisited (D10).
3. The remaining accuracy levers need new evidence, not more thresholds: more labelled home-room photos (a bigger field set), a stronger backbone (M7), or scraped listing data (M4). All are in the roadmap's deferred list.
4. Phase F only after asking Pranav for the Hugging Face account and confirming what may be published.

## Do not

Re-verify Phase A, commit the photos, publish a release, or loosen a test. See [docs/CONSTRAINTS.md](docs/CONSTRAINTS.md).

## Environment gotchas (Windows)

- Use `venv\Scripts\python.exe`; bare `python`/`python3` lacks dependencies or hangs.
- Set `PYTHONUTF8=1` and `PYTHONIOENCODING=utf-8` before scripts that print non-ASCII.
- `/tmp` in Git Bash is not `/tmp` in PowerShell; use the session scratchpad.
- Memory: the machine had 0.5 to 6 GB free of 24 GB while other sessions ran. One heavy process at a time; training slowed 4x under pressure. A dev `serve.py` on port 5000 may still be running from this session (about 1 GB); stop it if memory is tight.
- Stray zero-byte files named after the token after a `->` or a printed word appear in the repo root (`str`, `thr`, `wine` this session); delete after checking the size is 0.
- Commit message files need a blank line after the subject, or git treats the whole first paragraph as the subject.
