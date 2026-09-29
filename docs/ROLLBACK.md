# Rollback

How to undo a change. Code and model are versioned independently, so they roll
back independently.

## Code

1. Find the last good commit: `gh run list --branch main` (both `ci` and
   `docker-smoke` green) or `git log --oneline`.
2. Undo without rewriting history: `git revert <bad-sha>` (one commit) or
   `git revert <oldest-bad>^..<newest-bad>` (a range). Never force-push over
   commits others may have.
3. Re-check: the whole of [TEST_CHECKLIST.md](TEST_CHECKLIST.md) items 1 to 5,
   then wait for both workflows on the revert commit.

Each roadmap phase landed as small commits with the item id in the subject
(`B7`, `F2`, `T2`...), so a single feature reverts cleanly.

## Model weights

The served weights are named by `models/serving_manifest.json` (filename, URL,
sha256, size) and fetched from a GitHub release at startup, then verified.

- **Current:** release `model-v1`, `phase2_ema.pth`, sha256 `6ef66350...c2c6925dd`.
- **To roll back to a previous model:** `git revert` the commit that changed the
  manifest. The next start downloads and verifies the older file. Do not delete
  an old release asset while it is still the rollback target: keep N-1 published.
- **To point one machine at another file without a commit:** `MODEL_PATH` and
  `MODEL_URL` override the location; the checksum is enforced either way.

If a new model is ever published (for example the scoped 24 + other model from
M3), publish it as a **new** release (`model-v2`), never overwrite `model-v1`,
and change the manifest in its own commit so this rollback stays one revert.

## Behaviour switches that need no rollback

| Setting | Effect | Default |
|---|---|---|
| `REVIEW_MARGIN_MIN`, `REVIEW_ENTROPY_MAX` | enable the optional review rules | unset = off |
| `PREDICT_RATE_LIMIT` | inference rate limit | `30 per minute` |
| `?tta=0` on a request | single crop instead of TTA | TTA on |

Unset the variable and restart to return to the default.

## Local artefacts that are easy to lose

`models/*.pth` is git-ignored. `dist/phase2_ema.pth` is the export of the served
weights; the release copy is the durable one. `models/phase2_best.pth` (full
training checkpoint) exists only on this machine: back it up before wiping it.
