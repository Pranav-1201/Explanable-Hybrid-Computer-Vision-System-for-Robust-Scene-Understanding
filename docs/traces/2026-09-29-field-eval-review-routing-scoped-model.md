# feature: field eval set, review routing, scoped model (roadmap Phase B: M5, M2, M3)
- Status: done (M5 built, M2 built and off, M3 measured and not shipped)   - Opened: 2026-09-29
- Model/session: Claude Sonnet 5.5 (`claude-sonnet-5-5`), session 59024edd

## Found / scoped
Phase B acceptance (`FAANG_AUDIT_ROADMAP.md`): field-set home-class top-1 >= 90% with M3; the
cloister and nursery cases route to review; review precision reported; a before/after table committed
to `reports/`. The field photos were thought lost with the `E:\` drive; they were found in
`D:\Projects and Research papers\...` (62 + 10 = 72 files, 4 AVIF, matching the Phase A plan).

## Tried
- Phase A acceptance re-run through the live server: 72 files, 0 invalid, 0 format rejects.
- M5: labels drafted from filenames by script; all 72 matched a class (4 marked `ood`). Only 26 are
  home-class photos, so the set cannot support the 90% claim (D8).
- M2: margin and entropy thresholds from the validation split; tag precision +0.016, recall -0.015 on
  test, no change on the field set. Six scores compared by AUROC; none beats max softmax on val or test
  (`reports/review_routing.md`).
- M3: fine-tuned a 25-way head (15 epochs, val 90.86%). Test precision 0.898 -> 0.913, recall 0.799 ->
  0.708; field 0.875 -> 0.900 and 0.808 -> 0.692 (`reports/scoped_compare.md`).
- Process notes: a first Mahalanobis script used a triple `einsum` and ran for minutes on 1,340 x 67 x
  2,048; rewritten as one matrix product per class. RAM fell to about 0.5 GB while other sessions ran, and
  training slowed from 60 to 240 s per epoch until memory recovered.

## Worked
- `evaluation/field_labels.csv`, `collect_probs.py`, `field_metrics.py`, `field_eval.py`: a repeatable
  field evaluation with Wilson intervals, and a `--check` gate (tolerance 1 photo). Baseline recorded in
  `reports/field_eval_baseline.json`: 24 auto-tagged, 21 right, 3 wrong, 2 non-home false tags.
- `serving/routing.py`: one function decides review reason; optional rules off by default (D7).
- Why the unmet acceptance lines stayed unmet: the two target errors are confident, clear-margin
  predictions (0.91 and 0.99 calibrated confidence). A different threshold on the same probabilities
  cannot separate them, and retraining the head on the same backbone did not either (D10).

## Verified
- `field_eval.py --check` exits 0 against the baseline; `--tolerance -5` forces exit 1 (the gate can fail).
- Planted-bug runs: `serving/routing.py` 8 of 8 killed by `tests/test_routing.py`;
  `evaluation/field_metrics.py` 8 of 9 killed, the survivor is an equivalent mutant (an OOD label of -1
  never equals a prediction).
- Full suite and CI: see HANDOVER.md for the final counts.

## Left open
- Field home-class top-1 is 0.846 on 26 photos (interval roughly 0.66 to 0.94); the >= 90% criterion can
  only be judged with a larger labelled home-class set.
- The cloister and nursery errors remain. Options that each need new evidence: a stronger backbone (M7),
  more home-room training data (M4), or human review of high-confidence tags for those classes.
- Labels are the owner's filenames; an independent second labelling pass would harden the set.
