"""M5 / T3: score the served model on the 72 labelled field photos.

    venv\\Scripts\\python.exe evaluation/field_eval.py --field-root "<folder holding 'Testing dataset' and 'testing'>"
    ... --write-baseline     record today's numbers as the regression baseline
    ... --check              exit 1 if the model regressed against the baseline

The photos live outside the repo (docs/CONSTRAINTS.md #2); the ground truth is
evaluation/field_labels.csv. The model is deterministic, so any change from the
baseline means the model, the calibration or the routing policy changed. The
check tolerates `--tolerance` photos (default 1) so a re-export that flips one
borderline photo does not fail a build, and nothing larger slips through.

CI cannot run this (private photos); it is a local gate. See TEST_CHECKLIST.md.
"""
import argparse
import csv
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

from evaluation.field_metrics import field_metrics  # noqa: E402
from serving.routing import RoutingPolicy  # noqa: E402
from serving.scope import HOME_CLASS_LABELS  # noqa: E402

REPORTS = os.path.join(ROOT, "reports")
BASELINE = os.path.join(REPORTS, "field_eval_baseline.json")
LABELS = os.path.join(ROOT, "evaluation", "field_labels.csv")
GATED = ("right_tags", "wrong_tags", "non_home_false_tags", "reviewed", "warranted_reviews")


def load_policy():
    cfg = json.load(open(os.path.join(ROOT, "models", "calibration_config.json")))
    return RoutingPolicy(conf_min=float(cfg["rejection_threshold"]))


def score(field_root, weights=None):
    from evaluation.collect_probs import load_served_model, probs_for
    model, scaler, classes = load_served_model(weights)
    cidx = {c: i for i, c in enumerate(classes)}
    paths, labels, names = [], [], []
    for r in csv.DictReader(open(LABELS, encoding="utf-8")):
        folder = "Testing dataset" if r["group"] == "testing_dataset" else "testing"
        p = os.path.join(field_root, folder, r["filename"])
        if not os.path.isfile(p):
            raise SystemExit(f"missing field photo: {p}")
        paths.append(p)
        labels.append(cidx.get(r["true_class"], -1))
        names.append(r["filename"])
    probs = probs_for(model, scaler, paths)
    return probs, np.array(labels), names, classes


def fmt(v):
    return "n/a" if v is None else f"{v:.3f}"


def render(m, policy):
    lo, hi = m["tag_precision_ci95"]
    rlo, rhi = m["home_tag_recall_ci95"]
    return "\n".join([
        "# Field evaluation",
        "",
        f"Served model, TTA on, calibrated. Policy: confidence >= {policy.conf_min:.4f} and a home class.",
        f"{m['n']} photos: {m['n_labelled']} labelled with one of the 67 classes, {m['n_ood']} matching none.",
        f"{m['n_true_home']} of them are home-class photos (the only ones that can score a room tag).",
        "",
        "| Metric | Value | Counted over |",
        "|---|---|---|",
        f"| Top-1 (67-class) | {fmt(m['top1_labelled'])} | {m['n_labelled']} labelled photos |",
        f"| Home-class top-1 | {fmt(m['home_top1'])} | {m['n_true_home']} home photos |",
        f"| Auto-tagged | {m['auto_tagged']} | {m['n']} photos |",
        f"| Tag precision | {fmt(m['tag_precision'])} (95% CI {lo:.2f} to {hi:.2f}) | {m['auto_tagged']} auto-tagged |",
        f"| Home tag recall | {fmt(m['home_tag_recall'])} (95% CI {rlo:.2f} to {rhi:.2f}) | {m['n_true_home']} home photos |",
        f"| Non-home photos given a home tag | {m['non_home_false_tags']} | {m['non_home_photos']} non-home or OOD photos |",
        f"| Reviewed | {m['reviewed']} ({fmt(m['review_rate'])}) | {m['n']} photos |",
        f"| Review precision | {fmt(m['review_precision'])} | {m['reviewed']} reviewed |",
        f"| Review reasons | {m['reasons']} | |",
        "",
        "With this few photos the intervals above are wide: a one-photo change moves a",
        "proportion by several points and is not evidence of a better or worse model.",
        "",
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--field-root", required=True)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--write-baseline", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--tolerance", type=int, default=1)
    args = ap.parse_args()

    policy = load_policy()
    probs, labels, names, classes = score(args.field_root, args.weights)
    m = field_metrics(probs, labels, classes, list(HOME_CLASS_LABELS), policy)
    os.makedirs(REPORTS, exist_ok=True)
    md = render(m, policy)
    open(os.path.join(REPORTS, "field_eval.md"), "w", encoding="utf-8").write(md)
    print(md)

    if args.write_baseline:
        keep = {k: m[k] for k in m if not k.endswith("ci95")}
        json.dump({"policy_conf_min": policy.conf_min, "metrics": keep}, open(BASELINE, "w"), indent=2)
        print(f"baseline written -> {BASELINE}")
    if args.check:
        base = json.load(open(BASELINE))["metrics"]
        bad = []
        # More wrong tags / false tags is a regression; fewer right tags is too.
        if m["wrong_tags"] > base["wrong_tags"] + args.tolerance:
            bad.append(f"wrong_tags {base['wrong_tags']} -> {m['wrong_tags']}")
        if m["non_home_false_tags"] > base["non_home_false_tags"] + args.tolerance:
            bad.append(f"non_home_false_tags {base['non_home_false_tags']} -> {m['non_home_false_tags']}")
        if m["right_tags"] < base["right_tags"] - args.tolerance:
            bad.append(f"right_tags {base['right_tags']} -> {m['right_tags']}")
        if bad:
            print("REGRESSION vs baseline:", "; ".join(bad))
            sys.exit(1)
        print(f"no regression vs baseline (tolerance {args.tolerance} photo)")


if __name__ == "__main__":
    main()
