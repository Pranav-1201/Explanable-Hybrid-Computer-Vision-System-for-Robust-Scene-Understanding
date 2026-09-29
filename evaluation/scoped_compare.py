"""M3: does the scoped 24 + other model beat the served 67-class model?

    venv\\Scripts\\python.exe evaluation/scoped_compare.py --field-root "<photo root>" [--scoped models/scoped_best.pth]

Both models are scored with the same definitions (evaluation/field_metrics.py):
an auto-tag is right only when the predicted home class equals the true one, and
a tag on a non-home photo is always wrong. The scoped model is calibrated the way
the served one was: temperature fitted on the seed-42 validation split, rejection
threshold = 5th percentile of calibrated max-prob over correct validation
predictions. Nothing is tuned on the test or field sets.

Writes reports/scoped_compare.md and scoped_compare.json.
"""
import argparse
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from calibration.temperature_scaling import TemperatureScaler  # noqa: E402
from data.dataset_loader import MITIndoorDataset, get_transforms  # noqa: E402
from evaluation import field_eval  # noqa: E402
from evaluation.collect_probs import DEVICE, SEED, VAL_SPLIT, load_served_model, probs_for  # noqa: E402
from evaluation.field_metrics import field_metrics  # noqa: E402
from inference.tta import tta_predict  # noqa: E402
from models.cnn_baseline import CNNBaseline  # noqa: E402
from serving.routing import RoutingPolicy  # noqa: E402
from serving.scope import HOME_CLASS_LABELS, OTHER_CLASS  # noqa: E402
from training.train_scoped import label_map, scoped_classes  # noqa: E402
from utils.checkpoint import load_checkpoint  # noqa: E402

REPORTS = os.path.join(ROOT, "reports")


def load_scoped(path):
    meta = torch.load(path, map_location="cpu", weights_only=True)
    model = CNNBaseline(int(meta["num_classes"]), backbone=meta["backbone"], pretrained=False)
    load_checkpoint(path, model, device=DEVICE, load_ema=bool(meta["best_is_ema"]))
    return model.to(DEVICE).eval(), meta


@torch.no_grad()
def val_logits(model, items):
    tf = get_transforms(train=False)
    out = []
    for i in range(0, len(items), 64):
        x = torch.stack([tf(Image.open(p).convert("RGB")) for p, _ in items[i:i + 64]]).to(DEVICE)
        out.append(model(x).float().cpu())
    return torch.cat(out).numpy()


def calibrate(model, classes67):
    """(TemperatureScaler, rejection threshold) from the seed-42 validation split."""
    mapping = label_map(classes67)
    tr = MITIndoorDataset(root_dir=os.path.join(ROOT, "data", "MIT_Indoor", "train"))
    items = list(zip(tr.image_paths, tr.labels))
    idx = torch.randperm(len(items), generator=torch.Generator().manual_seed(SEED)).tolist()
    val = [items[i] for i in idx[:int(len(items) * VAL_SPLIT)]]
    logits = val_logits(model, val)
    labels = np.array([mapping[y] for _, y in val])
    scaler = TemperatureScaler()
    scaler.fit(logits, labels)
    z = scaler.calibrate_logits(logits)
    p = np.exp(z - z.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    correct = p.argmax(1) == labels
    return scaler, float(np.percentile(p.max(1)[correct], 5)), float(correct.mean())


def score_scoped(model, scaler, thr, classes67, items):
    mapping = np.array(label_map(classes67))
    paths = [p for p, _ in items]
    y67 = np.array([y for _, y in items])
    probs = np.stack([tta_predict(model, Image.open(p).convert("RGB"), DEVICE, scaler=scaler).numpy() for p in paths])
    y = np.where(y67 >= 0, mapping[np.clip(y67, 0, None)], -1)
    return probs, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--field-root", required=True)
    ap.add_argument("--scoped", default=os.path.join(ROOT, "models", "scoped_best.pth"))
    args = ap.parse_args()

    served, s_scaler, classes67 = load_served_model()
    scoped, meta = load_scoped(args.scoped)
    scoped_names = scoped_classes(classes67)
    assert meta["class_names"] == scoped_names, "checkpoint classes differ from scope.py"
    home67 = list(HOME_CLASS_LABELS)
    home25 = [c for c in scoped_names if c != OTHER_CLASS]
    p67 = RoutingPolicy(conf_min=float(json.load(open(os.path.join(ROOT, "models", "calibration_config.json")))["rejection_threshold"]))
    scaler25, thr25, val_acc25 = calibrate(scoped, classes67)
    p25 = RoutingPolicy(conf_min=thr25)
    print(f"scoped: T={scaler25.T:.4f} threshold={thr25:.4f} val acc (single crop)={val_acc25:.4f}", flush=True)

    te = MITIndoorDataset(root_dir=os.path.join(ROOT, "data", "MIT_Indoor", "test"))
    test_items = list(zip(te.image_paths, te.labels))
    import csv
    cidx = {c: i for i, c in enumerate(classes67)}
    field_items = []
    for r in csv.DictReader(open(field_eval.LABELS, encoding="utf-8")):
        folder = "Testing dataset" if r["group"] == "testing_dataset" else "testing"
        field_items.append((os.path.join(args.field_root, folder, r["filename"]), cidx.get(r["true_class"], -1)))

    result = {"scoped_T": scaler25.T, "scoped_threshold": thr25, "served_threshold": p67.conf_min}
    for tag, items in (("test", test_items), ("field", field_items)):
        print(f"{tag}: scoring {len(items)} photos with both models", flush=True)
        paths = [p for p, _ in items]
        y67 = np.array([y for _, y in items])
        a = field_metrics(probs_for(served, s_scaler, paths), y67, classes67, home67, p67)
        pr, y25 = score_scoped(scoped, scaler25, thr25, classes67, items)
        b = field_metrics(pr, y25, scoped_names, home25, p25)
        result[tag] = {"served_67": a, "scoped_25": b}

    os.makedirs(REPORTS, exist_ok=True)
    json.dump(result, open(os.path.join(REPORTS, "scoped_compare.json"), "w"), indent=2, default=str)
    rows = ["# Scoped (24 + other) vs served (67-class)", "",
            f"Scoped: T={scaler25.T:.4f}, threshold {thr25:.4f} (5th pct of correct validation confidence). "
            f"Served threshold {p67.conf_min:.4f}. Same metric definitions for both (evaluation/field_metrics.py).", ""]
    for tag in ("test", "field"):
        a, b = result[tag]["served_67"], result[tag]["scoped_25"]
        rows += [f"## {tag} (n={a['n']}, home photos={a['n_true_home']})", "",
                 "| Metric | served 67-class | scoped 25-class |", "|---|---|---|"]
        for k in ("auto_tagged", "right_tags", "wrong_tags", "tag_precision", "home_tag_recall",
                  "non_home_false_tags", "non_home_photos", "reviewed", "review_precision"):
            f = lambda v: "n/a" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))
            rows.append(f"| {k} | {f(a[k])} | {f(b[k])} |")
        rows += ["", f"Tag precision 95% CI: served {a['tag_precision_ci95'][0]:.3f} to {a['tag_precision_ci95'][1]:.3f}, "
                     f"scoped {b['tag_precision_ci95'][0]:.3f} to {b['tag_precision_ci95'][1]:.3f}.", ""]
    md = "\n".join(rows)
    open(os.path.join(REPORTS, "scoped_compare.md"), "w", encoding="utf-8").write(md)
    print(md)


if __name__ == "__main__":
    main()
