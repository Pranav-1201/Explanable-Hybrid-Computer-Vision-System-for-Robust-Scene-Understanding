"""Re-run real field-photo folders through a live server and write a CSV per
folder in the frontend's export format. Reports rows rejected for format.

    python scripts/field_rerun.py --base http://127.0.0.1:5000 --out <dir> <folder> [<folder> ...]
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from smoke_test import CHUNK, multipart, post  # noqa: E402

HEADER = ["filename", "room_tag", "prediction", "confidence", "in_scope", "review_reason", "error"]


def run_folder(base, folder, out_dir):
    names = sorted(n for n in os.listdir(folder) if os.path.isfile(os.path.join(folder, n)))
    rows = []
    for start in range(0, len(names), CHUNK):
        files = []
        for n in names[start:start + CHUNK]:
            with open(os.path.join(folder, n), "rb") as f:
                files.append(("images", n, f.read()))
        body, ctype = multipart(files)
        status, data = post(base + "/predict_batch", body, ctype, 600)
        if status != 200:
            raise SystemExit(f"{folder} chunk@{start}: HTTP {status} {data}")
        rows.extend(data["results"])
    out = os.path.join(out_dir, os.path.basename(os.path.normpath(folder)).replace(" ", "_") + ".csv")
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        for r in rows:
            w.writerow([r.get("filename"), r.get("label", ""), r.get("prediction", ""),
                        r.get("confidence", ""), r.get("in_scope", False),
                        r.get("review_reason") or "", r.get("error", "")])
    format_rejects = [r for r in rows if "Unsupported image format" in (r.get("error") or "")]
    invalid = [r for r in rows if r.get("review_reason") == "invalid"]
    tagged = sum(1 for r in rows if r.get("in_scope"))
    print(f"{folder}: files={len(names)} tagged={tagged} review={len(rows) - tagged} "
          f"invalid={len(invalid)} format_rejects={len(format_rejects)} -> {out}")
    for r in invalid:
        print(f"    invalid: {r['filename']}: {r.get('error')}")
    return len(format_rejects)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:5000")
    ap.add_argument("--out", required=True)
    ap.add_argument("folders", nargs="+")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    total = sum(run_folder(args.base, f, args.out) for f in args.folders)
    print(f"TOTAL format rejects: {total}")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
