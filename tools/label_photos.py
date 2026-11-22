"""Write images/hand_labels.csv for the photos made by tools/make_photos.py.

    python tools/label_photos.py

hand_labels.csv needs, for every hand, the NUMBER the notebook gives that hand
(the number show_detections() draws). To get exactly the same numbers, this
script runs the notebook's own find_hands() code (copied out of the notebook at
run time, so it can never drift from it). Each detected hand is then matched to
the real hand it lies on, using images/label_maps/ (which pixels belong to which
placed hand) and images/composite_truth.csv (that hand's owner and shape).

A real hand the detector missed is still written, with hand_index 9: the notebook
then counts it as a miss in Section 3.2 instead of silently dropping it.
"""
import csv
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = ROOT / "RPS1_Project_Report.ipynb"
MISSED = 9


def load_find_hands():
    """Run the notebook cell that defines find_hands() and return the function."""
    cells = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
    source = next("".join(c["source"]) for c in cells
                  if c["cell_type"] == "code" and "def find_hands" in "".join(c["source"]))
    namespace = {"np": np, "Path": Path}
    import os
    cwd = os.getcwd()
    os.chdir(ROOT)                      # the notebook uses paths relative to the repo
    exec(source, namespace)
    os.chdir(cwd)
    namespace["MODEL_PATH"] = ROOT / "models" / "hand_landmarker.task"
    return namespace["find_hands"]


def main():
    find_hands = load_find_hands()
    truth = list(csv.DictReader(open(ROOT / "images" / "composite_truth.csv")))
    by_photo = {}
    for r in truth:
        by_photo.setdefault((r["split"], r["filename"]), []).append(r)

    rows, missed, extra = [], 0, 0
    for (split, name), hands in sorted(by_photo.items()):
        found = find_hands(ROOT / "images" / split / name)
        label_map = cv2.imread(str(ROOT / "images" / "label_maps" / f"{Path(name).stem}.png"), 0)
        h, w = label_map.shape
        owner_of = {}                                   # placed hand id -> detection index
        for i, pts in enumerate(found):
            # look at the 21 points; the placed hand most of them land on is this hand
            ids = [label_map[min(h - 1, max(0, int(y * h))), min(w - 1, max(0, int(x * w)))] for x, y in pts]
            ids = [k for k in ids if k > 0]
            if not ids:
                extra += 1
                continue
            hand_id = max(set(ids), key=ids.count)
            if hand_id not in owner_of:                 # first detection of a hand wins
                owner_of[hand_id] = i
            else:
                extra += 1
        for r in hands:
            i = owner_of.get(int(r["hand_id"]), MISSED)
            missed += i == MISSED
            rows.append({"filename": name, "split": split, "hand_index": i,
                         "owner": r["owner"], "shape": r["shape"]})

    with open(ROOT / "images" / "hand_labels.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["filename", "split", "hand_index", "owner", "shape"])
        wr.writeheader()
        wr.writerows(rows)
    print(f"wrote {len(rows)} hand labels for {len(by_photo)} photos "
          f"({missed} real hands not detected, {extra} extra detections)")


if __name__ == "__main__":
    main()
