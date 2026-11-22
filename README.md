# RPS-1: AI-Assisted Rock-Paper-Scissors-Minus-One

AISE 3350A FW26 course project, Western University.

We upload a photo of the four hands. The notebook follows a real round: machine learning finds each hand, decides whether it belongs to our player or the opponent, and gives the probability that it is Rock, Paper or Scissors. Game theory then recommends which hand to keep.

**Start here:** open `RPS1_Project_Report.ipynb` and read the **STILL TO FINISH** section at the top.

## What's in this repo

| File / folder | What it is |
|---|---|
| `RPS1_Project_Report.ipynb` | The report and all the code: strategy, vision, simulations, tests |
| `requirements.txt` | Python packages to install |
| `models/hand_landmarker.task` | Google's pre-trained MediaPipe hand model (included; details in `models/README.md`) |
| `images/train/` | Photos for training the two classifiers (owner and shape) |
| `images/test/` | Photos **only** for measuring accuracy, never for training |
| `images/demo/` | The 5 photos we submit |
| `images/hand_labels.csv` | One row per hand: photo, hand number, owner (us/them), shape (R/P/S) |
| `images/photos.csv` | One row per photo: lighting, background, people, notes |
| `images/composite_truth.csv`, `images/label_maps/` | Where each hand was placed in the generated photos (used only for labelling) |
| `tools/make_photos.py` | Builds the four-hand photos from real hand photos (DrGFreeman dataset, CC BY-SA 4.0) |
| `tools/label_photos.py` | Writes `images/hand_labels.csv` using the notebook's own hand finder |

## Running

```bash
pip install -r requirements.txt
jupyter notebook RPS1_Project_Report.ipynb
```

Then **Kernel → Restart & Run All** (about 1–2 minutes). Photos, labels and the hand model are all in the repo, so the vision training, test results and demo photos run straight away.

To rebuild the photos from scratch: download the DrGFreeman "Rock-Paper-Scissors Images" dataset, then run `python tools/make_photos.py <folder containing rock/ paper/ scissors/>` and `python tools/label_photos.py`.

## Status

| Part | Status |
|---|---|
| Strategy: both stages, two cross-checked solvers | Done and tested |
| Uncertainty-aware decisions and vision-error simulation | Done |
| Our own strategy: adaptive mode with a 15-round warm-up, and its simulation | Done and tested |
| Vision: hand detection, owner classifier, shape classifier | Done: 135 labelled photos, trained, tested on 80 unseen photos (Section 3.2) |
| Real photos of our own setup | Recommended before the live competition (Section 2.2) |
| Display | Done |
