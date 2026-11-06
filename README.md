# RPS-1: AI-Assisted Rock-Paper-Scissors-Minus-One

AISE 3350A FW26 course project, Western University.

We upload a photo of the four hands. Machine learning finds each hand, decides whether it belongs to our player or the opponent, and gives the probability that it is Rock, Paper or Scissors. Game theory then recommends which hand to keep.

**Start here:** open `RPS1_Project_Report.ipynb` and read the **STILL TO FINISH** section at the top.

## What's in this repo

| File / folder | What it is |
|---|---|
| `RPS1_Project_Report.ipynb` | The report and all the code: strategy, vision, simulations, tests |
| `requirements.txt` | Python packages to install |
| `models/README.md` | Link to download the pre-trained hand model (`hand_landmarker.task`) into `models/` |
| `images/train/` | Photos for training the two classifiers (owner and shape) |
| `images/test/` | Photos **only** for measuring accuracy, never for training |
| `images/demo/` | The 5 photos we submit |
| `images/hand_labels.csv` | One row per hand: photo, hand number, owner (us/them), shape (R/P/S) |
| `images/photos.csv` | One row per photo: lighting, background, people, notes |

## Running

```bash
pip install -r requirements.txt
jupyter notebook RPS1_Project_Report.ipynb
```

Then **Kernel → Restart & Run All**. Everything runs without photos; the vision training and evaluation switch on automatically once photos, labels and the model file are added.

## Status

| Part | Status |
|---|---|
| Strategy: both stages, two cross-checked solvers | Done and tested |
| Uncertainty-aware decisions and vision-error simulation | Done |
| Adaptive (epsilon-safe) mode and its simulation | Done and tested |
| Vision: hand detection, owner classifier, shape classifier | Code done, tested on simulated hands; needs photos, labels and the model file |
| Display | Done |
