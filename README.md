# RPS-1: AI-Assisted Rock-Paper-Scissors-Minus-One

AISE 3350A — Fall 2026  
Western University

## Project Overview

This project combines computer vision, machine learning and game theory to assist a player in Rock-Paper-Scissors-Minus-One (RPS-1).

The system detects four hands, identifies their respective players, estimates Rock, Paper and Scissors probabilities, and calculates a strategy for selecting which hand to retain. Both deterministic and mixed strategies are supported.

## Repository Structure

| File / Folder | Description |
|---|---|
| `RPS1_Project_Report.ipynb` | Main project report, algorithms, experiments and evaluation |
| `live_app.py` | Core live application |
| `photo_demo.py` | Interactive demonstration using five official photographs |
| `webcam_demo.py` | Live webcam application with strategy probabilities and scoreboard |
| `test_photos.py` | Automated validation of the five demonstration photographs |
| `requirements.txt` | Required Python dependencies |
| `LIVE_DEMO.md` | Detailed demonstration and camera setup instructions |
| `images/train/` | 50 training photographs |
| `images/test/` | 80 independent evaluation photographs |
| `images/demo/` | Five real-world demonstration photographs |
| `images/hand_labels.csv` | Hand ownership and gesture annotations |
| `images/photos.csv` | Dataset metadata |
| `images/composite_truth.csv` | Generated image ground-truth information |
| `images/label_maps/` | Dataset annotation resources |
| `models/` | Pre-trained MediaPipe hand detection model |
| `tools/` | Dataset generation and annotation utilities |

## Installation

Python 3.12 is recommended.

```bash
pip install -r requirements.txt
```

## Running the Jupyter Notebook

```bash
jupyter notebook RPS1_Project_Report.ipynb
```

Select **Kernel → Restart & Run All** to execute the project pipeline.

## Photograph Demonstration

```bash
python photo_demo.py
```

The application loads the five official demonstration photographs and displays detected gestures, player ownership, retention probabilities, recommended hand removal and guaranteed expected payoff.

| Key | Action |
|---|---|
| N | Next photograph |
| S | Resample the strategy |
| R | Restart demonstration |
| Q | Quit |

Actual wins and losses are not reported for static photographs because the opponent's final retained hand is unknown.

## Live Webcam Demonstration

```bash
python webcam_demo.py
```

The live application processes camera frames, recommends which hand to remove
