from pathlib import Path
import cv2
import numpy as np
from live_app import load_notebook_code, Vision

DEMO_DIR = Path(__file__).resolve().parent / "images" / "demo"

expected = [
    ("R", "P", "R", "R"),
    ("S", "P", "P", "P"),
    ("P", "S", "P", "R"),
    ("S", "P", "P", "R"),
    ("R", "P", "P", "R"),
]

ns = load_notebook_code()
vision = Vision(ns)

passed = 0

for i, truth in enumerate(expected, 1):
    path = DEMO_DIR / f"demo{i}.jpg"
    print(f"\n--- DEMO {i} ---")

    frame = cv2.imread(str(path))

    if frame is None:
        print("FAIL: Image could not be loaded")
        continue

    hands = vision.analyse(frame)

    mine = sorted(
        [h for h in hands if h["ours"]],
        key=lambda h: h["x"]
    )

    theirs = sorted(
        [h for h in hands if not h["ours"]],
        key=lambda h: h["x"]
    )

    if len(mine) != 2 or len(theirs) != 2:
        print("FAIL: Expected two hands per player")
        continue

    detected = tuple(h["letter"] for h in mine + theirs)

    print("Expected:", truth)
    print("Detected:", detected)

    labels_ok = detected == truth

    A = ns["payoff_matrix"](
        [list(h["probs"]) for h in mine],
        [list(h["probs"]) for h in theirs]
    )

    p, value = ns["solve_game"](A)

    print("Payoff matrix:\n", A)
    print(f"Keep LEFT:  {p[0]:.1%}")
    print(f"Keep RIGHT: {p[1]:.1%}")
    print(f"Expected payoff guarantee: {value:.3f}")

    probabilities_ok = (
        np.all(np.isfinite(p))
        and np.all(np.array(p) >= -1e-8)
        and np.isclose(np.sum(p), 1.0)
    )

    guaranteed = np.min(np.array(p) @ A)

    strategy_ok = probabilities_ok and np.isclose(
        guaranteed, value, atol=1e-5
    )

    if labels_ok and strategy_ok:
        print("PASS")
        passed += 1
    else:
        print("FAIL")

print(f"\nFINAL RESULT: {passed}/{len(expected)} passed")

if passed != len(expected):
    raise SystemExit(1)
