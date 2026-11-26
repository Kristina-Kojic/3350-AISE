from pathlib import Path
import cv2
import numpy as np
from live_app import load_notebook_code, Vision

ns = load_notebook_code()
vision = Vision(ns)
expected = [("R","P","R","R"),("P","R","R","R"),("P","S","P","P"),("S","P","P","P"),("S","R","S","S")]
for i, truth in enumerate(expected, 1):
    path = Path(__file__).parent / f"photo{i}.jpg"
    frame = cv2.imread(str(path))
    hands = vision.analyse(frame)
    mine = sorted([h for h in hands if h["ours"]], key=lambda h:h["side"])
    theirs = sorted([h for h in hands if not h["ours"]], key=lambda h:h["x"])
    print(f"\nPHOTO {i}")
    for h in mine+theirs:
        print(h.get("side"), h["ours"], h["letter"], dict(zip(ns["SHAPES"], np.round(h["probs"],3))))
    if len(mine)!=2 or len(theirs)!=2:
        print("FAIL: expected two hands per player")
        continue
    A = ns["payoff_matrix"]([list(h["probs"]) for h in mine], [list(h["probs"]) for h in theirs])
    p, value = ns["solve_game"](A)
    print("Payoff matrix:", A, "Keep probabilities LEFT, RIGHT:", p)
    remove = 1-int(np.argmax(p))
    target = [0,1,0,1,0][i-1]
    print("Most likely removal:", ["LEFT","RIGHT"][remove], "Expected:", ["LEFT","RIGHT"][target], "PASS" if remove==target else "FAIL")
