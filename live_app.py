"""RPS-1 live: our notebook's vision + strategy on a live camera feed.

    python live_app.py                    # default camera
    python live_app.py --camera 1         # another camera
    python live_app.py --rotate           # camera mounted the other way round
    python live_app.py --source images/demo   # no camera: cycle through photos (for testing)

Keys
    N  next round          SPACE  reveal the decision (private mode)
    P  private mode on/off A      adaptive strategy on/off
    S  save a screenshot   R      reset the score        Q / Esc  quit

How it works
    This app does not have its own vision or strategy. It loads the code cells of
    RPS1_Project_Report.ipynb (hand finder, classifiers, game solvers, opponent
    model) and trains the classifiers on images/train exactly like the notebook.
    So what you see here is exactly what the report describes and measures.

A round, as the camera sees it
    1. SHOW      Stage-1 advice ("show Rock + Paper"). Both players show two hands.
    2. LOCK      All four hands found and steady for about a second: the decision
                 is computed once and fixed (keep LEFT or RIGHT).
    3. DECISION  Keep the advised hand, pull the other away.
    4. RESULT    One hand each left on the table: the camera reads both and
                 decides who won. Remove all hands (or press N) for the next round.
"""
import argparse
import json
import os
import queue
import sys
import tempfile
import threading
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "RPS1_Project_Report.ipynb"

# ---------------------------------------------------------------------------
# 1. Load the notebook's code (only the cells we need, no slow simulations)
# ---------------------------------------------------------------------------
NEEDED = ["SEED =", "def find_hands", "def assign_and_classify", "def load_labelled_hands",
          "def solve_game", "def recommend_stage1", "class OpponentModel"]


def load_notebook_code():
    os.environ.setdefault("MPLBACKEND", "Agg")          # no plot windows
    import matplotlib
    matplotlib.use("Agg")
    cells = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
    sources = ["".join(c["source"]) for c in cells if c["cell_type"] == "code"]
    ns = {"__name__": "rps_notebook"}
    cwd = os.getcwd()
    os.chdir(ROOT)                                       # notebook paths are relative
    try:
        for marker in NEEDED:
            src = next((s for s in sources if marker in s), None)
            if src is None:
                sys.exit(f"Could not find the notebook cell containing '{marker}'.")
            exec(compile(src, f"notebook cell [{marker}]", "exec"), ns)
    finally:
        os.chdir(cwd)
    if ns.get("OWNER_MODEL") is None or ns.get("SHAPE_MODEL") is None:
        sys.exit("The classifiers did not train: check images/train, images/hand_labels.csv "
                 "and models/hand_landmarker.task.")
    return ns


# ---------------------------------------------------------------------------
# 2. One frame -> hands with owner, side and shape probabilities
# ---------------------------------------------------------------------------
class Vision:
    def __init__(self, ns):
        self.ns = ns
        self.tmp = Path(tempfile.gettempdir()) / "rps1_live_frame.jpg"

    def analyse(self, frame_bgr):
        ns = self.ns
        cv2.imwrite(str(self.tmp), frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        hands = ns["find_hands"](self.tmp)               # the notebook's own hand finder
        if not hands:
            return []
        p_ours = ns["OWNER_MODEL"].predict_proba(np.array([ns["owner_features"](h) for h in hands]))[:, 1]
        ours = set(ns["pick_ours"](p_ours))
        cols = [list(ns["SHAPE_MODEL"].classes_).index(s) for s in ns["SHAPES"]]
        out = []
        for i, pts in enumerate(hands):
            probs = ns["SHAPE_MODEL"].predict_proba([ns["shape_features"](pts)])[0][cols]
            out.append({"points": pts, "ours": i in ours, "p_ours": float(p_ours[i]),
                        "probs": np.asarray(probs, float), "x": float(pts[0, 0])})
        # name the sides: our hands by the player's own left/right, theirs left-to-right in the image
        mine = sorted([h for h in out if h["ours"]], key=lambda h: h["x"])
        if not ns["IMAGE_LEFT_IS_PLAYERS_LEFT"]:
            mine.reverse()
        for h, side in zip(mine, ("LEFT", "RIGHT")):
            h["side"] = side
        theirs = sorted([h for h in out if not h["ours"]], key=lambda h: h["x"])
        for k, h in enumerate(theirs):
            h["side"] = str(k + 1)
        for h in out:
            h["letter"] = ns["SHAPES"][int(np.argmax(h["probs"]))]
            h["conf"] = float(np.max(h["probs"]))
        return out


class Worker(threading.Thread):
    """Runs the vision in the background so the video stays smooth."""

    def __init__(self, vision):
        super().__init__(daemon=True)
        self.vision, self.inbox = vision, queue.Queue(maxsize=1)
        self.result, self.result_id, self.ms = [], 0, 0.0
        self.lock = threading.Lock()
        self.stopping = False

    def submit(self, frame):
        try:
            self.inbox.put_nowait(frame.copy())
        except queue.Full:
            pass

    def run(self):
        while True:
            frame = self.inbox.get()
            start = time.perf_counter()
            try:
                hands = self.vision.analyse(frame)
            except Exception as err:                     # never crash the live view
                if self.stopping:
                    return
                print("vision error:", repr(err))
                hands = []
            with self.lock:
                self.result, self.result_id = hands, self.result_id + 1
                self.ms = 1000 * (time.perf_counter() - start)


# ---------------------------------------------------------------------------
# 3. The round: SHOW -> DECISION -> RESULT
# ---------------------------------------------------------------------------
BEATS = {("R", "S"), ("S", "P"), ("P", "R")}
STEADY = 4          # identical vision results in a row before we lock (about 0.5-1 s)


def signature(hands, n_ours, n_theirs):
    mine = [h for h in hands if h["ours"]]
    theirs = [h for h in hands if not h["ours"]]
    if len(mine) != n_ours or len(theirs) != n_theirs:
        return None
    return (tuple(h["letter"] for h in sorted(mine, key=lambda h: h["side"])),
            tuple(h["letter"] for h in sorted(theirs, key=lambda h: h["x"])))


class Game:
    def __init__(self, ns, seed=None):
        self.ns, self.rng = ns, np.random.default_rng(seed)
        self.model = ns["OpponentModel"]()
        self.adaptive, self.private = False, False
        self.score = {"win": 0, "tie": 0, "loss": 0}
        self.history = []
        self.new_round()

    def new_round(self):
        self.phase = "SHOW"
        self.advice = self.ns["recommend_stage1"](self.rng)
        self.steady_sig, self.steady_n = None, 0
        self.decision = None
        self.revealed = False
        self.result = None
        self.empty_since = None

    def _steady(self, sig):
        if sig is not None and sig == self.steady_sig:
            self.steady_n += 1
        else:
            self.steady_sig, self.steady_n = sig, (1 if sig else 0)
        return self.steady_n >= STEADY

    def update(self, hands, now):
        ns = self.ns
        if self.phase == "SHOW":
            sig = signature(hands, 2, 2)
            if self._steady(sig):
                mine = sorted([h for h in hands if h["ours"]], key=lambda h: h["side"])
                theirs = sorted([h for h in hands if not h["ours"]], key=lambda h: h["x"])
                my_probs = [list(h["probs"]) for h in mine]
                opp_probs = [list(h["probs"]) for h in theirs]
                A = ns["payoff_matrix"](my_probs, opp_probs)
                keep_probs, value = ns["solve_game"](A)
                mode = "Class strategy (equilibrium)"
                if self.adaptive:
                    eps = ns["final_safety_limit"](self.model.rounds_seen)
                    opp_letters = [h["letter"] for h in theirs]
                    keep_probs = ns["safe_exploit"](A, self.model.predict(opp_letters), eps)
                    mode = f"Adaptive (safety limit {eps:.2f})"
                keep = int(self.rng.choice(2, p=keep_probs))
                self.decision = {"keep": keep, "keep_probs": keep_probs, "mine": mine, "theirs": theirs,
                                 "guarantee": float(ns["guaranteed_payoff"](A, keep_probs[0])),
                                 "mode": mode, "time": now}
                self.phase, self.steady_sig, self.steady_n = "DECISION", None, 0
                self.revealed = not self.private
        elif self.phase == "DECISION":
            sig = signature(hands, 1, 1)
            if self._steady(sig):
                ours = next(h for h in hands if h["ours"])
                theirs = next(h for h in hands if not h["ours"])
                me, them = ours["letter"], theirs["letter"]
                outcome = "tie" if me == them else ("win" if (me, them) in BEATS else "loss")
                self.score[outcome] += 1
                # which of their two hands did the opponent keep? (same shape, closest position)
                locked = self.decision["theirs"]
                kept = min(range(2), key=lambda k: (locked[k]["letter"] != them,
                                                    abs(locked[k]["x"] - theirs["x"])))
                self.model.update([h["letter"] for h in locked], kept)
                advised = self.decision["mine"][self.decision["keep"]]["letter"]
                self.result = {"me": me, "them": them, "outcome": outcome,
                               "followed": me == advised, "advised": advised}
                self.history.append(outcome)
                self.phase, self.revealed = "RESULT", True
        elif self.phase == "RESULT":
            if not hands:
                self.empty_since = self.empty_since or now
                if now - self.empty_since > 1.5:
                    self.new_round()
            else:
                self.empty_since = None

    def progress(self):
        return min(1.0, self.steady_n / STEADY)


# ---------------------------------------------------------------------------
# 4. Drawing (OpenCV for the video, Pillow for clean text)
# ---------------------------------------------------------------------------
from PIL import Image, ImageDraw, ImageFont

W, H = 1360, 765                     # window size
PANEL = 440                          # side panel width
VIDEO_W, VIDEO_H = W - PANEL, H
BG = (17, 24, 39)                    # panel colours (RGB)
CARD = (31, 41, 55)
TEXT = (243, 244, 246)
MUTED = (156, 163, 175)
OURS = (42, 157, 143)                # teal, same as the notebook's charts
THEIRS = (231, 111, 81)              # orange
GOOD, BAD, WARN = (74, 222, 128), (248, 113, 113), (250, 204, 21)
NAME = {"R": "Rock", "P": "Paper", "S": "Scissors"}
CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10),
               (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18),
               (18, 19), (19, 20), (0, 17)]


def font(size, bold=False):
    import matplotlib
    folder = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(str(folder / name), size)
    except OSError:
        return ImageFont.load_default()


F = {k: font(*v) for k, v in {"h1": (30, True), "h2": (20, True), "big": (44, True), "body": (18, False),
                              "bodyb": (18, True), "small": (14, False), "tag": (16, True)}.items()}


def rounded(d, box, fill, r=14, outline=None, width=2):
    d.rounded_rectangle(box, radius=r, fill=fill, outline=outline, width=width)


def bar(d, x, y, w, h, frac, colour, back=(55, 65, 81)):
    rounded(d, (x, y, x + w, y + h), back, r=h // 2)
    if frac > 0:
        rounded(d, (x, y, x + max(h, int(w * frac)), y + h), colour, r=h // 2)


def fit_video(frame):
    """Scale the camera frame into the video area, keeping its shape (letterbox)."""
    h, w = frame.shape[:2]
    s = min(VIDEO_W / w, VIDEO_H / h)
    nw, nh = int(w * s), int(h * s)
    canvas = np.zeros((VIDEO_H, VIDEO_W, 3), np.uint8)
    x0, y0 = (VIDEO_W - nw) // 2, (VIDEO_H - nh) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = cv2.resize(frame, (nw, nh))
    return canvas, (x0, y0, nw, nh)


def draw_video(frame, hands, game, box):
    """Hand skeletons, labels and the decision highlight, drawn on the video."""
    x0, y0, nw, nh = box
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(img, "RGBA")
    # middle line: their side above, our side below
    y_mid = y0 + nh // 2
    for x in range(x0, x0 + nw, 24):                     # dark + light dashes: visible on any table
        d.line((x, y_mid + 1, x + 12, y_mid + 1), fill=(0, 0, 0, 110), width=3)
        d.line((x, y_mid, x + 12, y_mid), fill=(255, 255, 255, 150), width=1)
    for text, ty, colour in (("OPPONENT'S SIDE  ▲", y_mid - 30, THEIRS), ("YOUR SIDE  ▼", y_mid + 8, OURS)):
        tw = d.textlength(text, font=F["small"])
        rounded(d, (x0 + 10, ty, x0 + 30 + tw, ty + 22), colour + (200,), r=8)
        d.text((x0 + 20, ty + 3), text, font=F["small"], fill=(255, 255, 255))

    keep_side = None
    if game.decision and game.revealed and game.phase == "DECISION":
        keep_side = ("LEFT", "RIGHT")[game.decision["keep"]]
    for h in hands:
        pts = [(x0 + px * nw, y0 + py * nh) for px, py in h["points"]]
        colour = OURS if h["ours"] else THEIRS
        if keep_side and h["ours"]:                      # highlight the hand to keep
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
            ring = GOOD if h["side"] == keep_side else BAD
            d.rounded_rectangle((min(xs) - 18, min(ys) - 18, max(xs) + 18, max(ys) + 18),
                                radius=16, outline=ring + (255,), width=5)
        for a, b in CONNECTIONS:
            d.line((pts[a], pts[b]), fill=colour + (230,), width=4)
        for p in pts:
            d.ellipse((p[0] - 4, p[1] - 4, p[0] + 4, p[1] + 4), fill=(255, 255, 255, 240))
        who = f"YOU · {h['side']}" if h["ours"] else f"OPP · {h['side']}"
        label = f"{who}   {NAME[h['letter']]} {h['conf']:.0%}"
        wx, wy = pts[0]
        tw = d.textlength(label, font=F["tag"])
        ty = wy + 14 if h["ours"] and wy + 44 < y0 + nh else wy - 44
        ty = min(max(ty, y0 + 6), y0 + nh - 34)
        tx = min(max(wx - tw / 2 - 10, x0 + 6), x0 + nw - tw - 26)
        rounded(d, (tx, ty, tx + tw + 20, ty + 30), colour + (235,), r=10)
        d.text((tx + 10, ty + 5), label, font=F["tag"], fill=(255, 255, 255))
        if h["conf"] < 0.70:
            d.text((tx, ty + 32), "low confidence - check", font=F["small"], fill=WARN)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def draw_panel(game, hands, fps, vision_ms):
    img = Image.new("RGB", (PANEL, H), BG)
    d = ImageDraw.Draw(img)
    x, y, w = 24, 22, PANEL - 48
    d.text((x, y), "RPS-1  LIVE", font=F["h1"], fill=TEXT)
    chip = {"SHOW": ("STAGE 1 · SHOW", OURS), "DECISION": ("STAGE 2 · DECIDE", WARN),
            "RESULT": ("ROUND RESULT", GOOD)}[game.phase]
    tw = d.textlength(chip[0], font=F["small"])
    rounded(d, (x + w - tw - 20, y + 4, x + w, y + 32), chip[1], r=14)
    d.text((x + w - tw - 10, y + 10), chip[0], font=F["small"], fill=BG)
    y += 58

    # ---- main card: what to do now
    rounded(d, (x, y, x + w, y + 250), CARD, r=18)
    cx, cy = x + 20, y + 18
    if game.phase == "SHOW":
        d.text((cx, cy), "SHOW THESE TWO HANDS", font=F["small"], fill=MUTED)
        a, b = game.advice
        d.text((cx, cy + 26), f"{NAME[a]} + {NAME[b]}", font=F["big"], fill=TEXT)
        d.text((cx, cy + 92), "Then hold all four hands still.", font=F["body"], fill=MUTED)
        n_mine = sum(h["ours"] for h in hands)
        n_theirs = len(hands) - n_mine
        d.text((cx, cy + 130), f"Seeing  {n_mine}/2 yours   {n_theirs}/2 theirs", font=F["bodyb"],
               fill=GOOD if (n_mine, n_theirs) == (2, 2) else WARN)
        d.text((cx, cy + 168), "Locking decision", font=F["small"], fill=MUTED)
        bar(d, cx, cy + 192, w - 40, 14, game.progress(), OURS)
    elif game.phase == "DECISION" and not game.revealed:
        d.text((cx, cy), "DECISION READY", font=F["small"], fill=MUTED)
        d.text((cx, cy + 26), "Hidden", font=F["big"], fill=TEXT)
        d.text((cx, cy + 92), "Private mode is on.", font=F["body"], fill=MUTED)
        d.text((cx, cy + 122), "Only our player looks, then", font=F["body"], fill=MUTED)
        d.text((cx, cy + 150), "press SPACE to reveal.", font=F["bodyb"], fill=WARN)
    elif game.phase == "DECISION":
        dec = game.decision
        side = ("LEFT", "RIGHT")[dec["keep"]]
        shape = NAME[dec["mine"][dec["keep"]]["letter"]]
        d.text((cx, cy), "KEEP YOUR", font=F["small"], fill=MUTED)
        d.text((cx, cy + 22), f"{side} HAND", font=F["big"], fill=GOOD)
        d.text((cx, cy + 80), f"({shape})  ·  pull the other one away", font=F["body"], fill=TEXT)
        for k, lab in enumerate(("LEFT", "RIGHT")):
            yy = cy + 122 + 34 * k
            letter = dec["mine"][k]["letter"]
            d.text((cx, yy), f"{lab:<5} {NAME[letter]}", font=F["body"], fill=TEXT)
            bar(d, cx + 170, yy + 4, w - 250, 14, float(dec["keep_probs"][k]),
                GOOD if k == dec["keep"] else (107, 114, 128))
            d.text((x + w - 66, yy), f"{dec['keep_probs'][k]:.0%}", font=F["body"], fill=TEXT)
        d.text((cx, cy + 196), f"Guaranteed expected score {dec['guarantee']:+.2f} per round",
               font=F["small"], fill=MUTED)
    else:
        r = game.result
        colour, word = {"win": (GOOD, "YOU WIN"), "tie": (WARN, "TIE"), "loss": (BAD, "YOU LOSE")}[r["outcome"]]
        d.text((cx, cy), "THE CAMERA SAW", font=F["small"], fill=MUTED)
        d.text((cx, cy + 22), word, font=F["big"], fill=colour)
        d.text((cx, cy + 84), f"You {NAME[r['me']]}  vs  {NAME[r['them']]} Opponent", font=F["bodyb"], fill=TEXT)
        note = "You followed the advice." if r["followed"] else f"Advice was to keep {NAME[r['advised']]}."
        d.text((cx, cy + 120), note, font=F["body"], fill=MUTED)
        d.text((cx, cy + 168), "Remove all hands or press N", font=F["body"], fill=MUTED)
        d.text((cx, cy + 194), "for the next round.", font=F["body"], fill=MUTED)
    y += 270

    # ---- what the camera sees
    d.text((x, y), "WHAT THE CAMERA SEES", font=F["small"], fill=MUTED)
    y += 26
    rows = sorted(hands, key=lambda h: (not h["ours"], h["side"]))
    if not rows:
        d.text((x, y), "No hands yet", font=F["body"], fill=MUTED)
        y += 30
    for h in rows[:4]:
        colour = OURS if h["ours"] else THEIRS
        rounded(d, (x, y + 4, x + 10, y + 22), colour, r=4)
        who = f"You {h['side'].lower()}" if h["ours"] else f"Opponent {h['side']}"
        d.text((x + 20, y), who, font=F["body"], fill=TEXT)
        d.text((x + 175, y), NAME[h["letter"]], font=F["bodyb"], fill=TEXT)
        d.text((x + w - 52, y), f"{h['conf']:.0%}", font=F["body"],
               fill=TEXT if h["conf"] >= 0.70 else WARN)
        y += 30
    y = max(y, 470) + 10

    # ---- score
    rounded(d, (x, y, x + w, y + 92), CARD, r=18)
    s = game.score
    played = sum(s.values())
    for k, (lab, val, col) in enumerate((("WINS", s["win"], GOOD), ("TIES", s["tie"], WARN),
                                         ("LOSSES", s["loss"], BAD))):
        cx = x + 20 + k * (w - 40) // 3
        d.text((cx, y + 14), lab, font=F["small"], fill=MUTED)
        d.text((cx, y + 36), str(val), font=F["h1"], fill=col)
    avg = (s["win"] - s["loss"]) / played if played else 0.0
    y += 104
    d.text((x, y), f"{played} round{'' if played == 1 else 's'} · average {avg:+.2f} points per round", font=F["small"], fill=MUTED)
    mode = "Adaptive (learns their habits)" if game.adaptive else "Class strategy (unbeatable mix)"
    d.text((x, y + 22), f"Strategy: {mode}", font=F["small"], fill=MUTED)
    d.text((x, y + 44), f"Private mode: {'ON' if game.private else 'off'}", font=F["small"],
           fill=WARN if game.private else MUTED)
    d.text((x, y + 66), f"Video {fps:.0f} fps  ·  hands read every {vision_ms:.0f} ms", font=F["small"], fill=MUTED)

    # ---- footer
    d.line((x, H - 70, x + w, H - 70), fill=(55, 65, 81), width=1)
    d.text((x, H - 58), "N next  ·  SPACE reveal  ·  P private  ·  A adaptive", font=F["small"], fill=MUTED)
    d.text((x, H - 36), "S screenshot  ·  R reset  ·  Q quit", font=F["small"], fill=MUTED)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


# ---------------------------------------------------------------------------
# 5. Camera or test photos
# ---------------------------------------------------------------------------
class PhotoSource:
    """Pretends to be a camera by showing photos in turn (for testing without a camera)."""

    def __init__(self, folder, seconds=3.0):
        self.files = sorted(Path(folder).glob("*.jp*g")) + sorted(Path(folder).glob("*.png"))
        if not self.files:
            sys.exit(f"No photos in {folder}")
        self.seconds, self.start = seconds, time.time()

    def read(self):
        k = int((time.time() - self.start) / self.seconds) % len(self.files)
        time.sleep(1 / 30)
        return True, cv2.imread(str(self.files[k]))

    def release(self):
        pass


def open_camera(index):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW) if sys.platform == "win32" else cv2.VideoCapture(index)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        sys.exit(f"Could not open camera {index}. Close other apps using it (Zoom, Teams, Camera) "
                 f"or try --camera 1.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return cap


# ---------------------------------------------------------------------------
# 6. Main loop
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="RPS-1 live camera app")
    ap.add_argument("--camera", type=int, default=0, help="camera number (0, 1, ...)")
    ap.add_argument("--rotate", action="store_true", help="turn the picture 180 degrees")
    ap.add_argument("--source", help="folder of photos to use instead of a camera (testing)")
    ap.add_argument("--private", action="store_true", help="start in private mode")
    ap.add_argument("--adaptive", action="store_true", help="start with the adaptive strategy")
    ap.add_argument("--record", help="(testing) save every Nth rendered screen to this folder")
    ap.add_argument("--frames", type=int, default=0, help="(testing) stop after this many frames")
    args = ap.parse_args()

    print("Loading the notebook's code and training the classifiers (about 10-20 s)...")
    ns = load_notebook_code()
    worker = Worker(Vision(ns))
    worker.start()
    game = Game(ns)
    game.private, game.adaptive = args.private, args.adaptive
    source = PhotoSource(args.source) if args.source else open_camera(args.camera)
    headless = bool(args.record)
    if not headless:
        cv2.namedWindow("RPS-1 live", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("RPS-1 live", W, H)
    shots = ROOT / "captures"
    last_id, frames, t_prev, fps = -1, 0, time.time(), 0.0
    try:
        while True:
            ok, frame = source.read()
            if not ok or frame is None:
                continue
            if args.rotate:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            worker.submit(frame)
            with worker.lock:
                hands, rid, ms = worker.result, worker.result_id, worker.ms
            if rid != last_id:                           # a new vision result: advance the game
                game.update(hands, time.time())
                last_id = rid
            video, box = fit_video(frame)
            scaled = [dict(h) for h in hands]            # points are fractions, so they scale as-is
            screen = np.hstack([draw_video(video, scaled, game, box), draw_panel(game, hands, fps, ms)])
            now = time.time()
            fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-3)
            t_prev = now
            frames += 1
            if headless:
                if frames % 10 == 0:
                    Path(args.record).mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(Path(args.record) / f"screen_{frames:05d}_{game.phase}.jpg"), screen)
                if args.frames and frames >= args.frames:
                    break
                continue
            cv2.imshow("RPS-1 live", screen)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27) or cv2.getWindowProperty("RPS-1 live", cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == ord("n"):
                game.new_round()
            elif key == ord(" "):
                game.revealed = True
            elif key == ord("p"):
                game.private = not game.private
            elif key == ord("a"):
                game.adaptive = not game.adaptive
            elif key == ord("r"):
                game.score = {"win": 0, "tie": 0, "loss": 0}
                game.model = ns["OpponentModel"]()
            elif key == ord("s"):
                shots.mkdir(exist_ok=True)
                path = shots / f"screen_{time.strftime('%Y%m%d_%H%M%S')}.png"
                cv2.imwrite(str(path), screen)
                print("saved", path)
    finally:
        worker.stopping = True
        source.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
