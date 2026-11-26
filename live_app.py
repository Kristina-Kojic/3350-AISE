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

    def relabel(self, hands):
        """After the decision is locked, decide whose each hand is by matching it to the
        nearest of the four locked hands (a kept hand stays where it was), instead of
        trusting the owner classifier on its own. With exactly two hands left, the game
        rule 'one hand each' is enforced, so both can never be counted as ours."""
        locked = self.decision["mine"] + self.decision["theirs"]          # 2 ours, then 2 theirs
        if not hands or len(hands) > 4:
            return
        centre = lambda h: np.asarray(h["points"]).mean(axis=0)
        best, best_cost = None, np.inf
        from itertools import permutations
        for match in permutations(range(len(locked)), len(hands)):
            owners = [m < 2 for m in match]
            if len(hands) == 2 and owners[0] == owners[1]:
                continue                                   # showdown: exactly one each
            cost = sum(np.linalg.norm(centre(h) - centre(locked[m])) for h, m in zip(hands, match))
            if cost < best_cost:
                best, best_cost = match, cost
        if best is None:
            return
        for h, m in zip(hands, best):
            h["ours"] = m < 2
            h["side"] = locked[m]["side"]

    def update(self, hands, now):
        ns = self.ns
        if self.phase in ("DECISION", "RESULT"):
            self.relabel(hands)
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
# Pink theme (RGB)
PINK_TOP, PINK_BOTTOM = (255, 236, 244), (255, 200, 222)   # panel gradient
PLUM = (84, 22, 66)                  # main text
SOFT = (150, 90, 130)                # secondary text
HOT = (236, 72, 153)                 # accent pink
RASPBERRY = (200, 22, 86)            # "take away"
OURS = (255, 92, 168)                # our hands on the video
THEIRS = (139, 108, 214)             # opponent's hands on the video (lavender)
GOLD = (255, 196, 61)
WHITE = (255, 255, 255)
NAME = {"R": "Rock", "P": "Paper", "S": "Scissors"}
VERB = {("R", "S"): "crushes", ("S", "P"): "cuts", ("P", "R"): "covers"}
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


F = {k: font(*v) for k, v in {"huge": (76, True), "big": (50, True), "h1": (30, True), "h2": (24, True),
                              "body": (20, False), "bodyb": (20, True), "small": (15, False),
                              "smallb": (15, True), "tag": (17, True)}.items()}


def rounded(d, box, fill, r=14, outline=None, width=2):
    d.rounded_rectangle(box, radius=r, fill=fill, outline=outline, width=width)


def centred(d, y, text, fnt, fill, x0=0, width=PANEL):
    tw = d.textlength(text, font=fnt)
    d.text((x0 + (width - tw) / 2, y), text, font=fnt, fill=fill)


def fit_video(frame):
    """Scale the camera frame into the video area, keeping its shape (letterbox)."""
    h, w = frame.shape[:2]
    s = min(VIDEO_W / w, VIDEO_H / h)
    nw, nh = int(w * s), int(h * s)
    canvas = np.full((VIDEO_H, VIDEO_W, 3), (60, 30, 50), np.uint8)
    x0, y0 = (VIDEO_W - nw) // 2, (VIDEO_H - nh) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = cv2.resize(frame, (nw, nh))
    return canvas, (x0, y0, nw, nh)


def _tag(d, x, y, text, fill, box):
    """A rounded label kept inside the video area."""
    x0, y0, nw, nh = box
    tw = d.textlength(text, font=F["tag"])
    tx = min(max(x - tw / 2 - 12, x0 + 6), x0 + nw - tw - 30)
    ty = min(max(y, y0 + 6), y0 + nh - 36)
    rounded(d, (tx, ty, tx + tw + 24, ty + 32), fill + (240,), r=16)
    d.text((tx + 12, ty + 6), text, font=F["tag"], fill=WHITE)


def draw_video(frame, hands, game, box):
    """Each hand outlined and named, plus KEEP / TAKE AWAY / WINNER tags."""
    x0, y0, nw, nh = box
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(img, "RGBA")
    y_mid = y0 + nh // 2
    for x in range(x0, x0 + nw, 22):                     # soft divider between the two sides
        d.line((x, y_mid, x + 11, y_mid), fill=(255, 255, 255, 140), width=2)
    for text, ty, colour in (("OPPONENT", y_mid - 30, THEIRS), ("YOU", y_mid + 8, OURS)):
        tw = d.textlength(text, font=F["smallb"])
        rounded(d, (x0 + 10, ty, x0 + 30 + tw, ty + 22), colour + (210,), r=11)
        d.text((x0 + 20, ty + 3), text, font=F["smallb"], fill=WHITE)

    keep_side = None
    if game.phase == "DECISION" and game.revealed:
        keep_side = ("LEFT", "RIGHT")[game.decision["keep"]]
    winner = None
    if game.phase == "RESULT" and game.result:
        winner = {"win": "ours", "loss": "theirs", "tie": "both"}[game.result["outcome"]]

    for h in hands:
        pts = [(x0 + px * nw, y0 + py * nh) for px, py in h["points"]]
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        colour = OURS if h["ours"] else THEIRS
        top_tag = None
        if keep_side and h["ours"]:
            take = h["side"] != keep_side
            ring = RASPBERRY if take else HOT
            d.rounded_rectangle((min(xs) - 20, min(ys) - 20, max(xs) + 20, max(ys) + 20),
                                radius=22, outline=ring + (255,), width=6)
            if take:                                     # a big cross over the hand to take away
                cx, cy, r = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, 0.35 * (max(ys) - min(ys))
                d.line((cx - r, cy - r, cx + r, cy + r), fill=RASPBERRY + (200,), width=10)
                d.line((cx - r, cy + r, cx + r, cy - r), fill=RASPBERRY + (200,), width=10)
            top_tag = ("✕  TAKE AWAY", RASPBERRY) if take else None
        if winner and (winner == "both" or (winner == "ours") == h["ours"]):
            d.rounded_rectangle((min(xs) - 20, min(ys) - 20, max(xs) + 20, max(ys) + 20),
                                radius=22, outline=GOLD + (255,), width=6)
            top_tag = ("★  WINNER", GOLD) if winner != "both" else ("=  TIE", GOLD)
        for a, b in CONNECTIONS:
            d.line((pts[a], pts[b]), fill=colour + (235,), width=4)
        for p in pts:
            d.ellipse((p[0] - 4, p[1] - 4, p[0] + 4, p[1] + 4), fill=(255, 255, 255, 245))
        who = f"YOU · {h['side']}" if h["ours"] else "OPPONENT"
        label = f"{who}   {NAME[h['letter']]}"
        if h["ours"]:      # name tag under our hands, action tag above them
            _tag(d, (min(xs) + max(xs)) / 2, max(ys) + 26, label, colour, box)
            if top_tag:
                _tag(d, (min(xs) + max(xs)) / 2, min(ys) - 62, *top_tag, box)
        else:              # opponent: name tag above, result tag below
            _tag(d, (min(xs) + max(xs)) / 2, min(ys) - 62, label, colour, box)
            if top_tag:
                _tag(d, (min(xs) + max(xs)) / 2, max(ys) + 26, *top_tag, box)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def _panel_background():
    grad = np.linspace(0, 1, H)[:, None, None]
    arr = (np.array(PINK_TOP) * (1 - grad) + np.array(PINK_BOTTOM) * grad) * np.ones((H, PANEL, 3))
    img = Image.fromarray(arr.astype(np.uint8))
    d = ImageDraw.Draw(img, "RGBA")
    for cx, cy, r, a in ((380, 90, 120, 50), (40, 700, 160, 45), (420, 560, 70, 40)):   # soft bubbles
        d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(255, 255, 255, a))
    return img


_BACKGROUND = None


def _card(img, box):
    """White card with a soft pink shadow."""
    from PIL import ImageFilter
    shadow = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ImageDraw.Draw(shadow).rounded_rectangle((box[0], box[1] + 10, box[2], box[3] + 10), radius=34,
                                             fill=(214, 51, 132, 70))
    img.paste(shadow.filter(ImageFilter.GaussianBlur(14)), (0, 0), shadow.filter(ImageFilter.GaussianBlur(14)))
    ImageDraw.Draw(img).rounded_rectangle(box, radius=34, fill=WHITE)


def draw_panel(game, hands, fps, vision_ms):
    """Only what the player has to do now (and, at the end, who won)."""
    global _BACKGROUND
    if _BACKGROUND is None:
        _BACKGROUND = _panel_background()
    img = _BACKGROUND.copy().convert("RGBA")
    d = ImageDraw.Draw(img, "RGBA")
    round_no = len(game.history) + (0 if game.phase == "RESULT" else 1)
    centred(d, 30, f"ROUND {round_no}", F["smallb"], SOFT)
    card = (24, 70, PANEL - 24, H - 70)
    _card(img, card)
    d = ImageDraw.Draw(img, "RGBA")
    top = card[1] + 90                                   # content sits in the middle of the card

    if game.phase == "SHOW":
        n_mine = sum(h["ours"] for h in hands)
        ready = (n_mine, len(hands) - n_mine) == (2, 2)
        centred(d, top + 60, "Show your", F["h1"], SOFT)
        centred(d, top + 105, "two hands", F["big"], PLUM)
        x, w, y = card[0] + 50, card[2] - card[0] - 100, top + 230
        rounded(d, (x, y, x + w, y + 16), (255, 228, 240), r=8)
        if ready and game.progress() > 0:
            rounded(d, (x, y, x + max(16, int(w * game.progress())), y + 16), HOT, r=8)
    elif game.phase == "DECISION" and not game.revealed:
        centred(d, top + 90, "Ready", F["big"], PLUM)
        centred(d, top + 170, "press SPACE", F["h2"], HOT)
    elif game.phase == "DECISION":
        take_side = ("LEFT", "RIGHT")[1 - game.decision["keep"]]
        centred(d, top + 20, "TAKE AWAY", F["h1"], SOFT)
        centred(d, top + 62, "your", F["h2"], SOFT)
        rounded(d, (card[0] + 30, top + 110, card[2] - 30, top + 238), RASPBERRY, r=28)
        centred(d, top + 126, take_side, F["huge"], WHITE)
        centred(d, top + 258, "hand", F["h1"], RASPBERRY)
    else:
        word, colour = {"win": ("WIN", HOT), "loss": ("LOSE", THEIRS), "tie": ("TIE", SOFT)}[game.result["outcome"]]
        if game.result["outcome"] == "win":
            centred(d, top + 20, "★", F["huge"], GOLD)
        rounded(d, (card[0] + 30, top + 120, card[2] - 30, top + 248), colour, r=28)
        centred(d, top + 136, word, F["huge"], WHITE)
    if game.private:
        centred(d, H - 52, "private mode", F["small"], SOFT)
    return cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)



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
