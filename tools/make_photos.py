"""Build our four-hand table photos from real single-hand photos.

    python tools/make_photos.py PATH/TO/rps-cv-images

Where the hands come from
-------------------------
The "Rock-Paper-Scissors Images" dataset by Julien de la Bruere-Terreault
(DrGFreeman, CC-BY-SA 4.0): 2188 real photos of one hand each (Rock, Paper or
Scissors), taken from above on a green background. Download: Kaggle
"drgfreeman/rockpaperscissors", or the mirror
https://github.com/dicodingacademy/assets/releases/download/release/rockpaperscissors.zip

What this script does
---------------------
For each photo it cuts 4 real hands out of their green background (chroma key) and
places them on a table the way our capture setup sees them (camera above the
table, looking down):
  * our player's two hands come in from the BOTTOM, fingers pointing up;
  * the opponent's two hands come in from the TOP, fingers pointing down;
  * hands on the correct side are mirrored so they are real left/right hands
    (our LEFT hand is on the left of the image, the opponent's RIGHT hand is on
    the left of the image, because they face us).
Then it varies the conditions on purpose: lighting (bright / normal / dim),
table (plain / wood / dark / busy), how much the hands are tilted (small / large),
hand size, shadows, camera noise and JPEG compression.

No leakage between splits: the source hand photos are split 50/50 first, and
train photos only use one half, test and demo photos only the other half.

Outputs (relative to the repo root)
-----------------------------------
images/train/*.jpg, images/test/*.jpg, images/demo/demo1-5.jpg
images/photos.csv            one row per photo (conditions)
images/composite_truth.csv   one row per hand: where it was placed, owner, shape,
                             which source photo it came from
images/label_maps/*.png      for each photo, which pixels belong to hand 1-4
Then run tools/label_photos.py to write images/hand_labels.csv.

Everything is seeded, so running it again gives exactly the same photos.
"""
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

SEED = 3350
W, H = 960, 720                       # 4:3, like a webcam
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "images"
SPLITS = {"train": 50, "test": 80, "demo": 5}
CLASSES = {"rock": "R", "paper": "P", "scissors": "S"}
LIGHTING = ["bright", "normal", "dim"]
TABLES = ["plain", "wood", "dark", "busy"]
TILTS = {"small": 12, "large": 35}    # standard deviation of hand tilt, degrees
ARM_EXTRA = 320                       # px of forearm added so arms reach the photo edge
WRIST_X = 230                         # wrist position in the 300 px wide source photo


# ---------------------------------------------------------------------------
# 1. Cut a hand out of its green background
# ---------------------------------------------------------------------------
def cut_out(path):
    """Return (BGR image, alpha 0..1) of one source hand, arm extended to the right."""
    img = cv2.imread(str(path))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, (35, 50, 30), (90, 255, 255))
    fg = cv2.bitwise_not(green)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(fg)
    if n < 2:
        return None
    keep = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    mask = (lab == keep).astype(np.float32)
    if mask[:, -1].sum() < 10:          # the arm must leave the right edge
        return None
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    # remove green spill on the edges of the hand
    b, g, r = cv2.split(img.astype(np.float32))
    g = np.minimum(g, np.maximum(r, b) * 1.05)
    img = cv2.merge([b, g, r]).clip(0, 255).astype(np.uint8)
    # extend the forearm so it can reach the edge of the table photo
    img = np.hstack([img, np.repeat(img[:, -1:], ARM_EXTRA, axis=1)])
    mask = np.hstack([mask, np.repeat(mask[:, -1:], ARM_EXTRA, axis=1)])
    return img, mask


# ---------------------------------------------------------------------------
# 2. Tables and lighting
# ---------------------------------------------------------------------------
def smooth_noise(rng, scale, shape=(H, W)):
    small = rng.normal(0, 1, (max(2, shape[0] // scale), max(2, shape[1] // scale)))
    return cv2.resize(small, (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC)


def make_table(kind, rng):
    if kind == "plain":
        base = rng.choice([(225, 225, 225), (205, 220, 235), (190, 205, 215), (235, 232, 225)])
        bg = np.ones((H, W, 3)) * np.array(base, float)
        bg += smooth_noise(rng, 200)[..., None] * 6
    elif kind == "wood":
        # light oak (close to skin colour: a deliberately hard case), walnut, dark walnut, grey-washed
        base = np.array(rng.choice([(60, 105, 160), (35, 55, 85), (25, 35, 55), (110, 120, 125)]), float)
        y = np.arange(H)[:, None] + smooth_noise(rng, 60) * 25
        grain = np.sin(y / rng.uniform(5, 11)) * 0.5 + 0.5
        bg = base * (0.8 + 0.25 * grain[..., None]) + smooth_noise(rng, 15)[..., None] * 5
    elif kind == "dark":
        base = rng.choice([(40, 40, 40), (55, 45, 35), (30, 35, 45)])
        bg = np.ones((H, W, 3)) * np.array(base, float) + smooth_noise(rng, 100)[..., None] * 5
    else:  # busy: clutter on a light table
        bg = np.ones((H, W, 3)) * np.array(rng.choice([(200, 200, 195), (180, 190, 200)]), float)
        bg = bg.astype(np.uint8)
        for _ in range(int(rng.integers(18, 32))):
            colour = tuple(int(c) for c in rng.integers(20, 250, 3))
            x, y = int(rng.integers(0, W)), int(rng.integers(0, H))
            kind2 = rng.integers(0, 3)
            if kind2 == 0:
                cv2.rectangle(bg, (x, y), (x + int(rng.integers(30, 160)), y + int(rng.integers(20, 120))), colour, -1)
            elif kind2 == 1:
                cv2.circle(bg, (x, y), int(rng.integers(10, 60)), colour, -1)
            else:
                cv2.line(bg, (x, y), (int(rng.integers(0, W)), int(rng.integers(0, H))), colour, int(rng.integers(2, 8)))
        bg = bg.astype(float)
    return bg.clip(0, 255)


def apply_lighting(img, kind, rng):
    img = img.astype(float)
    # light from one side + vignette
    yy, xx = np.mgrid[0:H, 0:W]
    angle = rng.uniform(0, 2 * np.pi)
    side = 1 + 0.12 * ((xx / W - 0.5) * np.cos(angle) + (yy / H - 0.5) * np.sin(angle))
    vignette = 1 - 0.25 * (((xx - W / 2) / W) ** 2 + ((yy - H / 2) / H) ** 2)
    img *= (side * vignette)[..., None]
    tint = np.array([rng.uniform(0.92, 1.05), 1.0, rng.uniform(0.95, 1.08)])   # colour temperature
    if kind == "bright":
        img = img * rng.uniform(1.10, 1.25) * tint + 10
    elif kind == "normal":
        img = img * rng.uniform(0.92, 1.05) * tint
    else:  # dim: dark, noisy, slightly blurred (like a webcam in a dim room)
        img = img * rng.uniform(0.32, 0.48) * tint
        img = cv2.GaussianBlur(img, (3, 3), 0)
    noise = {"bright": 3, "normal": 4, "dim": 7}[kind]
    img += rng.normal(0, noise, img.shape)
    return img.clip(0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# 3. Place one hand on the table
# ---------------------------------------------------------------------------
def place(hand, ours, mirror, centre_x, wrist_y, tilt_deg, scale):
    """Return (BGR layer, alpha layer) of the whole table with one hand on it."""
    img, mask = hand
    h, w = mask.shape                   # source: fingers point LEFT, arm leaves on the RIGHT
    ys = np.nonzero(mask[:, WRIST_X] > 0.5)[0]
    wy = ys.mean() if len(ys) else h / 2
    if ours:   # rotate so fingers point UP and the arm goes DOWN
        img, mask = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE), cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE)
        anchor = np.array([h - 1 - wy, WRIST_X])
    else:      # fingers point DOWN, arm goes UP
        img, mask = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE), cv2.rotate(mask, cv2.ROTATE_90_COUNTERCLOCKWISE)
        anchor = np.array([wy, w - 1 - WRIST_X])
    if mirror:  # turns a right hand into a left hand
        img, mask = img[:, ::-1].copy(), mask[:, ::-1].copy()
        anchor[0] = img.shape[1] - 1 - anchor[0]
    # rotate by tilt_deg and scale around the wrist, then move the wrist to its spot
    M = cv2.getRotationMatrix2D((float(anchor[0]), float(anchor[1])), tilt_deg, scale)
    M[:, 2] += np.array([centre_x, wrist_y]) - anchor
    layer = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    alpha = cv2.warpAffine(mask, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=0)
    return layer.astype(float), alpha


def make_photo(pool, rng, lighting, table, tilt):
    """One table photo. pool: {letter: [paths]}. Returns (image, list of hand dicts)."""
    for _attempt in range(50):
        shapes = list(rng.choice(["R", "P", "S"], size=4))
        if rng.random() < 0.15:                      # extra doubles, especially Rock + Rock
            who = int(rng.integers(0, 2))
            s = "R" if rng.random() < 0.6 else str(rng.choice(["P", "S"]))
            shapes[2 * who] = shapes[2 * who + 1] = s
        # positions: index 0,1 = ours (image left, right); 2,3 = theirs (image left, right)
        spots = [(True, 0), (True, 1), (False, 0), (False, 1)]
        hands, layers = [], []
        scale = rng.uniform(0.95, 1.2)
        for k, ((ours, side), s) in enumerate(zip(spots, shapes)):
            src = pool[s][int(rng.integers(len(pool[s])))]
            cut = cut_out(src)
            if cut is None:
                break
            x = W * (0.30 if side == 0 else 0.70) + rng.normal(0, 25)
            y = H * (0.86 if ours else 0.14) + rng.normal(0, 12)
            t = float(np.clip(rng.normal(0, TILTS[tilt]), -60, 60))
            # our image-left hand is our LEFT hand (mirror the right-hand source);
            # the opponent faces us, so THEIR image-right hand is their left hand
            mirror = (side == 0) if ours else (side == 1)
            layer, alpha = place(cut, ours, mirror, x, y, t, scale * rng.uniform(0.95, 1.05))
            layers.append((layer, alpha))
            hands.append({"owner": "us" if ours else "them", "shape": s,
                          "position": ("our_" if ours else "opp_") + ("image_left" if side == 0 else "image_right"),
                          "source": src.name, "mirrored": int(mirror), "tilt_deg": round(t, 1)})
        if len(layers) < 4:
            continue
        alphas = np.stack([a for _, a in layers])
        overlap = ((alphas > 0.5).sum(axis=0) > 1).sum()
        visible = [(a > 0.5).sum() for a in alphas]
        if overlap > 0 or min(visible) < 4000:      # rule from our setup: hands must not touch
            continue
        img = make_table(table, rng)
        shadow = cv2.GaussianBlur(alphas.max(axis=0), (31, 31), 0)
        shadow = np.roll(shadow, (14, 10), axis=(0, 1))
        img *= (1 - 0.35 * shadow)[..., None]
        for layer, alpha in layers:
            img = img * (1 - alpha[..., None]) + layer * alpha[..., None]
        label_map = np.zeros((H, W), np.uint8)       # which pixel belongs to which hand
        for k, a in enumerate(alphas):
            label_map[a > 0.5] = k + 1
        return apply_lighting(img, lighting, rng), hands, label_map
    raise RuntimeError("could not place 4 hands without overlap")


def main(dataset):
    dataset = Path(dataset)
    rng = np.random.default_rng(SEED)
    pools = {"train": {}, "test": {}}
    for folder, letter in CLASSES.items():
        files = sorted((dataset / folder).glob("*.png"))
        order = rng.permutation(len(files))
        half = len(files) // 2
        pools["train"][letter] = [files[i] for i in order[:half]]
        pools["test"][letter] = [files[i] for i in order[half:]]

    photo_rows, truth_rows = [], []
    for split, count in SPLITS.items():
        folder = OUT / split
        folder.mkdir(parents=True, exist_ok=True)
        for old in folder.glob("*.jpg"):
            old.unlink()
        pool = pools["train"] if split == "train" else pools["test"]
        for n in range(count):
            # spread conditions evenly so every lighting/table/tilt is tested
            lighting = LIGHTING[n % 3]
            table = TABLES[(n // 3) % 4]
            tilt = "large" if (n // 12) % 2 else "small"
            if split == "demo":   # demo photos: one of each condition we care about
                lighting, table, tilt = [("normal", "wood", "small"), ("bright", "plain", "small"),
                                         ("dim", "dark", "small"), ("normal", "busy", "large"),
                                         ("bright", "wood", "large")][n]
            img, hands, label_map = make_photo(pool, rng, lighting, table, tilt)
            name = f"demo{n + 1}.jpg" if split == "demo" else f"{split}_{n + 1:03d}.jpg"
            cv2.imwrite(str(folder / name), img, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(82, 94))])
            maps = OUT / "label_maps"            # which pixel belongs to which hand (for labelling)
            maps.mkdir(exist_ok=True)
            cv2.imwrite(str(maps / f"{Path(name).stem}.png"), label_map)
            photo_rows.append({"filename": name, "split": split, "lighting": lighting,
                               "background": table + " table", "people": "composite of dataset hands",
                               "notes": f"{tilt} tilt"})
            for k, hnd in enumerate(hands):
                truth_rows.append({"filename": name, "split": split, "hand_id": k + 1, **hnd})
        print(f"{split}: {count} photos")

    with open(OUT / "photos.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(photo_rows[0]))
        wr.writeheader()
        wr.writerows(photo_rows)
    with open(OUT / "composite_truth.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(truth_rows[0]))
        wr.writeheader()
        wr.writerows(truth_rows)
    print("wrote images/photos.csv and images/composite_truth.csv")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
