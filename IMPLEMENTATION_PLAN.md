# RPS-1 Project: Step-by-Step Build Guide

> **What this doc is:** the exact order we build things in, what each piece does, and **why** we're doing it that way.
> **What it's not:** final code. The code blocks are skeletons (names + comments) so we all agree on the shape of things before writing real code.
> **Language:** we're assuming **Python** until the prof says otherwise (see Step 0 for why).

---

## The 30-second version

1. Our program has **5 parts**: 👁️ *see* the hands → 🔁 *track the game stage* → 🧠 *pick a move* → 📊 *prove the move is good* → 🖥️ *show it on screen*.
2. We build the **brain (strategy) and the eyes (camera) at the same time**, because they don't depend on each other.
3. We connect everything only once each part works on its own.
4. **One rule:** get a simple version working end-to-end *before* making any part fancy.

---

## The game in one minute (so everyone's on the same page)

- Both players show **two hands** at once (e.g. Rock + Scissors).
- Both players pull back **one hand** at once.
- The remaining hands are judged with normal RPS: **Rock beats Scissors, Scissors beats Paper, Paper beats Rock.**
- Win = **+1**, tie = **0**, loss = **−1**.

**The interesting part:** in stage 2 you've already *seen* your opponent's two hands, so you can make a smart choice about which hand to keep. That's where the game theory comes in.

---

## How our system works (one picture)

```
  Webcam  →  Find hands  →  Is it steady?  →  What stage are we in?  →  Pick our move  →  Show on screen
 (sensor)     (vision)      (stabilizer)       (state machine)           (strategy)          (display)
```

**Why this matches the course:** this is exactly the CPS loop from Week 1. A **sensor** (webcam) measures the physical world (hands), **computation** (our code) decides something, and an **output** (the screen) acts on it. Writing it this way in our report shows we understood the course, not just that we built an app.

**Why split it into separate pieces:**
- People can work on different pieces **at the same time** without breaking each other's code.
- We can **test each piece alone.** If something's broken, we know where to look.
- The simulations can **reuse the strategy code without the camera**, which saves a lot of work.

---

## Who works on what (in parallel)

| Team | Steps | Can start |
|---|---|---|
| 🧠 Strategy people | Steps 2, 3, 4 | Right away (no camera needed) |
| 👁️ Vision people | Step 5 | Right away |
| 🔁 Game flow + display | Steps 6, 7 | Right away (use keyboard as fake input) |
| Everyone | Steps 0, 1, 8, 9, 10 | Together |

---

# The Steps

Each step has the same layout:
**What** we're doing → **Why** → **Files** → **Skeleton** (if there's code) → **Done when**

---

## Step 0: Set up + ask the prof (everyone, this week)

**What:**
- Create the repo and add everyone as collaborators.
- Ask the prof/TA the open questions below.

**Why:** the answers change what we build. For example, if we're not allowed to use a pre-trained hand model, Step 5 becomes way more work, and we need to know that *now*, not in November.

**Questions to ask:**
- ❓ What language/libraries are allowed? Is a pre-trained model like MediaPipe OK?
- ❓ What's the demo setup (camera angle, lighting, the red-circle board from the slides)?
- ❓ Does our system tell a *human* what to play, act as the *referee*, or both?
- ❓ Competition format: how many rounds, and how are games started?
- ❓ Report requirements and deadlines?

**Why we're assuming Python for now:**

| Tool | What it does | Why we picked it |
|---|---|---|
| Python | The language | Best libraries for webcam + math; most tutorials online |
| OpenCV | Opens the webcam, draws on video | The standard tool for this; tons of examples |
| MediaPipe Hands | Finds 21 points on each hand | Already trained, so we skip collecting thousands of photos |
| NumPy | Math + random numbers | Needed for strategy probabilities and simulations |
| Matplotlib | Graphs | The prof wants numerical evidence, and graphs are how we show it |
| pytest | Automatic tests | Lets us check nothing broke after each change |

**Done when:** repo exists, everyone can push, questions are sent.

---

## Step 1: Shared rules + data types (everyone, together, ~1 meeting)

**What:** write the few small files that *every* other part uses: the shape names, the win/lose rules, and the settings.

**Why this is first:** it's like agreeing on units before an engineering project. If one person writes `"rock"`, another writes `"R"`, and another writes `0`, nothing will connect later. Deciding once saves hours of confusion.

**Files:** `common/types.py`, `common/rules.py`, `common/config.py`

**Skeleton: `common/types.py`**
```python
class Shape(Enum):          # the only way anyone refers to a hand shape
    ROCK = "R"
    PAPER = "P"
    SCISSORS = "S"
    UNKNOWN = "?"           # camera couldn't tell (WHY: better to admit "unsure" than guess wrong)

class Player(Enum):
    P1 = 1                  # us
    P2 = 2                  # opponent

@dataclass
class HandDetection:        # one hand the camera found
    shape: Shape
    player: Player
    confidence: float       # 0–1, how sure the camera is
    wrist_xy: tuple         # where it is in the frame (used to figure out whose hand it is)

@dataclass
class RoundRecord:          # one finished round (WHY: saved so we can do stats + spot opponent patterns)
    p1_pair: tuple          # e.g. (ROCK, SCISSORS)
    p2_pair: tuple
    p1_final: Shape
    p2_final: Shape
    result: int             # +1 we win, 0 tie, -1 we lose
```

**Skeleton: `common/rules.py`**
```python
def beats(a, b) -> bool:      # True if a beats b
def payoff(a, b) -> int:      # +1 / 0 / -1, exactly the payoff function from the Week 2 lecture
def make_pair(a, b) -> tuple: # always sort the two shapes
    # WHY sort: (S, R) and (R, S) are the same pair. Sorting means we never count them as different.
```

**Skeleton: `common/config.py`** (every "magic number" lives here)
```python
STABLE_FRAMES_REQUIRED = 8   # see Step 5 for why
PLAYER_SPLIT_Y = 0.5         # hands above this line = P1, below = P2
MAX_HANDS = 4
COUNTDOWN_SECONDS = 3
RANDOM_SEED = None
```
**Why one config file:** when we're tuning at the demo, we change one number in one place instead of hunting through 10 files.

**Done when:** `tests/test_rules.py` checks all 9 shape combos and passes.

---

## Step 2: Strategy math on paper (🧠 strategy people, no code)

**What:** work out the best way to play using the methods from the Week 2 lecture (payoff matrices + Nash equilibrium). Write it up in `docs/strategy_math.md`.

**Why paper first:**
- The prof said the strategy **needs support from theory**. This *is* that support, and it goes straight into the report.
- If the math is wrong, the code will be wrong too. Math is easier to fix on paper.

**What to work out (answers to the slide 33–34 questions):**

**1. Should we ever show the same shape twice (like Rock + Rock)?**
Our answer: **No.**
*Why:* with two different shapes you get a choice in stage 2. With doubles you have no choice, so the opponent can just keep whatever beats or ties you.

**2. What if both players show the same pair (e.g. RS vs RS)?**
Our answer: **Always keep the hand that beats your other hand** (here: Rock, since Rock beats Scissors).
*Why:* whatever the opponent keeps, that hand either wins or ties. It's never worse, so there's no reason to ever pick the other one (this is called a *dominant* option).

**3. What if the pairs are different (e.g. RS vs PS)?**
Our answer: the two pairs always share one shape (here: Scissors). **Keep the shared shape 2/3 of the time and the other one 1/3 of the time.**
*Why:* this is exactly the lecture example (slides 28–31). If we always played the "safe" shape, the opponent would figure it out and counter it. Mixing at exactly these odds means the opponent **can't do better no matter what they pick**. That's what a Nash equilibrium means.

**4. Which pair should we show in stage 1?**
Our answer: **RP, RS, or PS, each 1/3 of the time, at random.**
*Why:* none of the three pairs is better than the others overall (each one does well against one pair and badly against another, just like regular RPS). So the only safe thing is to be unpredictable.

> ⚠️ These are our current answers. The job in this step is to **prove them properly** with the payoff tables and write the proof out clearly.

**Done when:** `docs/strategy_math.md` has a payoff table + solution for every case above.

---

## Step 3: Strategy code (🧠 strategy people)

**What:** turn Step 2's math into code.

**Why a shared "Strategy" template:** our real strategy *and* all the fake opponents in Step 4 have the same two functions. That means we can swap any player for any other without changing the rest of the code.

**Files:** `strategy/base.py`, `strategy/equilibrium.py`, `strategy/optimal.py`

**Skeleton: `strategy/base.py`** (the template)
```python
class Strategy:
    def choose_pair(self, history) -> tuple:
        """Stage 1: which two shapes do we show?"""
    def choose_keep(self, my_pair, opp_pair, history) -> Shape:
        """Stage 2: we've seen their hands. Which of ours do we keep?"""
```

**Skeleton: `strategy/equilibrium.py`** (the lecture math, done automatically)
```python
def build_payoff_matrix(my_pair, opp_pair):
    """Make the 2x2 table like the lecture's RS-vs-PS table."""

def solve_2x2(matrix):
    """Returns: probability of keeping our FIRST hand.

    Step A: is one of our options always at least as good? -> play it 100% (question 2 above)
    Step B: otherwise, mix. For table [[a, b], [c, d]]:
                p = (d - c) / (a - b - c + d)
            WHY this formula: it's the lecture's method (set the two expected payoffs
            equal and solve for p), just written generally so it works for ANY pair.
    """
```

**Skeleton: `strategy/optimal.py`**
```python
class OptimalStrategy(Strategy):
    def choose_pair(self, history):
        # pick RP, RS, or PS at random, 1/3 each
    def choose_keep(self, my_pair, opp_pair, history):
        # build matrix -> solve_2x2 -> roll a random number using that probability
```

**First test to write:** the lecture example (RS vs PS) **must** give 1/3 for Rock.
*Why:* we know the right answer from class, so if the code disagrees, the code is wrong.

**Done when:** `tests/test_equilibrium.py` passes for every pair combo and matches the answers from Step 2.

---

## Step 4: Simulations (🧠 strategy people)

**What:** make our strategy play thousands of games against different fake opponents and graph the results.

**Why:** the prof said **numerical simulations must provide supporting evidence.** Step 2 is the theory, and this is the proof that the theory actually works.

**Files:** `simulation/fake_players.py`, `simulation/run_matches.py`, `simulation/plots.py`

**Fake opponents** (each one is just a `Strategy` from Step 3):
```python
class RandomPlayer(Strategy):         # does everything randomly
class AlwaysScissorsPlayer(Strategy): # the "obvious" move from lecture slide 11
class DoublesPlayer(Strategy):        # shows the same shape twice
class FixedPairPlayer(Strategy):      # always shows the same pair
class NoisyPlayer(Strategy):          # our strategy, but it "misreads" hands X% of the time
```
**Why these opponents:** each one tests a different claim from Step 2 (e.g. `DoublesPlayer` tests "doubles are bad"). `NoisyPlayer` pretends the camera makes mistakes, which is the **disturbance** idea from Week 3.

**Experiments:**

| # | Us vs | We expect | Proves |
|---|---|---|---|
| 1 | Optimal vs Optimal | Average ≈ 0 | The game is fair when both play perfectly |
| 2 | Optimal vs Random | Positive | We beat careless players |
| 3 | Optimal vs Always-Scissors | Positive | The "obvious" strategy is beatable |
| 4 | Optimal vs Doubles | Clearly positive | Doubles really are bad |
| 5 | Noisy (5/10/20% errors) vs Optimal | Lower than #1 | How much camera accuracy matters |

**Tips:**
- Run **10,000+ rounds** per experiment. *Why:* with only a few rounds, luck dominates and the average jumps around.
- Set a **random seed**. *Why:* we get the same result every run, so graphs in the report are repeatable.

**Done when:** all 5 graphs are saved in `docs/results/` and match what Step 2 predicted.

---

## Step 5: Camera + hand recognition (👁️ vision people)

**What:** point the webcam at the table and get R/P/S for each hand, plus whose hand it is.

**Why build it in small pieces:** camera stuff is where things go wrong most. Doing one hand first, then four, makes problems easy to find.

**Files:** `vision/camera.py`, `vision/detector.py`, `vision/classifier.py`, `vision/stabilizer.py`

**Build order:**
- [ ] **5a.** Show the webcam in a window.
- [ ] **5b.** Draw MediaPipe's hand points on one hand.
- [ ] **5c.** Print R/P/S for one hand.
- [ ] **5d.** Handle all 4 hands and decide which player each belongs to.
- [ ] **5e.** Add the stabilizer.
- [ ] **5f.** Measure accuracy (~50 photos per shape) and speed (frames per second).

**Skeleton: `vision/classifier.py`**
```python
def fingers_extended(landmarks) -> list[bool]:
    """Which of the 5 fingers are straight?
    A finger counts as straight if its TIP is farther from the WRIST than its middle joint.
    WHY distance and not just 'pointing up': hands come in from both the top AND bottom
    of the frame, so 'up' means different things for each player."""

def classify_shape(landmarks) -> Shape:
    """0–1 fingers straight           -> ROCK
       only index + middle straight   -> SCISSORS
       4–5 fingers straight           -> PAPER
       anything else                  -> UNKNOWN
    WHY simple rules first: easy to understand and debug. Only train an ML model
    if the rules turn out to be inaccurate."""

def assign_player(wrist_xy) -> Player:
    """Above the middle line = P1, below = P2.
    WHY: in the lecture photos each player sits on one side of the board."""
```

**Skeleton: `vision/stabilizer.py`**
```python
class Stabilizer:
    def update(self, detections):
        """Only report hands once they've looked the SAME for 8 frames in a row (~0.25 sec).
        WHY: the camera sometimes glitches for 1 frame, and players' hands look weird
        WHILE they're changing shape. Without this, we'd react to garbage."""
```

**Done when:** all 4 hands are labelled correctly about 90%+ of the time in normal lighting, and fast enough to feel live.

---

## Step 6: Game flow / state machine (🔁 game flow person)

**What:** code that keeps track of which stage the game is in and moves to the next one at the right time.

**Why a state machine:** the game happens in clear stages, and each stage waits for something specific. This is the **finite-state machine** model from Week 3, Lecture 1, so it's easy to explain in the report *and* hard to get tangled up in.

**The stages:**
```
WAITING → COUNTDOWN → STAGE 1 → DECIDING → STAGE 2 → RESULT → (back to WAITING)
                         ↓                     ↓
                      INVALID (wrong number of hands → redo the round)
```

| Stage | Waits for | Then goes to |
|---|---|---|
| WAITING | Spacebar pressed | COUNTDOWN |
| COUNTDOWN | Timer hits 0 | STAGE 1 |
| STAGE 1 | 2 steady hands per player | DECIDING |
| DECIDING | Strategy picks which hand to keep | STAGE 2 |
| STAGE 2 | 1 steady hand per player | RESULT |
| RESULT | A few seconds pass | WAITING |
| INVALID | Wrong hand count for too long | WAITING |

**Files:** `game/states.py`, `game/state_machine.py`

**Skeleton:**
```python
class GameStateMachine:
    def update(self, steady_hands, now):
        """Called every frame. Checks the table above and moves to the next stage if ready."""
    def judge(self) -> int:
        """Uses payoff() from common/rules.py"""
    def get_display_info(self) -> dict:
        """Everything the screen needs to show."""
```

**Important:** build and test this with **keyboard input first** (type R/P/S instead of using the camera).
*Why:* then you don't have to wait for the vision team, and if something breaks later you'll know it's the camera, not the game logic.

**Done when:** a full round played with the keyboard always gives the right winner.

---

## Step 7: Display (🔁 display person)

**What:** draw useful info on top of the webcam video.

**Show:**
- A label on each hand (R/P/S and P1/P2). *Why:* so we can instantly see if the camera is confused.
- The current stage + countdown. *Why:* so both players know when to go.
- **What we should play** ("SHOW: R + S", then "KEEP: S"). *Why:* this is the output of our whole system.
- The score.
- Frames per second in the corner. *Why:* timing matters (Week 1), and it's a nice number for the report.

**Controls:** `space` = start round, `r` = reset score, `q` = quit.

**Done when:** the text is big and clear enough to read from a couple of metres away at the demo.

---

## Step 8: Connect everything (everyone)

**What:** plug all the parts together in `main.py`.

**Why this comes late:** if each part already works alone, connecting them is mostly just passing data along. If we connected everything first, one bug anywhere would break everything, and we wouldn't know where.

```python
# main.py: the whole system in one loop (this is the CPS loop!)
while True:
    frame  = camera.read()                    # 1. SENSE   (webcam)
    hands  = detector.detect(frame)           # 2. PROCESS (find + classify hands)
    steady = stabilizer.update(hands)         # 3. FILTER  (ignore glitches)
    game.update(steady, now())                # 4. DECIDE  (state machine calls strategy)
    display.draw(frame, game.get_display_info())  # 5. ACT (show on screen)
    if display.key_pressed() == 'q': break
```

**Done when:** we play a full live round with real hands and it works start to finish.

---

## Step 9: Test + tune (everyone)

**Why:** the demo room won't be as nice as our testing spot. Better to find problems now.

- [ ] Different people's hands (sizes, skin tones, sleeves, watches)
- [ ] Bright, dim, and uneven lighting
- [ ] Someone going slow, or showing the wrong number of hands
- [ ] Measure the delay from hands moving to the answer appearing on screen
- [ ] Fresh install on a different laptop using only `requirements.txt` (*why:* the prof said the software must be **self-contained**)
- [ ] 20+ practice rounds with no crashes

---

## Step 10: Extras + report (only after Step 9 works)

**Stretch goal: adaptive strategy** (`strategy/adaptive.py`)
- Start with the optimal strategy, but watch the opponent's history.
- If they have an obvious habit (like always keeping scissors), switch to the move that beats it.
- If the habit disappears, go back to optimal.
- *Why it's a stretch goal:* it can win more against weak teams, but it's riskier, and the optimal strategy already can't be beaten on average.

**Report:** most of it is already written if we did the steps. It pulls from:
- the system diagram (top of this doc),
- the strategy math (Step 2),
- the simulation graphs (Step 4),
- the state machine diagram (Step 6),
- accuracy and timing numbers (Steps 5 and 9).

---

# Reference

## Folder map
```
rps1-project/
├── README.md, PLAN.md, IMPLEMENTATION_PLAN.md
├── requirements.txt     ← list of libraries (so anyone can install with one command)
├── main.py              ← Step 8
├── common/              ← Step 1 (types, rules, config)
├── strategy/            ← Step 3
├── simulation/          ← Step 4
├── vision/              ← Step 5
├── game/                ← Step 6
├── ui/                  ← Step 7
├── tests/               ← one test file per part
├── data/                ← our test photos (big videos go in .gitignore)
└── docs/                ← strategy math, result graphs, report notes
```

## Rough timeline (adjust once we know the real deadlines)

| Week | Goal |
|---|---|
| 1 | Steps 0 + 1 done; Step 2 started; webcam window working (5a) |
| 2 | Step 2 math done; one hand classified (5c); state machine with keyboard (Step 6) |
| 3 | Strategy code + tests (Step 3); 4 hands + players (5d–5e) |
| 4 | Simulations + graphs (Step 4); display (Step 7) |
| 5 | **Everything connected (Step 8)** ← most important milestone |
| 6–7 | Testing + tuning (Step 9) |
| 8+ | Stretch goals, report, demo practice (Step 10) |

## How we use Git (and why)

| Rule | Why |
|---|---|
| `main` always works; never push broken code to it | Anyone can grab `main` and it runs |
| One branch per task, e.g. `vision/finger-count` | People don't overwrite each other's work |
| Open a Pull Request; one teammate checks it before merging | Catches mistakes and keeps everyone in the loop |
| Clear commit messages ("Add stabilizer", not "stuff") | Easy to find when something broke |
| `git pull` before you start working | Avoids merge conflicts |
| Don't commit big videos | Keeps the repo fast to download |
