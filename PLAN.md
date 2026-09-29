# AISE 3350A – RPS-1 Project: Brainstorm + Plan

> This is our starting point, not a final design. Nothing here is locked in.
> No code yet since we don't know what language we're allowed or expected to use.
> Anything marked **TODO** or **❓** is something we still need to figure out.

---

## 1. What we're actually building

From the Week 2 lecture, our team has to build a software system that plays **Rock-Paper-Scissors-Minus-One (RPS-1)** with a webcam.

**Game rules (so we're all on the same page):**
1. Both players show **two hands** at the same time (e.g. Rock + Scissors).
2. Both players then remove **one hand** at the same time.
3. The hands left over are judged with normal RPS rules (R beats S, S beats P, P beats R).
4. Scoring: win = +1, tie = 0, loss = −1 (zero-sum).

**What the prof said the system needs:**
- [ ] A computer vision (CV) part that recognizes hand shapes
- [ ] A strategy for both stages (show two hands, then pull one back)
- [ ] A CV part that decides who won
- [ ] Uses a webcam, and a display
- [ ] Self-contained software (should run on its own without us babysitting it)
- [ ] The strategy has to be backed by theory **and** numerical simulations
- [ ] Live demo + competition at the end of the semester

We don't have to use the "optimal" strategy from lecture. We can come up with our own, but we have to justify it.

---

## 2. How this fits the course (CPS framing)

Since this is a CPS course, we should describe our system the way the lectures do, not just as "an app." This will probably matter for the report.

| CPS idea (from lectures) | What it is in our project |
|---|---|
| Physical world / plant | The table, both players' hands |
| Sensor | Webcam |
| Cyber / computation | Hand detection + game logic + strategy |
| Actuator / output | The display telling our player what to play (and showing the result) |
| Reference input | Our goal: pick the move that maximizes expected payoff |
| Disturbance | Bad lighting, blurry hands, hands overlapping, someone moving late |
| Feedback (closed loop) | We watch what the opponent shows in stage 1 and use it to decide stage 2 |

Other lecture ideas we can use:
- **Timing matters (Week 1).** If our system decides too slowly, it's useless in a live game. We need to know how fast detection + decision is.
- **Finite-state machine (Week 3, Lec 1).** The game naturally happens in stages, so the whole program flow can be an FSM (see section 5).
- **Game theory (Week 2).** Payoff matrices, Nash equilibrium, and mixed strategies are what our strategy is built on.

---

## 3. Big pieces of the project

We can split the work into 5 parts that different people can work on at the same time.

### Part A: Camera + hand recognition (CV)
**Goal:** Given a webcam frame, figure out each hand's shape (R, P, or S) and which player it belongs to.

To figure out:
- ❓ Which approach do we use?
  - Option 1: A pre-built hand landmark model (e.g. MediaPipe Hands gives 21 points per hand), then write simple rules like "how many fingers are extended."
  - Option 2: Train our own classifier on pictures of hands.
  - Option 3: A mix, using landmarks as features for a small ML model.
  - *Our leaning:* start with Option 1 because it's the most doable. Only move to ML if the rules aren't accurate enough.
- ❓ How do we tell which hands are Player 1's and which are Player 2's? From the lecture photos, the board has two circles and each player sits on one side, so we could split the frame into a top half and a bottom half.
- ❓ How do we handle up to 4 hands in the frame at once?
- ❓ What if a hand is blurry or halfway through changing shape? We probably need to require the same answer for a few frames in a row before trusting it.

TODO:
- [ ] Take our own test photos/videos on a setup like the lecture board
- [ ] Measure accuracy (how often it gets R/P/S right) and speed (frames per second)

### Part B: Game flow (state machine)
**Goal:** Know what stage of the game we're in and when to lock in each player's hands.

Rough stages:
```
WAITING  ->  STAGE 1 (both show 2 hands)  ->  DECIDE which hand to keep
         ->  STAGE 2 (both pull 1 hand)    ->  JUDGE winner  ->  SHOW RESULT  ->  back to WAITING
```

To figure out:
- ❓ How does the system know stage 1 is "done"? Options: 4 hands are detected and stable for X frames, or a countdown.
- ❓ How does it know stage 2 is done? Options: only 2 hands are left and stable.
- ❓ Do we need a countdown on screen so both players go at the same time?
- ❓ What happens with invalid rounds (someone shows 1 hand, or never removes one)?

### Part C: Strategy (game theory)
**Goal:** Decide (1) which two hands we show in stage 1, and (2) which hand we keep in stage 2 after seeing the opponent's hands.

What we have from lecture:
- In the RS vs PS example, the equilibrium was to play R with 1/3 probability and S with 2/3 (and for the other player, P 1/3 and S 2/3).
- Slides 33–34 left open questions: should you ever show the same shape twice? What if both players show the same pair? What if the pairs are different?

What we think so far (**still needs to be properly proven and simulated**):
- **Never show the same shape with both hands** (RR, PP, SS). It locks you in and a smart opponent can always beat or tie it.
- **Stage 1:** pick one of RP, RS, PS at random, each with probability 1/3, so the opponent can't predict us.
- **Stage 2, it depends on what the opponent shows:**
  - *They show a double (like RR):* keep whichever of our hands beats it (or ties if we can't beat it).
  - *They show the same pair as us:* one of our two hands always beats or ties the other one, so always keep that one.
  - *They show a different pair from us:* there's always one shape we both have (like S in RS vs PS). Keep that shared shape with probability 2/3 and the other one with 1/3. This matches the lecture example.

TODO:
- [ ] Write out the full payoff matrices for every pair-vs-pair case (there are 6×6 combos including doubles)
- [ ] Solve each stage-2 mini-game for its Nash equilibrium (like slides 28–31)
- [ ] Show why stage 1 should be 1/3 each
- [ ] ❓ Do we try to *exploit* opponents who aren't playing optimally? For example, if a team always keeps the same hand, we could track their history and adapt. This could win the competition, but it's riskier. Maybe mix: start optimal, then adapt if we detect a pattern.

### Part D: Simulations (proof our strategy works)
**Goal:** The prof wants numbers backing up the strategy.

Ideas:
- [ ] Simulate thousands of games of our strategy vs different fake opponents:
  - random opponent
  - "always keeps scissors" opponent (the naive one from lecture)
  - opponent that always shows the same pair
  - opponent using the same optimal strategy as us
- [ ] Plot win rate / average score over number of games
- [ ] Check the results match the theory (e.g. optimal vs optimal should average around 0)
- [ ] ❓ Also simulate what happens when CV gets a hand wrong X% of the time? That would tie the CPS "disturbance" idea in nicely.

### Part E: Display / UI
**Goal:** Show the live camera feed plus what our system is doing.

Should show:
- [ ] Detected hands with labels (R/P/S) drawn on the video
- [ ] Current game stage
- [ ] What our player should show / which hand to pull back
- [ ] Round result + running score

---

## 4. Rough folder structure for the repo

Just a plan, we'll rename things once we pick a language.

```
/
├── README.md          -> short overview + how to run (later)
├── PLAN.md            -> this file
├── docs/              -> report notes, math for the strategy, diagrams
├── vision/            -> Part A: camera + hand recognition
├── game/              -> Part B: state machine + judging winner
├── strategy/          -> Part C: decision logic
├── simulation/        -> Part D: simulation scripts + result plots
├── ui/                -> Part E: display
├── data/              -> our test photos/videos (maybe gitignore big files)
└── tests/             -> checks that each part works on its own
```

---

## 5. Step-by-step order to do things

**Step 0: Confirm with the prof / TA**
- [ ] ❓ What language/tools are we allowed? (If it's open, Python is probably easiest: OpenCV + MediaPipe for vision, NumPy/Matplotlib for simulations.)
- [ ] ❓ What's the exact setup at the demo: camera position, lighting, the board from the photos?
- [ ] ❓ Does our system *play* (tell a human what to show) or just *referee*? Or both?
- [ ] ❓ How is the competition run, and how many rounds?
- [ ] ❓ Deadlines and what goes in the report

**Step 1: Theory first (can start now, no code needed)**
- [ ] Do the payoff matrices and equilibrium math on paper (Part C)
- [ ] Write it up in `docs/`

**Step 2: Get the camera working**
- [ ] Open the webcam, detect hands, print R/P/S for one hand
- [ ] Then extend to 4 hands and split them by player

**Step 3: Simulations**
- [ ] Code up the strategy + fake opponents (Part D), no camera needed
- [ ] Make the plots

**Step 4: Game flow**
- [ ] Build the state machine, first with fake/typed inputs instead of the camera

**Step 5: Connect everything**
- [ ] Camera → state machine → strategy → display
- [ ] Measure how long each step takes (timing!)

**Step 6: Test a lot**
- [ ] Different people's hands, lighting, backgrounds
- [ ] Practice full games against each other

**Step 7: Report + demo prep**

---

## 6. Who does what (fill in)

| Part | Person(s) |
|---|---|
| A: Hand recognition | |
| B: Game flow | |
| C: Strategy math | |
| D: Simulations | |
| E: Display | |
| Report / docs | |

---

## 7. Risks we should keep in mind

- Hand detection might be unreliable in bad lighting. We need to test in conditions like the demo room.
- Detecting 4 hands at once is harder than detecting 1 or 2.
- If players aren't perfectly simultaneous, the system could read a hand mid-change.
- Scope creep: get a basic working version first, then add the fancy stuff (like opponent-adapting strategy).

---

## 8. Open questions list (quick summary)

- ❓ Language/tools?
- ❓ Exact demo setup?
- ❓ Does our system play, referee, or both?
- ❓ Rules-based CV vs trained model?
- ❓ How do we detect when each stage is "done"?
- ❓ Pure optimal strategy, or try to exploit opponents?
