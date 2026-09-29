# ✅ RPS-1 Project Tracker

**How to use:** when a task is done, change `[ ]` to `[x]` and commit. Each week is one section, so just look at the current week.

## 👥 Team

| Name | Main job |
|---|---|
| **Kristina** | Strategy math + code |
| **Romy** | Camera + hand recognition |
| **Rebecca** | Game flow + simulations |
| **Nicole** | Display + testing |

## 📌 Course deadlines (fill in when the prof posts them)

- [ ] Final report due: **TBD**
- [ ] Live demo / competition: **TBD**
- [ ] Other check-in (if any): **TBD**

---

## Week 1 · Sep 28 – Oct 4

- [x] Create GitHub repo + invite everyone · **Everyone** · due Wed Sep 30
- [ ] Upload PLAN.md + IMPLEMENTATION_PLAN.md + TRACKER.md · **Everyone** · due Wed Sep 30
- [ ] Email prof/TA the open questions (language, demo setup, rules) · **Everyone** · due Thu Oct 1
- [ ] Team meeting: agree on shapes, players, data types · **Everyone** · due Fri Oct 2
- [ ] Show webcam feed in a window · **Romy** · due Sun Oct 4
- [ ] Everyone installs Python + libraries, runs a 'hello webcam' test · **Everyone** · due Sun Oct 4
- [ ] Write common/types.py, rules.py, config.py · **Kristina** · due Sun Oct 4

## Week 2 · Oct 5 – Oct 11

- [ ] Write tests/test_rules.py (all 9 shape combos) · **Nicole** · due Tue Oct 6
- [ ] Draw MediaPipe hand points on 1 hand · **Romy** · due Wed Oct 7
- [ ] Draw the state machine diagram · **Rebecca** · due Thu Oct 8
- [ ] Write all pair-vs-pair payoff tables · **Kristina** · due Fri Oct 9
- [ ] Classify 1 hand as R/P/S (finger rules) · **Romy** · due Sun Oct 11
- [ ] Prove: never play doubles / same-pair / different-pair answers · **Kristina** · due Sun Oct 11
- [ ] Show why stage 1 should be 1/3 each pair · **Kristina** · due Sun Oct 11

## Week 3 · Oct 12 – Oct 18

- [ ] Write it all up in docs/strategy_math.md · **Kristina** · due Wed Oct 14
- [ ] Strategy/base.py (Strategy template) · **Kristina** · due Wed Oct 14
- [ ] Test: RS vs PS gives Rock = 1/3 · **Kristina** · due Fri Oct 16
- [ ] Game/state_machine.py with keyboard input · **Rebecca** · due Fri Oct 16
- [ ] Strategy/equilibrium.py (payoff matrix + solve_2x2) · **Kristina** · due Fri Oct 16
- [ ] Detect all 4 hands + assign to P1/P2 · **Romy** · due Sun Oct 18
- [ ] Strategy/optimal.py (OptimalStrategy) · **Kristina** · due Sun Oct 18

## Week 4 · Oct 19 – Oct 25

- [ ] Handle invalid rounds + save round history · **Rebecca** · due Wed Oct 21
- [ ] Stabilizer (same answer 8 frames in a row) · **Romy** · due Thu Oct 22
- [ ] Simulation/fake_players.py (5 fake opponents) · **Rebecca** · due Thu Oct 22
- [ ] Keyboard controls (space / r / q) · **Nicole** · due Fri Oct 23
- [ ] Simulation/run_matches.py (10,000+ rounds, fixed seed) · **Rebecca** · due Fri Oct 23
- [ ] Tests/test_state_machine.py · **Rebecca** · due Fri Oct 23
- [ ] Ui/display.py: labels, stage, countdown, score · **Nicole** · due Fri Oct 23

## Week 5 · Oct 26 – Nov 1

- [ ] Measure accuracy (~50 photos/shape) + FPS · **Romy** · due Wed Oct 28
- [ ] Run experiments 1–5 + save graphs to docs/results/ · **Rebecca** · due Wed Oct 28
- [ ] Show recommended move + FPS · **Nicole** · due Wed Oct 28

## Week 6 · Nov 2 – Nov 8

- [ ] Main.py: connect camera → game → strategy → display · **Everyone** · due Fri Nov 6

## Week 7 · Nov 9 – Nov 15

- [ ] First full live round works end-to-end · **Everyone** · due Fri Nov 13

## Week 8 · Nov 16 – Nov 22

- [ ] Fresh install on another laptop from requirements.txt · **Nicole** · due Fri Nov 20
- [ ] Measure end-to-end delay · **Romy** · due Fri Nov 20
- [ ] Test different hands, lighting, sleeves/watches · **Nicole** · due Fri Nov 20

## Week 9 · Nov 23 – Nov 29

- [ ] Play 20+ practice rounds, fix crashes · **Everyone** · due Fri Nov 27

## Week 10 · Nov 30 – Dec 6

- [ ] (Stretch) strategy/adaptive.py · **Kristina** · due Wed Dec 2
- [ ] Demo rehearsal · **Everyone** · due Fri Dec 4
- [ ] Write report (pull from docs/ + graphs) · **Everyone** · due Fri Dec 4

---

**Rule of thumb:** if you're stuck on something for more than a day, write `🚧 BLOCKED:` next to it and tell the group chat.
