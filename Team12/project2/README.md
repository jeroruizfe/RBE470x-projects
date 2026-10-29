# Team 12 -- RBE4701 Project 2 (Approximate Q-Learning Bomberman)

## Installation

The agent only uses the Python 3 standard library (`json`, `heapq`, `random`,
`collections`, `multiprocessing`).  The game itself needs the two packages
listed in the course README:

    pip3 install colorama pygame

No numpy / PyTorch / TensorFlow is required.

## Running the variants

Place the `team12` folder next to the course's `bomberman/` folder (as in the
course repository) and run:

    cd team12/project2
    python3 variant1.py      # ... variant5.py

The variant files are the course templates with two changes: the import path
names `../team12`, and `g.go(1)` makes the game run on its own instead of
waiting for Enter after every step.  The agent (`testcharacter.py`) lives in
this `project2` folder, so the variant files find it directly.

Note for Linux: the course repository currently names the engine folder
`Bomberman` while the templates import `../../bomberman`.  On a case-sensitive
file system create a link once: `ln -s Bomberman bomberman` in the repository
root.

## Files

All files below are in `team12/project2/`; `report.pdf` is in `team12/`.

| File | Purpose |
|---|---|
| `testcharacter.py` | The agent: state machine, exact one-tick simulator, features, linear Q-function, safety shield, approximate Q-learning update |
| `weights.json` | Learned Q-function weights (one vector per state-machine mode). **Must stay next to `testcharacter.py`.** If it is missing the agent falls back to the hand-set initial weights in `DEFAULT_WEIGHTS`. |
| `train.py` | Offline approximate Q-learning on all five variants; writes `weights.json` and `training_log.json` |
| `evaluate.py` | Headless evaluation: plays each variant with many random seeds (no pygame window) and prints win rates |
| `training_log.json` | Per-episode log of the training run used for the report |
| `map.txt`, `variant1.py` ... `variant5.py` | The Project 2 map and variant launchers |
| `../report.pdf` | Project report |

## Approximate Q-learning in one paragraph

Each tick the agent enumerates 18 actions (9 moves x {drop bomb, don't}),
simulates each one for one tick with an exact copy of the game rules (monsters
commit to their move before the character does, so their next step is known),
computes a feature vector f(s, a) and scores it with
Q(s, a) = w_mode . f(s, a).  The active weight vector is chosen by a three-state
machine (ADVANCE, BOMB_WAIT, DANGER).  Features include the planner distance to
the exit (walls cost one bomb cycle), the distance a bomb saves, the probability
of dying within the next 5 ticks (expectimax over the real monster rules),
blast-zone and monster-proximity terms.  Weights are learned with the standard
update w_i <- w_i + alpha [r + gamma max_a' Q(s', a') - Q(s, a)] f_i(s, a).

## Results (held-out seeds 5000-5199, `python3 evaluate.py 200 --start 5000`)

| Variant | Wins | Rate |
|---|---|---|
| 1 | 1/1 (deterministic) | 100% |
| 2 | 198/200 | 99.0% |
| 3 | 199/200 | 99.5% |
| 4 | 190/200 | 95.0% |
| 5 | 188/200 | 94.0% |

## Training and evaluation

    cd team12/project2
    python3 train.py 1600         # ~11 minutes on 2 cores, overwrites weights.json
    python3 evaluate.py 200       # 200 random seeds per variant
    python3 evaluate.py 100 4 5   # only variants 4 and 5
    python3 evaluate.py 100 --default-weights   # the untrained hand-set weights

`weights.json` was produced by `python3 train.py 1600` starting from the
hand-set `DEFAULT_WEIGHTS`.  `python3 train.py 1600 --init naive --alpha 0.1
--out naive.json --log naive_log.json` repeats the learning-from-scratch
experiment described in the report (it does not touch `weights.json`).

## Scenario description (from the course template)

### Your goal

In this scenario, you must plan the route of your agent from the top-left
corner to the exit. However, your route is obstructed - you need to use the bomb
to create a path to the exit.

### Variant 1: Alone in the world ##

In the first variant of this scenario, the world is deterministic and your agent
is alone in the environment.

### Variant 2: Random monster ##

In the second variant of this scenario, a stupid monster is present. The monster
chooses its next cell uniformly at random among the possible reachable cells.

### Variant 3: Self-preserving monster ##

In the third variant of this scenario, a smarter monster is present:
- The monster goes straight until it has reached an obstacle
- When it reaches an obstacle, it changes direction at random among the cells
  that are walkable and are not an explosion (if an agent, monster or character,
  touches an explosion, it dies)
- If the 8-distance of your agent to the monster is 1, the monster attacks your
  agent immediately and kills it

### Variant 4: Aggressive monster ##

In the fourth variant of this scenario, an aggressive monster is present:
- The monster goes straight until it has reached an obstacle
- When it reaches an obstacle, it changes direction at random among the cells
  that are walkable and are not an explosion (if an agent, monster or character,
  touches an explosion, it dies)
- If the 8-distance of your agent to the monster is 2, the monster moves towards
  your agent and attempts to kill it

### Variant 5: Stupid and Aggressive monsters together ##

In the fifth variant of this scenario, two monsters are present: an aggressive
one and a stupid one.
