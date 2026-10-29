# Team 12 -- headless evaluation of the agent on the five Project 2 variants.
#
# Rebuilds each variant exactly like project2/variantN.py (same map, same
# monsters, same start positions) but runs the game loop without the pygame
# window, with many different random seeds.
#
# Usage (from the team12/project2 folder):
#     python3 evaluate.py                 # 100 seeds per variant
#     python3 evaluate.py 200 3 5         # 200 seeds, only variants 3 and 5
#     python3 evaluate.py 100 --start 1000   # seeds 1000..1099
#     python3 evaluate.py 100 --default-weights   # untrained (hand-set) weights
#     python3 evaluate.py 100 --weights some.json
import sys
import os
import random
import time
import json

HERE = os.path.dirname(os.path.abspath(__file__))
for p in ('../../bomberman', '../../Bomberman'):
    sys.path.insert(0, os.path.join(HERE, p))
sys.path.insert(0, HERE)

from real_world import RealWorld
from events import Event
from monsters.stupid_monster import StupidMonster
from monsters.selfpreserving_monster import SelfPreservingMonster

MAP_FILE = os.path.join(HERE, 'map.txt')


def load_world(fname=MAP_FILE):
    """Same parser as Game.fromfile, without the GUI."""
    with open(fname) as fd:
        max_time = int(fd.readline().split()[1])
        bomb_time = int(fd.readline().split()[1])
        expl_duration = int(fd.readline().split()[1])
        expl_range = int(fd.readline().split()[1])
        width = len(fd.readline()) - 3
        rows = []
        row = fd.readline()
        while row and row[0] == '|':
            rows.append(row)
            row = fd.readline()
    w = RealWorld.from_params(width, len(rows), max_time, bomb_time, expl_duration, expl_range)
    for y, ln in enumerate(rows):
        for x in range(width):
            if ln[x + 1] == 'E':
                w.add_exit(x, y)
            elif ln[x + 1] == 'W':
                w.add_wall(x, y)
    return w


def add_monsters(w, variant):
    if variant == 2:
        w.add_monster(StupidMonster("stupid", "S", 3, 9))
    elif variant == 3:
        w.add_monster(SelfPreservingMonster("selfpreserving", "S", 3, 9, 1))
    elif variant == 4:
        w.add_monster(SelfPreservingMonster("aggressive", "A", 3, 13, 2))
    elif variant == 5:
        w.add_monster(StupidMonster("stupid", "S", 3, 5))
        w.add_monster(SelfPreservingMonster("aggressive", "A", 3, 13, 2))


def play(variant, seed, character=None, max_ticks=None):
    """Play one game. Returns (outcome, ticks) with outcome in
    {'win', 'killed_by_monster', 'own_bomb', 'timeout'}."""
    from testcharacter import TestCharacter
    random.seed(seed)
    w = load_world()
    add_monsters(w, variant)
    c = character or TestCharacter("me", "C", 0, 0)
    w.add_character(c)
    outcome, ticks = 'timeout', 0
    # same loop as Game.go(): world.next() then everybody decides
    while w.time > 0 and w.characters and any(w.characters.values()):
        w, events = w.next()
        ticks += 1
        for e in events:
            if e.tpe == Event.CHARACTER_KILLED_BY_MONSTER:
                outcome = 'killed_by_monster'
            elif e.tpe == Event.BOMB_HIT_CHARACTER:
                outcome = 'own_bomb'
            elif e.tpe == Event.CHARACTER_FOUND_EXIT and outcome == 'timeout':
                outcome = 'win'
        if outcome != 'timeout':
            break
        if max_ticks and ticks >= max_ticks:
            break
        w.next_decisions()
    return outcome, ticks


def _init_worker(weights):
    if weights is not None:
        import testcharacter
        testcharacter.TestCharacter._shared_weights = weights


def _job(args):
    v, seed = args
    t0 = time.time()
    out, ticks = play(v, seed)
    return v, seed, out, ticks, time.time() - t0


def main():
    argv = sys.argv[1:]
    start = 1
    if '--start' in argv:
        i = argv.index('--start')
        start = int(argv[i + 1])
        del argv[i:i + 2]
    weights = None
    if '--weights' in argv:              # evaluate another weights file
        i = argv.index('--weights')
        with open(argv[i + 1]) as fd:
            weights = json.load(fd)
        del argv[i:i + 2]
    if '--default-weights' in argv:      # the hand-set weights before training
        import testcharacter
        weights = {m: dict(testcharacter.DEFAULT_WEIGHTS[m]) for m in testcharacter.MODES}
    _init_worker(weights)
    args = [a for a in argv if not a.startswith('--')]
    n = int(args[0]) if args else 100
    variants = [int(a) for a in args[1:]] or [1, 2, 3, 4, 5]
    jobs = [(v, s) for v in variants for s in range(start, start + (1 if v == 1 else n))]
    try:
        from multiprocessing import Pool
        with Pool(initializer=_init_worker, initargs=(weights,)) as pool:
            results = pool.map(_job, jobs, chunksize=4)
    except (ImportError, OSError):
        results = [_job(j) for j in jobs]
    print(f"{'variant':>7} {'games':>5} {'wins':>5} {'rate':>6} {'monster':>8} {'own bomb':>8} "
          f"{'timeout':>7} {'avg ticks':>9} {'s/game':>6}")
    for v in variants:
        rs = [r for r in results if r[0] == v]
        wins = sum(r[2] == 'win' for r in rs)
        km = sum(r[2] == 'killed_by_monster' for r in rs)
        kb = sum(r[2] == 'own_bomb' for r in rs)
        to = sum(r[2] == 'timeout' for r in rs)
        wt = [r[3] for r in rs if r[2] == 'win']
        print(f"{v:>7} {len(rs):>5} {wins:>5} {wins / len(rs):>6.0%} {km:>8} {kb:>8} {to:>7} "
              f"{(sum(wt) / len(wt) if wt else 0):>9.0f} {sum(r[4] for r in rs) / len(rs):>6.2f}")
    if '--list' in sys.argv:
        for r in sorted(results):
            if r[2] != 'win':
                print('loss', r)


if __name__ == '__main__':
    main()
