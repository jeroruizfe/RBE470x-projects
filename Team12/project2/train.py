# Team 12 -- offline approximate Q-learning for the agent in testcharacter.py
#
# Q_mode(s, a) = w_mode . f(s, a)       (one weight vector per state-machine mode)
#
# Each training episode plays one of the five Project 2 variants (random seed)
# with an epsilon-greedy version of the agent.  After every action the agent
# performs the approximate Q-learning update
#
#     delta = r + gamma * max_a' Q(s', a') - Q(s, a)
#     w_i  <- w_i + alpha * delta * f_i(s, a) / (1 + |f(s, a)|^2)
#
# (r = -1 per tick, +100 on reaching the exit, -100 on death).  Episodes run
# in parallel batches; every batch starts from the current weights and the
# weight changes of the batch are averaged.  The result goes to weights.json.
#
# Usage (from the team12/project2 folder):
#     python3 train.py                       # 1200 episodes from DEFAULT_WEIGHTS
#     python3 train.py 600 --resume          # continue from weights.json
#     python3 train.py 800 --checkpoints ckpt  # also save weights every 200 episodes
#     python3 train.py 1600 --init naive --alpha 0.1 --out naive.json --log naive_log.json
#                                            # learn from a naive start (exit-seeking only)
import sys
import os
import json
import random
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import evaluate
import testcharacter as tc

VARIANT_MIX = [1, 2, 2, 3, 3, 4, 4, 4, 5, 5, 5, 5]   # harder variants more often
BATCH = 8
ALPHA = 0.02
GAMMA = 0.98
EPS_START, EPS_END = 0.10, 0.01


def run_episode(args):
    weights, variant, seed, eps, alpha = args
    c = tc.TestCharacter("me", "C", 0, 0, weights=weights, learn=True,
                         alpha=alpha, gamma=GAMMA, epsilon=eps, seed=seed)
    outcome, ticks = evaluate.play(variant, seed, character=c)
    if outcome == 'timeout' and c.prev is not None:
        c.prev = None
    delta = {m: {f: c.w[m][f] - weights[m][f] for f in tc.FEATURES} for m in tc.MODES}
    err = sum(abs(e) for e in c.td_errors) / max(1, len(c.td_errors))
    return variant, outcome, ticks, delta, err


# "naive" starting point used to check that learning works on its own:
# only head for the exit, every safety-related weight starts at zero.
NAIVE_WEIGHTS = {m: {f: (-10.0 if f == 'dist' else 0.0) for f in tc.FEATURES} for m in tc.MODES}


def option(name, default):
    if name in sys.argv:
        i = sys.argv.index(name)
        val = sys.argv[i + 1]
        del sys.argv[i:i + 2]
        return val
    return default


def main():
    alpha = float(option('--alpha', ALPHA))
    init = option('--init', 'default')
    out_file = option('--out', tc.WEIGHTS_FILE)
    log_file = option('--log', os.path.join(HERE, 'training_log.json'))
    ckpt_dir = option('--checkpoints', None)      # save weights every 200 episodes
    if ckpt_dir:
        os.makedirs(ckpt_dir, exist_ok=True)
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    episodes = int(args[0]) if args else 1200
    if '--resume' in sys.argv and os.path.exists(out_file):
        with open(out_file) as fd:
            loaded = json.load(fd)
        weights = {m: {f: float(loaded[m][f]) for f in tc.FEATURES} for m in tc.MODES}
    elif init == 'naive':
        weights = {m: dict(NAIVE_WEIGHTS[m]) for m in tc.MODES}
    else:
        weights = {m: dict(tc.DEFAULT_WEIGHTS[m]) for m in tc.MODES}
    if ckpt_dir:
        with open(os.path.join(ckpt_dir, 'weights_00000.json'), 'w') as fd:
            json.dump(weights, fd, indent=2)
    offset = 0
    if '--resume' in sys.argv and os.path.exists(log_file):
        with open(log_file) as fd:
            old_log = json.load(fd)
        offset = len(old_log)
    else:
        old_log = []
    rng = random.Random(2026 + offset)
    log = []
    t0 = time.time()
    from multiprocessing import Pool
    with Pool() as pool:
        done = 0
        while done < episodes:
            frac = done / max(1, episodes)
            eps = EPS_START + (EPS_END - EPS_START) * frac
            jobs = [(weights, rng.choice(VARIANT_MIX), rng.randrange(10 ** 6), eps, alpha)
                    for _ in range(BATCH)]
            results = pool.map(run_episode, jobs)
            for m in tc.MODES:
                for f in tc.FEATURES:
                    weights[m][f] += sum(r[3][m][f] for r in results) / len(results)
            for r in results:
                log.append({'episode': offset + done, 'variant': r[0], 'outcome': r[1],
                            'ticks': r[2], 'td_error': r[4], 'eps': eps})
                done += 1
            if done % 80 < BATCH:
                recent = log[-80:]
                wins = sum(e['outcome'] == 'win' for e in recent)
                print(f"episode {done:5d}  eps={eps:.3f}  recent win rate {wins}/{len(recent)}"
                      f"  mean |td error| {sum(e['td_error'] for e in recent) / len(recent):.2f}"
                      f"  ({time.time() - t0:.0f}s)", flush=True)
            with open(out_file, 'w') as fd:
                json.dump(weights, fd, indent=2)
            if ckpt_dir and done % 200 < BATCH:
                with open(os.path.join(ckpt_dir, f'weights_{offset + done:05d}.json'), 'w') as fd:
                    json.dump(weights, fd, indent=2)
    with open(log_file, 'w') as fd:
        json.dump(old_log + log, fd)
    print('saved', out_file)
    for m in tc.MODES:
        print(m, {f: round(v, 3) for f, v in weights[m].items()})


if __name__ == '__main__':
    main()
