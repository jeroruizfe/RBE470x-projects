# Team 12 -- RBE4701 Project 2 agent
#
# Approximate Q-learning Bomberman agent.
#
#   Q_mode(s, a) = sum_i  w_mode[i] * f_i(s, a)
#
# A small state machine (ADVANCE / BOMB_WAIT / DANGER) decides which learned
# weight vector is active.  Every tick the agent enumerates its 18 actions
# (9 moves x {bomb, no bomb}), simulates each one exactly for one tick with a
# light-weight copy of the game rules, extracts features from the result and
# takes the action with the highest Q value.  A safety shield removes actions
# that the simulator proves fatal (walking into an explosion or onto a monster,
# or ending up in a blast cross that cannot be left before it detonates).
#
# The weights are learned offline with approximate Q-learning (train.py) and
# stored in weights.json next to this file.

# This is necessary to find the main code (the engine folder sits two levels
# up from team12/project2; both spellings of its name are accepted)
import sys
import os
for _p in ('../../bomberman', '../../Bomberman'):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), _p))
# Import necessary stuff
from entity import CharacterEntity
from events import Event
from colorama import Fore, Back
import json
import heapq
import random
from collections import deque

HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_FILE = os.path.join(HERE, 'weights.json')

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
STUPID, CHASER = 0, 1
# Detection range of the self-preserving monsters, keyed by the names used in
# the variant files.  The sensed world hides the real monster class, so the
# name is the only clue; unknown names are treated as the most dangerous kind.
CHASER_RANGE = {'selfpreserving': 1, 'aggressive': 2}
DEFAULT_CHASER_RANGE = 2

# (dx, dy) moves; (0, 0) first so that ties prefer standing still
MOVES = [(0, 0), (0, 1), (-1, 1), (1, 1), (-1, 0), (1, 0), (0, -1), (-1, -1), (1, -1)]

WALL_COST = 16         # planner cost of an intact wall cell (one full bomb cycle)
LOOKAHEAD = 5          # depth of the survival lookahead (ticks)
STUPID_EXACT = 2       # lookahead levels where random walkers are branched exactly
NODE_BUDGET = 30000    # lookahead nodes per decision (keeps every tick fast)
THREAT_DIST = 6        # monster path distance that switches to DANGER mode
INF = 10 ** 6

# Rewards used by the approximate Q-learning update
REWARD_STEP = -1.0     # every tick
REWARD_WIN = 100.0     # reached the exit
REWARD_DEATH = -100.0  # killed by a monster or a bomb

MODES = ('advance', 'bomb_wait', 'danger')
FEATURES = ('bias',          # constant 1
            'dist',          # planner distance to the exit after the action / 10
            'bomb_gain',     # planner cost a bomb placed now saves / WALL_COST
            'risk',          # P(death within LOOKAHEAD ticks | best replies)
            'in_blast',      # standing in the cross of a live bomb
            'mon_prox',      # closeness of the nearest reachable monster (6 cells)
            'chaser_prox',   # closeness of the nearest reachable chaser (10 cells)
            'chaser_near',   # inside (detection range + 1) of a chaser
            'cornered',      # fraction of blocked neighbouring cells
            'stay')          # the action does not move

# Hand-written starting point for training (also used if weights.json is
# missing).  train.py refines these with approximate Q-learning.
DEFAULT_WEIGHTS = {
    'advance':   {'bias': 0.0, 'dist': -10.0, 'bomb_gain': 2.0, 'risk': -60.0, 'in_blast': -0.5,
                  'mon_prox': -2.0, 'chaser_prox': -2.0, 'chaser_near': -3.0, 'cornered': -0.5, 'stay': -0.3},
    'bomb_wait': {'bias': 0.0, 'dist': -10.0, 'bomb_gain': 0.0, 'risk': -60.0, 'in_blast': -2.0,
                  'mon_prox': -2.0, 'chaser_prox': -2.0, 'chaser_near': -3.0, 'cornered': -0.5, 'stay': 0.0},
    'danger':    {'bias': 0.0, 'dist': -10.0, 'bomb_gain': 1.0, 'risk': -60.0, 'in_blast': -1.0,
                  'mon_prox': -4.0, 'chaser_prox': -3.0, 'chaser_near': -6.0, 'cornered': -2.0, 'stay': -0.2},
}


def sign(v):
    return (v > 0) - (v < 0)


def cheb(ax, ay, bx, by):
    return max(abs(ax - bx), abs(ay - by))


# ---------------------------------------------------------------------------
# Light-weight exact simulator of one game tick
# ---------------------------------------------------------------------------
class Sim:
    """Compact copy of the world.  step() reproduces RealWorld.next() for one
    tick: explosions age, bombs explode, monsters move (with the direction
    they already chose -- monsters decide before the character, so the
    sensed world tells us their next move), then the character moves."""

    __slots__ = ('W', 'H', 'grid', 'exit', 'bombs', 'expl', 'mons', 'me',
                 'brange', 'btime', 'edur')

    @classmethod
    def from_world(cls, wrld, me):
        s = cls()
        s.W, s.H = wrld.width(), wrld.height()
        W = s.W
        s.grid = [False] * (W * s.H)
        for x in range(W):
            col = wrld.grid[x]
            for y in range(s.H):
                if col[y]:
                    s.grid[x + y * W] = True
        s.exit = wrld.exitcell
        s.bombs = tuple((b.x, b.y, b.timer) for b in wrld.bombs.values())
        s.expl = {e.x + e.y * W: e.timer for e in wrld.explosions.values()}
        mons = []
        for mlist in wrld.monsters.values():
            for m in mlist:
                if m.name == 'stupid':
                    mons.append((STUPID, m.x, m.y, m.dx, m.dy, 0))
                else:
                    mons.append((CHASER, m.x, m.y, m.dx, m.dy,
                                 CHASER_RANGE.get(m.name, DEFAULT_CHASER_RANGE)))
        s.mons = tuple(mons)
        s.me = (me.x, me.y)
        s.brange = wrld.expl_range
        s.btime = wrld.bomb_time
        s.edur = wrld.expl_duration
        return s

    def copy_with(self, **kw):
        n = Sim()
        for k in Sim.__slots__:
            setattr(n, k, kw[k] if k in kw else getattr(self, k))
        return n

    def wall(self, x, y):
        return self.grid[x + y * self.W]

    def cross(self, bx, by, grid=None):
        """Cells a bomb at (bx, by) would set on fire (ignores creatures,
        which only make the blast shorter -- so this is conservative)."""
        grid = self.grid if grid is None else grid
        W, H = self.W, self.H
        cells = [(bx, by)]
        bombpos = {(b[0], b[1]) for b in self.bombs}
        for ddx, ddy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            x, y, r = bx + ddx, by + ddy, 0
            while r < self.brange and 0 <= x < W and 0 <= y < H:
                if (x, y) == self.exit or (x, y) in bombpos:
                    break
                cells.append((x, y))
                if grid[x + y * W]:
                    break
                x, y, r = x + ddx, y + ddy, r + 1
        return cells

    def step(self, dx, dy, bomb):
        """Advance one tick with the character doing (dx, dy, bomb).
        Returns (new_state, status) with status 1 = exit, -1 = dead, 0 = alive."""
        W, H = self.W, self.H
        grid = self.grid
        copied = False
        # 1. explosions age; expired ones clear the cell (that is how walls go)
        expl = {}
        for idx, t in self.expl.items():
            t -= 1
            if t < 0:
                if grid[idx]:
                    if not copied:
                        grid = list(grid)
                        copied = True
                    grid[idx] = False
            else:
                expl[idx] = t
        # 2. bombs tick and explode
        mons = list(self.mons)
        me = self.me
        dead = False
        bombs = []
        bombpos = {(b[0], b[1]) for b in self.bombs}
        for (bx, by, t) in self.bombs:
            t -= 1
            if t >= 0:
                bombs.append((bx, by, t))
                continue
            expl[bx + by * W] = self.edur
            hit = False
            if any(m[1] == bx and m[2] == by for m in mons):
                mons = [m for m in mons if not (m[1] == bx and m[2] == by)]
                hit = True
            if me == (bx, by):
                dead = True
                hit = True
            if hit:
                continue
            for ddx, ddy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                x, y, r = bx + ddx, by + ddy, 0
                while r < self.brange and 0 <= x < W and 0 <= y < H:
                    if (x, y) == self.exit or (x, y) in bombpos:
                        break
                    expl[x + y * W] = self.edur
                    if grid[x + y * W]:
                        break
                    stop = False
                    if any(m[1] == x and m[2] == y for m in mons):
                        mons = [m for m in mons if not (m[1] == x and m[2] == y)]
                        stop = True
                    if me == (x, y):
                        dead = True
                        stop = True
                    if stop:
                        break
                    x, y, r = x + ddx, y + ddy, r + 1
        if dead:
            return None, -1
        # 3. monsters move with their committed direction
        nm = []
        for (k, x, y, mdx, mdy, rg) in mons:
            nx = min(max(x + mdx, 0), W - 1)
            ny = min(max(y + mdy, 0), H - 1)
            if (nx != x or ny != y) and not grid[nx + ny * W]:
                if (nx + ny * W) in expl:
                    continue                      # walked into fire
                if (nx, ny) == me:
                    dead = True
                nm.append((k, nx, ny, mdx, mdy, rg))
            else:
                nm.append((k, x, y, mdx, mdy, rg))
        if dead:
            return None, -1
        # 4. the character: maybe drop a bomb, then move
        if bomb and not bombs:
            bombs.append((me[0], me[1], self.btime))
        nx = min(max(me[0] + dx, 0), W - 1)
        ny = min(max(me[1] + dy, 0), H - 1)
        status = 0
        if (nx, ny) != me and not grid[nx + ny * W]:
            me = (nx, ny)
            if (nx + ny * W) in expl:
                return None, -1
            if any(m[1] == nx and m[2] == ny for m in nm):
                return None, -1
            if (nx, ny) == self.exit:
                status = 1
        n = Sim()
        n.W, n.H, n.grid, n.exit = W, H, grid, self.exit
        n.bombs, n.expl, n.mons, n.me = tuple(bombs), expl, tuple(nm), me
        n.brange, n.btime, n.edur = self.brange, self.btime, self.edur
        return n, status

    # ---- monster decision models (exact copies of the monster code) ----
    def empty(self, x, y):
        if (x, y) == self.exit or self.grid[x + y * self.W]:
            return False
        if (x + y * self.W) in self.expl:
            return False
        for b in self.bombs:
            if b[0] == x and b[1] == y:
                return False
        for m in self.mons:
            if m[1] == x and m[2] == y:
                return False
        return self.me != (x, y)

    def monster_choices(self, m):
        """List of (probability, (dx, dy)) for the monster's next decision."""
        k, x, y, dx, dy, rg = m
        W, H = self.W, self.H
        if k == STUPID:
            opts = []
            for ddx in (-1, 0, 1):
                if 0 <= x + ddx < W:
                    for ddy in (-1, 0, 1):
                        if 0 <= y + ddy < H and not self.grid[(x + ddx) + (y + ddy) * W]:
                            opts.append((ddx, ddy))
            p = 1.0 / len(opts)
            return [(p, o) for o in opts]
        cx, cy = self.me
        found = abs(cx - x) <= rg and abs(cy - y) <= rg
        nx, ny = x + dx, y + dy
        if not (0 <= nx < W and 0 <= ny < H):
            must = True
        else:
            must = (((nx + ny * W) in self.expl) or self.grid[nx + ny * W]
                    or (nx, ny) == self.exit
                    or any(o[1] == nx and o[2] == ny for o in self.mons))
        if found and not must:
            return [(1.0, (sign(cx - x), sign(cy - y)))]
        if (dx == 0 and dy == 0) or must:
            opts = []
            for ddx in (-1, 0, 1):
                if 0 <= x + ddx < W:
                    for ddy in (-1, 0, 1):
                        if 0 <= y + ddy < H:
                            if (x + ddx, y + ddy) == self.exit or self.empty(x + ddx, y + ddy):
                                opts.append((ddx, ddy))
            if not opts:
                return [(1.0, (0, 0))]
            p = 1.0 / len(opts)
            return [(p, o) for o in opts]
        return [(1.0, (dx, dy))]


# ---------------------------------------------------------------------------
# Planner: Dijkstra distance-to-exit where walls cost a bomb cycle
# ---------------------------------------------------------------------------
def scheduled_walls(s, bombs):
    """{wall index: ticks until it is gone} for walls that a live bomb will
    destroy or that are already on fire.  A bomb with timer t explodes after
    t + 1 ticks and its fire (duration edur) clears the wall edur + 1 ticks
    later; a fire with timer e clears its cell after e + 1 ticks."""
    out = {}
    for (bx, by, t) in bombs:
        for (x, y) in s.cross(bx, by):
            idx = x + y * s.W
            if s.grid[idx]:
                out[idx] = min(out.get(idx, INF), t + s.edur + 2)
    for idx, e in s.expl.items():
        if s.grid[idx]:
            out[idx] = min(out.get(idx, INF), e + 1)
    return out


def exit_distance_map(s, bombs):
    """dist[idx] = cost to reach the exit from cell idx (8-connected)."""
    W, H = s.W, s.H
    sched = scheduled_walls(s, bombs)
    dist = [INF] * (W * H)
    ex, ey = s.exit
    dist[ex + ey * W] = 0
    pq = [(0, ex, ey)]
    while pq:
        d, x, y = heapq.heappop(pq)
        if d > dist[x + y * W]:
            continue
        # cost of the step INTO (x, y) is paid by the neighbour, so the
        # neighbour's distance is d + cost(x, y) where cost depends on (x, y)
        idx = x + y * W
        if s.grid[idx]:
            c = max(1, sched[idx]) if idx in sched else WALL_COST
        else:
            c = 1
        if (x, y) == (ex, ey):
            c = 1
        for ddx in (-1, 0, 1):
            nx = x + ddx
            if not 0 <= nx < W:
                continue
            for ddy in (-1, 0, 1):
                ny = y + ddy
                if (ddx or ddy) and 0 <= ny < H:
                    nd = d + c
                    nidx = nx + ny * W
                    if nd < dist[nidx]:
                        dist[nidx] = nd
                        heapq.heappush(pq, (nd, nx, ny))
    return dist


def bfs_map(s, sources):
    """8-connected walking distance from the source cells through free cells."""
    W, H = s.W, s.H
    dist = [INF] * (W * H)
    q = deque()
    for (x, y) in sources:
        dist[x + y * W] = 0
        q.append((x, y))
    while q:
        x, y = q.popleft()
        d = dist[x + y * W] + 1
        for ddx in (-1, 0, 1):
            nx = x + ddx
            if not 0 <= nx < W:
                continue
            for ddy in (-1, 0, 1):
                ny = y + ddy
                if (ddx or ddy) and 0 <= ny < H:
                    nidx = nx + ny * W
                    # walls on fire are about to open, treat them as open
                    if (not s.grid[nidx] or nidx in s.expl) and d < dist[nidx]:
                        dist[nidx] = d
                        q.append((nx, ny))
    return dist


def blast_escape_ok(s):
    """Shield: can the character leave every live blast cross before it
    explodes?  (bomb with timer t leaves exactly t more moves)."""
    if not s.bombs:
        return True
    danger = {}
    for (bx, by, t) in s.bombs:
        for c in s.cross(bx, by):
            danger[c] = min(danger.get(c, INF), t)
    if s.me not in danger:
        return True
    W, H = s.W, s.H
    budget = danger[s.me]
    seen = {s.me}
    frontier = [s.me]
    for step in range(1, budget + 1):
        nxt = []
        for (x, y) in frontier:
            for ddx in (-1, 0, 1):
                for ddy in (-1, 0, 1):
                    nx, ny = x + ddx, y + ddy
                    if (nx, ny) in seen or not (0 <= nx < W and 0 <= ny < H):
                        continue
                    if s.grid[nx + ny * W]:
                        continue
                    if (nx + ny * W) in s.expl and s.expl[nx + ny * W] >= step - 1:
                        continue
                    if (nx, ny) not in danger:
                        return True
                    if danger[(nx, ny)] < step:
                        continue
                    seen.add((nx, ny))
                    nxt.append((nx, ny))
        frontier = nxt
    return False


# ---------------------------------------------------------------------------
# Survival lookahead (expectimax over the exact monster models)
# ---------------------------------------------------------------------------
def survival(s, depth, memo, tag, level, budget):
    """Probability of being alive `depth` ticks from now if we play the best
    replies, given the monsters' true (partly random) decision rules.

    Chasers are modelled exactly at every level.  Random walkers ("stupid"
    monsters) are branched exactly for the first STUPID_EXACT levels; deeper
    than that they are frozen in place and every tick we spend next to one
    costs its 1/n chance of stepping onto us.  `budget` caps the number of
    expanded nodes per decision so a tick never takes too long."""
    if depth == 0:
        return 1.0
    key = (tag, level, depth, s.me, s.mons)
    v = memo.get(key)
    if v is not None:
        return v
    budget[0] -= 1
    if budget[0] <= 0 and depth > 1:
        depth = 1
    mx, my = s.me
    rel, frozen = [], []
    for m in s.mons:
        if cheb(m[1], m[2], mx, my) > depth + 1:
            continue                       # cannot reach us in time
        if m[0] == STUPID and level >= STUPID_EXACT:
            frozen.append((m[0], m[1], m[2], 0, 0, m[5]))
        else:
            rel.append(m)
    if not rel and not frozen and not s.bombs and not s.expl:
        memo[key] = 1.0
        return 1.0
    base = s.copy_with(mons=tuple(rel) + tuple(frozen))
    frozen_p = [1.0 / len(base.monster_choices(f)) for f in frozen]
    choice_lists = [base.monster_choices(m) for m in rel]
    combos = [(1.0, [])]
    for cl in choice_lists:
        combos = [(p * q, dirs + [d]) for (p, dirs) in combos for (q, d) in cl]
    # order our replies: moves away from the nearest monster first
    near = rel + frozen
    if near:
        n = min(near, key=lambda m: cheb(m[1], m[2], mx, my))
        order = sorted(MOVES, key=lambda d: -cheb(mx + d[0], my + d[1], n[1], n[2]))
    else:
        order = MOVES
    total = 0.0
    for p, dirs in combos:
        mons = tuple((m[0], m[1], m[2], d[0], d[1], m[5]) for m, d in zip(rel, dirs)) + tuple(frozen)
        sd = base.copy_with(mons=mons)
        best = 0.0
        for (dx, dy) in order:
            s2, st = sd.step(dx, dy, False)
            if st == 1:
                v2 = 1.0
            elif st == -1:
                v2 = 0.0
            else:
                pen = 1.0
                for f, fp in zip(frozen, frozen_p):
                    if cheb(f[1], f[2], s2.me[0], s2.me[1]) <= 1:
                        pen *= 1.0 - fp
                v2 = pen * survival(s2, depth - 1, memo, tag, level + 1, budget)
            if v2 > best:
                best = v2
                if best >= 0.999999:
                    break
        total += p * best
    memo[key] = total
    return total


# ---------------------------------------------------------------------------
# The agent
# ---------------------------------------------------------------------------
class TestCharacter(CharacterEntity):

    _shared_weights = None

    def __init__(self, name, avatar, x, y, weights=None, learn=False,
                 alpha=0.02, gamma=0.98, epsilon=0.0, seed=None):
        super().__init__(name, avatar, x, y)
        if weights is None:
            weights = TestCharacter.load_weights()
        self.w = {m: dict(weights[m]) for m in MODES}
        self.learn = learn
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.rng = random.Random(seed)
        self.prev = None          # (mode, features, q) of the last action
        self.mode = 'advance'
        self.td_errors = []

    # ---------------- weights ----------------
    @classmethod
    def load_weights(cls):
        if cls._shared_weights is None:
            w = {m: dict(DEFAULT_WEIGHTS[m]) for m in MODES}
            try:
                with open(WEIGHTS_FILE) as fd:
                    data = json.load(fd)
                for m in MODES:
                    for f in FEATURES:
                        if f in data.get(m, {}):
                            w[m][f] = float(data[m][f])
            except (OSError, ValueError):
                pass
            cls._shared_weights = w
        return cls._shared_weights

    def q(self, mode, feats):
        w = self.w[mode]
        return sum(w[f] * feats[f] for f in FEATURES)

    # ---------------- state machine ----------------
    @staticmethod
    def pick_mode(s, mon_dist):
        if s.mons and min(mon_dist[s.me[0] + s.me[1] * s.W]) <= THREAT_DIST:
            return 'danger'
        if s.bombs:
            return 'bomb_wait'
        return 'advance'

    # ---------------- features ----------------
    def candidates(self, s):
        """Simulate every action and compute its features.
        Returns (mode, list of (action, features, safe))."""
        W = s.W
        me = s.me
        # walking distance from each monster, measured where the monsters
        # will be after this tick (their move is already known)
        mon_after = []
        for (k, x, y, mdx, mdy, rg) in s.mons:
            nx = min(max(x + mdx, 0), W - 1)
            ny = min(max(y + mdy, 0), s.H - 1)
            if s.grid[nx + ny * W]:
                nx, ny = x, y
            mon_after.append((k, nx, ny, rg))
        mon_maps = [bfs_map(s, [(m[1], m[2])]) for m in mon_after]
        per_cell = [[mm[i] for mm in mon_maps] for i in range(W * s.H)] if mon_maps else None
        mode = self.pick_mode(s, per_cell) if per_cell else ('bomb_wait' if s.bombs else 'advance')

        d0 = exit_distance_map(s, s.bombs)
        can_bomb = not s.bombs
        if can_bomb:
            db = exit_distance_map(s, s.bombs + ((me[0], me[1], s.btime),))
            gain = max(0.0, min(1.0, (d0[me[0] + me[1] * W] - db[me[0] + me[1] * W]) / WALL_COST))
        out = []
        memo = {}
        budget = [NODE_BUDGET]
        for bomb in ((False, True) if can_bomb else (False,)):
            dmap = db if bomb else d0
            for (dx, dy) in MOVES:
                s1, st = s.step(dx, dy, bomb)
                if st == 1:
                    out.append(((dx, dy, bomb), None, 'win'))
                    continue
                if st == -1:
                    out.append(((dx, dy, bomb), None, 'dead'))
                    continue
                m1 = s1.me
                i1 = m1[0] + m1[1] * W
                f = dict.fromkeys(FEATURES, 0.0)
                f['bias'] = 1.0
                f['bomb_gain'] = gain if bomb else 0.0
                f['dist'] = min(dmap[i1], 200) / 10.0
                crossing = set()
                for (bx, by, _t) in s1.bombs:
                    crossing.update(s1.cross(bx, by))
                f['in_blast'] = 1.0 if m1 in crossing else 0.0
                safe = blast_escape_ok(s1)
                # monster features
                if per_cell:
                    near = min(per_cell[i1])
                    f['mon_prox'] = max(0.0, (THREAT_DIST + 1 - near) / (THREAT_DIST + 1))
                    for (k, x, y, rg), mm in zip(mon_after, mon_maps):
                        if k == CHASER and mm[i1] < INF:
                            f['chaser_prox'] = max(f['chaser_prox'], max(0.0, (10 - mm[i1]) / 10.0))
                            if cheb(x, y, m1[0], m1[1]) <= rg + 1:
                                f['chaser_near'] = 1.0
                # survival lookahead (monsters and bombs)
                if s1.mons or s1.bombs or s1.expl:
                    f['risk'] = 1.0 - survival(s1, LOOKAHEAD, memo, bomb, 0, budget)
                blocked = 0
                for ddx in (-1, 0, 1):
                    for ddy in (-1, 0, 1):
                        if ddx or ddy:
                            nx, ny = m1[0] + ddx, m1[1] + ddy
                            if not (0 <= nx < W and 0 <= ny < s.H) or s.grid[nx + ny * W]:
                                blocked += 1
                f['cornered'] = blocked / 8.0
                f['stay'] = 1.0 if (dx, dy) == (0, 0) else 0.0
                out.append(((dx, dy, bomb), f, 'ok' if safe else 'unsafe'))
        return mode, out

    def choose(self, mode, cands):
        wins = [c for c in cands if c[2] == 'win']
        if wins:
            return wins[0][0], None, None
        ok = [c for c in cands if c[2] == 'ok']
        if not ok:
            ok = [c for c in cands if c[2] == 'unsafe']
        if not ok:
            return (0, 0, False), None, None
        scored = [(self.q(mode, c[1]), c) for c in ok]
        if self.learn and self.rng.random() < self.epsilon:
            qv, c = self.rng.choice(scored)
        else:
            qv, c = max(scored, key=lambda t: t[0])
        return c[0], c[1], qv

    def best_q(self, mode, cands):
        if any(c[2] == 'win' for c in cands):
            return REWARD_WIN
        ok = [c for c in cands if c[2] == 'ok'] or [c for c in cands if c[2] == 'unsafe']
        if not ok:
            return REWARD_DEATH
        return max(self.q(mode, c[1]) for c in ok)

    def td_update(self, target):
        mode, feats, qv = self.prev
        err = target - qv
        err = max(-50.0, min(50.0, err))
        self.td_errors.append(err)
        w = self.w[mode]
        norm = 1.0 + sum(v * v for v in feats.values())     # normalised step size
        for f in FEATURES:
            w[f] += self.alpha * err * feats[f] / norm

    # ---------------- main entry points ----------------
    def do(self, wrld):
        me = wrld.me(self)
        s = Sim.from_world(wrld, me)
        if wrld.exit_at(me.x, me.y):
            self.move(0, 0)
            return
        mode, cands = self.candidates(s)
        self.mode = mode
        if self.learn and self.prev is not None:
            self.td_update(REWARD_STEP + self.gamma * self.best_q(mode, cands))
        (dx, dy, bomb), feats, qv = self.choose(mode, cands)
        self.prev = (mode, feats, qv) if feats is not None else None
        if bomb:
            self.place_bomb()
        self.move(dx, dy)

    def done(self, wrld):
        if not (self.learn and self.prev is not None):
            return
        won = False
        for e in wrld.events:
            if e.tpe == Event.CHARACTER_FOUND_EXIT and e.character.name == self.name:
                won = True
        self.td_update(REWARD_WIN if won else REWARD_DEATH)
        self.prev = None

