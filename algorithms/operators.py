"""
algorithms/operators.py
Defines candidate neighborhood generation and genetic recombination operators.
"""

import random
from typing import List, Tuple
from core.state import State
from core.problem import CommuteProblem


def _time_neighbors(state: State, problem: CommuteProblem, max_steps: int = 2) -> List[float]:
    """
    Departure times reachable by shifting +-1..max_steps grid steps.
    All results stay on the discrete grid and inside [t_min, t_max].
    """
    times = []
    for k in range(1, max_steps + 1):
        for sign in (-1, 1):
            t = round(state.departure_time + sign * k * problem.t_step, 2)
            if problem.t_min - 1e-9 <= t <= problem.t_max + 1e-9:
                times.append(t)
    return times


def _find_detours(
    problem: CommuteProblem,
    path: Tuple[str, ...],
    i: int,
    j: int,
    extra_hops: int = 3,
    limit: int = 3,
    budget: int = 3000,
) -> List[Tuple[str, ...]]:
    """
    Finds up to `limit` simple sub-paths from path[i] to path[j] that differ
    from the current segment path[i..j].

    Loop-freeness guarantee: the DFS never enters a node that is kept outside
    the replaced segment (path[:i] and path[j+1:]) nor a node already used
    in the segment under construction. So prefix + segment + suffix is
    always a simple path.
    """
    src, dst = path[i], path[j]
    original = path[i:j + 1]
    forbidden_outside = set(path[:i]) | set(path[j + 1:])
    max_len = (j - i) + extra_hops  # max number of hops of the new segment

    found: List[Tuple[str, ...]] = []
    expansions = 0
    stack_path = [src]
    on_path = {src}

    def dfs(u: str) -> None:
        nonlocal expansions
        if len(found) >= limit or expansions >= budget:
            return
        expansions += 1
        if u == dst:
            seg = tuple(stack_path)
            if seg != original:
                found.append(seg)
            return
        if len(stack_path) - 1 >= max_len:
            return
        nbrs = problem.network.get_neighbors(u)
        random.shuffle(nbrs)
        for v in nbrs:
            if v in on_path or v in forbidden_outside:
                continue
            stack_path.append(v)
            on_path.add(v)
            dfs(v)
            on_path.discard(v)
            stack_path.pop()
            if len(found) >= limit or expansions >= budget:
                return

    dfs(src)
    return found


def _path_neighbors(state: State, problem: CommuteProblem) -> List[Tuple[str, ...]]:
    """
    Local topological moves: for every pair of positions (i, j) on the path,
    replace the sub-path between them with an alternative detour.
    """
    path = state.path
    result: List[Tuple[str, ...]] = []
    for i in range(len(path) - 1):
        for j in range(i + 1, len(path)):
            for seg in _find_detours(problem, path, i, j):
                new_path = path[:i] + seg + path[j + 1:]
                if problem.is_valid_path(new_path):  # safety net
                    result.append(new_path)
    return result


def get_neighbors(state: State, problem: CommuteProblem) -> List[State]:
    """
    Generates a set of valid neighboring configurations for the current state.

    Three kinds of moves are produced:
      1. Time moves:   same path, departure_time +- 1 or 2 steps.
      2. Path moves:   same departure time, one sub-path replaced by a
                       loop-free detour between two nodes of the path.
      3. Mixed moves:  a path move combined with a +-1 step time shift
                       (lets the search change route and time together).

    Every returned state is valid (simple path origin -> destination over
    existing edges, departure time on the grid inside [t_min, t_max]) and
    differs from `state`.
    """
    neighbors: List[State] = []
    seen = {state}

    def add(path: Tuple[str, ...], t: float) -> None:
        cand = State(path=path, departure_time=t)
        if cand not in seen and problem.is_valid_state(cand):
            seen.add(cand)
            neighbors.append(cand)

    # 1) time moves
    for t in _time_neighbors(state, problem):
        add(state.path, t)

    # 2) path moves
    new_paths = _path_neighbors(state, problem)
    for p in new_paths:
        add(p, state.departure_time)

    # 3) mixed moves (a few random samples to keep the neighborhood small)
    one_step = _time_neighbors(state, problem, max_steps=1)
    if new_paths and one_step:
        for p in random.sample(new_paths, min(3, len(new_paths))):
            add(p, random.choice(one_step))

    return neighbors


def _snap_time(t: float, problem: CommuteProblem) -> float:
    """Rounds t to the nearest point of the grid t_min + k*t_step and clamps it into [t_min, t_max]."""
    k = round((t - problem.t_min) / problem.t_step)
    snapped = problem.t_min + k * problem.t_step
    if snapped > problem.t_max + 1e-9:
        snapped -= problem.t_step
    snapped = max(problem.t_min, min(snapped, problem.t_max))
    return round(snapped, 2)


def _splice_at_junctions(
    p1: Tuple[str, ...], p2: Tuple[str, ...], problem: CommuteProblem
) -> List[Tuple[str, ...]]:
    """
    One-point topological crossover. For every intermediate node c that appears
    in both paths: child = p1[:index of c in p1] + p2[index of c in p2:].
    Only children that pass is_valid_path (simple, connected) are kept: the
    prefix of p1 and the suffix of p2 may share nodes, which would create a loop.
    """
    common = (set(p1) & set(p2)) - {problem.origin, problem.destination}
    children = []
    for c in common:
        child = p1[:p1.index(c)] + p2[p2.index(c):]
        if problem.is_valid_path(child):
            children.append(child)
    return children


def crossover(parent1: State, parent2: State, problem: CommuteProblem) -> State:
    """
    Recombines two parents into one valid offspring.

    Route:  choose a random shared intermediate junction c, and join the first
            part of one parent's path (origin -> c) with the second part of the
            other's (c -> destination). Parent order is randomised, so both
            directions of splicing are possible.
    Time:   arithmetic mean of both departure times, snapped to the time grid.
    Fallback: if the parents share no usable junction (e.g. two completely
            disjoint routes), the child takes the route of a random parent and
            receives one random neighbor move (mutation), so the output is
            always valid and never a broken path.
    """
    t_child = _snap_time((parent1.departure_time + parent2.departure_time) / 2.0, problem)

    a, b = (parent1, parent2) if random.random() < 0.5 else (parent2, parent1)
    candidates = _splice_at_junctions(a.path, b.path, problem)
    if not candidates:
        candidates = _splice_at_junctions(b.path, a.path, problem)

    if candidates:
        child = State(path=random.choice(candidates), departure_time=t_child)
        if problem.is_valid_state(child):
            return child

    # Fallback: disjoint parents -> mutated copy of one parent
    base = random.choice((parent1, parent2))
    base = State(path=base.path, departure_time=t_child)
    if not problem.is_valid_state(base):
        base = random.choice((parent1, parent2))
    mutants = get_neighbors(base, problem)
    return random.choice(mutants) if mutants else base