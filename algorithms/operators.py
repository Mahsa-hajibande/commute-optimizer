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


def crossover(parent1: State, parent2: State, problem: CommuteProblem) -> State:
    """
    Recombines structural features from two parent configurations into an offspring.

    Guidelines:
    - Exchange routing sub-paths using shared intermediate waypoints.
    - Blend or select departure times from the parents in a way that respects valid step intervals.
    - Verify that the resulting child path remains simple and loop-free.

    Encouraged:
    - Finding valid topological crossover points where parent routes intersect.
    - Having a fallback mechanism (e.g., returning a mutated parent) if parents share no junctions.

    Prohibited:
    - Returning paths with invalid edges or broken topologies.

    Args:
        parent1: First parent State.
        parent2: Second parent State.
        problem: Problem instance providing graph topology and constraints.

    Returns:
        State: A newly generated child State.
    """
    # TODO: Implement genetic crossover
    raise NotImplementedError("Implement crossover in algorithms/operators.py")