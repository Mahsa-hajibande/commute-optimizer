"""
algorithms/beam_search.py
Local Beam Search maintaining K parallel candidate states.
"""

from typing import Dict, List, Tuple
from algorithms.base import LocalSearchSolver
from core.state import State
from core.problem import CommuteProblem
from algorithms.operators import get_neighbors, random_state


class LocalBeamSearchSolver(LocalSearchSolver):
    def __init__(self, problem: CommuteProblem, **kwargs):
        super().__init__(problem, **kwargs)
        self.beam_width: int = self.params.get("beam_width", 5)
        self.max_iterations: int = self.params.get("max_iterations", 150)

    def solve(self) -> Tuple[State, float]:
        """
        Executes Local Beam Search.

        - The beam starts with the free-flow shortest path plus random valid states.
        - Every iteration, the neighbors of ALL beam states are pooled together with the
          current beam (so the best state is never lost) and ranked by cost. The K best
          are kept: the K searches share information, they are not independent.
        - Diversity: at most `max_per_path` states with the same route may enter the beam
          (first pass, in cost order). Remaining slots are filled with the best leftovers.
          This prevents all K beams from collapsing into time-shifted copies of one route.
        - Stops after `patience` iterations without improving the best cost, or at
          max_iterations. self.history stores the best cost of the beam at every step.
        """
        patience: int = self.params.get("patience", 10)
        max_per_path: int = self.params.get("max_per_path", 2)
        K = self.beam_width
        self.history = []

        beam: List[State] = [self.problem.generate_initial_state()]
        seen = set(beam)
        attempts = 0
        while len(beam) < K and attempts < 20 * K:
            attempts += 1
            cand = random_state(self.problem)
            if cand not in seen:
                seen.add(cand)
                beam.append(cand)

        cost: Dict[State, float] = {s: self.problem.evaluate(s) for s in beam}
        best_state = min(beam, key=lambda s: cost[s])
        best_cost = cost[best_state]
        self.history.append(best_cost)
        stale = 0

        for _ in range(self.max_iterations):
            pool = set(beam)
            for st in beam:
                pool.update(get_neighbors(st, self.problem))
            for st in pool:
                if st not in cost:
                    cost[st] = self.problem.evaluate(st)

            ranked = sorted(pool, key=lambda s: (cost[s], s.path, s.departure_time))
            # pass 1: best states with a cap on how many share the same route
            new_beam: List[State] = []
            per_path: Dict[Tuple[str, ...], int] = {}
            for st in ranked:
                if len(new_beam) == K:
                    break
                if per_path.get(st.path, 0) < max_per_path:
                    new_beam.append(st)
                    per_path[st.path] = per_path.get(st.path, 0) + 1
            # pass 2: fill empty slots with the best leftovers
            if len(new_beam) < K:
                chosen = set(new_beam)
                for st in ranked:
                    if len(new_beam) == K:
                        break
                    if st not in chosen:
                        new_beam.append(st)
            beam = new_beam

            if cost[ranked[0]] < best_cost - 1e-9:
                best_state, best_cost = ranked[0], cost[ranked[0]]
                stale = 0
            else:
                stale += 1
            self.history.append(best_cost)
            if stale >= patience:
                break

        return best_state, best_cost