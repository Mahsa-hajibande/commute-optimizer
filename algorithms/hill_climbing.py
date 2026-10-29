"""
algorithms/hill_climbing.py
Hill Climbing search with sideways move limits and random restarts.
"""

import random
from typing import Tuple
from algorithms.base import LocalSearchSolver
from core.state import State
from core.problem import CommuteProblem
from algorithms.operators import get_neighbors, random_state


class HillClimbingSolver(LocalSearchSolver):
    def __init__(self, problem: CommuteProblem, **kwargs):
        super().__init__(problem, **kwargs)
        self.max_sideways: int = self.params.get("max_sideways", 100)
        self.max_restarts: int = self.params.get("max_restarts", 5)

    def solve(self) -> Tuple[State, float]:
        """
        Steepest-descent Hill Climbing with sideways moves and random restarts.

        - Each step evaluates the whole neighborhood and moves to the best neighbor.
        - If the best neighbor is strictly better -> move, sideways counter resets.
        - If it ties with the current cost (plateau) -> sideways move to an unvisited
          tied neighbor, at most `max_sideways` times in a row. A visited-set stops
          cycling on the plateau.
        - Otherwise (local optimum or plateau budget used up) the run ends and a
          random restart begins, up to `max_restarts` times.
        - self.history gets the best cost found so far (over all restarts) at every step.
        """
        eps = 1e-9
        self.history = []
        best_state, best_cost = None, float("inf")

        for restart in range(self.max_restarts + 1):
            current = self.problem.generate_initial_state() if restart == 0 else random_state(self.problem)
            cur_cost = self.problem.evaluate(current)
            visited = {current}
            sideways = 0

            if cur_cost < best_cost:
                best_state, best_cost = current, cur_cost
            self.history.append(best_cost)

            while True:
                neighbors = get_neighbors(current, self.problem)
                if not neighbors:
                    break
                scored = [(self.problem.evaluate(n), n) for n in neighbors]
                best_nb_cost = min(c for c, _ in scored)

                if best_nb_cost < cur_cost - eps:
                    # strict improvement: steepest descent step (random tie-break)
                    ties = [n for c, n in scored if c <= best_nb_cost + eps]
                    current = random.choice(ties)
                    cur_cost = best_nb_cost
                    visited.add(current)
                    sideways = 0
                elif abs(best_nb_cost - cur_cost) <= eps and sideways < self.max_sideways:
                    # plateau: sideways move to an unvisited neighbor of equal cost
                    flat = [n for c, n in scored if abs(c - cur_cost) <= eps and n not in visited]
                    if not flat:
                        break
                    current = random.choice(flat)
                    visited.add(current)
                    sideways += 1
                else:
                    break  # local optimum

                if cur_cost < best_cost:
                    best_state, best_cost = current, cur_cost
                self.history.append(best_cost)

        return best_state, best_cost