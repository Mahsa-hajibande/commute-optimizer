"""
algorithms/simulated_annealing.py
Stochastic optimization governed by a temperature schedule.
"""

import math
import random
from typing import Tuple
from algorithms.base import LocalSearchSolver
from core.state import State
from core.problem import CommuteProblem
from algorithms.operators import get_neighbors


class SimulatedAnnealingSolver(LocalSearchSolver):
    def __init__(self, problem: CommuteProblem, **kwargs):
        super().__init__(problem, **kwargs)
        self.initial_temp: float = self.params.get("initial_temp", 100.0)
        self.cooling_rate: float = self.params.get("cooling_rate", 0.995)
        self.min_temp: float = self.params.get("min_temp", 1e-3)
        self.max_iterations: int = self.params.get("max_iterations", 2000)

    def solve(self) -> Tuple[State, float]:
        """
        Executes Simulated Annealing optimization.

        - Start from the free-flow shortest path at t_min.
        - Each iteration: pick ONE random neighbor, compute dE = f(neighbor) - f(current).
        - dE <= 0 (improvement) -> always accept.
        - dE  > 0 (worse)       -> accept with Metropolis probability exp(-dE / T).
        - Geometric cooling: T <- cooling_rate * T. Stops when T < min_temp
          or after max_iterations.
        - self.history holds the best cost found so far at every iteration, and the
          best state ever visited is returned (not just the final current state).
        """
        self.history = []
        current = self.problem.generate_initial_state()
        cur_cost = self.problem.evaluate(current)
        best_state, best_cost = current, cur_cost
        self.history.append(best_cost)

        T = self.initial_temp
        for _ in range(self.max_iterations):
            if T < self.min_temp:
                break

            neighbors = get_neighbors(current, self.problem)
            if not neighbors:
                break
            candidate = random.choice(neighbors)
            cand_cost = self.problem.evaluate(candidate)
            delta = cand_cost - cur_cost

            if delta <= 0 or random.random() < math.exp(-delta / T):
                current, cur_cost = candidate, cand_cost
                if cur_cost < best_cost:
                    best_state, best_cost = current, cur_cost

            self.history.append(best_cost)
            T *= self.cooling_rate

        return best_state, best_cost