"""
algorithms/genetic_algorithm.py
Population-based optimization using selection, recombination, and mutation.
"""

import random
from typing import Dict, List, Tuple
from algorithms.base import LocalSearchSolver
from core.state import State
from core.problem import CommuteProblem
from algorithms.operators import get_neighbors, crossover, random_state


class GeneticAlgorithmSolver(LocalSearchSolver):
    def __init__(self, problem: CommuteProblem, **kwargs):
        super().__init__(problem, **kwargs)
        self.population_size: int = self.params.get("population_size", 30)
        self.generations: int = self.params.get("generations", 100)
        self.crossover_rate: float = self.params.get("crossover_rate", 0.8)
        self.mutation_rate: float = self.params.get("mutation_rate", 0.2)
        self.elitism: int = self.params.get("elitism", 2)

    @staticmethod
    def fitness(cost: float) -> float:
        """Fitness to MAXIMISE: lower travel time -> higher fitness (always in (0, 1])."""
        return 1.0 / (1.0 + cost)

    def _tournament(self, population: List[State], cost: Dict[State, float], k: int = 3) -> State:
        """Tournament selection: sample k individuals, return the one with the highest fitness."""
        contenders = random.sample(population, min(k, len(population)))
        return max(contenders, key=lambda s: self.fitness(cost[s]))

    def solve(self) -> Tuple[State, float]:
        """
        Executes Genetic Algorithm optimization.

        - Initial population: the free-flow shortest path + random valid states
          (random simple routes with random departure times).
        - Fitness: 1 / (1 + travel time).
        - Selection: tournament (size `tournament_size`, default 3).
        - Recombination: with probability crossover_rate the child is crossover(p1, p2);
          otherwise it is a copy of the first parent.
        - Mutation: with probability mutation_rate the child is replaced by a random
          neighbor of itself (departure-time shift or loop-free detour), so mutation
          always yields a valid state.
        - Elitism: the best `elitism` individuals are copied unchanged to the next
          generation, so the best cost never gets worse.
        - self.history stores the best cost of the population at every generation
          (index 0 = initial population).
        """
        tournament_size: int = self.params.get("tournament_size", 3)
        self.history = []
        cost: Dict[State, float] = {}

        def f(st: State) -> float:
            if st not in cost:
                cost[st] = self.problem.evaluate(st)
            return cost[st]

        # --- initial population
        population: List[State] = [self.problem.generate_initial_state()]
        seen = set(population)
        attempts = 0
        while len(population) < self.population_size and attempts < 20 * self.population_size:
            attempts += 1
            cand = random_state(self.problem)
            if cand not in seen:
                seen.add(cand)
                population.append(cand)
        while len(population) < self.population_size:  # tiny search space: allow duplicates
            population.append(random_state(self.problem))

        population.sort(key=f)
        best_state, best_cost = population[0], f(population[0])
        self.history.append(best_cost)

        for _ in range(self.generations):
            n_elite = min(self.elitism, len(population))
            next_gen: List[State] = list(population[:n_elite])  # elitism

            while len(next_gen) < self.population_size:
                p1 = self._tournament(population, cost, tournament_size)
                p2 = self._tournament(population, cost, tournament_size)

                if random.random() < self.crossover_rate:
                    child = crossover(p1, p2, self.problem)
                else:
                    child = p1

                if random.random() < self.mutation_rate:
                    mutants = get_neighbors(child, self.problem)
                    if mutants:
                        child = random.choice(mutants)

                next_gen.append(child)

            population = sorted(next_gen, key=f)
            if f(population[0]) < best_cost:
                best_state, best_cost = population[0], f(population[0])
            self.history.append(best_cost)

        return best_state, best_cost