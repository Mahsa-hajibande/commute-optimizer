import random, statistics as st
from main import build_problem, load_configuration
from algorithms.simulated_annealing import SimulatedAnnealingSolver as SA
from algorithms.beam_search import LocalBeamSearchSolver as BS
from algorithms.genetic_algorithm import GeneticAlgorithmSolver as GA

cfg = load_configuration("config.json")
CASES = [("toy_graph", "rush_hour", 41.77), ("metropolis_graph", "dual_peak_rush_hour", 31.69)]
def prob(m, s): return build_problem(f"data/maps/{m}.json", f"data/scenarios/{s}.json", cfg["problem"])
def run(cls, pr, params, n):
    out = []
    for seed in range(n):
        random.seed(seed); out.append(cls(pr, **params).solve()[1])
    return out
def show(label, res, opt):
    hits = sum(abs(c - opt) < 0.01 for c in res)
    print(f"  {label:34s} mean={st.mean(res):6.2f}  worst={max(res):6.2f}  optimum reached {hits}/{len(res)}")

print("=== Simulated Annealing: T0 and cooling rate (10 seeds) ===")
for m, s, opt in CASES:
    pr = prob(m, s); print(m, "optimum", opt)
    for T0, a in [(120, 0.992), (120, 0.5), (120, 0.999), (0.01, 0.992), (5000, 0.992)]:
        p = {**cfg["simulated_annealing"], "initial_temp": T0, "cooling_rate": a}
        show(f"T0={T0}, alpha={a}", run(SA, pr, p, 10), opt)

print("\n=== Local Beam Search: beam width (10 seeds) ===")
for m, s, opt in CASES:
    pr = prob(m, s); print(m, "optimum", opt)
    for K in (1, 2, cfg["beam_search"].get("beam_width", 8)):
        show(f"K={K}", run(BS, pr, {**cfg["beam_search"], "beam_width": K}, 10), opt)

print("\n=== Genetic Algorithm ablation (10 seeds) ===")
for m, s, opt in CASES:
    pr = prob(m, s); print(m, "optimum", opt)
    for label, extra in [("default", {}), ("no crossover", {"crossover_rate": 0.0}),
                         ("no mutation", {"mutation_rate": 0.0}),
                         ("no crossover + no mutation", {"crossover_rate": 0.0, "mutation_rate": 0.0})]:
        show(label, run(GA, pr, {**cfg["genetic_algorithm"], **extra}, 10), opt)