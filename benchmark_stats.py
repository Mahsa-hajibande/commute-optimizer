"""
benchmark_stats.py  (helper script, run from the project root)
Runs every algorithm N times with different seeds on one map/scenario and prints
a markdown table: best cost, success rate, departure time, runtime, steps to convergence.
Usage: python benchmark_stats.py [map] [scenario] [N]
"""
import sys, time, random, statistics as st
from collections import Counter
from main import build_problem, load_configuration, SOLVER_REGISTRY

map_path = sys.argv[1] if len(sys.argv) > 1 else "data/maps/toy_graph.json"
scenario = sys.argv[2] if len(sys.argv) > 2 else "data/scenarios/rush_hour.json"
N = int(sys.argv[3]) if len(sys.argv) > 3 else 20

config = load_configuration("config.json")
problem = build_problem(map_path, scenario, config["problem"])

def steps_to_converge(history):
    """Number of steps until the final best value is first reached."""
    best = min(history)
    return next(i for i, v in enumerate(history) if v <= best + 1e-9)

overall_best = float("inf")
results = {}
variants = [(key, name, cls, config.get(cfg_key, {}), name) for key, (name, cls, cfg_key) in SOLVER_REGISTRY.items()]
# extra row: plain hill climbing without random restarts (shows local optima)
hc = SOLVER_REGISTRY["hc"]
variants.insert(1, ("hc0", "Hill Climbing (no restarts)", hc[1], {**config.get(hc[2], {}), "max_restarts": 0}, "Hill Climbing (no restarts)"))

for key, name, cls, params, label in variants:
    costs, times, steps, deps = [], [], [], []
    for seed in range(N):
        random.seed(seed)
        solver = cls(problem, **params)
        t0 = time.perf_counter()
        state, cost = solver.solve()
        times.append(time.perf_counter() - t0)
        costs.append(cost); steps.append(steps_to_converge(solver.history)); deps.append(state.format_time())
    results[label] = (costs, times, steps, deps)
    overall_best = min(overall_best, min(costs))

print(f"\nMap: {map_path} | Scenario: {scenario} | {N} runs (seeds 0..{N-1})")
print(f"Best cost found by any run: {overall_best:.2f} min\n")
print("| Algorithm | Best f(x*) | Mean f | Worst f | Hits best | Departure (most common) | Mean time (s) | Mean steps to converge |")
print("|---|---|---|---|---|---|---|---|")
for label, (costs, times, steps, deps) in results.items():
    hits = sum(abs(c - overall_best) < 1e-6 for c in costs)
    dep = Counter(d for d, c in zip(deps, costs) if abs(c - min(costs)) < 1e-6).most_common(1)[0][0]
    print(f"| {label} | {min(costs):.2f} | {st.mean(costs):.2f} | {max(costs):.2f} | {hits}/{N} | {dep} | {st.mean(times):.3f} | {st.mean(steps):.1f} |")