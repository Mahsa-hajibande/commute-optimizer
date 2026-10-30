import sys
from main import build_problem, load_configuration
from core.state import State

map_path = sys.argv[1] if len(sys.argv) > 1 else "data/maps/toy_graph.json"
scenario = sys.argv[2] if len(sys.argv) > 2 else "data/scenarios/rush_hour.json"
cfg = load_configuration("config.json")
pr = build_problem(map_path, scenario, cfg["problem"])

paths = []
def dfs(u, cur):
    if u == pr.destination:
        paths.append(tuple(cur)); return
    for v in pr.network.get_neighbors(u):
        if v not in cur:
            dfs(v, cur + [v])
dfs(pr.origin, [pr.origin])

times = pr.get_valid_time_domain()
best = min((pr.evaluate(State(p, t)), p, t) for p in paths for t in times)
print(f"{map_path} | {scenario}")
print(f"simple paths: {len(paths)} | time points: {len(times)} | |X| = {len(paths) * len(times)}")
print(f"global optimum: {best[0]:.2f} min, departure {best[2]}, path {' -> '.join(best[1])}")