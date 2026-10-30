"""
plot_zoom.py  (helper script, run from the project root)
Convergence plot in two panels: full range, and a zoom on the first steps
(the full range is dominated by Simulated Annealing's long run).
Usage: python plot_zoom.py [map] [scenario] [seed] [zoom_steps]
Saves convergence_comparison.png
"""
import sys, random
import matplotlib.pyplot as plt
from main import build_problem, load_configuration, SOLVER_REGISTRY

map_path = sys.argv[1] if len(sys.argv) > 1 else "data/maps/toy_graph.json"
scenario = sys.argv[2] if len(sys.argv) > 2 else "data/scenarios/rush_hour.json"
seed = int(sys.argv[3]) if len(sys.argv) > 3 else 1
zoom = int(sys.argv[4]) if len(sys.argv) > 4 else 30

config = load_configuration("config.json")
problem = build_problem(map_path, scenario, config["problem"])
palette = {"Hill Climbing": "#E67E22", "Simulated Annealing": "#9B59B6",
           "Local Beam Search": "#2980B9", "Genetic Algorithm": "#27AE60"}

histories = {}
random.seed(seed)
for key, (name, cls, cfg_key) in SOLVER_REGISTRY.items():
    solver = cls(problem, **config.get(cfg_key, {}))
    solver.solve()
    histories[name] = solver.history

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, xmax, title in ((axes[0], None, "Full run"), (axes[1], zoom, f"Zoom: first {zoom} steps")):
    for name, h in histories.items():
        xs = list(range(len(h)))
        if xmax is not None:
            xs, h = xs[:xmax + 1], h[:xmax + 1]
        ax.plot(xs, h, label=name, color=palette.get(name), linewidth=2.2, alpha=0.9,
                marker="o" if xmax is not None else None, markersize=3)
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Iteration / generation step (step 0 = initial state)")
    ax.set_ylabel("Best commute duration f(x) so far (minutes)")
    ax.grid(True, linestyle="--", alpha=0.6)
axes[0].legend(frameon=True, loc="upper right")
fig.suptitle(f"Search convergence ({map_path.split('/')[-1]}, {scenario.split('/')[-1]}, seed {seed})",
             fontweight="bold")
plt.tight_layout()
plt.savefig("convergence_comparison.png", dpi=200, bbox_inches="tight")
print("saved convergence_comparison.png")
for n, h in histories.items():
    print(f"{n:22s} start={h[0]:.2f} final={h[-1]:.2f} steps={len(h)-1}")