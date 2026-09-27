import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

SPREAD_RATES = {
    "Tomato___Late_blight": 0.35,
    "Tomato___Early_blight": 0.20,
    "Tomato___Bacterial_spot": 0.18,
    "Grape___Black_rot": 0.28,
    "Potato___Late_blight": 0.32,
    "default": 0.15,
}

def run_ca(disease_class: str, grid_size: int = 80, days: int = 14,
           initial_infection: float = 0.05) -> dict:
    """
    Cellular Automata disease spread simulation.
    States: 0=Healthy, 1=Infected, 2=Necrotic
    Returns snapshot percentages at Day 0, 3, 7, 14.
    """
    spread_rate = SPREAD_RATES.get(disease_class, SPREAD_RATES["default"])

    grid = np.zeros((grid_size, grid_size), dtype=int)
    num_infected = int(grid_size * grid_size * initial_infection)
    infected_idx = np.random.choice(grid_size * grid_size, num_infected, replace=False)
    for idx in infected_idx:
        grid[idx // grid_size, idx % grid_size] = 1

    snapshots = {}
    percentages = {}

    for day in range(days + 1):
        if day in [0, 3, 7, 14]:
            snapshots[f"day_{day}"] = grid.copy()
            total = grid_size * grid_size
            percentages[f"day_{day}"] = {
                "infected": float(np.sum(grid == 1) / total * 100),
                "necrotic": float(np.sum(grid == 2) / total * 100),
                "healthy":  float(np.sum(grid == 0) / total * 100),
            }

        if day < days:
            new_grid = grid.copy()
            infected_pos = np.argwhere(grid == 1)

            for pos in infected_pos:
                r, c = pos
                if np.random.random() < 0.10:
                    new_grid[r, c] = 2  # turns necrotic

                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < grid_size and 0 <= nc < grid_size:
                            if grid[nr, nc] == 0 and np.random.random() < spread_rate:
                                new_grid[nr, nc] = 1
            grid = new_grid

    return {"snapshots": snapshots, "percentages": percentages, "spread_rate": spread_rate}


def plot_spread(result: dict, save_path: str = None):
    cmap = ListedColormap(["#2ecc71", "#e74c3c", "#2c2c2c"])
    days = [0, 3, 7, 14]

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    for ax, day in zip(axes, days):
        snap = result["snapshots"][f"day_{day}"]
        ax.imshow(snap, cmap=cmap, vmin=0, vmax=2)
        pct = result["percentages"][f"day_{day}"]
        ax.set_title(
            f"Day {day}\n"
            f"Infected: {pct['infected']:.1f}%\n"
            f"Necrotic: {pct['necrotic']:.1f}%",
            fontsize=9
        )
        ax.axis("off")

    plt.suptitle("FytoPod — Cellular Automata Disease Spread Forecast", fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()