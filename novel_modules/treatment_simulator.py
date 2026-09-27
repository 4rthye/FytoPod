import numpy as np
import matplotlib.pyplot as plt

def simulate_treatment_impact(health_score: float, severity_score: float,
                               stress_predicted: bool = False,
                               days: int = 14) -> dict:
    """
    Simulates plant health trajectory over 14 days
    under treated and untreated conditions.
    """
    t = np.arange(0, days + 1)
    severity = severity_score
    stress_penalty = 0.01 if stress_predicted else 0.0

    # Untreated: exponential decay
    untreated = health_score * np.exp(-0.08 * severity * t) - stress_penalty * t
    untreated = np.clip(untreated, 0, 100)

    # Treated: slight dip then recovery
    treated = np.where(
        t <= 3,
        health_score - (severity * 5 * t / 3),
        (health_score - severity * 5) + (0.04 * health_score * (t - 3))
    )
    treated = np.clip(treated, 0, 100)

    return {
        "days": t.tolist(),
        "untreated": untreated.tolist(),
        "treated": treated.tolist(),
        "final_untreated": round(float(untreated[-1]), 1),
        "final_treated":   round(float(treated[-1]), 1),
    }


def plot_treatment_impact(result: dict, save_path: str = None):
    t         = result["days"]
    untreated = result["untreated"]
    treated   = result["treated"]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(t, treated, color="#27ae60", linewidth=2.5,
            marker="o", markersize=4, label="With Treatment")
    ax.plot(t, untreated, color="#e74c3c", linewidth=2.5,
            marker="o", markersize=4, label="Without Treatment", linestyle="--")

    ax.fill_between(t, untreated, treated, alpha=0.1, color="#27ae60")
    ax.axhline(y=30, color="#f39c12", linestyle=":", linewidth=1.5,
               label="Critical Threshold (30%)")

    ax.annotate(f"Day 14: {result['final_treated']:.0f}%",
                xy=(14, result["final_treated"]),
                xytext=(11, result["final_treated"] + 5),
                fontsize=9, color="#27ae60",
                arrowprops=dict(arrowstyle="->", color="#27ae60"))

    ax.annotate(f"Day 14: {result['final_untreated']:.0f}%",
                xy=(14, result["final_untreated"]),
                xytext=(11, result["final_untreated"] - 8),
                fontsize=9, color="#e74c3c",
                arrowprops=dict(arrowstyle="->", color="#e74c3c"))

    ax.set_xlabel("Days", fontsize=11)
    ax.set_ylabel("Plant Health Score (%)", fontsize=11)
    ax.set_title("FytoPod — Treated vs Untreated Health Trajectory", fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 105)
    ax.set_xlim(0, 14)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()