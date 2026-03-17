"""
Experiment 07 — Visualization Suite
====================================
PURPOSE:
    Generate publication-quality plots from all experiment results.
    Run this AFTER experiments 01-06 have been completed.

OUTPUT:
    results/figures/
        fig1_layer_accuracy.png     — Probing accuracy by layer (Exp 02)
        fig2_logit_lens.png         — Truth trajectory / flip layer (Exp 06a)
        fig3_cross_model.png        — Cross-model transfer heatmap (Exp 04)
        fig4_deception_types.png    — Within-type vs cross-type (Exp 05)
        fig5_cosine_similarity.png  — Lie direction similarity matrix (Exp 05)
        fig6_controls.png           — All baselines and controls summary

USAGE:
    %run experiments/07_visualizations/generate_plots.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import json
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ── Configuration ──────────────────────────────────────────────────────────
RESULTS_DIR = "results"
FIGURES_DIR = "results/figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

# Color palette (colorblind-friendly)
COLORS = {
    "primary": "#2563EB",       # blue
    "secondary": "#DC2626",     # red
    "accent": "#059669",        # green
    "neutral": "#6B7280",       # gray
    "highlight": "#F59E0B",     # amber
    "bg": "#F8FAFC",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})


def load_json(filename):
    path = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# ── Figure 1: Layer Accuracy (Exp 02) ─────────────────────────────────────

def plot_layer_accuracy():
    data = load_json("exp02_confound_free.json")
    if not data:
        print("Skipping fig1: exp02 results not found")
        return

    results = data["results_per_layer"]
    layers = sorted([int(k) for k in results.keys()])
    accs = [results[str(l)]["balanced_accuracy"] * 100 for l in layers]

    fig, ax = plt.subplots(figsize=(10, 5))

    bars = ax.bar(range(len(layers)), accs, color=[
        COLORS["neutral"] if l == 0 else COLORS["primary"] for l in layers
    ], edgecolor="white", linewidth=0.5)

    # Highlight best layer
    best_idx = np.argmax([a if layers[i] != 0 else 0 for i, a in enumerate(accs)])
    bars[best_idx].set_color(COLORS["secondary"])
    bars[best_idx].set_edgecolor(COLORS["secondary"])

    ax.axhline(y=50, color=COLORS["neutral"], linestyle="--", alpha=0.7, label="Chance (50%)")
    ax.axhline(y=data.get("length_baseline", 0.5) * 100, color=COLORS["highlight"],
               linestyle=":", alpha=0.7, label=f"Length baseline ({data.get('length_baseline', 0.5)*100:.0f}%)")

    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels([f"L{l}" for l in layers])
    ax.set_xlabel("Layer")
    ax.set_ylabel("Balanced Accuracy (%)")
    ax.set_title(f"Confound-Free Deception Detection by Layer\n"
                 f"(n={data['n_balanced']} per class, same prompt)")
    ax.set_ylim(40, 100)
    ax.legend(loc="upper left")

    # Annotate best
    ax.annotate(f"{accs[best_idx]:.1f}%", xy=(best_idx, accs[best_idx]),
                xytext=(0, 8), textcoords="offset points", ha="center",
                fontweight="bold", color=COLORS["secondary"])

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig1_layer_accuracy.png"), dpi=200)
    plt.close(fig)
    print("Saved fig1_layer_accuracy.png")


# ── Figure 2: Logit Lens Trajectory (Exp 06a) ─────────────────────────────

def plot_logit_lens():
    data = load_json("exp06a_logit_lens.json")
    if not data:
        print("Skipping fig2: exp06a results not found")
        return

    n_layers = data["n_layers"]
    layers = list(range(n_layers + 1))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    # Left: When model LIES
    if data["avg_lie_correct_rank"]:
        ax = axes[0]
        ax.plot(layers, data["avg_lie_correct_rank"], color=COLORS["accent"],
                linewidth=2, label="Correct answer", marker="o", markersize=3)
        ax.plot(layers, data["avg_lie_wrong_rank"], color=COLORS["secondary"],
                linewidth=2, label="Wrong answer", marker="s", markersize=3)

        if data["median_flip_layer"] is not None:
            ax.axvline(x=data["median_flip_layer"], color=COLORS["highlight"],
                       linestyle="--", alpha=0.8, label=f"Flip layer ({data['median_flip_layer']:.0f})")

        ax.set_xlabel("Layer")
        ax.set_ylabel("Average Rank (lower = more likely)")
        ax.set_title(f"When Model LIES (n={data['n_lie_trajectories']})")
        ax.legend()
        ax.invert_yaxis()

    # Right: When model RESISTS
    if data["avg_resist_correct_rank"]:
        ax = axes[1]
        ax.plot(layers, data["avg_resist_correct_rank"], color=COLORS["accent"],
                linewidth=2, label="Correct answer", marker="o", markersize=3)
        ax.plot(layers, data["avg_resist_wrong_rank"], color=COLORS["secondary"],
                linewidth=2, label="Wrong answer", marker="s", markersize=3)

        ax.set_xlabel("Layer")
        ax.set_title(f"When Model RESISTS (n={data['n_resist_trajectories']})")
        ax.legend()

    fig.suptitle("Logit Lens: Truth Trajectory Across Layers", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig2_logit_lens.png"), dpi=200)
    plt.close(fig)
    print("Saved fig2_logit_lens.png")


# ── Figure 3: Cross-Model Transfer Heatmap (Exp 04) ───────────────────────

def plot_cross_model():
    data = load_json("exp04_cross_model.json")
    if not data:
        print("Skipping fig3: exp04 results not found")
        return

    results = data["results"]
    models = list(results.get("within_model", {}).keys())
    if len(models) < 2:
        print("Skipping fig3: not enough models")
        return

    n = len(models)
    matrix = np.zeros((n, n))

    for i, src in enumerate(models):
        for j, tgt in enumerate(models):
            if src == tgt:
                matrix[i, j] = results["within_model"][src]["accuracy"] * 100
            else:
                key = f"{src}->{tgt}"
                if key in results.get("cross_model", {}):
                    matrix[i, j] = results["cross_model"][key]["best_accuracy"] * 100

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=100)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([m.capitalize() for m in models])
    ax.set_yticklabels([m.capitalize() for m in models])
    ax.set_xlabel("Test Model")
    ax.set_ylabel("Train Model")
    ax.set_title("Cross-Model Deception Transfer (Best Accuracy %)")

    # Annotate cells
    for i in range(n):
        for j in range(n):
            val = matrix[i, j]
            color = "white" if val < 40 or val > 80 else "black"
            polarity = ""
            if i != j:
                key = f"{models[i]}->{models[j]}"
                if key in results.get("cross_model", {}):
                    if results["cross_model"][key].get("inverted_polarity", False):
                        polarity = "\n(inverted)"
            ax.text(j, i, f"{val:.0f}%{polarity}", ha="center", va="center",
                    color=color, fontweight="bold", fontsize=11)

    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig3_cross_model.png"), dpi=200)
    plt.close(fig)
    print("Saved fig3_cross_model.png")


# ── Figure 4: Deception Types (Exp 05) ────────────────────────────────────

def plot_deception_types():
    data = load_json("exp05_deception_types.json")
    if not data:
        print("Skipping fig4: exp05 results not found")
        return

    results = data["results"]
    within = results.get("within_type", {})
    cross = results.get("cross_type", {})

    if not within:
        print("Skipping fig4: no within-type results")
        return

    types = list(within.keys())
    type_labels = [t.replace("_", " ").title() for t in types]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Within-type accuracy
    ax = axes[0]
    accs = [within[t]["accuracy"] * 100 for t in types]
    colors = [COLORS["primary"], COLORS["secondary"], COLORS["accent"]]
    bars = ax.bar(range(len(types)), accs, color=colors[:len(types)], edgecolor="white")
    ax.axhline(y=50, color=COLORS["neutral"], linestyle="--", alpha=0.7)
    ax.set_xticks(range(len(types)))
    ax.set_xticklabels(type_labels, rotation=15, ha="right")
    ax.set_ylabel("Balanced Accuracy (%)")
    ax.set_title("Within-Type Detection Accuracy")
    ax.set_ylim(40, 100)
    for i, acc in enumerate(accs):
        ax.text(i, acc + 1, f"{acc:.0f}%", ha="center", fontweight="bold")

    # Right: Cross-type transfer matrix
    if cross:
        ax = axes[1]
        n = len(types)
        matrix = np.zeros((n, n))
        for i, src in enumerate(types):
            for j, tgt in enumerate(types):
                if src == tgt:
                    matrix[i, j] = within[src]["accuracy"] * 100
                else:
                    key = f"{src}->{tgt}"
                    if key in cross:
                        matrix[i, j] = cross[key] * 100

        im = ax.imshow(matrix, cmap="RdYlGn", vmin=40, vmax=100)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(type_labels, rotation=15, ha="right")
        ax.set_yticklabels(type_labels)
        ax.set_xlabel("Test Type")
        ax.set_ylabel("Train Type")
        ax.set_title("Cross-Type Transfer Accuracy (%)")

        for i in range(n):
            for j in range(n):
                color = "white" if matrix[i, j] > 75 or matrix[i, j] < 45 else "black"
                ax.text(j, i, f"{matrix[i, j]:.0f}%", ha="center", va="center",
                        color=color, fontweight="bold")

        fig.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle("Deception Types: Internal Representations", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig4_deception_types.png"), dpi=200)
    plt.close(fig)
    print("Saved fig4_deception_types.png")


# ── Figure 5: Cosine Similarity Matrix (Exp 05) ──────────────────────────

def plot_cosine_similarity():
    data = load_json("exp05_deception_types.json")
    if not data:
        print("Skipping fig5: exp05 results not found")
        return

    results = data["results"]
    cosines = results.get("cosine_similarity", {})
    baseline = results.get("random_cosine_baseline", {})

    if not cosines:
        print("Skipping fig5: no cosine similarity data")
        return

    types = list(results.get("within_type", {}).keys())
    type_labels = [t.replace("_", " ").title() for t in types]
    n = len(types)

    matrix = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            key = f"{types[i]}_vs_{types[j]}"
            if key in cosines:
                val = cosines[key]["cosine"] if isinstance(cosines[key], dict) else cosines[key]
                matrix[i, j] = val
                matrix[j, i] = val

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix, cmap="coolwarm", vmin=-0.3, vmax=1.0)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(type_labels, rotation=15, ha="right")
    ax.set_yticklabels(type_labels)
    ax.set_title("Cosine Similarity Between Lie Direction Vectors")

    for i in range(n):
        for j in range(n):
            color = "white" if abs(matrix[i, j]) > 0.5 else "black"
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                    color=color, fontweight="bold")

    fig.colorbar(im, ax=ax, shrink=0.8)

    # Add baseline annotation
    if baseline:
        ax.text(0.02, -0.12, f"Random baseline: {baseline.get('expected_cosine', 0):.3f} "
                f"(std={baseline.get('std', 0):.3f}, dim={baseline.get('dim', '?')})",
                transform=ax.transAxes, fontsize=9, color=COLORS["neutral"])

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig5_cosine_similarity.png"), dpi=200)
    plt.close(fig)
    print("Saved fig5_cosine_similarity.png")


# ── Figure 6: Controls Summary ────────────────────────────────────────────

def plot_controls_summary():
    data = load_json("exp02_confound_free.json")
    if not data:
        print("Skipping fig6: exp02 results not found")
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    controls = {
        "Best Layer\n(Deception Signal)": data["best_accuracy"] * 100,
        "Layer 0\n(Embedding)": data.get("layer_0_accuracy", 0.5) * 100,
        "Length\nBaseline": data.get("length_baseline", 0.5) * 100,
        "Chance\nLevel": 50.0,
    }

    names = list(controls.keys())
    values = list(controls.values())
    colors_list = [COLORS["secondary"], COLORS["neutral"], COLORS["neutral"], COLORS["neutral"]]

    bars = ax.bar(range(len(names)), values, color=colors_list, edgecolor="white", width=0.6)
    ax.axhline(y=50, color=COLORS["neutral"], linestyle="--", alpha=0.5)

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names)
    ax.set_ylabel("Balanced Accuracy (%)")
    ax.set_title(f"Controls Summary — Confound-Free Detection\n"
                 f"(n={data['n_balanced']} per class, p < 0.001)")
    ax.set_ylim(0, 100)

    for i, v in enumerate(values):
        ax.text(i, v + 1.5, f"{v:.1f}%", ha="center", fontweight="bold",
                color=colors_list[i])

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig6_controls.png"), dpi=200)
    plt.close(fig)
    print("Saved fig6_controls.png")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Generating publication-quality figures...")
    print("=" * 60)

    plot_layer_accuracy()
    plot_logit_lens()
    plot_cross_model()
    plot_deception_types()
    plot_cosine_similarity()
    plot_controls_summary()

    print("\nDone! Figures saved to results/figures/")


if __name__ == "__main__":
    main()
