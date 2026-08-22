"""Beautified rendering of Appendix A1: Position-wise accuracy by evidence slot.

Reads outputs/metrics/position_metrics.csv (single source of truth for the
numbers; this script never types them by hand). Only the visual style differs
from src/plot_paper_figures.py:fig_a1_position_bars; the data array is
constructed from the same CSV.
"""
import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import rcParams
import numpy as np

BASE = os.environ.get("RAGPOS_BASE", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CSV = os.path.join(BASE, "outputs/metrics/position_metrics.csv")
FIG = os.path.join(BASE, "figures")

DISPLAY = [
    ("gpt-5.4-medium",            "GPT-5.4-medium",     "#4C78A8"),
    ("deepseek-chat",             "DeepSeek-Chat",      "#72B7B2"),
    ("gpt-5.4-mini",              "GPT-5.4-mini",       "#54A24B"),
    ("claude-haiku-4-5-20251001", "Claude-Haiku-4.5",   "#B279A2"),
    ("claude-sonnet-4-6",         "Claude-Sonnet-4.6",  "#9D755D"),
    ("deepseek-reasoner",         "DeepSeek-Reasoner",  "#F2A65A"),
    ("gemini-2.5-flash",          "Gemini-2.5-Flash",   "#E45756"),
]
POSITIONS = ["E1", "E2", "E3", "E5", "E6"]


def pick_serif():
    for name in ("Times New Roman", "Liberation Serif", "DejaVu Serif"):
        try:
            fm.findfont(fm.FontProperties(family=name), fallback_to_default=False)
            return name
        except Exception:
            continue
    return "DejaVu Serif"


def load():
    by_model = defaultdict(dict)
    with open(CSV) as f:
        for r in csv.DictReader(f):
            by_model[r["model"]][r["position"]] = float(r["accuracy"])
    return by_model


def main():
    data = load()

    # Build the (n_models x n_positions) array strictly from the CSV map.
    raw = np.array([[data[mid][p] for p in POSITIONS] for mid, _, _ in DISPLAY])

    # Cross-check: this matches the legacy plot's data structure.
    assert raw.shape == (7, 5), raw.shape

    serif = pick_serif()
    rcParams.update({
        "font.family": "serif",
        "font.serif": [serif],
        "axes.titlesize": 13,
        "axes.labelsize": 11.5,
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5,
        "legend.fontsize": 9.5,
        "axes.linewidth": 0.6,
        "axes.edgecolor": "#444444",
    })

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    n_models = len(DISPLAY)
    n_pos = len(POSITIONS)
    group_w = 0.78
    bar_w = group_w / n_models
    x = np.arange(n_pos)

    for i, (_, label, color) in enumerate(DISPLAY):
        offsets = x - group_w / 2 + (i + 0.5) * bar_w
        ax.bar(
            offsets, raw[i], bar_w,
            color=color,
            edgecolor="white",
            linewidth=0.4,
            label=label,
        )

    # axes / spines
    ax.set_xlabel("Slot of the Correct Evidence", labelpad=6)
    ax.set_ylabel("Accuracy", labelpad=6)
    ax.set_xticks(x)
    ax.set_xticklabels(POSITIONS)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylim(0.0, 1.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["bottom"].set_linewidth(0.6)
    ax.tick_params(axis="both", which="both", length=3, width=0.5, colors="#333333")

    ax.yaxis.grid(True, color="#B0B0B0", linewidth=0.5, alpha=0.25)
    ax.set_axisbelow(True)

    ax.set_title("Appendix A1: Position-wise Accuracy by Evidence Slot",
                 pad=10, color="#222222")

    leg = ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.06),
        ncol=4,
        frameon=False,
        handlelength=1.2,
        handletextpad=0.5,
        columnspacing=1.4,
    )
    for text in leg.get_texts():
        text.set_color("#222222")

    plt.tight_layout()
    out_pdf = os.path.join(FIG, "appendix_fig_a1_position_bars_beautified.pdf")
    out_png = os.path.join(FIG, "appendix_fig_a1_position_bars_beautified.png")
    plt.savefig(out_pdf, bbox_inches="tight", pad_inches=0.04)
    plt.savefig(out_png, bbox_inches="tight", pad_inches=0.04, dpi=300)
    plt.close()

    # Echo the data array back to stdout so the human can verify zero numerical drift.
    print("[ok] wrote", out_pdf)
    print("[ok] wrote", out_png)
    print("\nData array used (rows = models, cols = positions):")
    print("position:", POSITIONS)
    for (_, label, _), row in zip(DISPLAY, raw):
        print(f"  {label:<22s}", " ".join(f"{v:.4f}" for v in row))


if __name__ == "__main__":
    main()
