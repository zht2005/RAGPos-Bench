"""Beautified rendering of Figure 3: Position sensitivity slope chart.

Reads outputs/metrics/by_variant_metrics.csv (correct_front / correct_middle /
correct_end). The data array is constructed from the CSV; this script never
types numbers by hand. Only the visual style differs from
src/plot_paper_figures.py:fig3_position_slope.
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
CSV = os.path.join(BASE, "outputs/metrics/by_variant_metrics.csv")
FIG = os.path.join(BASE, "figures")

DISPLAY = [
    ("gpt-5.4-medium",            "GPT-5.4-medium",     "#4C78A8", False),
    ("deepseek-chat",             "DeepSeek-Chat",      "#72B7B2", True),
    ("gpt-5.4-mini",              "GPT-5.4-mini",       "#54A24B", True),
    ("claude-haiku-4-5-20251001", "Claude-Haiku-4.5",   "#B279A2", False),
    ("claude-sonnet-4-6",         "Claude-Sonnet-4.6",  "#9D755D", False),
    ("deepseek-reasoner",         "DeepSeek-Reasoner",  "#F2A65A", False),
    ("gemini-2.5-flash",          "Gemini-2.5-Flash",   "#E45756", True),
]
VARIANTS = ["correct_front", "correct_middle", "correct_end"]
VLABELS = ["Front (E1)", "Middle (E3)", "End (E6)"]


def pick_serif():
    for name in ("Times New Roman", "Liberation Serif", "DejaVu Serif"):
        try:
            fm.findfont(fm.FontProperties(family=name), fallback_to_default=False)
            return name
        except Exception:
            continue
    return "DejaVu Serif"


def load():
    data = defaultdict(dict)
    with open(CSV) as f:
        for r in csv.DictReader(f):
            if r["variant"] in VARIANTS:
                data[r["model"]][r["variant"]] = float(r["accuracy"])
    return data


def main():
    data = load()
    raw = np.array([[data[mid][v] for v in VARIANTS] for mid, _, _, _ in DISPLAY])
    assert raw.shape == (7, 3), raw.shape

    serif = pick_serif()
    rcParams.update({
        "font.family": "serif",
        "font.serif": [serif],
        "axes.titlesize": 12.5,
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

    x = np.arange(3)
    for i, (_, label, color, sig) in enumerate(DISPLAY):
        y = raw[i]
        ax.plot(
            x, y,
            color=color,
            lw=2.4 if sig else 1.4,
            alpha=1.0 if sig else 0.65,
            marker="o",
            markersize=6.5 if sig else 5.0,
            markeredgecolor="white",
            markeredgewidth=0.6,
            label=(label + r" $\star$") if sig else label,
            zorder=5 if sig else 3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(VLABELS)
    ax.set_xlabel("Released Layout (Gold-Evidence Slot)", labelpad=6)
    ax.set_ylabel("Accuracy", labelpad=6)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8])
    ax.set_ylim(0.0, 0.85)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["bottom"].set_linewidth(0.6)
    ax.tick_params(axis="both", which="both", length=3, width=0.5, colors="#333333")
    ax.yaxis.grid(True, color="#B0B0B0", linewidth=0.5, alpha=0.25)
    ax.set_axisbelow(True)

    ax.set_title(
        "Accuracy across V1-V3 layouts "
        r"($\star$ = significant V1-to-V3 decline, $p<0.01$)",
        pad=10, color="#222222",
    )

    leg = ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.10),
        ncol=4,
        frameon=False,
        handlelength=1.6,
        handletextpad=0.5,
        columnspacing=1.2,
    )
    for t in leg.get_texts():
        t.set_color("#222222")

    plt.tight_layout()
    out_pdf = os.path.join(FIG, "fig3_position_slope.pdf")
    out_png = os.path.join(FIG, "fig3_position_slope.png")
    plt.savefig(out_pdf, bbox_inches="tight", pad_inches=0.04)
    plt.savefig(out_png, bbox_inches="tight", pad_inches=0.04, dpi=300)
    plt.close()

    print("[ok] wrote", out_pdf)
    print("[ok] wrote", out_png)
    print("\nData array used (rows = models, cols = variants):")
    print("variants:", VARIANTS)
    for (_, label, _, _), row in zip(DISPLAY, raw):
        print(f"  {label:<22s}", " ".join(f"{v:.4f}" for v in row))


if __name__ == "__main__":
    main()
