"""Beautified rendering of Figure 5: DeepSeek-Chat vs DeepSeek-Reasoner radar.

Reads outputs/metrics/overall_metrics.csv. Data array is sourced from the CSV
exclusively. Style matches src/beautify_appendix_a1.py and src/beautify_fig3.py:
muted palette, serif font, light grid, frameless legend.
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import rcParams
import numpy as np

BASE = ".."
CSV = os.path.join(BASE, "outputs/metrics/overall_metrics.csv")
FIG = os.path.join(BASE, "figures")

# orange -> blue: pair the chat (teal) with a calm blue for the reasoner.
COLOR_CHAT = "#72B7B2"   # muted teal
COLOR_REASONER = "#4C78A8"  # muted steel blue

METRICS = [
    ("Accuracy", "accuracy", False),
    ("CAR",      "CAR",      False),
    ("CEU",      "CEU",      False),
    ("1 - PSR",  "PSR",      True),
    ("1 - PBR",  "PBR",      True),
    ("1 - EAR",  "EAR",      True),
]


def pick_serif():
    for name in ("Times New Roman", "Liberation Serif", "DejaVu Serif"):
        try:
            fm.findfont(fm.FontProperties(family=name), fallback_to_default=False)
            return name
        except Exception:
            continue
    return "DejaVu Serif"


def load():
    by_m = {}
    with open(CSV) as f:
        for r in csv.DictReader(f):
            by_m[r["model"]] = r
    return by_m


def vals(rec):
    return [(1.0 - float(rec[k]) if inv else float(rec[k]))
            for _, k, inv in METRICS]


def main():
    by_m = load()
    chat = by_m["deepseek-chat"]
    reas = by_m["deepseek-reasoner"]
    v_chat = vals(chat)
    v_reas = vals(reas)

    serif = pick_serif()
    rcParams.update({
        "font.family": "serif",
        "font.serif": [serif],
        "axes.titlesize": 12.5,
        "axes.labelsize": 11.0,
        "xtick.labelsize": 10.0,
        "ytick.labelsize": 9.0,
        "legend.fontsize": 9.5,
    })

    angles = np.linspace(0, 2 * np.pi, len(METRICS), endpoint=False).tolist()
    angles += angles[:1]
    v_chat_c = v_chat + v_chat[:1]
    v_reas_c = v_reas + v_reas[:1]

    fig, ax = plt.subplots(figsize=(5.6, 5.0), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor("white")

    ax.plot(angles, v_chat_c, color=COLOR_CHAT, lw=2.2, label="DeepSeek-Chat",
            marker="o", markersize=4.5, markeredgecolor="white", markeredgewidth=0.5)
    ax.fill(angles, v_chat_c, color=COLOR_CHAT, alpha=0.20)

    ax.plot(angles, v_reas_c, color=COLOR_REASONER, lw=2.2, label="DeepSeek-Reasoner",
            marker="o", markersize=4.5, markeredgecolor="white", markeredgewidth=0.5)
    ax.fill(angles, v_reas_c, color=COLOR_REASONER, alpha=0.18)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m[0] for m in METRICS])
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8"], color="#666666")
    ax.set_ylim(0, 1.0)

    ax.spines["polar"].set_color("#999999")
    ax.spines["polar"].set_linewidth(0.6)
    ax.grid(color="#B0B0B0", linewidth=0.5, alpha=0.35)

    ax.set_title(
        "Reasoning Does Not Necessarily Improve\nRAG Conflict Resolution",
        pad=22, color="#222222",
    )

    leg = ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=2,
        frameon=False,
        handlelength=1.6,
        handletextpad=0.5,
        columnspacing=2.0,
    )
    for t in leg.get_texts():
        t.set_color("#222222")

    plt.tight_layout()
    out_pdf = os.path.join(FIG, "fig5_deepseek_radar.pdf")
    out_png = os.path.join(FIG, "fig5_deepseek_radar.png")
    plt.savefig(out_pdf, bbox_inches="tight", pad_inches=0.04)
    plt.savefig(out_png, bbox_inches="tight", pad_inches=0.04, dpi=300)
    plt.close()

    print("[ok] wrote", out_pdf)
    print("[ok] wrote", out_png)
    print("\nData array used (axes oriented so larger = better):")
    print("axis:", [m[0] for m in METRICS])
    print(f"  DeepSeek-Chat       {' '.join(f'{v:.4f}' for v in v_chat)}")
    print(f"  DeepSeek-Reasoner   {' '.join(f'{v:.4f}' for v in v_reas)}")


if __name__ == "__main__":
    main()
