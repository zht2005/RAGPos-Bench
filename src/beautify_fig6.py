"""Figure 6: Task-type breakdown (model x task_type heatmap, two panels: Accuracy and EAR).

Reads outputs/metrics/task_type_breakdown.csv. Style matches the rest of the
beautified figures (muted RdBu_r palette, serif font, light grid).
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

BASE = ".."
CSV = os.path.join(BASE, "outputs/metrics/task_type_breakdown.csv")
FIG = os.path.join(BASE, "figures")

DISPLAY = [
    ("gpt-5.4-medium",            "GPT-5.4-medium"),
    ("deepseek-chat",             "DeepSeek-Chat"),
    ("gpt-5.4-mini",              "GPT-5.4-mini"),
    ("claude-haiku-4-5-20251001", "Claude-Haiku-4.5"),
    ("claude-sonnet-4-6",         "Claude-Sonnet-4.6"),
    ("deepseek-reasoner",         "DeepSeek-Reasoner"),
    ("gemini-2.5-flash",          "Gemini-2.5-Flash"),
]
TASK_TYPES = ["entity", "numerical", "temporal", "compositional", "other"]
TASK_LABELS = ["Entity", "Numerical", "Temporal", "Compositional", "Other"]


def pick_serif():
    for name in ("Times New Roman", "Liberation Serif", "DejaVu Serif"):
        try:
            fm.findfont(fm.FontProperties(family=name), fallback_to_default=False)
            return name
        except Exception:
            continue
    return "DejaVu Serif"


def load():
    by_mt = defaultdict(dict)
    with open(CSV) as f:
        for r in csv.DictReader(f):
            by_mt[r["model"]][r["task_type"]] = {
                "accuracy": float(r["accuracy"]),
                "ear": float(r["ear"]),
                "ceu": float(r["ceu"]),
            }
    return by_mt


def main():
    data = load()
    acc = np.array([[data[mid][t]["accuracy"] for t in TASK_TYPES]
                    for mid, _ in DISPLAY])
    ear = np.array([[data[mid][t]["ear"] for t in TASK_TYPES]
                    for mid, _ in DISPLAY])

    serif = pick_serif()
    rcParams.update({
        "font.family": "serif",
        "font.serif": [serif],
        "axes.titlesize": 11.5,
        "axes.labelsize": 10.5,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
    })

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.0))

    def heat(ax, mat, title, cmap_name, fmt="{:.2f}"):
        im = ax.imshow(mat, cmap=cmap_name, aspect="auto",
                       vmin=mat.min(), vmax=mat.max())
        ax.set_xticks(range(len(TASK_TYPES)))
        ax.set_xticklabels(TASK_LABELS, rotation=20, ha="right")
        ax.set_yticks(range(len(DISPLAY)))
        ax.set_yticklabels([d[1] for d in DISPLAY])
        ax.set_title(title, pad=8, color="#222")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                norm = (v - mat.min()) / (mat.max() - mat.min() + 1e-12)
                txt_color = "white" if (norm < 0.20 or norm > 0.80) else "#222"
                ax.text(j, i, fmt.format(v), ha="center", va="center",
                        color=txt_color, fontsize=8.5)
        cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        cbar.ax.tick_params(labelsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    heat(axes[0], acc, "Accuracy by task type ($\\uparrow$)", "RdBu_r")
    heat(axes[1], ear, "Error Adoption Rate by task type ($\\downarrow$)", "RdBu")

    plt.tight_layout()
    out_pdf = os.path.join(FIG, "fig6_task_type_heatmap.pdf")
    out_png = os.path.join(FIG, "fig6_task_type_heatmap.png")
    plt.savefig(out_pdf, bbox_inches="tight", pad_inches=0.04)
    plt.savefig(out_png, bbox_inches="tight", pad_inches=0.04, dpi=300)
    plt.close()

    print("[ok] wrote", out_pdf)
    print("[ok] wrote", out_png)
    print("\nAccuracy:")
    for (mid, lab), row in zip(DISPLAY, acc):
        print(f"  {lab:<22s}", " ".join(f"{v:.3f}" for v in row))
    print("\nEAR:")
    for (mid, lab), row in zip(DISPLAY, ear):
        print(f"  {lab:<22s}", " ".join(f"{v:.3f}" for v in row))


if __name__ == "__main__":
    main()
