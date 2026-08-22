"""Generate the four PRICAI 2026 figures (Fig 2, 3, 4, 5) and two appendix figures (A1, A2)
strictly from the seven-model CSVs. No new inference, no old-batch data.

Outputs:
  figures/fig2_main_heatmap.{pdf,png}
  figures/fig3_position_slope.{pdf,png}
  figures/fig4_source_heatmap.{pdf,png}
  figures/fig5_deepseek_radar.{pdf,png}
  figures/appendix_fig_a1_position_bars.{pdf,png}
  figures/appendix_fig_a2_variant_lines.{pdf,png}
"""
import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = os.environ.get("RAGPOS_BASE", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
METRICS = os.path.join(BASE, "outputs", "metrics")
FIG = os.path.join(BASE, "figures")
os.makedirs(FIG, exist_ok=True)

DISPLAY = {
    "gpt-5.4-medium": "GPT-5.4-medium",
    "gpt-5.4-mini": "GPT-5.4-mini",
    "deepseek-chat": "DeepSeek-Chat",
    "deepseek-reasoner": "DeepSeek-Reasoner",
    "gemini-2.5-flash": "Gemini-2.5-Flash",
    "claude-haiku-4-5-20251001": "Claude-Haiku-4.5",
    "claude-sonnet-4-6": "Claude-Sonnet-4.6",
}

ORDER = [
    "gpt-5.4-medium", "deepseek-chat", "gpt-5.4-mini",
    "claude-haiku-4-5-20251001", "claude-sonnet-4-6",
    "deepseek-reasoner", "gemini-2.5-flash",
]

PALETTE = {
    "gpt-5.4-medium":            "#1f77b4",
    "deepseek-chat":             "#2ca02c",
    "gpt-5.4-mini":              "#17becf",
    "claude-haiku-4-5-20251001": "#9467bd",
    "claude-sonnet-4-6":         "#8c564b",
    "deepseek-reasoner":         "#ff7f0e",
    "gemini-2.5-flash":          "#d62728",
}


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------- Figure 2
def fig2_main_heatmap():
    rows = read_csv(os.path.join(METRICS, "overall_metrics.csv"))
    by_model = {r["model"]: r for r in rows}
    metric_cols = ["accuracy", "PSR", "PBR", "CAR", "EAR", "CEU"]
    arrows = {"accuracy": r"Acc $\uparrow$", "PSR": r"PSR $\downarrow$",
              "PBR": r"PBR $\downarrow$", "CAR": r"CAR $\uparrow$",
              "EAR": r"EAR $\downarrow$", "CEU": r"CEU $\uparrow$"}
    higher_better = {"accuracy", "CAR", "CEU"}

    raw = np.array([[float(by_model[m][c]) for c in metric_cols] for m in ORDER])

    norm = np.zeros_like(raw)
    for j, c in enumerate(metric_cols):
        col = raw[:, j]
        v = (col - col.min()) / (col.max() - col.min() + 1e-12)
        norm[:, j] = v if c in higher_better else 1.0 - v

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    im = ax.imshow(norm, cmap="RdBu_r", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(metric_cols)))
    ax.set_xticklabels([arrows[c] for c in metric_cols], fontsize=10)
    ax.set_yticks(range(len(ORDER)))
    ax.set_yticklabels([DISPLAY[m] for m in ORDER], fontsize=10)

    for i in range(raw.shape[0]):
        for j in range(raw.shape[1]):
            txt_color = "white" if norm[i, j] < 0.30 or norm[i, j] > 0.85 else "black"
            ax.text(j, i, f"{raw[i, j]:.3f}", ha="center", va="center",
                    color=txt_color, fontsize=9)

    cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Per-column normalized score (warmer = better)", fontsize=8)
    cbar.ax.tick_params(labelsize=8)

    ax.set_title("Main results: per-model performance on six diagnostic metrics",
                 fontsize=11, pad=10)
    plt.tight_layout()
    for ext in ("pdf", "png"):
        plt.savefig(os.path.join(FIG, f"fig2_main_heatmap.{ext}"),
                    dpi=200 if ext == "png" else None, bbox_inches="tight")
    plt.close()
    print("[ok] fig2_main_heatmap")


# ---------------------------------------------------------------- Figure 3
def fig3_position_slope():
    rows = read_csv(os.path.join(METRICS, "by_variant_metrics.csv"))
    pick = {"correct_front": "Front (E1)", "correct_middle": "Middle (E3)",
            "correct_end": "End (E6)"}
    data = defaultdict(dict)
    for r in rows:
        if r["variant"] in pick:
            data[r["model"]][r["variant"]] = float(r["accuracy"])

    sig_models = {"deepseek-chat", "gpt-5.4-mini", "gemini-2.5-flash"}

    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    x = np.arange(3)
    xlabels = [pick["correct_front"], pick["correct_middle"], pick["correct_end"]]

    for m in ORDER:
        y = [data[m]["correct_front"], data[m]["correct_middle"], data[m]["correct_end"]]
        is_sig = m in sig_models
        ax.plot(x, y,
                color=PALETTE[m],
                lw=2.6 if is_sig else 1.3,
                alpha=1.0 if is_sig else 0.55,
                marker="o", markersize=7 if is_sig else 5,
                label=DISPLAY[m] + (" *" if is_sig else ""),
                zorder=4 if is_sig else 2)
        ax.text(2.05, y[2], DISPLAY[m].replace("-", "‑"),
                color=PALETTE[m], fontsize=8, va="center",
                fontweight="bold" if is_sig else "normal")

    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=10)
    ax.set_ylabel("Accuracy", fontsize=10)
    ax.set_xlabel("Slot of the correct evidence", fontsize=10)
    ax.set_ylim(0, 0.85)
    ax.grid(axis="y", alpha=0.25)
    ax.set_title("Position sensitivity: accuracy as the correct evidence moves\nfrom Front to End "
                 "(* = paired-bootstrap significant at p < 0.05, V1 vs V3)",
                 fontsize=10, pad=8)
    ax.legend(loc="lower left", fontsize=7, ncol=2, frameon=False)
    plt.tight_layout()
    for ext in ("pdf", "png"):
        plt.savefig(os.path.join(FIG, f"fig3_position_slope.{ext}"),
                    dpi=200 if ext == "png" else None, bbox_inches="tight")
    plt.close()
    print("[ok] fig3_position_slope")


# ---------------------------------------------------------------- Figure 4
def fig4_source_heatmap():
    rows = read_csv(os.path.join(METRICS, "by_source_metrics.csv"))
    sources = ["hotpotqa", "musique", "squad"]
    src_label = ["HotpotQA", "MuSiQue", "SQuAD"]
    by_ms = defaultdict(dict)
    for r in rows:
        by_ms[r["model"]][r["source"]] = float(r["accuracy"])

    raw = np.array([[by_ms[m][s] for s in sources] for m in ORDER])

    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    im = ax.imshow(raw, cmap="RdBu_r", vmin=0, vmax=raw.max() * 1.05, aspect="auto")
    ax.set_xticks(range(3))
    ax.set_xticklabels(src_label, fontsize=10)
    ax.set_yticks(range(len(ORDER)))
    ax.set_yticklabels([DISPLAY[m] for m in ORDER], fontsize=10)

    for i in range(raw.shape[0]):
        for j in range(raw.shape[1]):
            v = raw[i, j]
            vmax = raw.max() * 1.05
            r = v / vmax if vmax > 0 else 0
            ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                    color="white" if (r < 0.18 or r > 0.82) else "black", fontsize=9)

    means = raw.mean(axis=0)
    for j, m in enumerate(means):
        ax.text(j, raw.shape[0] + 0.05, f"avg {m:.3f}", ha="center",
                va="top", fontsize=8, color="#444")

    cbar = plt.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cbar.set_label("Accuracy", fontsize=8)
    cbar.ax.tick_params(labelsize=8)

    ax.set_title("Source-level difficulty across QA datasets\n"
                 "(MuSiQue, the multi-hop benchmark, is consistently the hardest)",
                 fontsize=10, pad=8)
    plt.tight_layout()
    for ext in ("pdf", "png"):
        plt.savefig(os.path.join(FIG, f"fig4_source_heatmap.{ext}"),
                    dpi=200 if ext == "png" else None, bbox_inches="tight")
    plt.close()
    print("[ok] fig4_source_heatmap")


# ---------------------------------------------------------------- Figure 5
def fig5_deepseek_radar():
    rows = read_csv(os.path.join(METRICS, "overall_metrics.csv"))
    by_m = {r["model"]: r for r in rows}
    chat = by_m["deepseek-chat"]
    reas = by_m["deepseek-reasoner"]

    metrics = [("Accuracy", "accuracy", False),
               ("CAR", "CAR", False),
               ("CEU", "CEU", False),
               ("1 - PSR", "PSR", True),
               ("1 - PBR", "PBR", True),
               ("1 - EAR", "EAR", True)]

    def vals(rec):
        return [(1.0 - float(rec[k]) if inv else float(rec[k])) for _, k, inv in metrics]

    v_chat = vals(chat) + [vals(chat)[0]]
    v_reas = vals(reas) + [vals(reas)[0]]
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6.0, 5.0), subplot_kw=dict(polar=True))
    ax.plot(angles, v_chat, color="#2ca02c", lw=2.2, label="DeepSeek-Chat")
    ax.fill(angles, v_chat, color="#2ca02c", alpha=0.18)
    ax.plot(angles, v_reas, color="#ff7f0e", lw=2.2, label="DeepSeek-Reasoner")
    ax.fill(angles, v_reas, color="#ff7f0e", alpha=0.18)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m[0] for m in metrics], fontsize=9)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8"], fontsize=8, color="#666")
    ax.set_ylim(0, 1.0)
    ax.set_title("Reasoning Does Not Necessarily Improve\nRAG Conflict Resolution",
                 fontsize=11, pad=18)
    ax.legend(loc="upper right", bbox_to_anchor=(1.30, 1.10), fontsize=9, frameon=False)
    plt.tight_layout()
    for ext in ("pdf", "png"):
        plt.savefig(os.path.join(FIG, f"fig5_deepseek_radar.{ext}"),
                    dpi=200 if ext == "png" else None, bbox_inches="tight")
    plt.close()
    print("[ok] fig5_deepseek_radar")


# ---------------------------------------------------------------- Appendix A1
def fig_a1_position_bars():
    rows = read_csv(os.path.join(METRICS, "position_metrics.csv"))
    positions = ["E1", "E2", "E3", "E5", "E6"]
    data = defaultdict(dict)
    for r in rows:
        data[r["model"]][r["position"]] = float(r["accuracy"])

    x = np.arange(len(positions))
    w = 0.11
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    for i, m in enumerate(ORDER):
        ax.bar(x + (i - 3) * w, [data[m].get(p, 0) for p in positions],
               w, color=PALETTE[m], label=DISPLAY[m])
    ax.set_xticks(x)
    ax.set_xticklabels(positions, fontsize=10)
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Slot of the correct evidence")
    ax.set_ylim(0, 1.0)
    ax.set_title("Appendix A1: Position-wise accuracy by evidence slot")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=7, ncol=4, loc="upper left", bbox_to_anchor=(0, 1.18))
    plt.tight_layout()
    for ext in ("pdf", "png"):
        plt.savefig(os.path.join(FIG, f"appendix_fig_a1_position_bars.{ext}"),
                    dpi=200 if ext == "png" else None, bbox_inches="tight")
    plt.close()
    print("[ok] appendix_a1")


# ---------------------------------------------------------------- Appendix A2
def fig_a2_variant_lines():
    rows = read_csv(os.path.join(METRICS, "by_variant_metrics.csv"))
    variants = ["correct_front", "correct_middle", "correct_end",
                "conflict_before_correct", "correct_before_conflict",
                "distractor_dominant"]
    short = ["V1", "V2", "V3", "V4", "V5", "V6"]
    data = defaultdict(dict)
    for r in rows:
        data[r["model"]][r["variant"]] = float(r["accuracy"])

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    for m in ORDER:
        y = [data[m][v] for v in variants]
        ax.plot(short, y, marker="o", lw=1.8, color=PALETTE[m], label=DISPLAY[m])
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Variant")
    ax.set_ylabel("Accuracy")
    ax.set_title("Appendix A2: Variant-level accuracy by model")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=7, ncol=4, loc="upper left", bbox_to_anchor=(0, 1.18))
    plt.tight_layout()
    for ext in ("pdf", "png"):
        plt.savefig(os.path.join(FIG, f"appendix_fig_a2_variant_lines.{ext}"),
                    dpi=200 if ext == "png" else None, bbox_inches="tight")
    plt.close()
    print("[ok] appendix_a2")


if __name__ == "__main__":
    fig2_main_heatmap()
    fig3_position_slope()
    fig4_source_heatmap()
    fig5_deepseek_radar()
    fig_a1_position_bars()
    fig_a2_variant_lines()
    print("All figures written to", FIG)
