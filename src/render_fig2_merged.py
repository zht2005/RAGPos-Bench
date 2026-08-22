"""Render Figure 2 as a 2-panel composite:
  (a) Overall diagnostic metrics heatmap (was Figure 2)
  (b) Source-level accuracy heatmap (was Figure 4)
Reads:
  outputs/metrics/overall_metrics.csv
  outputs/metrics/by_source_metrics.csv
Writes:
  figures/fig2_overall_and_source.pdf
  figures/fig2_overall_and_source.png
"""
import csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

BASE = os.environ.get("RAGPOS_BASE", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
METRICS = os.path.join(BASE, "outputs", "metrics")
FIG = os.path.join(BASE, "figures")

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


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


# ---------- panel (a): overall heatmap ----------
overall = {r["model"]: r for r in read_csv(os.path.join(METRICS, "overall_metrics.csv"))}
metric_cols = ["accuracy", "PSR", "PBR", "CAR", "EAR", "CEU"]
arrows = {"accuracy": r"Acc $\uparrow$", "PSR": r"PSR $\downarrow$",
          "PBR": r"PBR $\downarrow$", "CAR": r"CAR $\uparrow$",
          "EAR": r"EAR $\downarrow$", "CEU": r"CEU $\uparrow$"}
higher_better = {"accuracy", "CAR", "CEU"}
raw = np.array([[float(overall[m][c]) for c in metric_cols] for m in ORDER])
norm = np.zeros_like(raw)
for j, c in enumerate(metric_cols):
    col = raw[:, j]
    v = (col - col.min()) / (col.max() - col.min() + 1e-12)
    norm[:, j] = v if c in higher_better else 1.0 - v

# ---------- panel (b): source-level ----------
by_src = read_csv(os.path.join(METRICS, "by_source_metrics.csv"))
sources = ["hotpotqa", "musique", "squad"]
src_label = ["HotpotQA", "MuSiQue", "SQuAD"]
src_data = defaultdict(dict)
for r in by_src:
    src_data[r["model"]][r["source"]] = float(r["accuracy"])
src_raw = np.array([[src_data[m][s] for s in sources] for m in ORDER])

# ---------- figure ----------
plt.rcParams.update({"font.family": "serif", "font.size": 9})
fig, (axA, axB) = plt.subplots(
    1, 2, figsize=(7.0, 4.2),
    gridspec_kw={"width_ratios": [6.0, 3.2], "wspace": 0.18}
)

# --- (a) ---
imA = axA.imshow(norm, cmap="viridis", vmin=0, vmax=1, aspect="auto")
axA.set_xticks(range(len(metric_cols)))
axA.set_xticklabels([arrows[c] for c in metric_cols], fontsize=8.5)
axA.set_yticks(range(len(ORDER)))
axA.set_yticklabels([DISPLAY[m] for m in ORDER], fontsize=8.5)
for i in range(raw.shape[0]):
    for j in range(raw.shape[1]):
        txt_color = "white" if norm[i, j] < 0.55 else "black"
        axA.text(j, i, f"{raw[i, j]:.3f}", ha="center", va="center",
                 color=txt_color, fontsize=7.5)
axA.set_title("(a) Overall diagnostic metrics", fontsize=10, pad=6)
axA.tick_params(length=0)
for s in axA.spines.values():
    s.set_visible(False)

# --- (b) ---
imB = axB.imshow(src_raw, cmap="viridis", vmin=0, vmax=src_raw.max() * 1.05, aspect="auto")
axB.set_xticks(range(len(sources)))
axB.set_xticklabels(src_label, fontsize=8.5)
axB.set_yticks(range(len(ORDER)))
axB.set_yticklabels([])  # share with panel (a)
for i in range(src_raw.shape[0]):
    for j in range(src_raw.shape[1]):
        v = src_raw[i, j]
        axB.text(j, i, f"{v:.3f}", ha="center", va="center",
                 color="white" if v < 0.45 else "black", fontsize=7.5)
means = src_raw.mean(axis=0)
for j, m in enumerate(means):
    axB.text(j, src_raw.shape[0] + 0.05, f"avg {m:.2f}", ha="center",
             va="top", fontsize=7, color="#444")
axB.set_title("(b) Source-level accuracy", fontsize=10, pad=6)
axB.tick_params(length=0)
for s in axB.spines.values():
    s.set_visible(False)

plt.tight_layout()
for ext in ("pdf", "png"):
    plt.savefig(os.path.join(FIG, f"fig2_overall_and_source.{ext}"),
                dpi=200 if ext == "png" else None, bbox_inches="tight",
                pad_inches=0.03)
plt.close()
print(f"[ok] {FIG}/fig2_overall_and_source.{{pdf,png}}")
