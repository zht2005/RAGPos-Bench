"""Plot paper Figure 7 from the paired mitigation result table."""

import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


BASE = os.environ.get(
    "RAGPOS_BASE", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
INPUT = os.path.join(BASE, "outputs", "mitigation", "mitigation_delta.csv")
FIGURES = os.path.join(BASE, "figures")

MODELS = (
    ("gpt-5.4-medium", "GPT-5.4-medium", "#4C78A8"),
    ("deepseek-chat", "DeepSeek-Chat", "#72B7B2"),
    ("gpt-5.4-mini", "GPT-5.4-mini", "#54A24B"),
    ("claude-haiku-4-5-20251001", "Claude-Haiku-4.5", "#B279A2"),
    ("claude-sonnet-4-6", "Claude-Sonnet-4.6", "#9D755D"),
    ("deepseek-reasoner", "DeepSeek-Reasoner", "#F2A65A"),
    ("gemini-2.5-flash", "Gemini-2.5-Flash", "#E45756"),
)
METRICS = (
    ("ACC", r"$\Delta$Acc $\uparrow$", 1.0),
    ("CAR", r"$\Delta$CAR $\uparrow$", 1.0),
    ("PBR", r"$\Delta$PBR $\downarrow$ (sign-flipped)", -1.0),
    ("EAR", r"$\Delta$EAR $\downarrow$ (sign-flipped)", -1.0),
)


def main():
    with open(INPUT, newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    by_key = {(row["model"], row["metric"]): row for row in rows}

    fig, ax = plt.subplots(figsize=(8.9, 4.9))
    x = np.arange(len(METRICS))
    width = 0.105

    for index, (model, label, color) in enumerate(MODELS):
        offsets = x + (index - (len(MODELS) - 1) / 2) * width
        values = []
        significant = []
        for metric, _, direction in METRICS:
            row = by_key[(model, metric)]
            values.append(direction * float(row["delta"]))
            significant.append(row["significant"] == "yes")
        bars = ax.bar(offsets, values, width=width, color=color, label=label)
        for bar, is_significant in zip(bars, significant):
            if not is_significant:
                continue
            value = bar.get_height()
            offset = 0.012 if value >= 0 else -0.018
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + offset,
                "*",
                ha="center",
                va="bottom" if value >= 0 else "top",
                fontsize=10,
                fontweight="bold",
            )

    ax.axhline(0, color="#777777", linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label, _ in METRICS])
    ax.set_ylabel(r"$\Delta$ (conflict-aware $-$ baseline) -- higher is better")
    ax.set_title(
        "Conflict-aware prompting: per-model improvement over baseline\n"
        r"(* = two-sided paired-bootstrap significant at $p<0.05$)"
    )
    ax.grid(axis="y", alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.17), ncol=4,
              frameon=False, fontsize=8.5)
    ax.set_axisbelow(True)
    fig.tight_layout()

    os.makedirs(FIGURES, exist_ok=True)
    for extension in ("pdf", "png"):
        path = os.path.join(FIGURES, f"fig7_mitigation_delta.{extension}")
        fig.savefig(path, bbox_inches="tight", pad_inches=0.04,
                    dpi=300 if extension == "png" else None)
        print(f"[ok] wrote {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
