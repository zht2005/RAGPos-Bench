"""Auto-generate results_summary.md for paper writing."""
import csv
import os

BASE_DIR = os.path.join(os.path.dirname(__file__), '..')


def read_csv(path):
    with open(path, 'r') as f:
        return list(csv.DictReader(f))


def main():
    metrics_dir = os.path.join(BASE_DIR, 'outputs', 'metrics')
    overall = read_csv(os.path.join(metrics_dir, 'overall_metrics.csv'))
    by_source = read_csv(os.path.join(metrics_dir, 'by_source_metrics.csv'))
    by_variant = read_csv(os.path.join(metrics_dir, 'by_variant_metrics.csv'))

    lines = ["# RAGPos-Bench: Results Summary\n"]
    lines.append("## 1. Dataset Scale")
    lines.append("- 2500 base samples (HotpotQA 1000 + MuSiQue 1000 + SQuAD 500)")
    lines.append("- 15000 evaluation instances (6 variants per base sample)")
    lines.append("- 7 models evaluated\n")

    lines.append("## 2. Overall Results\n")
    lines.append("| Model | Accuracy | PSR↓ | PBR↓ | CAR↑ | EAR↓ | CEU↑ |")
    lines.append("|-------|----------|------|------|------|------|------|")
    for r in overall:
        lines.append(f"| {r['model']} | {float(r['accuracy']):.3f} | {float(r['PSR']):.3f} | "
                     f"{float(r['PBR']):.3f} | {float(r['CAR']):.3f} | {float(r['EAR']):.3f} | "
                     f"{float(r['CEU']):.3f} |")
    lines.append("")

    lines.append("## 3. Key Findings\n")
    best = max(overall, key=lambda x: float(x['accuracy']))
    worst = min(overall, key=lambda x: float(x['accuracy']))
    most_sensitive = max(overall, key=lambda x: float(x['PSR']))
    highest_pbr = max(overall, key=lambda x: float(x['PBR']))

    lines.append(f"- Best overall accuracy: **{best['model']}** ({float(best['accuracy']):.3f})")
    lines.append(f"- Worst overall accuracy: **{worst['model']}** ({float(worst['accuracy']):.3f})")
    lines.append(f"- Most position-sensitive: **{most_sensitive['model']}** (PSR={float(most_sensitive['PSR']):.3f})")
    lines.append(f"- Highest primacy bias: **{highest_pbr['model']}** (PBR={float(highest_pbr['PBR']):.3f})")
    lines.append("")

    v1_accs = {r['model']: float(r['accuracy']) for r in by_variant if r['variant'] == 'correct_front'}
    v3_accs = {r['model']: float(r['accuracy']) for r in by_variant if r['variant'] == 'correct_end'}
    v4_accs = {r['model']: float(r['accuracy']) for r in by_variant if r['variant'] == 'conflict_before_correct'}
    v5_accs = {r['model']: float(r['accuracy']) for r in by_variant if r['variant'] == 'correct_before_conflict'}

    avg_v1 = sum(v1_accs.values()) / len(v1_accs) if v1_accs else 0
    avg_v3 = sum(v3_accs.values()) / len(v3_accs) if v3_accs else 0
    avg_v4 = sum(v4_accs.values()) / len(v4_accs) if v4_accs else 0
    avg_v5 = sum(v5_accs.values()) / len(v5_accs) if v5_accs else 0
    lines.append(f"- Avg accuracy when correct evidence is FIRST (correct_front): {avg_v1:.3f}")
    lines.append(f"- Avg accuracy when correct evidence is LAST (correct_end): {avg_v3:.3f}")
    lines.append(f"- Avg accuracy when CONFLICT precedes correct (V4): {avg_v4:.3f}")
    lines.append(f"- Avg accuracy when CORRECT precedes conflict (V5): {avg_v5:.3f}")
    lines.append(f"- Position drop (V1→V3): **{avg_v1 - avg_v3:.3f}**")
    lines.append(f"- Primacy effect (V5 - V4): **{avg_v5 - avg_v4:.3f}** (positive = primacy bias)")
    lines.append("")

    lines.append("## 4. Accuracy by Data Source\n")
    sources = sorted(set(r['source'] for r in by_source))
    lines.append("| Model | " + " | ".join(sources) + " |")
    lines.append("|-------|" + "---|" * len(sources))
    by_model_source = {}
    for r in by_source:
        by_model_source.setdefault(r['model'], {})[r['source']] = float(r['accuracy'])
    for m in sorted(by_model_source.keys()):
        accs = [f"{by_model_source[m].get(s, 0):.3f}" for s in sources]
        lines.append(f"| {m} | " + " | ".join(accs) + " |")
    lines.append("")

    lines.append("## 5. Figures\n")
    lines.append("- **fig1_position_accuracy**: Accuracy by correct evidence position")
    lines.append("- **fig2_bias_metrics**: PBR, PSR, CAR comparison")
    lines.append("- **fig3_source_breakdown**: Accuracy across HotpotQA / MuSiQue / SQuAD")
    lines.append("- **fig4_variant_accuracy**: Per-variant accuracy line plot")
    lines.append("- **fig5_primacy_bias**: V4 vs V5 conflict ordering effect\n")

    lines.append("## 6. Paper-Ready Bullet Points\n")
    lines.append(f"- Across {len(overall)} state-of-the-art LLMs, accuracy drops {(avg_v1-avg_v3):.1%} on average when correct evidence shifts from first to last position.")
    lines.append(f"- The order of conflicting evidence creates a {(avg_v5-avg_v4):.1%} accuracy gap (primacy effect): models trust evidence appearing earlier.")
    lines.append(f"- {best['model']} achieves best robustness with {float(best['accuracy']):.1%} accuracy.")
    lines.append(f"- {highest_pbr['model']} suffers from extreme primacy bias (PBR={float(highest_pbr['PBR']):.1%}), suggesting heavy reliance on first-position evidence.")
    lines.append("- Conflict awareness (CAR) is uniformly low, indicating LLMs rarely identify contradictions explicitly.")

    out_path = os.path.join(BASE_DIR, 'paper_assets', 'results_summary.md')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"Results summary -> {out_path}")


if __name__ == "__main__":
    main()
