"""Generate all figures using matplotlib."""
import csv
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from utils import load_jsonl, match_answer

BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
FIG_DIR = os.path.join(BASE_DIR, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)


def read_csv(path):
    with open(path, 'r') as f:
        return list(csv.DictReader(f))


def fig1():
    rows = read_csv(os.path.join(BASE_DIR, 'outputs/metrics/position_metrics.csv'))
    models = sorted(set(r["model"] for r in rows))
    positions = sorted(set(r["position"] for r in rows))
    data = defaultdict(dict)
    for r in rows:
        data[r["model"]][r["position"]] = float(r["accuracy"])
    x = np.arange(len(positions))
    w = 0.8 / len(models)
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, m in enumerate(models):
        ax.bar(x + i*w, [data[m].get(p, 0) for p in positions], w, label=m[:12])
    ax.set_xlabel('Correct Evidence Position')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy by Correct Evidence Position')
    ax.set_xticks(x + w*len(models)/2)
    ax.set_xticklabels(positions)
    ax.legend(fontsize=7, ncol=2)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig1_position_accuracy.pdf'))
    plt.savefig(os.path.join(FIG_DIR, 'fig1_position_accuracy.png'), dpi=150)
    plt.close()


def fig2():
    rows = read_csv(os.path.join(BASE_DIR, 'outputs/metrics/overall_metrics.csv'))
    models = [r["model"] for r in rows]
    short = [m[:12] for m in models]
    feb = [float(r["PBR"]) for r in rows]
    psr = [float(r["PSR"]) for r in rows]
    car = [float(r["CAR"]) for r in rows]
    x = np.arange(len(models))
    w = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w, feb, w, label='PBR', color='#e74c3c')
    ax.bar(x, psr, w, label='PSR', color='#3498db')
    ax.bar(x + w, car, w, label='CAR', color='#2ecc71')
    ax.set_xlabel('Model')
    ax.set_ylabel('Score')
    ax.set_title('Front Evidence Bias / Position Sensitivity / Conflict Awareness')
    ax.set_xticks(x)
    ax.set_xticklabels(short, rotation=30, ha='right', fontsize=8)
    ax.legend()
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig2_bias_metrics.pdf'))
    plt.savefig(os.path.join(FIG_DIR, 'fig2_bias_metrics.png'), dpi=150)
    plt.close()


def fig3():
    rows = read_csv(os.path.join(BASE_DIR, 'outputs/metrics/by_source_metrics.csv'))
    models = sorted(set(r["model"] for r in rows))
    sources = sorted(set(r["source"] for r in rows))
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(sources))
    w = 0.8 / len(models)
    da = defaultdict(dict)
    for r in rows:
        da[r["model"]][r["source"]] = float(r["accuracy"])
    for i, m in enumerate(models):
        ax.bar(x + i*w, [da[m].get(s, 0) for s in sources], w, label=m[:12])
    ax.set_title('Accuracy by Data Source')
    ax.set_xticks(x + w*len(models)/2)
    ax.set_xticklabels(sources, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Source')
    ax.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig3_source_breakdown.pdf'))
    plt.savefig(os.path.join(FIG_DIR, 'fig3_source_breakdown.png'), dpi=150)
    plt.close()
def fig4():
    rows = read_csv(os.path.join(BASE_DIR, 'outputs/metrics/by_variant_metrics.csv'))
    models = sorted(set(r["model"] for r in rows))
    variants = ["correct_front", "correct_middle", "correct_end",
                "conflict_before_correct", "correct_before_conflict",
                "distractor_dominant"]
    short_v = ["V1", "V2", "V3", "V4", "V5", "V6"]
    fig, ax = plt.subplots(figsize=(10, 5))
    for m in models:
        mr = {r["variant"]: float(r["accuracy"]) for r in rows if r["model"] == m}
        ax.plot(short_v, [mr.get(v, 0) for v in variants], marker='o', label=m[:12])
    ax.set_xlabel('Variant')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy Across Position Variants')
    ax.legend(fontsize=7, ncol=2)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig4_variant_accuracy.pdf'))
    plt.savefig(os.path.join(FIG_DIR, 'fig4_variant_accuracy.png'), dpi=150)
    plt.close()


def fig5():
    instances = load_jsonl(os.path.join(BASE_DIR, 'data/eval_instances.jsonl'))
    cbc = [i for i in instances if i["variant"] == "conflict_before_correct"]
    cfc = [i for i in instances if i["variant"] == "correct_before_conflict"]
    pred_dir = os.path.join(BASE_DIR, 'outputs/parsed_predictions')
    models, cbc_acc, cfc_acc = [], [], []
    for fname in sorted(os.listdir(pred_dir)):
        if not fname.endswith('.jsonl'):
            continue
        model = fname.replace('.jsonl', '')
        preds = {p["instance_id"]: p for p in load_jsonl(os.path.join(pred_dir, fname))}
        cbc_correct = sum(1 for i in cbc if i["instance_id"] in preds and
                          match_answer(preds[i["instance_id"]]["answer"], i["gold_answer"], "general"))
        cfc_correct = sum(1 for i in cfc if i["instance_id"] in preds and
                          match_answer(preds[i["instance_id"]]["answer"], i["gold_answer"], "general"))
        models.append(model[:12])
        cbc_acc.append(cbc_correct / len(cbc) if cbc else 0)
        cfc_acc.append(cfc_correct / len(cfc) if cfc else 0)
    x = np.arange(len(models))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w/2, cfc_acc, w, label='Correct-Before-Conflict (V5)', color='#2ecc71')
    ax.bar(x + w/2, cbc_acc, w, label='Conflict-Before-Correct (V4)', color='#e74c3c')
    ax.set_xlabel('Model')
    ax.set_ylabel('Accuracy')
    ax.set_title('Primacy Bias: Order of Conflicting Evidence')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha='right', fontsize=8)
    ax.legend()
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig5_primacy_bias.pdf'))
    plt.savefig(os.path.join(FIG_DIR, 'fig5_primacy_bias.png'), dpi=150)
    plt.close()


def main():
    print("Generating figures...")
    fig1(); print("  fig1 done")
    fig2(); print("  fig2 done")
    fig3(); print("  fig3 done")
    fig4(); print("  fig4 done")
    fig5(); print("  fig5 done")
    print(f"All figures saved to {FIG_DIR}")


if __name__ == "__main__":
    main()
