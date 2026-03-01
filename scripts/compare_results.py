"""Generate comparison table from all experiment results.

Reads eval_results.json from each Exp*/ folder and produces
a combined markdown + CSV results table.

Usage:
    python scripts/compare_results.py
"""

import json
import os
from pathlib import Path


PROJECT_ROOT = Path("/home/siu856569517/Taminul/co2_farm")


def load_experiment(exp_dir):
    """Load eval results from an experiment directory."""
    results_file = exp_dir / "eval_results" / "eval_results.json"
    if not results_file.exists():
        return None
    with open(results_file) as f:
        data = json.load(f)
    return data


def format_table():
    """Collect all experiment results and format as table."""
    experiments = sorted(PROJECT_ROOT.glob("Exp*_*"))
    if not experiments:
        print("No Exp*_ directories found!")
        return

    rows = []
    for exp_dir in experiments:
        data = load_experiment(exp_dir)
        if data is None:
            print(f"  ⚠ No results in {exp_dir.name}")
            continue

        seg = data.get("segmentation", {})
        cls = data.get("classification", {})
        model_name = exp_dir.name.split("_", 1)[1] if "_" in exp_dir.name else exp_dir.name

        row = {
            "Model": model_name,
            "Params (M)": f"{data.get('params_m', '?')}",
            # Segmentation
            "mIoU": f"{seg.get('iou', 0):.4f}",
            "Dice": f"{seg.get('dice', 0):.4f}",
            "Precision": f"{seg.get('precision', 0):.4f}",
            "Recall": f"{seg.get('recall', 0):.4f}",
            "BF1": f"{seg.get('boundary_f1', 0):.4f}",
            "HD (px)": f"{seg.get('hausdorff', 0):.2f}",
            # Classification
            "Acc": f"{cls.get('accuracy', 0):.4f}",
            "BalAcc": f"{cls.get('balanced_accuracy', 0):.4f}",
            "F1": f"{cls.get('macro_f1', 0):.4f}",
            "AUC": f"{cls.get('macro_auc_roc', 0):.4f}",
            "Kappa": f"{cls.get('cohen_kappa', 0):.4f}",
        }
        rows.append(row)

    if not rows:
        print("No results found!")
        return

    # Print markdown table
    headers = list(rows[0].keys())
    print("\n" + "=" * 120)
    print("  MODEL COMPARISON TABLE")
    print("=" * 120)

    # Header
    header_line = "| " + " | ".join(headers) + " |"
    sep_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    print(header_line)
    print(sep_line)

    # Rows
    for row in rows:
        line = "| " + " | ".join(row.values()) + " |"
        print(line)

    print()

    # Save to file
    output_md = PROJECT_ROOT / "comparison_results.md"
    with open(output_md, "w") as f:
        f.write("# RumiFormer — Model Comparison Results\n\n")
        f.write(f"Experiments: {len(rows)}\n\n")
        f.write("## Segmentation + Classification Metrics\n\n")
        f.write(header_line + "\n")
        f.write(sep_line + "\n")
        for row in rows:
            line = "| " + " | ".join(row.values()) + " |"
            f.write(line + "\n")
        f.write("\n\n---\n*Generated automatically from experiment results.*\n")
    print(f"📄 Table saved to: {output_md}")

    # Also save CSV
    output_csv = PROJECT_ROOT / "comparison_results.csv"
    with open(output_csv, "w") as f:
        f.write(",".join(headers) + "\n")
        for row in rows:
            f.write(",".join(row.values()) + "\n")
    print(f"📊 CSV saved to: {output_csv}")


if __name__ == "__main__":
    format_table()
