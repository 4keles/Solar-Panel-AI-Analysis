#!/usr/bin/env python3
"""
Comprehensive HTML training report generator.

Usage:
  python scripts/generate_report.py \
      --weights models/v1.0.3/best.pt \
      --run-dir  runs/detect/solar_mvp3 \
      --data     data/processed_data/solar/unifiedV2_dataset/data.yaml \
      --version  v1.0.3 \
      --split    test

Output: reports/<version>/report.html
"""

from __future__ import annotations

import argparse
import base64
import json
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
from ultralytics import YOLO


# ── helpers ──────────────────────────────────────────────────────────────────

def img_to_b64(path: Path) -> str | None:
    if path and path.exists():
        return base64.b64encode(path.read_bytes()).decode()
    return None


def fig_to_b64(fig: plt.Figure) -> str:
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=130)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


# ── training curves ───────────────────────────────────────────────────────────

def plot_training_curves(results_csv: Path) -> str | None:
    if not results_csv.exists():
        return None

    df = pd.read_csv(results_csv)
    df.columns = [c.strip() for c in df.columns]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("Training Curves", fontsize=14, fontweight="bold")

    plots = [
        ("train/box_loss",    "Box Loss (train)",      "tab:blue"),
        ("train/cls_loss",    "Cls Loss (train)",      "tab:orange"),
        ("train/dfl_loss",    "DFL Loss (train)",      "tab:red"),
        ("metrics/mAP50(B)",  "mAP@50",                "tab:green"),
        ("metrics/mAP50-95(B)", "mAP@50-95",          "tab:purple"),
        ("metrics/recall(B)", "Recall",                "tab:cyan"),
    ]
    for ax, (col, title, color) in zip(axes.flat, plots):
        if col in df.columns:
            ax.plot(df["epoch"], df[col], color=color, linewidth=1.5)
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("Epoch")
            ax.grid(alpha=0.3)
        else:
            ax.set_visible(False)

    fig.tight_layout()
    return fig_to_b64(fig)


# ── per-class bar chart ───────────────────────────────────────────────────────

def plot_per_class(per_class: dict, metric: str, title: str, color: str) -> str:
    names = list(per_class.keys())
    vals  = [per_class[n].get(metric, 0) for n in names]

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.barh(names, vals, color=color, alpha=0.8)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9)
    ax.set_xlim(0, 1.1)
    ax.set_xlabel(metric)
    ax.set_title(title, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    return fig_to_b64(fig)


# ── radar chart — overall overview ───────────────────────────────────────────

def plot_radar(overall: dict) -> str:
    metrics = ["precision", "recall", "f1", "mAP50", "mAP50_95"]
    labels  = ["Precision", "Recall", "F1", "mAP50", "mAP50-95"]
    vals    = [overall.get(m, 0) for m in metrics]
    vals   += [vals[0]]   # close polygon

    angles = [n / len(metrics) * 2 * 3.14159 for n in range(len(metrics))]
    angles += [angles[0]]

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw={"polar": True})
    ax.plot(angles, vals, "o-", linewidth=2, color="tab:blue")
    ax.fill(angles, vals, alpha=0.25, color="tab:blue")
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_title("Overall Performance", fontweight="bold", pad=15)
    fig.tight_layout()
    return fig_to_b64(fig)


# ── validation ────────────────────────────────────────────────────────────────

def run_validation(weights: Path, data: Path, split: str, imgsz: int, device: str) -> dict:
    model   = YOLO(str(weights))
    results = model.val(data=str(data), split=split, imgsz=imgsz, device=device, verbose=False)

    rd = getattr(results, "results_dict", {})
    names = results.names  # {0: 'bird_drop', ...}

    overall = {
        "mAP50":     float(rd.get("metrics/mAP50(B)", 0)),
        "mAP50_95":  float(rd.get("metrics/mAP50-95(B)", 0)),
        "precision": float(rd.get("metrics/precision(B)", 0)),
        "recall":    float(rd.get("metrics/recall(B)", 0)),
    }
    p = overall["precision"]; r = overall["recall"]
    overall["f1"] = 2 * p * r / (p + r) if (p + r) else 0.0

    # Per-class metrics
    per_class = {}
    box = results.box
    if hasattr(box, "ap_class_index") and box.ap_class_index is not None:
        for i, cls_idx in enumerate(box.ap_class_index):
            cname = names.get(int(cls_idx), f"cls{cls_idx}")
            per_class[cname] = {
                "mAP50":    float(box.ap50[i])  if hasattr(box, "ap50")  else 0.0,
                "mAP50_95": float(box.ap[i])    if hasattr(box, "ap")    else 0.0,
                "precision":float(box.p[i])     if hasattr(box, "p")     else 0.0,
                "recall":   float(box.r[i])     if hasattr(box, "r")     else 0.0,
            }
            pr = per_class[cname]["precision"]; rr = per_class[cname]["recall"]
            per_class[cname]["f1"] = 2*pr*rr/(pr+rr) if (pr+rr) else 0.0

    return {"overall": overall, "per_class": per_class, "val_run_dir": str(results.save_dir)}


# ── HTML builder ─────────────────────────────────────────────────────────────

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="tr">
<head>
<meta charset="UTF-8">
<title>Solar Panel OD — {version} Report</title>
<style>
  body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 0; background: #f4f6f9; color: #222; }}
  header {{ background: #1a1a2e; color: white; padding: 24px 40px; }}
  header h1 {{ margin: 0; font-size: 1.7rem; }}
  header p  {{ margin: 4px 0 0; opacity: .7; font-size: .9rem; }}
  main {{ max-width: 1200px; margin: 30px auto; padding: 0 20px; }}
  h2 {{ border-left: 5px solid #4a90d9; padding-left: 12px; margin-top: 40px; }}
  .cards {{ display: flex; gap: 16px; flex-wrap: wrap; margin: 16px 0; }}
  .card {{ background: white; border-radius: 10px; padding: 20px 28px; flex: 1;
           min-width: 150px; box-shadow: 0 2px 8px rgba(0,0,0,.08); text-align: center; }}
  .card .val {{ font-size: 2rem; font-weight: bold; color: #4a90d9; }}
  .card .lbl {{ font-size: .85rem; color: #666; margin-top: 4px; }}
  .card.good  .val {{ color: #27ae60; }}
  .card.warn  .val {{ color: #e67e22; }}
  .card.bad   .val {{ color: #e74c3c; }}
  table {{ border-collapse: collapse; width: 100%; background: white;
           border-radius: 10px; overflow: hidden; box-shadow: 0 2px 8px rgba(0,0,0,.08); }}
  th {{ background: #1a1a2e; color: white; padding: 12px 16px; text-align: left; font-size: .9rem; }}
  td {{ padding: 10px 16px; border-bottom: 1px solid #eee; font-size: .9rem; }}
  tr:last-child td {{ border-bottom: none; }}
  tr:hover td {{ background: #f0f4ff; }}
  .bar {{ display: inline-block; height: 10px; border-radius: 5px; background: #4a90d9; vertical-align: middle; margin-right: 6px; }}
  .imgs-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(340px, 1fr)); gap: 16px; }}
  .img-box {{ background: white; border-radius: 10px; padding: 12px;
              box-shadow: 0 2px 8px rgba(0,0,0,.08); }}
  .img-box p {{ margin: 8px 0 0; font-size: .8rem; color: #666; text-align: center; }}
  img {{ max-width: 100%; border-radius: 6px; }}
  .badge {{ display: inline-block; padding: 2px 10px; border-radius: 12px;
            font-size: .75rem; font-weight: bold; }}
  .badge-green {{ background: #d4edda; color: #155724; }}
  .badge-red   {{ background: #f8d7da; color: #721c24; }}
  .badge-yellow{{ background: #fff3cd; color: #856404; }}
  footer {{ text-align: center; color: #999; font-size: .8rem; margin: 50px 0 20px; }}
  .chart-row {{ display: flex; gap: 16px; flex-wrap: wrap; }}
  .chart-box {{ background: white; border-radius: 10px; padding: 16px;
                box-shadow: 0 2px 8px rgba(0,0,0,.08); flex: 1; min-width: 300px; }}
</style>
</head>
<body>
<header>
  <h1>Solar Panel Object Detection — {version}</h1>
  <p>Generated: {generated_at} &nbsp;|&nbsp; Split: <b>{split}</b> &nbsp;|&nbsp; Model: <b>{weights_name}</b></p>
</header>
<main>

<h2>Overall Metrics</h2>
<div class="cards">
  <div class="card {map50_cls}"><div class="val">{mAP50:.3f}</div><div class="lbl">mAP@50</div></div>
  <div class="card {map5095_cls}"><div class="val">{mAP50_95:.3f}</div><div class="lbl">mAP@50-95</div></div>
  <div class="card"><div class="val">{precision:.3f}</div><div class="lbl">Precision</div></div>
  <div class="card"><div class="val">{recall:.3f}</div><div class="lbl">Recall</div></div>
  <div class="card {f1_cls}"><div class="val">{f1:.3f}</div><div class="lbl">F1</div></div>
</div>

{radar_section}

{curves_section}

<h2>Per-Class Metrics</h2>
{per_class_table}

{per_class_charts}

<h2>Confusion Matrix</h2>
{confusion_section}

<h2>Sample Predictions</h2>
{predictions_section}

<h2>Training Config</h2>
<div style="background:white;border-radius:10px;padding:20px;box-shadow:0 2px 8px rgba(0,0,0,.08);">
<pre style="margin:0;font-size:.85rem;white-space:pre-wrap;">{args_yaml}</pre>
</div>

</main>
<footer>Solar Panel OD Report &mdash; {version} &mdash; {generated_at}</footer>
</body>
</html>"""


def score_class(val: float, good: float = 0.7, warn: float = 0.5) -> str:
    if val >= good: return "good"
    if val >= warn: return "warn"
    return "bad"


def per_class_table_html(per_class: dict) -> str:
    if not per_class:
        return "<p><em>Per-class metrics not available.</em></p>"

    rows = ""
    for cname, m in sorted(per_class.items()):
        f1_badge = ("badge-green" if m["f1"] >= 0.7
                    else "badge-yellow" if m["f1"] >= 0.5
                    else "badge-red")
        bar_w = int(m["mAP50"] * 120)
        rows += f"""
        <tr>
          <td><b>{cname}</b></td>
          <td><span class="bar" style="width:{bar_w}px;"></span>{m['mAP50']:.3f}</td>
          <td>{m['mAP50_95']:.3f}</td>
          <td>{m['precision']:.3f}</td>
          <td>{m['recall']:.3f}</td>
          <td><span class="badge {f1_badge}">{m['f1']:.3f}</span></td>
        </tr>"""
    return f"""
    <table>
      <thead><tr>
        <th>Class</th><th>mAP@50</th><th>mAP@50-95</th>
        <th>Precision</th><th>Recall</th><th>F1</th>
      </tr></thead>
      <tbody>{rows}</tbody>
    </table>"""


def img_section(b64: str | None, caption: str) -> str:
    if not b64:
        return ""
    return f'<div class="img-box"><img src="data:image/png;base64,{b64}"><p>{caption}</p></div>'


def chart_section(b64: str | None, title: str) -> str:
    if not b64:
        return ""
    return f'<div class="chart-box"><img src="data:image/png;base64,{b64}"><p style="text-align:center;font-size:.85rem;color:#555;">{title}</p></div>'


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate comprehensive HTML report")
    parser.add_argument("--weights",  type=Path, required=True)
    parser.add_argument("--run-dir",  type=Path, default=None,
                        help="Training run directory (for curves & sample images)")
    parser.add_argument("--data",     type=Path, required=True)
    parser.add_argument("--version",  default="v?.?.?")
    parser.add_argument("--split",    choices=["train", "val", "test"], default="test")
    parser.add_argument("--imgsz",    type=int, default=640)
    parser.add_argument("--device",   default="0")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    report_dir = root / "reports" / args.version
    report_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running validation ({args.split})...")
    val = run_validation(args.weights, args.data, args.split, args.imgsz, args.device)
    overall    = val["overall"]
    per_class  = val["per_class"]

    # Save JSON
    summary = {
        "version": args.version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "split": args.split,
        "weights": str(args.weights),
        "overall": overall,
        "per_class": per_class,
    }
    (report_dir / "val_summary.json").write_text(json.dumps(summary, indent=2))
    print("val_summary.json saved.")

    # Training curves
    curves_b64 = None
    if args.run_dir:
        csv_path = args.run_dir / "results.csv"
        print("Plotting training curves...")
        curves_b64 = plot_training_curves(csv_path)

    # Radar chart
    print("Generating charts...")
    radar_b64 = plot_radar(overall)

    # Per-class bar charts
    pc_map50_b64 = plot_per_class(per_class, "mAP50",     "mAP@50 per Class",     "#4a90d9") if per_class else None
    pc_f1_b64    = plot_per_class(per_class, "f1",        "F1 per Class",          "#27ae60") if per_class else None
    pc_rec_b64   = plot_per_class(per_class, "recall",    "Recall per Class",      "#e67e22") if per_class else None

    # Confusion matrix
    cm_b64 = cm_norm_b64 = None
    if args.run_dir:
        cm_b64      = img_to_b64(args.run_dir / "confusion_matrix.png")
        cm_norm_b64 = img_to_b64(args.run_dir / "confusion_matrix_normalized.png")
    if not cm_b64:  # fallback to val run dir
        val_rd = Path(val["val_run_dir"])
        cm_b64      = img_to_b64(val_rd / "confusion_matrix.png")
        cm_norm_b64 = img_to_b64(val_rd / "confusion_matrix_normalized.png")

    # Sample predictions (val batches)
    pred_b64s = []
    search_dirs = []
    if args.run_dir:
        search_dirs.append(args.run_dir)
    search_dirs.append(Path(val["val_run_dir"]))
    for sd in search_dirs:
        for i in range(3):
            b = img_to_b64(sd / f"val_batch{i}_pred.jpg")
            lbl = img_to_b64(sd / f"val_batch{i}_labels.jpg")
            if b and b not in pred_b64s:
                pred_b64s.append((b, lbl, f"Batch {i}"))
        if pred_b64s:
            break

    # args.yaml
    args_yaml = ""
    if args.run_dir and (args.run_dir / "args.yaml").exists():
        args_yaml = (args.run_dir / "args.yaml").read_text()

    # ── Build HTML ────────────────────────────────────────────────────────────
    radar_section = f"""
    <div class="chart-row">
      {chart_section(radar_b64, "Overall radar")}
    </div>""" if radar_b64 else ""

    curves_section = f"""
    <h2>Training Curves</h2>
    <div class="chart-box" style="padding:16px;">
      <img src="data:image/png;base64,{curves_b64}" style="width:100%;">
    </div>""" if curves_b64 else ""

    per_class_charts = ""
    if pc_map50_b64 or pc_f1_b64:
        inner = chart_section(pc_map50_b64, "mAP@50 per Class")
        inner += chart_section(pc_f1_b64,   "F1 per Class")
        inner += chart_section(pc_rec_b64,  "Recall per Class")
        per_class_charts = f'<div class="chart-row" style="margin-top:20px;">{inner}</div>'

    confusion_section = '<div class="imgs-grid">'
    confusion_section += img_section(cm_b64,      "Confusion Matrix (counts)")
    confusion_section += img_section(cm_norm_b64, "Confusion Matrix (normalized)")
    confusion_section += "</div>"

    predictions_section = '<div class="imgs-grid">'
    for b64, lbl_b64, caption in pred_b64s[:3]:
        predictions_section += img_section(lbl_b64, f"{caption} — Ground Truth")
        predictions_section += img_section(b64,     f"{caption} — Predictions")
    predictions_section += "</div>"

    html = HTML_TEMPLATE.format(
        version=args.version,
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        split=args.split,
        weights_name=args.weights.name,
        mAP50=overall["mAP50"],
        mAP50_95=overall["mAP50_95"],
        precision=overall["precision"],
        recall=overall["recall"],
        f1=overall["f1"],
        map50_cls=score_class(overall["mAP50"]),
        map5095_cls=score_class(overall["mAP50_95"], 0.4, 0.25),
        f1_cls=score_class(overall["f1"]),
        radar_section=radar_section,
        curves_section=curves_section,
        per_class_table=per_class_table_html(per_class),
        per_class_charts=per_class_charts,
        confusion_section=confusion_section,
        predictions_section=predictions_section,
        args_yaml=args_yaml or "(args.yaml bulunamadı)",
    )

    report_path = report_dir / "report.html"
    report_path.write_text(html, encoding="utf-8")
    print(f"\nReport saved → {report_path}")
    print(f"Browser'da aç: file://{report_path.resolve()}")


if __name__ == "__main__":
    main()
