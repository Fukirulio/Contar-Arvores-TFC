import os
import sys
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

BASE = Path(__file__).resolve().parent.parent
RUN_DIR = BASE / "runs" / "detect" / "tree_neon_v12s_100ep"
OUT_PNG = BASE / "results" / "curvas_treino_yolo12s.png"


def main():
    os.chdir(BASE)

    csv_path = RUN_DIR / "results.csv"
    if not csv_path.exists():
        print(f"ERRO: {csv_path} não encontrado. Corre train_yolo12s_100ep.py primeiro.")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    print(f"Épocas: {len(df)}")

    epoch = df["epoch"] if "epoch" in df.columns else df.index + 1

    fig = plt.figure(figsize=(16, 9))
    fig.suptitle("YOLO12s — NEON fine-tune 100 épocas", fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    loss_pairs = [
        ("train/box_loss", "val/box_loss", "Box Loss"),
        ("train/cls_loss", "val/cls_loss", "Class Loss"),
        ("train/dfl_loss", "val/dfl_loss", "DFL Loss"),
    ]
    for i, (tcol, vcol, title) in enumerate(loss_pairs):
        ax = fig.add_subplot(gs[0, i])
        if tcol in df.columns:
            ax.plot(epoch, df[tcol], label="train", color="#1f77b4", linewidth=1.5)
        if vcol in df.columns:
            ax.plot(epoch, df[vcol], label="val", color="#ff7f0e", linewidth=1.5, linestyle="--")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Época")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    metric_pairs = [
        ("metrics/mAP50(B)", "mAP@50"),
        ("metrics/mAP50-95(B)", "mAP@50-95"),
        ("metrics/precision(B)", "Precision"),
    ]
    for i, (col, title) in enumerate(metric_pairs):
        ax = fig.add_subplot(gs[1, i])
        if col in df.columns:
            ax.plot(epoch, df[col], color="#2ca02c", linewidth=1.5)
            best_val = df[col].max()
            best_ep = df.loc[df[col].idxmax(), "epoch"] if "epoch" in df.columns else df[col].idxmax() + 1
            ax.axhline(best_val, color="red", linestyle=":", linewidth=1, alpha=0.7)
            ax.set_title(f"{title}\n(best={best_val:.4f} @ ep{best_ep:.0f})", fontsize=10)
        else:
            ax.set_title(title, fontsize=10)
            ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
        ax.set_xlabel("Época")
        ax.grid(True, alpha=0.3)

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(OUT_PNG), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Guardado: {OUT_PNG}")

    train_loss_end = df.get("train/box_loss", pd.Series([0])).iloc[-1]
    val_loss_end = df.get("val/box_loss", pd.Series([0])).iloc[-1]
    diverge = val_loss_end > train_loss_end * 1.5
    print(f"box_loss final — train: {train_loss_end:.4f}  val: {val_loss_end:.4f}")
    print(f"Divergência: {'sim (possível overfitting)' if diverge else 'não'}")

    if "metrics/mAP50(B)" in df.columns:
        peak = df["metrics/mAP50(B)"].max()
        peak_ep = df.loc[df["metrics/mAP50(B)"].idxmax(), "epoch"] if "epoch" in df.columns else "?"
        print(f"mAP50 máximo: {peak:.4f} @ época {peak_ep}")


if __name__ == "__main__":
    main()
