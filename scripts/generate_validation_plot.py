from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

BASE = Path(__file__).resolve().parent.parent
YOLO_PATH = BASE / "runs" / "detect" / "tree_neon_v12s_100ep" / "weights" / "best.pt"
DF_PATH   = BASE / "models" / "deepforest_neon_finetuned.pt"
VAL_IMGS  = BASE / "neon_yolo" / "images" / "val"
VAL_LBLS  = BASE / "neon_yolo" / "labels" / "val"
OUT       = BASE / "results" / "validation_predicted_vs_actual.png"

img_files = sorted(VAL_IMGS.glob("*.png"))
print(f"Imagens val: {len(img_files)}")

gt, yolo_preds, df_preds = [], [], []

print("A carregar YOLO12s 100ep...")
from ultralytics import YOLO
yolo = YOLO(str(YOLO_PATH))

print("A inferir YOLO12s...")
for img_path in img_files:
    lbl = VAL_LBLS / (img_path.stem + ".txt")
    n_gt = sum(1 for ln in open(lbl) if ln.strip()) if lbl.exists() else 0
    res  = yolo(str(img_path), verbose=False, conf=0.25)
    n_yolo = len(res[0].boxes) if res[0].boxes is not None else 0
    gt.append(n_gt)
    yolo_preds.append(n_yolo)

print("A carregar DeepForest fine-tuned...")
from deepforest import main as df_main
df_model = df_main.deepforest()
df_model.load_model()
ckpt = torch.load(str(DF_PATH), map_location="cpu")
weights = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
df_model.model.load_state_dict(weights)
df_model.model.eval()

print("A inferir DeepForest...")
for img_path in img_files:
    boxes = df_model.predict_image(path=str(img_path))
    n_df  = len(boxes) if boxes is not None and len(boxes) > 0 else 0
    df_preds.append(n_df)

gt         = np.array(gt)
yolo_preds = np.array(yolo_preds)
df_preds   = np.array(df_preds)


def metrics(true, pred):
    mae = np.mean(np.abs(true - pred))
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - np.mean(true)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return mae, r2


mae_y, r2_y = metrics(gt, yolo_preds)
mae_d, r2_d = metrics(gt, df_preds)
print(f"YOLO12s 100ep  — MAE={mae_y:.2f}  R²={r2_y:.3f}")
print(f"DeepForest FT  — MAE={mae_d:.2f}  R²={r2_d:.3f}")

plt.style.use("seaborn-v0_8-whitegrid")
GREEN = "#2d6a2d"

max_val = int(max(gt.max(), yolo_preds.max(), df_preds.max())) + 5
diag    = [0, max_val]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Predição vs Ground Truth — NEON val set (n=39)", fontsize=13, fontweight="bold")

for ax, preds, title in [
    (axes[0], yolo_preds, f"YOLO12s 100ep\nMAE={mae_y:.2f}  R²={r2_y:.3f}"),
    (axes[1], df_preds,   f"DeepForest fine-tuned\nMAE={mae_d:.2f}  R²={r2_d:.3f}"),
]:
    ax.scatter(gt, preds, color=GREEN, alpha=0.7, s=60, edgecolors="white", linewidths=0.5, zorder=3)
    ax.plot(diag, diag, "k--", linewidth=1.2, alpha=0.6, label="Previsão perfeita (y=x)")
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)
    ax.set_xlabel("Ground Truth (árvores)", fontsize=11)
    ax.set_ylabel("Predição (árvores)", fontsize=11)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=9)
    ax.set_aspect("equal")

plt.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(str(OUT), dpi=150, bbox_inches="tight")
plt.close(fig)

from PIL import Image
img = Image.open(OUT)
print(f"Guardado: {OUT}  ({img.size[0]}x{img.size[1]} px, {OUT.stat().st_size/1024:.1f} KB)")
