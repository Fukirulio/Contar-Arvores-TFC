from pathlib import Path
import numpy as np
from ultralytics import YOLO

BASE       = Path(__file__).resolve().parent.parent
YOLO_PATH  = BASE / "runs" / "detect" / "tree_neon_v12s_100ep" / "weights" / "best.pt"
TRAIN_IMGS = BASE / "neon_yolo" / "images" / "train"
TRAIN_LBLS = BASE / "neon_yolo" / "labels" / "train"
VAL_IMGS   = BASE / "neon_yolo" / "images" / "val"
VAL_LBLS   = BASE / "neon_yolo" / "labels" / "val"

def count_gt(lbl_path):
    if not lbl_path.exists():
        return 0
    return sum(1 for ln in open(lbl_path) if ln.strip())

def evaluate(model, img_dir, lbl_dir):
    files = sorted(img_dir.glob("*.png"))
    gt, preds = [], []
    for img in files:
        lbl = lbl_dir / (img.stem + ".txt")
        res = model(str(img), verbose=False, conf=0.25)
        n_pred = len(res[0].boxes) if res[0].boxes is not None else 0
        gt.append(count_gt(lbl))
        preds.append(n_pred)
    gt    = np.array(gt)
    preds = np.array(preds)
    mae   = float(np.mean(np.abs(gt - preds)))
    ss_res = np.sum((gt - preds) ** 2)
    ss_tot = np.sum((gt - np.mean(gt)) ** 2)
    r2    = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    return len(files), mae, r2

print("A carregar YOLO12s 100ep...")
model = YOLO(str(YOLO_PATH))

print(f"A avaliar treino  ({len(list(TRAIN_IMGS.glob('*.png')))} imagens)...")
n_train, mae_train, r2_train = evaluate(model, TRAIN_IMGS, TRAIN_LBLS)

print(f"A avaliar validação ({len(list(VAL_IMGS.glob('*.png')))} imagens)...")
n_val, mae_val, r2_val = evaluate(model, VAL_IMGS, VAL_LBLS)

print()
print("=" * 52)
print("  OVERFITTING CHECK — YOLO12s 100ep (NEON)")
print("=" * 52)
print(f"  {'Conjunto':<12} | {'Imagens':>7} | {'MAE':>6} | {'R²':>6}")
print(f"  {'-'*12}-+-{'-'*7}-+-{'-'*6}-+-{'-'*6}")
print(f"  {'Treino':<12} | {n_train:>7} | {mae_train:>6.2f} | {r2_train:>6.3f}")
print(f"  {'Validação':<12} | {n_val:>7} | {mae_val:>6.2f} | {r2_val:>6.3f}")
print("=" * 52)

gap = abs(mae_train - mae_val)
pct = gap / mae_val * 100 if mae_val > 0 else 0
print()
print(f"  Diferença MAE: {gap:.2f} ({pct:.0f}% relativamente ao val)")
print()
if pct > 50:
    print("  ⚠  OVERFITTING MODERADO — MAE de treino muito inferior ao de validação.")
else:
    print("  ✓  GENERALIZAÇÃO SAUDÁVEL — MAE de treino e validação são similares.")
print()
