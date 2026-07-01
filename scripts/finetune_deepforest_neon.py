import os
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
import csv
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent.parent
CSV_ALL = BASE / "neon_deepforest.csv"
TRAIN_IMG_DIR = BASE / "neon_yolo" / "images" / "train"
VAL_IMG_DIR = BASE / "neon_yolo" / "images" / "val"
TRAIN_CSV = BASE / "neon_train.csv"
VAL_CSV = BASE / "neon_val.csv"
MODEL_OUT = BASE / "models" / "deepforest_neon_finetuned.pt"


def make_split_csvs():
    df = pd.read_csv(CSV_ALL)
    train_imgs = {f.name for f in TRAIN_IMG_DIR.glob("*.png")}
    val_imgs = {f.name for f in VAL_IMG_DIR.glob("*.png")}
    df_train = df[df["image_path"].isin(train_imgs)].copy()
    df_val = df[df["image_path"].isin(val_imgs)].copy()
    df_train.to_csv(TRAIN_CSV, index=False)
    df_val.to_csv(VAL_CSV, index=False)
    print(f"Treino: {len(df_train)} anotações, {df_train['image_path'].nunique()} imagens")
    print(f"Val:    {len(df_val)} anotações, {df_val['image_path'].nunique()} imagens")
    return df_val


def evaluate_deepforest(model, val_imgs, gt):
    import cv2
    IOU = 0.5

    def iou(b1, b2):
        x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
        x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
        inter = max(0, x2-x1) * max(0, y2-y1)
        a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
        a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
        u = a1 + a2 - inter
        return inter/u if u > 0 else 0

    def match(pred, gt_b):
        pairs = sorted(
            [(iou(p,g), pi, gi) for pi,p in enumerate(pred) for gi,g in enumerate(gt_b) if iou(p,g) >= IOU],
            reverse=True)
        mp, mg, tp = set(), set(), 0
        for _, pi, gi in pairs:
            if pi not in mp and gi not in mg:
                mp.add(pi); mg.add(gi); tp += 1
        return tp, len(pred)-len(mp), len(gt_b)-len(mg)

    tp_t = fp_t = fn_t = 0
    abs_err = []

    for img_name in val_imgs:
        path = str(VAL_IMG_DIR / img_name)
        if cv2.imread(path) is None:
            continue
        gt_boxes = gt.get(img_name, [])
        if not gt_boxes:
            continue
        try:
            preds = model.predict_image(path=path)
            pred_boxes = preds[["xmin","ymin","xmax","ymax"]].values.tolist() if preds is not None and not preds.empty else []
        except Exception as e:
            print(f"  Erro {img_name}: {e}")
            continue
        tp, fp, fn = match(pred_boxes, gt_boxes)
        tp_t += tp; fp_t += fp; fn_t += fn
        abs_err.append(abs(len(pred_boxes) - len(gt_boxes)))

    p = tp_t/(tp_t+fp_t) if (tp_t+fp_t) > 0 else 0
    r = tp_t/(tp_t+fn_t) if (tp_t+fn_t) > 0 else 0
    f1 = 2*p*r/(p+r) if (p+r) > 0 else 0
    mae = sum(abs_err)/len(abs_err) if abs_err else 0
    return {"precision": p, "recall": r, "f1": f1, "mae": mae}


def main():
    os.chdir(BASE)

    df_val = make_split_csvs()

    from deepforest import main as df_main
    model = df_main.deepforest()
    model.load_model()

    model.config["train"]["csv_file"] = str(TRAIN_CSV)
    model.config["train"]["root_dir"] = str(TRAIN_IMG_DIR)
    model.config["validation"]["csv_file"] = str(VAL_CSV)
    model.config["validation"]["root_dir"] = str(VAL_IMG_DIR)
    model.config["train"]["epochs"] = 15
    model.config["train"]["batch_size"] = 4
    model.config["train"]["lr"] = 0.0001

    print(f"Fine-tune: {TRAIN_CSV.name}  |  Val: {VAL_CSV.name}")
    model.create_trainer(enable_progress_bar=True, enable_model_summary=False)
    model.trainer.fit(model)

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(MODEL_OUT))
    print(f"Modelo guardado: {MODEL_OUT}")

    gt = {}
    for name, grp in df_val.groupby("image_path"):
        gt[name] = grp[["xmin","ymin","xmax","ymax"]].values.tolist()

    val_imgs = sorted(f.name for f in VAL_IMG_DIR.glob("*.png"))
    res = evaluate_deepforest(model, val_imgs, gt)

    print(f"\n{'Modelo':<28}{'Precision':>10}{'Recall':>10}{'F1':>8}{'MAE':>8}")
    print("-" * 58)
    print(f"{'DeepForest zero-shot':<28}{'0.590':>10}{'0.626':>10}{'0.608':>8}{'8.87':>8}")
    print(f"{'DeepForest fine-tuned':<28}{res['precision']:>10.3f}{res['recall']:>10.3f}{res['f1']:>8.3f}{res['mae']:>8.2f}")

    delta_f1 = res['f1'] - 0.608
    delta_mae = res['mae'] - 8.87
    print(f"\n  Delta F1  : {delta_f1:+.3f}  ({'melhor' if delta_f1 > 0 else 'pior'})")
    print(f"  Delta MAE : {delta_mae:+.2f}  ({'melhor' if delta_mae < 0 else 'pior'})")


if __name__ == "__main__":
    main()
