import os
import sys
import torch
from pathlib import Path
from ultralytics import YOLO

BASE = Path(__file__).resolve().parent.parent
DATA_YAML = BASE / "neon_yolo" / "data.yaml"
RUN_NAME = "tree_neon_v12s_100ep"


def evaluate_on_val(weights_path):
    from scripts.eval_all_models import (
        load_ground_truth, predict_yolo, evaluate, IMAGES_FOLDER
    )
    import os as _os; _os.chdir(BASE)
    gt = load_ground_truth()
    imgs = sorted(f.name for f in IMAGES_FOLDER.glob("*.png"))
    model = YOLO(str(weights_path))
    return evaluate("YOLO12s-100ep", predict_yolo, model, imgs, gt)


def main():
    os.chdir(BASE)

    if torch.cuda.is_available():
        device = 0
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("AVISO: CPU — vai demorar muito")

    model = YOLO("yolo12s.pt")
    results = model.train(
        data=str(DATA_YAML),
        epochs=100,
        imgsz=640,
        batch=8,
        device=device,
        name=RUN_NAME,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        cos_lr=True,
        patience=20,
        degrees=0.0,
        flipud=0.5,
        fliplr=0.5,
        mosaic=1.0,
        scale=0.5,
        cache=False,
        workers=0,
        save=True,
        save_period=25,
        plots=True,
        verbose=True,
    )

    m = results.results_dict
    p     = m.get("metrics/precision(B)", 0.0)
    r     = m.get("metrics/recall(B)", 0.0)
    map50 = m.get("metrics/mAP50(B)", 0.0)
    map95 = m.get("metrics/mAP50-95(B)", 0.0)
    print(f"P={p:.4f}  R={r:.4f}  mAP50={map50:.4f}  mAP50-95={map95:.4f}")
    print(f"Pesos: {BASE / 'runs' / 'detect' / RUN_NAME / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
