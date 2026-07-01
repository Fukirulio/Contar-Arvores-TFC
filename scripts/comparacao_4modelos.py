import os
import sys
from pathlib import Path

import torch
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent.parent
DEMO_DIR = BASE / "demo"
RESULTS_DIR = BASE / "results"

YOLO12s_30  = BASE / "runs" / "detect" / "neon_yolo12s" / "weights" / "best.pt"
YOLO12s_100 = BASE / "runs" / "detect" / "tree_neon_v12s_100ep" / "weights" / "best.pt"
DF_FINETUNED = BASE / "models" / "deepforest_neon_finetuned.pt"

COLORS = {
    "df_zero":  (0.2, 0.6, 1.0),
    "df_fine":  (0.0, 0.85, 0.4),
    "yolo_30":  (1.0, 0.5, 0.0),
    "yolo_100": (0.9, 0.1, 0.2),
}


def draw_boxes(img_bgr, boxes_xyxy, color_rgb, thickness=2):
    out = img_bgr.copy()
    c = tuple(int(x * 255) for x in reversed(color_rgb))
    for box in boxes_xyxy:
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        cv2.rectangle(out, (x1, y1), (x2, y2), c, thickness)
    return out


def predict_yolo(model, img_path):
    res = model(str(img_path), verbose=False)
    return res[0].boxes.xyxy.cpu().numpy().tolist()


def predict_df(df_model, img_path):
    try:
        preds = df_model.predict_image(path=str(img_path))
        if preds is None or preds.empty:
            return []
        return preds[["xmin", "ymin", "xmax", "ymax"]].values.tolist()
    except Exception as e:
        print(f"  DeepForest erro: {e}")
        return []


def process_image(img_path, models_preds):
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        print(f"ERRO: não consegue ler {img_path}")
        return None

    h, w = img_bgr.shape[:2]
    max_dim = 800
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)))
        for i, (lbl, ck, boxes) in enumerate(models_preds):
            models_preds[i] = (lbl, ck, [[b[0]*scale, b[1]*scale, b[2]*scale, b[3]*scale] for b in boxes])

    n = len(models_preds)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    fig.suptitle(img_path.stem, fontsize=13, fontweight="bold")

    for ax, (label, color_key, boxes) in zip(axes, models_preds):
        rendered = cv2.cvtColor(draw_boxes(img_bgr, boxes, COLORS[color_key]), cv2.COLOR_BGR2RGB)
        ax.imshow(rendered)
        ax.set_title(f"{label}\n({len(boxes)} det.)", fontsize=10,
                     color=[c * 0.7 for c in COLORS[color_key]])
        ax.axis("off")

    plt.tight_layout()
    return fig


def main():
    os.chdir(BASE)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    demo_imgs = sorted(DEMO_DIR.glob("*.png"))
    if not demo_imgs:
        print(f"ERRO: sem imagens em {DEMO_DIR}")
        sys.exit(1)

    print(f"A carregar modelos ({len(demo_imgs)} imagens de demo)...")

    from deepforest import main as df_main
    df_zero = df_main.deepforest()
    df_zero.load_model()

    if DF_FINETUNED.exists():
        df_fine = df_main.deepforest()
        df_fine.load_model()
        ckpt = torch.load(str(DF_FINETUNED), map_location="cpu")
        weights = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        df_fine.model.load_state_dict(weights)
    else:
        print(f"AVISO: {DF_FINETUNED} não existe — fallback para zero-shot")
        df_fine = df_zero

    from ultralytics import YOLO
    if not YOLO12s_30.exists():
        print(f"ERRO: {YOLO12s_30} não existe"); sys.exit(1)
    if not YOLO12s_100.exists():
        print(f"ERRO: {YOLO12s_100} não existe"); sys.exit(1)

    yolo_30  = YOLO(str(YOLO12s_30))
    yolo_100 = YOLO(str(YOLO12s_100))

    for img_path in demo_imgs:
        print(f"\n{img_path.name}")
        boxes_df_zero = predict_df(df_zero, img_path)
        boxes_df_fine = predict_df(df_fine, img_path)
        boxes_y30     = predict_yolo(yolo_30, img_path)
        boxes_y100    = predict_yolo(yolo_100, img_path)
        print(f"  df_zero={len(boxes_df_zero)}  df_fine={len(boxes_df_fine)}"
              f"  yolo30={len(boxes_y30)}  yolo100={len(boxes_y100)}")

        models_preds = [
            ("DeepForest\nzero-shot",  "df_zero",  boxes_df_zero),
            ("DeepForest\nfine-tuned", "df_fine",  boxes_df_fine),
            ("YOLO12s\n30 épocas",     "yolo_30",  boxes_y30),
            ("YOLO12s\n100 épocas",    "yolo_100", boxes_y100),
        ]
        fig = process_image(img_path, models_preds)
        if fig is None:
            continue
        out_path = RESULTS_DIR / f"comparacao_4modelos_neon_{img_path.stem}.png"
        fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Guardado: {out_path}")

    print(f"\nFicheiros em {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
