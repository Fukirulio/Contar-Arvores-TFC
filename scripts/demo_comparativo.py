import sys
import os
from pathlib import Path

import torch
import cv2
import numpy as np
from ultralytics import YOLO
from deepforest import main as df_main

RESULTS_FOLDER = "results"
BASE = Path(__file__).resolve().parent.parent
YOLO_MODEL_PATH = BASE / "runs" / "detect" / "tree_neon_v12s_100ep" / "weights" / "best.pt"
DF_FINETUNED_PATH = BASE / "models" / "deepforest_neon_finetuned.pt"

YOLO_COLOR = (255, 0, 0)
DEEPFOREST_COLOR = (0, 200, 0)


def predict_yolo(model, image_path):
    results = model(image_path, verbose=False)
    return results[0].boxes.xyxy.cpu().numpy().tolist()


def predict_deepforest(model, image_path):
    boxes_df = model.predict_image(path=image_path)
    if boxes_df is None or boxes_df.empty:
        return []
    return boxes_df[["xmin", "ymin", "xmax", "ymax"]].values.tolist()


def annotate(image, boxes, color, label):
    out = image.copy()
    for box in boxes:
        x1, y1, x2, y2 = map(int, box[:4])
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
    cv2.putText(out, label, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    return out


def add_header(image, text):
    bar = np.zeros((50, image.shape[1], 3), dtype=np.uint8)
    cv2.putText(bar, text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    return np.vstack([bar, image])


def main():
    if len(sys.argv) < 2:
        print("Uso: python scripts/demo_comparativo.py caminho/imagem.png")
        return

    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f"ERRO: imagem não encontrada: {image_path}")
        return

    image = cv2.imread(image_path)
    if image is None:
        print(f"ERRO: não foi possível ler: {image_path}")
        return

    os.makedirs(RESULTS_FOLDER, exist_ok=True)

    yolo_model = YOLO(str(YOLO_MODEL_PATH))

    deepforest_model = df_main.deepforest()
    deepforest_model.load_model()
    if DF_FINETUNED_PATH.exists():
        ckpt = torch.load(str(DF_FINETUNED_PATH), map_location="cpu")
        weights = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        deepforest_model.model.load_state_dict(weights)
        df_label = "DeepForest fine-tuned"
    else:
        print("AVISO: fine-tuned não encontrado, a usar zero-shot")
        df_label = "DeepForest zero-shot"

    try:
        yolo_boxes = predict_yolo(yolo_model, image_path)
    except Exception as e:
        print(f"Erro YOLO: {e}")
        yolo_boxes = []

    try:
        deepforest_boxes = predict_deepforest(deepforest_model, image_path)
    except Exception as e:
        print(f"Erro DeepForest: {e}")
        deepforest_boxes = []

    yolo_count = len(yolo_boxes)
    df_count = len(deepforest_boxes)
    print(f"YOLO12s 100ep: {yolo_count}  |  {df_label}: {df_count}")

    yolo_img = annotate(image, yolo_boxes, YOLO_COLOR, f"YOLO12s 100ep: {yolo_count}")
    df_img = annotate(image, deepforest_boxes, DEEPFOREST_COLOR, f"{df_label}: {df_count}")
    side_by_side = np.hstack([yolo_img, df_img])

    image_name = os.path.splitext(os.path.basename(image_path))[0]
    header_text = f"{image_name}  |  YOLO12s: {yolo_count}  vs  {df_label}: {df_count}"
    final_image = add_header(side_by_side, header_text)

    out_path = os.path.join(RESULTS_FOLDER, f"demo_{image_name}.png")
    cv2.imwrite(out_path, final_image)
    print(f"Guardado: {out_path}")


if __name__ == "__main__":
    main()
