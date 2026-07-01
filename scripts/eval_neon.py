import os

import pandas as pd
import cv2
from ultralytics import YOLO
from deepforest import main as df_main

YOLO_MODEL_PATH = "runs/detect/tree_detection-4/weights/best.pt"
IMAGES_FOLDER = "neon_yolo/images/val"
GROUND_TRUTH_CSV = "neon_deepforest.csv"
IOU_THRESHOLD = 0.5


def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def match_boxes(pred_boxes, gt_boxes, iou_threshold=IOU_THRESHOLD):
    pairs = []
    for pi, pbox in enumerate(pred_boxes):
        for gi, gbox in enumerate(gt_boxes):
            iou = compute_iou(pbox, gbox)
            if iou >= iou_threshold:
                pairs.append((iou, pi, gi))
    pairs.sort(reverse=True)
    matched_pred, matched_gt, tp = set(), set(), 0
    for iou, pi, gi in pairs:
        if pi in matched_pred or gi in matched_gt:
            continue
        matched_pred.add(pi)
        matched_gt.add(gi)
        tp += 1
    return tp, len(pred_boxes) - len(matched_pred), len(gt_boxes) - len(matched_gt)


def load_ground_truth(csv_path):
    if not os.path.exists(csv_path):
        print(f"ERRO: ground truth não encontrado: {csv_path}")
        return None
    df = pd.read_csv(csv_path)
    return {
        image_path: group[["xmin", "ymin", "xmax", "ymax"]].values.tolist()
        for image_path, group in df.groupby("image_path")
    }


def predict_yolo(model, image_path):
    results = model(image_path, verbose=False)
    return results[0].boxes.xyxy.cpu().numpy().tolist()


def predict_deepforest(model, image_path):
    boxes_df = model.predict_image(path=image_path)
    if boxes_df is None or boxes_df.empty:
        return []
    return boxes_df[["xmin", "ymin", "xmax", "ymax"]].values.tolist()


def evaluate_model(name, predict_fn, model, image_files, ground_truth):
    print(f"\n{name}:")
    tp_tot = fp_tot = fn_tot = 0
    abs_errors = []
    evaluated = 0

    for i, image_file in enumerate(image_files, start=1):
        image_path = os.path.join(IMAGES_FOLDER, image_file)
        gt_boxes = ground_truth.get(image_file)
        if gt_boxes is None:
            continue
        if cv2.imread(image_path) is None:
            continue
        try:
            pred_boxes = predict_fn(model, image_path)
        except Exception as e:
            print(f"  [{i}] {image_file}: erro ({e})")
            continue
        tp, fp, fn = match_boxes(pred_boxes, gt_boxes)
        tp_tot += tp
        fp_tot += fp
        fn_tot += fn
        abs_errors.append(abs(len(pred_boxes) - len(gt_boxes)))
        evaluated += 1
        print(f"  [{i}/{len(image_files)}] {image_file}: pred={len(pred_boxes)} gt={len(gt_boxes)} TP={tp}")

    p = tp_tot / (tp_tot + fp_tot) if (tp_tot + fp_tot) > 0 else 0.0
    r = tp_tot / (tp_tot + fn_tot) if (tp_tot + fn_tot) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    mae = sum(abs_errors) / len(abs_errors) if abs_errors else 0.0
    return {"modelo": name, "imagens": evaluated, "precision": p, "recall": r, "f1": f1, "mae": mae}


def main():
    if not os.path.isdir(IMAGES_FOLDER):
        print(f"ERRO: pasta de imagens não encontrada: {IMAGES_FOLDER}")
        return

    ground_truth = load_ground_truth(GROUND_TRUTH_CSV)
    if ground_truth is None:
        return

    image_files = sorted(
        f for f in os.listdir(IMAGES_FOLDER) if f.lower().endswith((".png", ".jpg", ".jpeg"))
    )
    print(f"Imagens: {len(image_files)}")

    if not image_files:
        print("ERRO: nenhuma imagem encontrada")
        return
    if not os.path.exists(YOLO_MODEL_PATH):
        print(f"ERRO: modelo YOLO não encontrado: {YOLO_MODEL_PATH}")
        return

    yolo_model = YOLO(YOLO_MODEL_PATH)
    deepforest_model = df_main.deepforest()
    deepforest_model.load_model()

    results = [
        evaluate_model("YOLOv8", predict_yolo, yolo_model, image_files, ground_truth),
        evaluate_model("DeepForest", predict_deepforest, deepforest_model, image_files, ground_truth),
    ]

    print("\nModelo        Imagens   Precision    Recall        F1       MAE")
    print("-" * 64)
    for r in results:
        print(f"{r['modelo']:<12}{r['imagens']:>8}{r['precision']:>12.3f}"
              f"{r['recall']:>10.3f}{r['f1']:>10.3f}{r['mae']:>10.3f}")


if __name__ == "__main__":
    main()
