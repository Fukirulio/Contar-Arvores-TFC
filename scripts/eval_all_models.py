"""
Avalia todos os modelos YOLO (OAM + NEON) e DeepForest no conjunto NEON val.
Compara precision, recall, F1 e MAE numa tabela única.
"""
import os
import sys
from pathlib import Path

import cv2
import pandas as pd
from ultralytics import YOLO

BASE = Path(__file__).resolve().parent.parent
IMAGES_FOLDER = BASE / "neon_yolo" / "images" / "val"
GROUND_TRUTH_CSV = BASE / "neon_deepforest.csv"
IOU_THRESHOLD = 0.5


def compute_iou(b1, b2):
    x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a1 = max(0.0, b1[2] - b1[0]) * max(0.0, b1[3] - b1[1])
    a2 = max(0.0, b2[2] - b2[0]) * max(0.0, b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def match_boxes(pred, gt):
    pairs = sorted(
        [(compute_iou(p, g), pi, gi) for pi, p in enumerate(pred) for gi, g in enumerate(gt)
         if compute_iou(p, g) >= IOU_THRESHOLD],
        reverse=True,
    )
    mp, mg, tp = set(), set(), 0
    for _, pi, gi in pairs:
        if pi not in mp and gi not in mg:
            mp.add(pi); mg.add(gi); tp += 1
    return tp, len(pred) - len(mp), len(gt) - len(mg)


def load_ground_truth():
    if not GROUND_TRUTH_CSV.exists():
        print(f"ERRO: ground truth não encontrado: {GROUND_TRUTH_CSV}")
        sys.exit(1)
    df = pd.read_csv(GROUND_TRUTH_CSV)
    return {
        name: grp[["xmin", "ymin", "xmax", "ymax"]].values.tolist()
        for name, grp in df.groupby("image_path")
    }


def predict_yolo(model, image_path):
    res = model(str(image_path), verbose=False)
    return res[0].boxes.xyxy.cpu().numpy().tolist()


def predict_deepforest(df_model, image_path):
    boxes_df = df_model.predict_image(path=str(image_path))
    if boxes_df is None or boxes_df.empty:
        return []
    return boxes_df[["xmin", "ymin", "xmax", "ymax"]].values.tolist()


def evaluate(label, predict_fn, model, image_files, gt):
    print(f"\n  [{label}] a avaliar {len(image_files)} imagens...")
    tp_tot = fp_tot = fn_tot = 0
    abs_errors = []
    evaluated = 0

    for img_file in image_files:
        path = IMAGES_FOLDER / img_file
        gt_boxes = gt.get(img_file)
        if gt_boxes is None:
            continue
        if cv2.imread(str(path)) is None:
            continue
        try:
            pred = predict_fn(model, path)
        except Exception as e:
            print(f"    ERRO {img_file}: {e}")
            continue
        tp, fp, fn = match_boxes(pred, gt_boxes)
        tp_tot += tp; fp_tot += fp; fn_tot += fn
        abs_errors.append(abs(len(pred) - len(gt_boxes)))
        evaluated += 1

    p = tp_tot / (tp_tot + fp_tot) if (tp_tot + fp_tot) > 0 else 0.0
    r = tp_tot / (tp_tot + fn_tot) if (tp_tot + fn_tot) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    mae = sum(abs_errors) / len(abs_errors) if abs_errors else 0.0

    return {"modelo": label, "imgs": evaluated, "precision": p, "recall": r, "f1": f1, "mae": mae}


def find_neon_models():
    """Encontra pesos treinados no NEON (neon_* e runs explícitos)."""
    detect_dir = BASE / "runs" / "detect"
    models = {}
    if not detect_dir.exists():
        return models
    for run_dir in sorted(detect_dir.iterdir()):
        if run_dir.name.startswith("neon_"):
            best = run_dir / "weights" / "best.pt"
            if best.exists():
                label = run_dir.name.replace("neon_", "NEON/").replace("yolo", "YOLO")
                models[label] = best
    # modelos NEON com nome fora do padrão neon_*
    extra = [
        ("NEON/YOLO12s 100ep", detect_dir / "tree_neon_v12s_100ep" / "weights" / "best.pt"),
    ]
    for label, path in extra:
        if path.exists():
            models[label] = path
    return models


def find_oam_models():
    """Encontra o melhor modelo treinado no OAM (tree_detection-4)."""
    candidates = [
        ("OAM/yolov8s", BASE / "runs" / "detect" / "tree_detection-4" / "weights" / "best.pt"),
        ("OAM/yolov8s-2", BASE / "runs" / "detect" / "tree_detection2" / "weights" / "best.pt"),
    ]
    return {label: path for label, path in candidates if path.exists()}


def main():
    os.chdir(BASE)

    if not IMAGES_FOLDER.exists():
        print(f"ERRO: pasta de imagens não encontrada: {IMAGES_FOLDER}")
        sys.exit(1)

    gt = load_ground_truth()
    image_files = sorted(f.name for f in IMAGES_FOLDER.glob("*.png"))
    print(f"Imagens NEON val: {len(image_files)}")
    print(f"Ground truth (ficheiros únicos): {len(gt)}")

    if not image_files:
        print("ERRO: sem imagens em neon_yolo/images/val/")
        sys.exit(1)

    neon_models = find_neon_models()
    oam_models = find_oam_models()
    all_yolo = {**oam_models, **neon_models}

    results = []

    for label, path in all_yolo.items():
        print(f"\nA carregar {label} de {path.parent.parent.name}/...")
        model = YOLO(str(path))
        r = evaluate(label, predict_yolo, model, image_files, gt)
        results.append(r)
        print(f"    -> P={r['precision']:.3f}  R={r['recall']:.3f}  F1={r['f1']:.3f}  MAE={r['mae']:.2f}")

    try:
        import torch
        from deepforest import main as df_main

        print("\nA carregar DeepForest zero-shot...")
        df_model = df_main.deepforest()
        df_model.load_model()
        r = evaluate("DeepForest zero-shot", predict_deepforest, df_model, image_files, gt)
        results.append(r)
        print(f"    -> P={r['precision']:.3f}  R={r['recall']:.3f}  F1={r['f1']:.3f}  MAE={r['mae']:.2f}")

        df_ft_path = BASE / "models" / "deepforest_neon_finetuned.pt"
        if df_ft_path.exists():
            print("\nA carregar DeepForest fine-tuned NEON...")
            df_ft = df_main.deepforest()
            df_ft.load_model()
            ckpt = torch.load(str(df_ft_path), map_location="cpu")
            weights = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
            df_ft.model.load_state_dict(weights)
            r_ft = evaluate("DeepForest fine-tuned", predict_deepforest, df_ft, image_files, gt)
            results.append(r_ft)
            print(f"    -> P={r_ft['precision']:.3f}  R={r_ft['recall']:.3f}  F1={r_ft['f1']:.3f}  MAE={r_ft['mae']:.2f}")
        else:
            print(f"\nDeepForest fine-tuned nao encontrado em {df_ft_path} — ignorado.")
    except ImportError:
        print("\nDeepForest nao instalado — ignorado.")
    except Exception as e:
        print(f"\nDeepForest erro: {e} — ignorado.")

    print(f"\n{'='*70}")
    print("TABELA COMPARATIVA FINAL — NEON val set")
    print(f"{'='*70}")
    header = f"{'Modelo':<20}{'Imagens':>8}{'Precision':>12}{'Recall':>10}{'F1':>8}{'MAE':>8}"
    print(header)
    print("-" * len(header))
    for r in sorted(results, key=lambda x: -x["f1"]):
        print(
            f"{r['modelo']:<20}{r['imgs']:>8}"
            f"{r['precision']:>12.3f}{r['recall']:>10.3f}"
            f"{r['f1']:>8.3f}{r['mae']:>8.2f}"
        )
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
