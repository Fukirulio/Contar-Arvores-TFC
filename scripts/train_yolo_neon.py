import os
import random
import shutil
import sys
import torch
from pathlib import Path

from ultralytics import YOLO

BASE = Path(__file__).resolve().parent.parent
NEON_DIR = BASE / "neon_yolo"
DATA_YAML = NEON_DIR / "data.yaml"

MODELS_TO_TRY = ["yolov8s.pt", "yolo11s.pt", "yolo12s.pt"]
TRAIN_SPLIT = 0.80
RANDOM_SEED = 42


def ensure_split():
    val_imgs = NEON_DIR / "images" / "val"
    val_labels = NEON_DIR / "labels" / "val"
    train_imgs = NEON_DIR / "images" / "train"
    train_labels = NEON_DIR / "labels" / "train"

    if train_imgs.exists() and any(train_imgs.iterdir()):
        n_train = len(list(train_imgs.glob("*.png")))
        n_val = len(list(val_imgs.glob("*.png")))
        print(f"Split já existe: {n_train} treino | {n_val} validação")
        return

    train_imgs.mkdir(parents=True, exist_ok=True)
    train_labels.mkdir(parents=True, exist_ok=True)

    imgs = sorted(val_imgs.glob("*.png"))
    if not imgs:
        print("ERRO: nenhuma imagem em neon_yolo/images/val/")
        sys.exit(1)

    random.seed(RANDOM_SEED)
    random.shuffle(imgs)
    n_train = int(len(imgs) * TRAIN_SPLIT)

    for img in imgs[:n_train]:
        shutil.move(str(img), str(train_imgs / img.name))
        lbl = val_labels / (img.stem + ".txt")
        if lbl.exists():
            shutil.move(str(lbl), str(train_labels / lbl.name))

    print(f"Split criado: {n_train} treino | {len(imgs) - n_train} validação (seed={RANDOM_SEED})")


def create_data_yaml():
    content = (
        f"train: {(NEON_DIR / 'images' / 'train').as_posix()}\n"
        f"val: {(NEON_DIR / 'images' / 'val').as_posix()}\n\n"
        "nc: 1\n"
        'names: ["tree"]\n'
    )
    DATA_YAML.write_text(content, encoding="utf-8")
    print(f"data.yaml: {DATA_YAML}")


def check_model_available(model_name):
    try:
        YOLO(model_name)
        return True
    except Exception as e:
        print(f"  {model_name} indisponível: {e}")
        return False


def train_model(model_name, device):
    print(f"\nTreinar {model_name}...")
    model = YOLO(model_name)
    results = model.train(
        data=str(DATA_YAML),
        epochs=30,
        imgsz=640,
        batch=4,
        device=device,
        name=f"neon_{model_name.replace('.pt', '')}",
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        cos_lr=True,
        patience=10,
        degrees=0.0,
        flipud=0.5,
        fliplr=0.5,
        mosaic=1.0,
        scale=0.5,
        cache=False,
        workers=0,
        save=True,
        save_period=10,
        plots=True,
        verbose=True,
    )
    return results


def print_metrics(model_name, results):
    m = results.results_dict
    p     = m.get("metrics/precision(B)", 0.0)
    r     = m.get("metrics/recall(B)", 0.0)
    map50 = m.get("metrics/mAP50(B)", 0.0)
    map95 = m.get("metrics/mAP50-95(B)", 0.0)
    print(f"  {model_name}: P={p:.4f}  R={r:.4f}  mAP50={map50:.4f}  mAP50-95={map95:.4f}")


def main():
    os.chdir(BASE)

    if torch.cuda.is_available():
        device = 0
        print(f"GPU: {torch.cuda.get_device_name(0)}  ({torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB)")
    else:
        device = "cpu"
        print("AVISO: sem GPU — treino em CPU")

    ensure_split()
    create_data_yaml()

    available = [m for m in MODELS_TO_TRY if check_model_available(m)]
    if not available:
        print("ERRO: nenhum modelo YOLO disponível")
        sys.exit(1)

    print(f"Modelos: {available}")
    all_results = []

    for model_name in available:
        try:
            results = train_model(model_name, device)
            print_metrics(model_name, results)
            all_results.append((model_name, results.results_dict))
        except Exception as e:
            print(f"ERRO ao treinar {model_name}: {e}")

    print("\nResumo:")
    print(f"{'Modelo':<14}{'Precision':>12}{'Recall':>10}{'mAP50':>10}{'mAP50-95':>12}")
    for name, m in all_results:
        p     = m.get("metrics/precision(B)", 0.0)
        r     = m.get("metrics/recall(B)", 0.0)
        map50 = m.get("metrics/mAP50(B)", 0.0)
        map95 = m.get("metrics/mAP50-95(B)", 0.0)
        print(f"{name.replace('.pt',''):<14}{p:>12.4f}{r:>10.4f}{map50:>10.4f}{map95:>12.4f}")

    print(f"\nPesos em: {BASE / 'runs' / 'detect'}/")
    print("Para avaliação completa: python scripts/eval_all_models.py")


if __name__ == "__main__":
    main()
