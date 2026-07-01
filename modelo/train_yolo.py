import torch
from ultralytics import YOLO

if __name__ == "__main__":
    # verificar GPU
    if torch.cuda.is_available():
        device = 0
        print(f"GPU encontrada: {torch.cuda.get_device_name(0)}")
        print(f"VRAM disponível: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        device = "cpu"
        print("AVISO: GPU não encontrada, a correr em CPU (muito lento)")

    model = YOLO("yolov8s.pt")  # 's' em vez de 'n' — melhor precisão, ainda leve

    model.train(
        data="data.yaml",
        epochs=50,
        imgsz=640,
        batch=8,
        device=device,
        name="tree_detection",

        # otimizador e learning rate
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,          # lr final = lr0 * lrf
        cos_lr=True,       # cosine annealing schedule

        # early stopping — para se não melhorar em 15 épocas
        patience=15,

        # augmentação (útil para imagens aéreas)
        degrees=0.0,       # sem rotação aleatória (aérea já é top-down)
        flipud=0.5,        # flip vertical
        fliplr=0.5,        # flip horizontal
        mosaic=1.0,        # mosaic augmentation
        scale=0.5,         # scale jitter

        # performance
        cache=False,       # mudar para True se tiveres RAM suficiente (>= 32GB)
        workers=4,

        # guardar melhor e último checkpoint
        save=True,
        save_period=10,    # checkpoint a cada 10 épocas

        # logs
        plots=True,        # gera gráficos de loss/metrics
        verbose=True,
    )
