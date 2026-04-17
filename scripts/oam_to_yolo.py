import os
import numpy as np
import cv2
import rasterio as rio
import random

# ========================
# CAMINHOS
# ========================
images_folder = "oam/images"
masks_folder = "oam/masks"

dataset_images_train = "dataset/images/train"
dataset_images_val = "dataset/images/val"
dataset_labels_train = "dataset/labels/train"
dataset_labels_val = "dataset/labels/val"

# ========================
# CRIAR PASTAS
# ========================
for p in [dataset_images_train, dataset_images_val, dataset_labels_train, dataset_labels_val]:
    os.makedirs(p, exist_ok=True)

# ========================
# FUNÇÃO PARA LER TIFF
# ========================
def load_tif(path):
    with rio.open(path) as src:
        img = src.read()

        # usar apenas 3 bandas
        if img.shape[0] >= 3:
            img = np.transpose(img[:3], (1, 2, 0))
        else:
            img = img[0]

    return img

# ========================
# ENCONTRAR PARES (tif + png)
# ========================
image_files = [f for f in os.listdir(images_folder) if f.endswith(".tif")]

files = []
for img in image_files:
    mask_name = img.replace(".tif", ".png")

    if os.path.exists(os.path.join(masks_folder, mask_name)):
        files.append(img)

print(f"Pares válidos encontrados: {len(files)}")

# embaralhar
random.shuffle(files)

# split 80/20
split = int(len(files) * 0.8)
train_files = files[:split]
val_files = files[split:]

# ========================
# PROCESSAMENTO
# ========================
def process(files, img_out, label_out):
    for file in files:

        image_path = os.path.join(images_folder, file)
        mask_name = file.replace(".tif", ".png")
        mask_path = os.path.join(masks_folder, mask_name)

        # carregar dados
        image = load_tif(image_path)
        mask = cv2.imread(mask_path, 0)

        if image is None or mask is None:
            print(f"Erro ao carregar: {file}")
            continue

        h, w = mask.shape

        # binarizar mask
        mask = (mask > 0).astype(np.uint8) * 255

        # encontrar árvores (contornos)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # caminhos saída
        image_out_path = os.path.join(img_out, file.replace(".tif", ".png"))
        label_path = os.path.join(label_out, file.replace(".tif", ".txt"))

        # guardar imagem
        cv2.imwrite(image_out_path, image)

        # criar label YOLO
        with open(label_path, "w") as f:
            for cnt in contours:
                x, y, bw, bh = cv2.boundingRect(cnt)

                # remover ruído
                if bw < 10 or bh < 10:
                    continue

                x_center = (x + bw / 2) / w
                y_center = (y + bh / 2) / h
                width = bw / w
                height = bh / h

                f.write(f"0 {x_center} {y_center} {width} {height}\n")

        print(f"{file} processado")

# ========================
# EXECUTAR
# ========================
process(train_files, dataset_images_train, dataset_labels_train)
process(val_files, dataset_images_val, dataset_labels_val)

print("\nDataset YOLO pronto!")