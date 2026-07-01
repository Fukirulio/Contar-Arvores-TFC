import os
import numpy as np
import cv2
import rasterio as rio
import random

images_folder = "oam/images"
masks_folder = "oam/masks"

dataset_images_train = "dataset/images/train"
dataset_images_val = "dataset/images/val"
dataset_labels_train = "dataset/labels/train"
dataset_labels_val = "dataset/labels/val"

for p in [dataset_images_train, dataset_images_val, dataset_labels_train, dataset_labels_val]:
    os.makedirs(p, exist_ok=True)


def load_tif(path):
    with rio.open(path) as src:
        img = src.read()
        if img.shape[0] >= 3:
            img = np.transpose(img[:3], (1, 2, 0))
        else:
            img = img[0]
    return img


image_files = [f for f in os.listdir(images_folder) if f.endswith(".tif")]

files = []
for img in image_files:
    mask_name = img.replace(".tif", ".png")
    if os.path.exists(os.path.join(masks_folder, mask_name)):
        files.append(img)

print(f"Pares válidos: {len(files)}")

random.shuffle(files)
split = int(len(files) * 0.8)
train_files = files[:split]
val_files = files[split:]


def process(files, img_out, label_out):
    for file in files:
        image_path = os.path.join(images_folder, file)
        mask_path = os.path.join(masks_folder, file.replace(".tif", ".png"))

        image = load_tif(image_path)
        mask = cv2.imread(mask_path, 0)

        if image is None or mask is None:
            print(f"Erro ao carregar: {file}")
            continue

        h, w = mask.shape
        mask = (mask > 0).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cv2.imwrite(os.path.join(img_out, file.replace(".tif", ".png")), image)

        with open(os.path.join(label_out, file.replace(".tif", ".txt")), "w") as f:
            for cnt in contours:
                x, y, bw, bh = cv2.boundingRect(cnt)
                # ignora contornos de ruído abaixo de 10 px em qualquer dimensão
                if bw < 10 or bh < 10:
                    continue
                f.write(f"0 {(x + bw/2)/w} {(y + bh/2)/h} {bw/w} {bh/h}\n")


process(train_files, dataset_images_train, dataset_labels_train)
process(val_files, dataset_images_val, dataset_labels_val)
print(f"Treino: {len(train_files)}  Val: {len(val_files)}")
