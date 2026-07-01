import os
import csv
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import rasterio as rio

ANNOTATIONS_FOLDER = "evaluation/annotations"
IMAGES_FOLDER = "evaluation/RGB"

CSV_OUTPUT = "neon_deepforest.csv"
YOLO_LABELS_FOLDER = "neon_yolo/labels/val"
YOLO_IMAGES_FOLDER = "neon_yolo/images/val"

DEFAULT_LABEL = "Tree"
CLASS_ID = 0


def load_tif(path):
    with rio.open(path) as src:
        img = src.read()
    if img.shape[0] >= 3:
        img = np.transpose(img[:3], (1, 2, 0))
    else:
        img = np.repeat(img[0][:, :, None], 3, axis=2)
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
    return img


def parse_voc_annotation(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    filename = root.findtext("filename")
    if not filename:
        filename = os.path.splitext(os.path.basename(xml_path))[0] + ".tif"
    boxes = []
    for obj in root.findall("object"):
        label = obj.findtext("name") or DEFAULT_LABEL
        bndbox = obj.find("bndbox")
        if bndbox is None:
            continue
        try:
            xmin = float(bndbox.findtext("xmin"))
            ymin = float(bndbox.findtext("ymin"))
            xmax = float(bndbox.findtext("xmax"))
            ymax = float(bndbox.findtext("ymax"))
        except (TypeError, ValueError):
            continue
        boxes.append((xmin, ymin, xmax, ymax, label))
    return filename, boxes


def find_image_file(filename, xml_file):
    """Tenta o nome do XML referencia; se não existir, tenta o nome base do ficheiro XML."""
    candidate = os.path.join(IMAGES_FOLDER, filename)
    if os.path.exists(candidate):
        return candidate
    fallback = os.path.join(IMAGES_FOLDER, os.path.splitext(xml_file)[0] + ".tif")
    return fallback if os.path.exists(fallback) else None


def main():
    os.makedirs(YOLO_LABELS_FOLDER, exist_ok=True)
    os.makedirs(YOLO_IMAGES_FOLDER, exist_ok=True)

    if not os.path.isdir(ANNOTATIONS_FOLDER):
        print(f"ERRO: pasta de anotações não encontrada: {ANNOTATIONS_FOLDER}")
        return
    if not os.path.isdir(IMAGES_FOLDER):
        print(f"ERRO: pasta de imagens não encontrada: {IMAGES_FOLDER}")
        return

    xml_files = sorted(f for f in os.listdir(ANNOTATIONS_FOLDER) if f.lower().endswith(".xml"))
    print(f"Anotações XML: {len(xml_files)}")

    csv_rows = []
    processed = 0
    skipped = 0

    for i, xml_file in enumerate(xml_files, start=1):
        xml_path = os.path.join(ANNOTATIONS_FOLDER, xml_file)
        try:
            filename, boxes = parse_voc_annotation(xml_path)
        except ET.ParseError as e:
            print(f"  [{i}/{len(xml_files)}] XML inválido '{xml_file}': {e}")
            skipped += 1
            continue

        if not boxes:
            print(f"  [{i}/{len(xml_files)}] '{xml_file}' sem anotações, ignorado")
            skipped += 1
            continue

        image_path = find_image_file(filename, xml_file)
        if image_path is None:
            print(f"  [{i}/{len(xml_files)}] Imagem não encontrada para '{xml_file}'")
            skipped += 1
            continue

        try:
            image = load_tif(image_path)
        except Exception as e:
            print(f"  [{i}/{len(xml_files)}] Erro ao ler '{image_path}': {e}")
            skipped += 1
            continue

        h, w = image.shape[:2]
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        png_name = image_name + ".png"

        cv2.imwrite(os.path.join(YOLO_IMAGES_FOLDER, png_name), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

        with open(os.path.join(YOLO_LABELS_FOLDER, image_name + ".txt"), "w") as f:
            for xmin, ymin, xmax, ymax, label in boxes:
                x_center = ((xmin + xmax) / 2) / w
                y_center = ((ymin + ymax) / 2) / h
                box_width = (xmax - xmin) / w
                box_height = (ymax - ymin) / h
                f.write(f"{CLASS_ID} {x_center} {y_center} {box_width} {box_height}\n")
                csv_rows.append([png_name, int(xmin), int(ymin), int(xmax), int(ymax), label])

        processed += 1
        print(f"  [{i}/{len(xml_files)}] {xml_file} -> {len(boxes)} árvores")

    with open(CSV_OUTPUT, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "xmin", "ymin", "xmax", "ymax", "label"])
        writer.writerows(csv_rows)

    print(f"\nProcessadas: {processed}  |  Ignoradas: {skipped}  |  Anotações: {len(csv_rows)}")
    print(f"CSV: {CSV_OUTPUT}  |  Labels: {YOLO_LABELS_FOLDER}/  |  Imagens: {YOLO_IMAGES_FOLDER}/")


if __name__ == "__main__":
    main()
