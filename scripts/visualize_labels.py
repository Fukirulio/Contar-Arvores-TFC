import os
import cv2

images_folder = "dataset/images/train"
labels_folder = "dataset/labels/train"
output_folder = "dataset/preview"

os.makedirs(output_folder, exist_ok=True)

files = os.listdir(images_folder)[:5]

for file in files:
    img_path = os.path.join(images_folder, file)
    label_path = os.path.join(labels_folder, file.replace(".png", ".txt"))

    image = cv2.imread(img_path)
    h, w, _ = image.shape

    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            for line in f.readlines():
                cls, x, y, bw, bh = map(float, line.split())

                x1 = int((x - bw/2) * w)
                y1 = int((y - bh/2) * h)
                x2 = int((x + bw/2) * w)
                y2 = int((y + bh/2) * h)

                cv2.rectangle(image, (x1, y1), (x2, y2), (0,255,0), 2)

    out_path = os.path.join(output_folder, file)
    cv2.imwrite(out_path, image)

    print(f"Preview gerado: {file}")

print("\nAbre a pasta dataset/preview para ver os resultados")