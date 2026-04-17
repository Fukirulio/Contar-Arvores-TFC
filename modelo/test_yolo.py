from ultralytics import YOLO
import cv2
import os

model = YOLO("runs/detect/tree_detection2/weights/best.pt")

input_folder = "dataset/images/val"
output_folder = "results"

os.makedirs(output_folder, exist_ok=True)

for file in os.listdir(input_folder)[:10]:
    path = os.path.join(input_folder, file)

    results = model(path)

    # desenhar resultados
    img = results[0].plot()

    # contar árvores
    num_trees = len(results[0].boxes)

    # salvar
    out_path = os.path.join(output_folder, file)
    cv2.imwrite(out_path, img)

    print(f"{file} → {num_trees} árvores")

print("\nResultados guardados na pasta 'results'")