from deepforest import main
import cv2
import os

model = main.deepforest()
model.load_model()  

input_folder = "dataset/images/val"
output_folder = "results_deepforest"

os.makedirs(output_folder, exist_ok=True)

for file in os.listdir(input_folder)[:10]:
    path = os.path.join(input_folder, file)

    boxes = model.predict_image(path=path)

 
    image = cv2.imread(path)

    count = 0

    if boxes is not None:
        for _, row in boxes.iterrows():
            x1, y1, x2, y2 = int(row.xmin), int(row.ymin), int(row.xmax), int(row.ymax)

            cv2.rectangle(image, (x1, y1), (x2, y2), (0,255,0), 2)
            count += 1

  
    out_path = os.path.join(output_folder, file)
    cv2.imwrite(out_path, image)

    print(f"{file} → {count} árvores")

print("\nResultados guardados em 'results_deepforest'")