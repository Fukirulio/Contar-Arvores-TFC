import os
import glob
import cv2
from ultralytics import YOLO

if __name__ == "__main__":

    # encontrar o run mais recente automaticamente
    runs = sorted(glob.glob("runs/detect/tree_detection*/weights/best.pt"))
    if not runs:
        raise FileNotFoundError("Nenhum modelo treinado encontrado em runs/detect/")

    model_path = runs[-1]
    run_name = model_path.split(os.sep)[2]
    print(f"Modelo carregado: {model_path}  (run: {run_name})")

    model = YOLO(model_path)

    # --- métricas de validação ---
    print("\n--- A calcular métricas no conjunto de validação ---")
    metrics = model.val(data="data.yaml", imgsz=640, verbose=True)

    print("\n=== RESULTADOS ===")
    print(f"mAP50      : {metrics.box.map50:.4f}")
    print(f"mAP50-95   : {metrics.box.map:.4f}")
    print(f"Precision  : {metrics.box.mp:.4f}")
    print(f"Recall     : {metrics.box.mr:.4f}")

    # --- visualizações em imagens de validação ---
    input_folder = "dataset/images/val"
    output_folder = f"results/{run_name}"
    os.makedirs(output_folder, exist_ok=True)

    val_files = [f for f in os.listdir(input_folder) if f.endswith(".png")][:20]

    print(f"\n--- A gerar visualizações para {len(val_files)} imagens ---")

    total_trees = 0
    for file in val_files:
        path = os.path.join(input_folder, file)
        results = model(path, verbose=False)

        img = results[0].plot()
        num_trees = len(results[0].boxes)
        total_trees += num_trees

        # escrever contagem na imagem
        cv2.putText(img, f"Arvores: {num_trees}", (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        out_path = os.path.join(output_folder, file)
        cv2.imwrite(out_path, img)
        print(f"  {file} → {num_trees} árvores")

    print(f"\nTotal de árvores detetadas: {total_trees}")
    print(f"Média por imagem          : {total_trees / len(val_files):.1f}")
    print(f"Resultados guardados em   : {output_folder}/")
