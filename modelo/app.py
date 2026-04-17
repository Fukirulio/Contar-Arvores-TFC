import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from deepforest import main
from PIL import Image

st.set_page_config(layout="wide")
st.title("Detecção e Contagem de Árvores")


@st.cache_resource
def load_yolo():
    return YOLO("runs/detect/tree_detection2/weights/best.pt")

@st.cache_resource
def load_deepforest():
    model = main.deepforest()
    model.load_model()
    return model

yolo_model = load_yolo()
deepforest_model = load_deepforest()


uploaded_file = st.file_uploader("Carregar imagem", type=["png", "jpg", "jpeg"])

zoom = st.slider("Zoom", 0.2, 1.0, 0.5)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("YOLO (Treinado)")

        results = yolo_model(image_np)
        yolo_img = results[0].plot()
        yolo_count = len(results[0].boxes)

        resized = cv2.resize(
            yolo_img,
            (int(yolo_img.shape[1]*zoom), int(yolo_img.shape[0]*zoom))
        )

        st.image(resized, channels="BGR")
        st.success(f"Árvores detectadas: {yolo_count}")


    with col2:
        st.subheader("DeepForest (Pré-treinado)")

        df_boxes = deepforest_model.predict_image(image=image_np)

        df_img = image_np.copy()
        df_count = 0

        if df_boxes is not None:
            for _, row in df_boxes.iterrows():
                x1, y1, x2, y2 = int(row.xmin), int(row.ymin), int(row.xmax), int(row.ymax)
                cv2.rectangle(df_img, (x1, y1), (x2, y2), (0,255,0), 2)
                df_count += 1

        resized_df = cv2.resize(
            df_img,
            (int(df_img.shape[1]*zoom), int(df_img.shape[0]*zoom))
        )

        st.image(resized_df)
        st.info(f"Árvores detectadas: {df_count}")

    st.markdown("---")
    st.subheader("Comparação")

    diff = yolo_count - df_count

    st.write(f"YOLO: {yolo_count} árvores")
    st.write(f"DeepForest: {df_count} árvores")

    if diff > 0:
        st.write(f"YOLO detectou mais {diff} árvores")
    elif diff < 0:
        st.write(f"DeepForest detectou mais {abs(diff)} árvores")
    else:
        st.write("Ambos detectaram o mesmo número de árvores")