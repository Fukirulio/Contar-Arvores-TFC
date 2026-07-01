import io
import os
from datetime import datetime
from pathlib import Path

import torch
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from rasterio.io import MemoryFile
from ultralytics import YOLO
from deepforest import main as df_main

BASE = Path(__file__).resolve().parent.parent
YOLO_MODEL_PATH = str(BASE / "runs" / "detect" / "tree_neon_v12s_100ep" / "weights" / "best.pt")
DF_FINETUNED_PATH = BASE / "models" / "deepforest_neon_finetuned.pt"
GT_CSV_PATH = BASE / "neon_ground_truth.csv"
RESULTS_FOLDER = "results"

YOLO_COLOR = (0, 0, 255)
DEEPFOREST_COLOR = (0, 200, 0)
DF_FINE_COLOR = (220, 100, 0)

GITHUB_URL = "https://github.com/Fukirulio/Contar-Arvores-TFC"

OPCOES_MODELO = [
    "Ambos (YOLO12s vs DeepForest fine-tuned)",
    "YOLO12s NEON (100 epocas)",
    "DeepForest zero-shot",
    "DeepForest fine-tuned NEON",
]

st.set_page_config(page_title="TreeDetect", page_icon="🌳", layout="wide")

st.markdown(
    """
    <style>
    .main-header {
        background: linear-gradient(90deg, #0f3d2e 0%, #1f6f4f 60%, #2e9c63 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
    }
    .main-header h1 { color: #ffffff; margin: 0; font-size: 2.1rem; }
    .main-header p  { color: #d7f5e3; margin: 0.2rem 0 0 0; font-size: 1rem; }
    div[data-testid="stMetric"] {
        background-color: rgba(46, 156, 99, 0.08);
        border: 1px solid rgba(46, 156, 99, 0.3);
        border-radius: 10px;
        padding: 0.6rem 0.8rem;
    }
    .img-caption { text-align: center; font-weight: 600; padding: 0.3rem 0; }
    .img-caption-yolo        { color: #3366ff; }
    .img-caption-deepforest  { color: #1f9c46; }
    .img-caption-df-fine     { color: #cc5500; }
    .stTabs [data-baseweb="tab"] { font-size: 1.05rem; font-weight: 600; }
    </style>
    <div class="main-header">
        <h1>🌳 TreeDetect</h1>
        <p>Deteção e contagem automática de árvores em imagens aéreas — YOLO12s vs DeepForest</p>
    </div>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_ground_truth():
    if not GT_CSV_PATH.exists():
        return {}
    df = pd.read_csv(GT_CSV_PATH)
    return dict(zip(df["image_name"], df["ground_truth_count"].astype(int)))


def lookup_gt(filename):
    """Devolve o ground truth count para o nome de ficheiro, ou None se não for NEON."""
    stem = Path(filename).stem
    gt_map = load_ground_truth()
    return gt_map.get(stem + ".png") or gt_map.get(stem + ".PNG")


@st.cache_resource
def load_yolo():
    return YOLO(YOLO_MODEL_PATH)


@st.cache_resource
def load_deepforest_zero():
    model = df_main.deepforest()
    model.load_model()
    return model


@st.cache_resource
def load_deepforest_finetuned():
    model = df_main.deepforest()
    model.load_model()
    if DF_FINETUNED_PATH.exists():
        ckpt = torch.load(str(DF_FINETUNED_PATH), map_location="cpu")
        weights = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        model.model.load_state_dict(weights)
    return model


def load_image(uploaded_file):
    name = uploaded_file.name.lower()
    data = uploaded_file.read()
    if name.endswith((".tif", ".tiff")):
        with MemoryFile(data) as memfile:
            with memfile.open() as src:
                arr = src.read()
        if arr.shape[0] >= 3:
            arr = np.transpose(arr[:3], (1, 2, 0))
        else:
            arr = np.repeat(arr[0][:, :, None], 3, axis=2)
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)
        return arr
    image = Image.open(io.BytesIO(data)).convert("RGB")
    return np.array(image)


def run_yolo(model, image_rgb):
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    results = model(image_bgr, verbose=False)
    return results[0].boxes.xyxy.cpu().numpy().tolist()


def run_deepforest(model, image_rgb):
    boxes_df = model.predict_image(image=image_rgb.astype("float32"))
    if boxes_df is None or boxes_df.empty:
        return []
    return boxes_df[["xmin", "ymin", "xmax", "ymax"]].values.tolist()


def draw_boxes(image_rgb, boxes, color):
    out = image_rgb.copy()
    for box in boxes:
        x1, y1, x2, y2 = map(int, box[:4])
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
    return out


def resize_for_display(image_rgb, zoom):
    h, w = image_rgb.shape[:2]
    return cv2.resize(image_rgb, (max(1, int(w * zoom)), max(1, int(h * zoom))))


def _needs_yolo(opcao):
    return opcao in ("Ambos (YOLO12s vs DeepForest fine-tuned)", "YOLO12s NEON (100 epocas)")

def _needs_df_zero(opcao):
    return opcao == "DeepForest zero-shot"

def _needs_df_fine(opcao):
    return opcao in ("Ambos (YOLO12s vs DeepForest fine-tuned)", "DeepForest fine-tuned NEON")


tab_deteccao, tab_metricas, tab_sobre = st.tabs(
    ["🔍 Deteção", "📊 Métricas e Benchmark", "ℹ️ Sobre o Projeto"]
)


with tab_deteccao:
    col_upload, col_options = st.columns([2, 1])

    with col_upload:
        uploaded_file = st.file_uploader(
            "Carregar imagem", type=["png", "jpg", "jpeg", "tif", "tiff"]
        )

    with col_options:
        modelo_escolhido = st.selectbox("Modelo", OPCOES_MODELO)
        zoom = st.slider("Zoom", 0.2, 1.0, 0.5)

    if uploaded_file is not None:
        gt_count = lookup_gt(uploaded_file.name)
        if gt_count is not None:
            st.success(f"Imagem NEON reconhecida — Ground truth: **{gt_count}** arvores anotadas manualmente")

        try:
            image_rgb = load_image(uploaded_file)
        except Exception as e:
            st.error(f"Não foi possível ler a imagem: {e}")
            image_rgb = None

        if image_rgb is not None:
            progress = st.progress(0, text="A carregar modelos...")

            try:
                yolo_model     = load_yolo()             if _needs_yolo(modelo_escolhido)    else None
                progress.progress(35, text="A carregar DeepForest zero-shot...")
                df_zero_model  = load_deepforest_zero()  if _needs_df_zero(modelo_escolhido)  else None
                progress.progress(65, text="A carregar DeepForest fine-tuned...")
                df_fine_model  = load_deepforest_finetuned() if _needs_df_fine(modelo_escolhido) else None
                progress.progress(85, text="A processar imagem...")
            except Exception as e:
                progress.empty()
                st.error(f"Erro ao carregar modelos: {e}")
                st.stop()

            yolo_boxes, df_zero_boxes, df_fine_boxes = [], [], []

            with st.spinner("A detetar arvores..."):
                try:
                    if yolo_model is not None:
                        yolo_boxes = run_yolo(yolo_model, image_rgb)
                except Exception as e:
                    st.error(f"Erro YOLO: {e}")
                try:
                    if df_zero_model is not None:
                        df_zero_boxes = run_deepforest(df_zero_model, image_rgb)
                except Exception as e:
                    st.error(f"Erro DeepForest zero-shot: {e}")
                try:
                    if df_fine_model is not None:
                        df_fine_boxes = run_deepforest(df_fine_model, image_rgb)
                except Exception as e:
                    st.error(f"Erro DeepForest fine-tuned: {e}")

            progress.empty()

            export_image = None

            if modelo_escolhido == "Ambos (YOLO12s vs DeepForest fine-tuned)":
                col1, col2 = st.columns(2)
                yolo_img = draw_boxes(image_rgb, yolo_boxes, YOLO_COLOR)
                df_fine_img = draw_boxes(image_rgb, df_fine_boxes, DF_FINE_COLOR)
                with col1:
                    st.markdown('<p class="img-caption img-caption-yolo">YOLO12s 100ep (NEON)</p>', unsafe_allow_html=True)
                    st.image(resize_for_display(yolo_img, zoom))
                with col2:
                    st.markdown('<p class="img-caption img-caption-df-fine">DeepForest fine-tuned (NEON)</p>', unsafe_allow_html=True)
                    st.image(resize_for_display(df_fine_img, zoom))
                export_image = np.hstack([yolo_img, df_fine_img])

                st.markdown("---")
                m1, m2, m3 = st.columns(3)
                diff = len(yolo_boxes) - len(df_fine_boxes)
                m1.metric("YOLO12s detectou", len(yolo_boxes))
                m2.metric("DeepForest fine-tuned detectou", len(df_fine_boxes))
                m3.metric("Diferença", abs(diff), delta=diff, delta_color="normal")

            elif modelo_escolhido == "YOLO12s NEON (100 epocas)":
                yolo_img = draw_boxes(image_rgb, yolo_boxes, YOLO_COLOR)
                st.markdown('<p class="img-caption img-caption-yolo">YOLO12s 100ep (NEON)</p>', unsafe_allow_html=True)
                st.image(resize_for_display(yolo_img, zoom))
                export_image = yolo_img
                st.markdown("---")
                st.metric("YOLO12s detectou", len(yolo_boxes))

            elif modelo_escolhido == "DeepForest zero-shot":
                df_zero_img = draw_boxes(image_rgb, df_zero_boxes, DEEPFOREST_COLOR)
                st.markdown('<p class="img-caption img-caption-deepforest">DeepForest zero-shot</p>', unsafe_allow_html=True)
                st.image(resize_for_display(df_zero_img, zoom))
                export_image = df_zero_img
                st.markdown("---")
                st.metric("DeepForest zero-shot detectou", len(df_zero_boxes))

            else:
                df_fine_img = draw_boxes(image_rgb, df_fine_boxes, DF_FINE_COLOR)
                st.markdown('<p class="img-caption img-caption-df-fine">DeepForest fine-tuned (NEON)</p>', unsafe_allow_html=True)
                st.image(resize_for_display(df_fine_img, zoom))
                export_image = df_fine_img
                st.markdown("---")
                st.metric("DeepForest fine-tuned detectou", len(df_fine_boxes))

            if gt_count is not None:
                st.markdown("---")
                st.markdown("### 📊 Comparação com Ground Truth")

                if modelo_escolhido == "Ambos (YOLO12s vs DeepForest fine-tuned)":
                    count_a, label_a = len(yolo_boxes),    "YOLO12s 100ep"
                    count_b, label_b = len(df_fine_boxes), "DeepForest fine-tuned"
                    col_gt, col_a, col_b = st.columns(3)
                    col_gt.metric("Ground Truth", gt_count)
                    col_a.metric(label_a, count_a, delta=count_a - gt_count, delta_color="inverse")
                    col_b.metric(label_b, count_b, delta=count_b - gt_count, delta_color="inverse")
                    err_a, err_b = abs(count_a - gt_count), abs(count_b - gt_count)
                    winner, winner_err = (label_a, err_a) if err_a <= err_b else (label_b, err_b)

                elif modelo_escolhido == "YOLO12s NEON (100 epocas)":
                    count_a, label_a = len(yolo_boxes), "YOLO12s 100ep"
                    col_gt, col_a = st.columns(2)
                    col_gt.metric("Ground Truth", gt_count)
                    col_a.metric(label_a, count_a, delta=count_a - gt_count, delta_color="inverse")
                    winner, winner_err = label_a, abs(count_a - gt_count)

                elif modelo_escolhido == "DeepForest zero-shot":
                    count_a, label_a = len(df_zero_boxes), "DeepForest zero-shot"
                    col_gt, col_a = st.columns(2)
                    col_gt.metric("Ground Truth", gt_count)
                    col_a.metric(label_a, count_a, delta=count_a - gt_count, delta_color="inverse")
                    winner, winner_err = label_a, abs(count_a - gt_count)

                else:
                    count_a, label_a = len(df_fine_boxes), "DeepForest fine-tuned"
                    col_gt, col_a = st.columns(2)
                    col_gt.metric("Ground Truth", gt_count)
                    col_a.metric(label_a, count_a, delta=count_a - gt_count, delta_color="inverse")
                    winner, winner_err = label_a, abs(count_a - gt_count)

                pct_err = winner_err / gt_count * 100 if gt_count > 0 else 0
                st.info(f"Modelo mais proximo do valor real: **{winner}** (erro de {winner_err} arvores, {pct_err:.1f}%)")

            st.markdown("---")
            if export_image is not None and st.button("💾 Exportar resultado"):
                os.makedirs(RESULTS_FOLDER, exist_ok=True)
                base_name = os.path.splitext(uploaded_file.name)[0]
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                out_path = os.path.join(RESULTS_FOLDER, f"app_{base_name}_{timestamp}.png")
                cv2.imwrite(out_path, cv2.cvtColor(export_image, cv2.COLOR_RGB2BGR))
                st.success(f"Resultado guardado em: {out_path}")
    else:
        st.info("Carrega uma imagem (png, jpg ou tif) para começar a deteção.")


with tab_metricas:
    st.markdown("## Resultados Quantitativos — NEON Benchmark (39 imagens val)")
    st.caption("Avaliação com IoU=0.5 no conjunto val do NEON Tree Evaluation Dataset.")

    benchmark_df = pd.DataFrame(
        [
            ["DeepForest fine-tuned", "NEON (fine-tune)", "NEON val", 0.635, 0.675, 0.654, 8.08],
            ["DeepForest zero-shot",  "NEON (pre-treino)", "NEON val", 0.590, 0.626, 0.608, 8.87],
            ["YOLO12s 30ep",          "NEON",              "NEON val", 0.635, 0.495, 0.556, 8.44],
            ["YOLOv8s 30ep",          "NEON",              "NEON val", 0.721, 0.434, 0.542, 12.72],
            ["YOLO11s 30ep",          "NEON",              "NEON val", 0.712, 0.399, 0.512, 14.00],
            ["YOLOv8s (OAM)",         "OAM-TCD",           "NEON val", 0.209, 0.007, 0.014, 30.05],
        ],
        columns=["Modelo", "Treino", "Teste", "Precision", "Recall", "F1", "MAE"],
    )

    st.dataframe(
        benchmark_df.style.format(
            {"Precision": "{:.3f}", "Recall": "{:.3f}", "F1": "{:.3f}", "MAE": "{:.2f}"}
        ).background_gradient(subset=["F1"], cmap="Greens"),
        width="stretch",
        hide_index=True,
    )

    st.markdown("### F1-score por modelo")
    chart_df = pd.DataFrame(
        {"F1": [0.654, 0.608, 0.556, 0.542, 0.512, 0.014]},
        index=[
            "DF fine-tuned",
            "DF zero-shot",
            "YOLO12s 30ep",
            "YOLOv8s 30ep",
            "YOLO11s 30ep",
            "YOLOv8s OAM",
        ],
    )
    st.bar_chart(chart_df)

    st.markdown("---")
    st.markdown("### Conclusões")
    st.markdown(
        """
        - **DeepForest fine-tuned** é o melhor modelo global (F1=0.654, MAE=8.08), superando o zero-shot em +0.046 F1.
        - **YOLO12s 30ep** tem o melhor MAE entre os modelos YOLO (8.44), mas recall mais baixo que o DeepForest.
        - **YOLOv8s treinado no OAM** generaliza muito mal para o domínio NEON (F1=0.014) — evidência clara de domain shift.
        - O fine-tuning do DeepForest com apenas 15 épocas e 155 imagens melhorou ambas as métricas.
        """
    )


with tab_sobre:
    st.markdown("# Contagem Automática de Árvores em Imagens Aéreas")
    st.markdown("#### Trabalho Final de Curso — Universidade Lusófona 2025/26")

    st.markdown(
        """
        **Autor:** Victor Lopes Soares de Oliveira (22300040)
        **Orientador:** Prof. Luís Campos
        """
    )

    st.markdown("---")
    st.markdown(
        """
        - Deteção e contagem automática de árvores em ortomosaicos aéreos usando deep learning.
        - Comparação entre modelos YOLO (treinados no NEON) e DeepForest (zero-shot e fine-tuned).
        - Avaliação cruzada entre domínios (OAM-TCD e NEON) para medir capacidade de generalização.
        """
    )

    st.markdown("### Datasets")
    st.markdown(
        """
        - **OAM-TCD** — imagens aéreas de baixa altitude (OpenAerialMap Tree Crown Detection).
        - **NEON Tree Evaluation** — benchmark do National Ecological Observatory Network (EUA),
          tiles 400×400 px de múltiplos biomas temperados e boreais.
        """
    )

    st.markdown("### Modelos")
    st.markdown(
        """
        - **YOLO12s 100ep** — treinado no dataset NEON (100 épocas), mAP50=0.676.
        - **DeepForest zero-shot** — pré-treinado pela equipa Weecology no NEON (sem adaptação).
        - **DeepForest fine-tuned** — fine-tune de 15 épocas no split NEON local, F1=0.654 (melhor modelo).
        """
    )

    st.markdown("---")
    st.markdown(f"[Repositório no GitHub]({GITHUB_URL})")
