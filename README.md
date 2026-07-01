# Contagem Automática de Árvores em Imagens Aéreas

Trabalho Final de Curso — Licenciatura em Ciência de Dados
Universidade Lusófona | 2025/2026
Autor: Victor Lopes Soares de Oliveira (22300040)
Orientador: Prof. Luís Campos

---

## Sobre o projecto

Pipeline de Ciência de Dados para deteção e contagem automática de árvores individuais em imagens aéreas RGB de alta resolução. Compara cinco variantes da família YOLO (YOLOv8s, YOLO11s, YOLO12s) com o modelo especializado DeepForest, treinados e avaliados no benchmark NeonTreeEvaluation.

---

## Resultados principais

**NeonTreeEvaluation — 39 imagens de validação, IoU=0.5**

| Modelo               | Precision | Recall | F1    | MAE   |
|----------------------|-----------|--------|-------|-------|
| DeepForest fine-tuned| 0.635     | 0.675  | 0.654 |  8.08 |
| YOLO12s 100ep        | 0.658     | 0.647  | 0.653 |  4.59 |
| DeepForest zero-shot | 0.590     | 0.626  | 0.608 |  8.87 |
| YOLO12s 30ep         | 0.635     | 0.495  | 0.556 |  8.44 |
| YOLOv8s 30ep         | 0.721     | 0.434  | 0.542 | 12.72 |
| YOLO11s 30ep         | 0.712     | 0.399  | 0.512 | 14.00 |
| YOLOv8s (OAM→NEON)  | 0.209     | 0.007  | 0.014 | 30.05 |

O YOLO12s com 100 épocas atinge **R²=0.958** na tarefa de contagem, com MAE de 4.59 árvores por imagem de 400×400 pixels. A diferença de MAE entre treino (4.70) e validação (4.59) é de 2%, confirmando generalização sem overfitting.

---

## Datasets

**OAM-TCD** (NeurIPS 2024) — 3 764 imagens 2048×2048 px com máscaras de segmentação binárias, usado para treino exploratório inicial.
Disponível: https://github.com/Restor-Foundation/tcd

**NeonTreeEvaluation** (PLOS Computational Biology 2021) — 194 imagens 400×400 px com 6 633 bounding boxes manuais em 22 sites ecológicos dos EUA (biomas temperados, mediterrânicos, boreais e tropicais), usado para treino supervisionado e benchmark de validação.
Disponível: https://zenodo.org/records/5914554

---

## Estrutura do repositório

```
├── modelo/
│   └── app.py                         # Aplicação Streamlit (4 modelos, 3 tabs)
├── scripts/
│   ├── oam_to_yolo.py                 # OAM-TCD: extrai bboxes de máscaras → formato YOLO
│   ├── convert_neon.py                # NEON: converte XML Pascal VOC + .tif → PNG + YOLO
│   ├── train_yolo_neon.py             # Treina YOLOv8s, YOLO11s, YOLO12s no NEON (30ep)
│   ├── train_yolo12s_100ep.py         # Treina YOLO12s no NEON (100ep, batch=8)
│   ├── finetune_deepforest_neon.py    # Fine-tune DeepForest 15 épocas no NEON
│   ├── eval_all_models.py             # Avalia todos os modelos (IoU=0.5, P/R/F1/MAE)
│   ├── eval_neon.py                   # Avaliação YOLO OAM + DeepForest zero-shot no NEON
│   ├── demo_comparativo.py            # Comparação YOLO12s vs DeepForest numa imagem
│   ├── comparacao_4modelos.py         # 4 modelos × 3 imagens demo → PNG lado a lado
│   ├── generate_curves.py             # Curvas de treino (loss e métricas por época)
│   ├── generate_eda.py                # Figura de análise exploratória (4 subplots)
│   ├── generate_validation_plot.py    # Scatter GT vs predição + MAE e R²
│   ├── overfitting_check.py           # Compara MAE/R² em treino vs validação
│   └── visualize_labels.py            # Visualização das anotações YOLO sobre imagens
├── demo/
│   ├── demo_neon_alta_densidade.png   # NIWO_011_2018 (GT=153 árvores)
│   ├── demo_neon_media_densidade.png  # TEAK_049_2018 (GT=25 árvores)
│   └── demo_neon_floresta.png         # BONA_012_2019 (GT=82 árvores)
├── results/
│   ├── comparacao_4modelos_neon_*.png # Comparações visuais 4 modelos × 3 demos
│   ├── curvas_treino_yolo12s.png      # Curvas de treino YOLO12s 100ep
│   ├── eda_datasets.png               # Análise exploratória (histogramas, sites, datasets)
│   └── validation_predicted_vs_actual.png  # Scatter GT vs predição
├── evaluation/
│   └── annotations/                   # 226 ficheiros XML Pascal VOC (NEON)
├── neon_yolo/
│   ├── labels/train/                  # 155 ficheiros .txt YOLO (treino)
│   ├── labels/val/                    # 39 ficheiros .txt YOLO (validação)
│   └── data.yaml                      # Configuração do dataset NEON para YOLO
├── models/
│   └── README.md                      # Instruções para obter/re-treinar o DeepForest
├── neon_deepforest.csv                # Ground truth NEON em formato DeepForest
├── neon_ground_truth.csv              # Contagem GT por imagem (para lookup na app)
├── neon_train.csv / neon_val.csv      # Splits para fine-tune do DeepForest
├── data.yaml                          # Configuração dataset OAM para YOLO
├── data_neon.yaml                     # Configuração dataset NEON (path absoluto)
└── requirements.txt
```

---

## Instalação

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

---

## Correr a aplicação

```bash
streamlit run modelo/app.py
```

Abre em `http://localhost:8501` com três separadores:

- **Deteção** — upload de imagem (PNG/JPG/TIF), selecção do modelo, visualização das bounding boxes e comparação automática com ground truth para imagens NEON
- **Métricas e Benchmark** — tabela comparativa dos 6 modelos com gradiente de cor e gráfico de barras F1
- **Sobre o Projecto** — descrição, datasets e link para o repositório

---

## Modelos

Os pesos YOLO ficam em `runs/detect/` (não incluídos no repositório — re-treinar com os scripts):

```bash
python scripts/train_yolo_neon.py          # YOLOv8s, YOLO11s, YOLO12s × 30ep
python scripts/train_yolo12s_100ep.py      # YOLO12s × 100ep (melhor configuração)
```

O modelo DeepForest fine-tuned (245 MB) não está incluído. Ver `models/README.md`.

```bash
python scripts/finetune_deepforest_neon.py  # Fine-tune 15 épocas no NEON
```

---

## Reproduzir a avaliação

```bash
# Avaliação completa de todos os modelos
python scripts/eval_all_models.py

# Verificação de overfitting
python scripts/overfitting_check.py

# Gerar figuras para o relatório
python scripts/generate_eda.py
python scripts/generate_curves.py
python scripts/generate_validation_plot.py
python scripts/comparacao_4modelos.py
```
