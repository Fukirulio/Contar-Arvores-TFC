# Contagem Automática de Árvores em Imagens Aéreas

Trabalho Final de Curso — Licenciatura em Ciência de Dados  
Universidade Lusófona | 2025/2026  
Autor: Victor Lopes Soares de Oliveira (22300040)  
Orientador: Prof. Luís Campos

---

## Descrição

Pipeline de deteção e contagem automática de árvores em imagens aéreas, com comparação entre dois modelos:

- **YOLOv8n** — treinado com o dataset OAM-TCD
- **DeepForest** — modelo pré-treinado em modo zero-shot

A solução inclui uma aplicação web em Streamlit que permite carregar uma imagem aérea e visualizar as deteções de ambos os modelos lado a lado.

---

## Estrutura do repositório

```
├── modelo/
│   ├── app.py                  # Aplicação Streamlit
│   ├── train_yolo.py           # Treino do modelo YOLOv8n
│   ├── test_yolo.py            # Teste do YOLOv8n em imagens de validação
│   └── test_deepforest.py      # Teste do DeepForest em imagens de validação
├── scripts/
│   ├── prepare_dataset.py      # Pré-processamento OAM-TCD → formato YOLO
│   └── preview_dataset.py      # Visualização das anotações geradas
├── data.yaml                   # Configuração do dataset para YOLOv8
├── requirements.txt
└── README.md
```

---

## Instalação

```bash
# Clonar o repositório
git clone https://github.com/SEU_USERNAME/tree-detection-tfc.git
cd tree-detection-tfc

# Criar ambiente virtual
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/Mac

# Instalar dependências
pip install -r requirements.txt
```

---

## Como usar

### 1. Preparar o dataset

Coloca as imagens TIF na pasta `oam/images/` e as máscaras PNG em `oam/masks/`.  
Depois corre:

```bash
python scripts/prepare_dataset.py
```

Isto gera automaticamente as anotações YOLO em `dataset/images/` e `dataset/labels/`.

### 2. Treinar o modelo

```bash
python modelo/train_yolo.py
```

O modelo treinado ficará em `runs/detect/tree_detection/weights/best.pt`.

### 3. Testar os modelos individualmente

```bash
python modelo/test_yolo.py
python modelo/test_deepforest.py
```

Os resultados ficam nas pastas `results/` e `results_deepforest/`.

### 4. Correr a aplicação Streamlit

**Importante:** antes de correr a app, certifica-te de que o caminho do modelo treinado em `app.py` está correto:
```python
YOLO("runs/detect/tree_detection2/weights/best.pt")
```

Depois:
```bash
streamlit run modelo/app.py
```

A aplicação abre automaticamente no browser em `http://localhost:8501`.

---

## Dataset

- **OAM-TCD** (Open Aerial Map Tree Crown Detection)
- 4608 imagens TIF de 2048×2048 pixels com máscaras de segmentação PNG
- Divisão: 80% treino / 20% validação
- As anotações YOLO são geradas automaticamente a partir das máscaras via `scripts/prepare_dataset.py`

**Nota:** as imagens e máscaras originais não estão incluídas no repositório devido ao seu tamanho. O dataset está disponível em [https://openaerialmap.org](https://openaerialmap.org).

---

## Configuração de treino

| Parâmetro | Valor |
|-----------|-------|
| Modelo base | YOLOv8n (COCO pretrained) |
| Épocas | 5–10 |
| Resolução | 640px |
| Batch size | 8 |
| Device | GPU (CUDA) |

---

## Dependências principais

| Biblioteca | Uso |
|-----------|-----|
| ultralytics | YOLOv8 treino e inferência |
| deepforest-pytorch | Modelo DeepForest zero-shot |
| streamlit | Interface web |
| opencv-python | Processamento de imagens |
| rasterio | Leitura de imagens GeoTIFF |
| Pillow | Manipulação de imagens |
