from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "results" / "eda_datasets.png"

plt.style.use("seaborn-v0_8-whitegrid")
GREEN = "#2d6a2d"
BLUE  = "#2d4a8a"

df_neon = pd.read_csv(BASE / "neon_deepforest.csv")
trees_per_img = df_neon.groupby("image_path").size().values

areas = []
for split in ("train", "val"):
    for txt in (BASE / "neon_yolo" / "labels" / split).glob("*.txt"):
        with open(txt) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    w, h = float(parts[3]), float(parts[4])
                    areas.append(w * h * 400 * 400)
areas = np.array(areas)

site_counts = Counter()
for split in ("train", "val"):
    for img in (BASE / "neon_yolo" / "images" / split).glob("*.png"):
        parts = img.stem.split("_")
        site = parts[1] if parts[0].isdigit() else parts[0]
        site_counts[site] += 1
sites_sorted = sorted(site_counts.items(), key=lambda x: -x[1])
site_labels = [s for s, _ in sites_sorted]
site_vals   = [v for _, v in sites_sorted]

bar_labels = [
    "OAM-TCD: imagens",
    "OAM-TCD: copas anotadas",
    "NEON: imagens anotadas",
    "NEON: bounding boxes manuais",
]
bar_values = [3764, 204802, 194, 6633]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Análise Exploratória dos Dados", fontsize=15, fontweight="bold", y=1.01)

ax = axes[0, 0]
ax.hist(trees_per_img, bins=25, color=GREEN, edgecolor="white", linewidth=0.6)
mean_val = trees_per_img.mean()
ax.axvline(mean_val, color="red", linestyle="--", linewidth=1.5, label=f"Média: {mean_val:.1f}")
ax.set_title(f"Distribuição de árvores por imagem (NEON, n={len(trees_per_img)})", fontsize=11)
ax.set_xlabel("Número de árvores")
ax.set_ylabel("Frequência")
ax.legend(fontsize=9)

ax = axes[0, 1]
ax.hist(areas, bins=30, color=GREEN, edgecolor="white", linewidth=0.6)
mean_area = areas.mean()
ax.axvline(mean_area, color="red", linestyle="--", linewidth=1.5, label=f"Média: {mean_area:.0f} px²")
ax.set_title(f"Distribuição de tamanho de copa (NEON, n={len(areas):,} boxes)", fontsize=11)
ax.set_xlabel("Área da copa (px²)")
ax.set_ylabel("Frequência")
ax.legend(fontsize=9)

ax = axes[1, 0]
bars = ax.bar(site_labels, site_vals, color=GREEN, edgecolor="white", linewidth=0.6)
ax.set_title(f"Imagens por site NEON ({len(site_labels)} sites, total={sum(site_vals)})", fontsize=11)
ax.set_xlabel("Site")
ax.set_ylabel("Número de imagens")
ax.set_xticks(range(len(site_labels)))
ax.set_xticklabels(site_labels, rotation=45, ha="right", fontsize=8)
for bar, val in zip(bars, site_vals):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
            str(val), ha="center", va="bottom", fontsize=7)

ax = axes[1, 1]
colors_mapped = [BLUE if "OAM" in lbl else GREEN for lbl in bar_labels]
hbars = ax.barh(bar_labels, bar_values, color=colors_mapped, edgecolor="white", linewidth=0.6)
ax.set_xscale("log")
ax.set_title("Comparação dos datasets", fontsize=11)
ax.set_xlabel("Contagem (escala logarítmica)")
for bar, val in zip(hbars, bar_values):
    ax.text(val * 1.05, bar.get_y() + bar.get_height() / 2,
            f"{val:,}", va="center", fontsize=9)
ax.set_xlim(left=100)

plt.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(str(OUT), dpi=150, bbox_inches="tight")
plt.close(fig)

from PIL import Image
img = Image.open(OUT)
print(f"Guardado: {OUT}  ({img.size[0]}x{img.size[1]} px, {OUT.stat().st_size/1024:.1f} KB)")
