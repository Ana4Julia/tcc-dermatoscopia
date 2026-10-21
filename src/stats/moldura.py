"""
Contagem das imagens com moldura do dermatoscópio e comparação da detecção
entre as quatro versões do pré-processamento.

Para cada imagem, aplica campo_de_visao() em cada versão e registra se houve
moldura e a fração da imagem excluída. Depois, compara cada versão com a
versão sem tratamento pelo índice de Jaccard entre as regiões excluídas.

Saídas: results/moldura.csv (uma linha por imagem e versão)
        results/moldura_resumo.csv

Uso:
    python -m src.stats.moldura
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.segmentation.fundo import campo_de_visao

RAIZ = Path(__file__).resolve().parents[2]
DIR_PROCESSADO = RAIZ / "data" / "processed"
DIR_RESULTADOS = RAIZ / "results"
RAMOS = ["sem_tratamento", "dullrazor_bilateral", "clahe",
         "dullrazor_bilateral_clahe"]


def main() -> None:
    ids = sorted(p.stem for p in (DIR_PROCESSADO / RAMOS[0]).glob("*.png"))
    linhas = []
    for id_img in tqdm(ids, desc="moldura"):
        fora = {}
        for ramo in RAMOS:
            img = cv2.imread(str(DIR_PROCESSADO / ramo / f"{id_img}.png"))
            fora[ramo] = ~campo_de_visao(img)
        ref = fora["sem_tratamento"]
        for ramo in RAMOS:
            m = fora[ramo]
            uniao = (m | ref).sum()
            linhas.append({
                "image_id": id_img, "pre_processamento": ramo,
                "tem_moldura": bool(m.any()),
                "fracao_excluida": float(m.mean()),
                "jaccard_vs_sem_tratamento": (float((m & ref).sum() / uniao)
                                              if uniao else 1.0),
            })

    df = pd.DataFrame(linhas)
    DIR_RESULTADOS.mkdir(exist_ok=True)
    df.to_csv(DIR_RESULTADOS / "moldura.csv", index=False)

    com = df[df["tem_moldura"]]
    resumo = df.groupby("pre_processamento").agg(
        imagens_com_moldura=("tem_moldura", "sum"),
        pct_com_moldura=("tem_moldura", lambda s: 100 * s.mean()),
    )
    resumo["fracao_excluida_mediana"] = com.groupby("pre_processamento")[
        "fracao_excluida"].median()
    diferente = df.pivot(index="image_id", columns="pre_processamento",
                         values="jaccard_vs_sem_tratamento") < 0.99
    resumo["imagens_com_deteccao_diferente"] = diferente.sum()
    resumo = resumo.reindex(RAMOS)
    resumo.to_csv(DIR_RESULTADOS / "moldura_resumo.csv")
    print(resumo.round(3).to_string())

    alguma = diferente.any(axis=1)
    print(f"\nImagens em que a moldura detectada muda entre as versões: "
          f"{alguma.sum()} de {len(ids)} ({100 * alguma.mean():.1f}%)")


if __name__ == "__main__":
    main()
