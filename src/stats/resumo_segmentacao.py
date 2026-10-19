"""
Resumo da segmentação: média, desvio padrão e mediana do Jaccard e do Dice,
e % de imagens com Jaccard >= 0,5, para cada configuração. Também separa as
imagens com mais e com menos de 30% de pelos.

Saídas: results/segmentacao_resumo.csv e results/segmentacao_por_pelos.csv

Uso:
    python -m src.stats.resumo_segmentacao
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
DIR_META = RAIZ / "data" / "metadata"
DIR_RESULTADOS = RAIZ / "results"

RAMOS = ["sem_tratamento", "dullrazor_bilateral", "clahe",
         "dullrazor_bilateral_clahe"]
ALGORITMOS = ["otsu", "kmeans", "snakes"]
LIMITE_PELOS = 30.0
LIMITE_JACCARD = 0.5


def _resumir(grupo: pd.DataFrame) -> pd.Series:
    return pd.Series({
        "n": len(grupo),
        "jaccard_media": grupo["jaccard"].mean(),
        "jaccard_dp": grupo["jaccard"].std(),
        "jaccard_mediana": grupo["jaccard"].median(),
        "dice_media": grupo["dice"].mean(),
        "dice_dp": grupo["dice"].std(),
        "pct_jaccard_05": 100 * (grupo["jaccard"] >= LIMITE_JACCARD).mean(),
    })


def main() -> None:
    seg = pd.read_csv(DIR_META / "segmentacao.csv")
    seg = seg[seg["erro"].fillna("") == ""]
    prep = pd.read_csv(DIR_META / "preprocessamento.csv")[["image_id", "pct_pelos"]]
    seg = seg.merge(prep, on="image_id", how="left")
    seg["grupo_pelos"] = (seg["pct_pelos"] > LIMITE_PELOS).map(
        {True: f"> {LIMITE_PELOS:.0f}% de pelos", False: f"<= {LIMITE_PELOS:.0f}% de pelos"})

    chave = ["pre_processamento", "algoritmo"]
    resumo = seg.groupby(chave).apply(_resumir, include_groups=False).reset_index()
    resumo["pre_processamento"] = pd.Categorical(resumo["pre_processamento"], RAMOS)
    resumo["algoritmo"] = pd.Categorical(resumo["algoritmo"], ALGORITMOS)
    resumo = resumo.sort_values(chave)

    por_pelos = (seg.groupby(["grupo_pelos"] + chave)
                    .apply(_resumir, include_groups=False).reset_index())

    DIR_RESULTADOS.mkdir(exist_ok=True)
    resumo.to_csv(DIR_RESULTADOS / "segmentacao_resumo.csv", index=False)
    por_pelos.to_csv(DIR_RESULTADOS / "segmentacao_por_pelos.csv", index=False)

    with pd.option_context("display.width", 140):
        print("Jaccard (média ± dp | mediana) e % de imagens com Jaccard >= 0,5:\n")
        for _, l in resumo.iterrows():
            print(f"  {l.pre_processamento:<27} {l.algoritmo:<7} "
                  f"J = {l.jaccard_media:.3f} ± {l.jaccard_dp:.3f} | "
                  f"mediana {l.jaccard_mediana:.3f} | "
                  f"D = {l.dice_media:.3f} | J>=0,5: {l.pct_jaccard_05:.1f}%")
    print("\nSalvo em results/segmentacao_resumo.csv e results/segmentacao_por_pelos.csv")


if __name__ == "__main__":
    main()
