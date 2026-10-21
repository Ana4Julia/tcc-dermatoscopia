"""
Análise complementar: ICC entre as 8 configurações sem o Snakes
(Otsu e K-Means nas quatro versões do pré-processamento).

Mostra quanto da instabilidade entre as 12 configurações se deve ao Snakes.
Usa o mesmo modelo de ICC e a mesma classificação de src/stats/icc.py.

Saída: results/icc_sem_snakes.csv

Uso:
    python -m src.stats.icc_sem_snakes
"""

from __future__ import annotations

import pandas as pd

from src.descriptors.extracao import NOMES
from src.stats.icc import (DIR_META, DIR_RESULTADOS, LIMITE_JACCARD, RAMOS,
                           _icc, categoria, classe_koo_li)

ALGORITMOS = ["otsu", "kmeans"]


def main() -> None:
    desc = pd.read_csv(DIR_META / "descritores.csv", low_memory=False)
    desc = desc[desc["erro"].fillna("") == ""]
    seg = pd.read_csv(DIR_META / "segmentacao.csv")
    desc["config"] = list(zip(desc["pre_processamento"], desc["algoritmo"]))
    seg["config"] = list(zip(seg["pre_processamento"], seg["algoritmo"]))

    configs = [(r, a) for r in RAMOS for a in ALGORITMOS]
    jac = seg.pivot_table(index="image_id", columns="config", values="jaccard")
    ok = (jac.reindex(columns=configs) >= LIMITE_JACCARD).all(axis=1)
    ok = set(ok[ok].index)

    linhas = []
    for descritor in NOMES:
        base = desc.pivot_table(index="image_id", columns="config",
                                values=descritor).reindex(columns=configs)
        for analise, largo in (("principal", base),
                               ("jaccard_05", base[base.index.isin(ok)])):
            largo = largo.dropna().copy()
            largo.columns = [f"{r}|{a}" for r, a in largo.columns]
            resultado = _icc(largo)
            for tipo, (icc, inf, sup) in resultado.items():
                c_inf, c_sup = classe_koo_li(inf), classe_koo_li(sup)
                linhas.append({
                    "analise": analise, "descritor": descritor,
                    "categoria": categoria(descritor), "n_imagens": len(largo),
                    "tipo": tipo, "icc": icc, "ic95_inf": inf, "ic95_sup": sup,
                    "classe": c_inf if c_inf == c_sup else f"{c_inf} a {c_sup}",
                })

    res = pd.DataFrame(linhas)
    DIR_RESULTADOS.mkdir(exist_ok=True)
    res.to_csv(DIR_RESULTADOS / "icc_sem_snakes.csv", index=False)
    abs_ = res[res["tipo"] == "concordancia_absoluta"]
    print(abs_.groupby(["analise", "categoria"])["icc"].median().round(2))


if __name__ == "__main__":
    main()
