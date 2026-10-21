"""
Efeito da detecção da moldura sobre o ICC.

A moldura foi detectada separadamente em cada versão do pré-processamento.
Este script recalcula o ICC entre as 12 configurações (análise principal)
sem as imagens em que a região excluída mudou entre as versões
(Jaccard < 0,99 em relação à versão sem tratamento, em alguma versão) e
compara com o resultado original de results/icc.csv.

Requer: results/moldura.csv (gerado por src.stats.moldura) e results/icc.csv.
Saída: results/icc_sem_moldura_diferente.csv

Uso:
    python -m src.stats.icc_moldura
"""

from __future__ import annotations

import pandas as pd

from src.descriptors.extracao import NOMES
from src.stats.icc import (ALGORITMOS, DIR_META, DIR_RESULTADOS, RAMOS, _icc,
                           categoria, classe_koo_li)

LIMITE_IGUAL = 0.99   # Jaccard mínimo para considerar a moldura igual


def _classe(inf: float, sup: float) -> str:
    c_inf, c_sup = classe_koo_li(inf), classe_koo_li(sup)
    return c_inf if c_inf == c_sup else f"{c_inf} a {c_sup}"


def main() -> None:
    moldura = pd.read_csv(DIR_RESULTADOS / "moldura.csv")
    jac = moldura.pivot(index="image_id", columns="pre_processamento",
                        values="jaccard_vs_sem_tratamento")
    diferentes = set(jac[(jac < LIMITE_IGUAL).any(axis=1)].index)
    print(f"Imagens com moldura diferente entre as versões: {len(diferentes)}")

    desc = pd.read_csv(DIR_META / "descritores.csv", low_memory=False)
    desc = desc[desc["erro"].fillna("") == ""]
    desc["config"] = list(zip(desc["pre_processamento"], desc["algoritmo"]))
    configs = [(r, a) for r in RAMOS for a in ALGORITMOS]

    original = pd.read_csv(DIR_RESULTADOS / "icc.csv")
    original = original[(original["analise"] == "principal")
                        & (original["recorte"] == "todas")
                        & (original["tipo"] == "concordancia_absoluta")]
    original = original.set_index("descritor")

    linhas = []
    for descritor in NOMES:
        largo = desc.pivot_table(index="image_id", columns="config",
                                 values=descritor).reindex(columns=configs)
        largo = largo[~largo.index.isin(diferentes)].dropna().copy()
        largo.columns = [f"{r}|{a}" for r, a in largo.columns]
        icc, inf, sup = _icc(largo)["concordancia_absoluta"]
        orig = original.loc[descritor]
        linhas.append({
            "descritor": descritor, "categoria": categoria(descritor),
            "n_imagens": len(largo),
            "icc_original": orig["icc"], "classe_original": orig["classe"],
            "icc_sem_moldura_diferente": icc, "ic95_inf": inf, "ic95_sup": sup,
            "classe_sem_moldura_diferente": _classe(inf, sup),
        })

    res = pd.DataFrame(linhas)
    res["diferenca"] = res["icc_sem_moldura_diferente"] - res["icc_original"]
    DIR_RESULTADOS.mkdir(exist_ok=True)
    res.to_csv(DIR_RESULTADOS / "icc_sem_moldura_diferente.csv", index=False)

    print(res.groupby("categoria")[["icc_original",
                                    "icc_sem_moldura_diferente"]].median().round(2))
    i = res["diferenca"].abs().idxmax()
    print(f"\nMaior diferença: {res.loc[i, 'diferenca']:+.3f} ({res.loc[i, 'descritor']})")
    mudou = res[res["classe_original"] != res["classe_sem_moldura_diferente"]]
    print(f"Descritores que mudaram de classe: {len(mudou)}")
    if len(mudou):
        print(mudou[["descritor", "icc_original", "classe_original",
                     "icc_sem_moldura_diferente",
                     "classe_sem_moldura_diferente"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
