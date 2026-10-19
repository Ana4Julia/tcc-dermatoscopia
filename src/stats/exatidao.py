"""
Exatidão dos descritores: ICC entre o descritor calculado com a máscara de
cada configuração e o mesmo descritor calculado com a máscara padrão-ouro
(mesmo pré-processamento). Também calcula o ICC entre os 4 pré-processamentos
usando só a máscara padrão-ouro, para medir o efeito do pré-processamento sozinho.

Análises: todas as imagens e só as imagens com Jaccard >= 0,5.
Saídas: results/exatidao.csv e results/exatidao_resumo.csv

Uso:
    python -m src.stats.exatidao
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.descriptors.extracao import NOMES
from src.stats.icc import _icc, categoria, classe_koo_li

RAIZ = Path(__file__).resolve().parents[2]
DIR_META = RAIZ / "data" / "metadata"
DIR_RESULTADOS = RAIZ / "results"

RAMOS = ["sem_tratamento", "dullrazor_bilateral", "clahe",
         "dullrazor_bilateral_clahe"]
ALGORITMOS = ["otsu", "kmeans", "snakes"]
LIMITE_JACCARD = 0.5
N_MINIMO = 30


def _linha(analise, recorte, ramo, alg, descritor, largo, extra=None) -> list[dict]:
    largo = largo.dropna()
    if len(largo) < N_MINIMO or np.isclose(largo.values.std(), 0):
        resultado = {}
    else:
        largo = largo.copy()
        largo.columns = [str(c) for c in largo.columns]
        resultado = _icc(largo)
    linhas = []
    for tipo in ("concordancia_absoluta", "consistencia"):
        icc, inf, sup = resultado.get(tipo, (np.nan,) * 3)
        c_inf, c_sup = classe_koo_li(inf), classe_koo_li(sup)
        linha = {"analise": analise, "recorte": recorte,
                 "pre_processamento": ramo, "algoritmo": alg,
                 "descritor": descritor, "categoria": categoria(descritor),
                 "n_imagens": len(largo), "tipo": tipo,
                 "icc": icc, "ic95_inf": inf, "ic95_sup": sup,
                 "classe": (c_inf if c_inf == c_sup
                            else f"{c_inf} a {c_sup}" if c_inf else "")}
        linha.update(extra or {})
        linhas.append(linha)
    return linhas


def main() -> None:
    desc = pd.read_csv(DIR_META / "descritores.csv", low_memory=False)
    desc = desc[desc["erro"].fillna("") == ""]
    gt = pd.read_csv(DIR_META / "descritores_gt.csv", low_memory=False)
    gt = gt[gt["erro"].fillna("") == ""]
    seg = pd.read_csv(DIR_META / "segmentacao.csv")
    jac = seg.set_index(["image_id", "pre_processamento", "algoritmo"])["jaccard"]

    print(f"{desc['image_id'].nunique()} imagens (algoritmos) | "
          f"{gt['image_id'].nunique()} imagens (padrão-ouro)")

    linhas = []
    for descritor in NOMES:
        ref = gt.pivot_table(index="image_id", columns="pre_processamento",
                             values=descritor)
        # 1) cada configuração x padrão-ouro (mesmo pré-processamento)
        for ramo in RAMOS:
            for alg in ALGORITMOS:
                v = desc[(desc.pre_processamento == ramo) & (desc.algoritmo == alg)]
                v = v.set_index("image_id")[descritor]
                largo = pd.concat({"algoritmo": v, "padrao_ouro": ref.get(ramo)},
                                  axis=1).dropna()
                dif = (largo["algoritmo"] - largo["padrao_ouro"]).median()
                extra = {"ref_mediana": largo["padrao_ouro"].median(),
                         "alg_mediana": largo["algoritmo"].median(),
                         "dif_mediana": dif}
                linhas += _linha("principal", "config_x_gt", ramo, alg,
                                 descritor, largo, extra)
                j = jac.xs((ramo, alg), level=["pre_processamento", "algoritmo"])
                ok = j[j >= LIMITE_JACCARD].index
                linhas += _linha("jaccard_05", "config_x_gt", ramo, alg,
                                 descritor, largo[largo.index.isin(ok)], extra)
        # 2) padrão-ouro entre os 4 pré-processamentos (efeito só do pré-proc.)
        linhas += _linha("principal", "gt_entre_preprocessamentos", "-",
                         "padrao_ouro", descritor, ref.reindex(columns=RAMOS))
        print(f"  {descritor}")

    DIR_RESULTADOS.mkdir(exist_ok=True)
    res = pd.DataFrame(linhas)
    res.to_csv(DIR_RESULTADOS / "exatidao.csv", index=False)

    sel = res[(res.analise == "principal") & (res.recorte == "config_x_gt")
              & (res.tipo == "concordancia_absoluta")].copy()
    sel["config"] = sel.pre_processamento + "|" + sel.algoritmo
    resumo = sel.pivot_table(index=["categoria", "descritor"], columns="config",
                             values="icc")
    resumo = resumo[[f"{r}|{a}" for r in RAMOS for a in ALGORITMOS]]
    gt_pre = res[(res.recorte == "gt_entre_preprocessamentos")
                 & (res.tipo == "concordancia_absoluta")].set_index("descritor").icc
    resumo["gt_entre_preprocessamentos"] = [gt_pre.get(d) for _, d in resumo.index]
    resumo.reset_index().to_csv(DIR_RESULTADOS / "exatidao_resumo.csv", index=False)

    # Resumo no terminal: mediana por categoria
    print("\nMediana do ICC(A,1) configuração x padrão-ouro, por categoria "
          "(análise principal | Jaccard >= 0,5):")
    for analise in ("principal", "jaccard_05"):
        s = res[(res.analise == analise) & (res.recorte == "config_x_gt")
                & (res.tipo == "concordancia_absoluta")]
        t = s.pivot_table(index=["pre_processamento", "algoritmo"],
                          columns="categoria", values="icc", aggfunc="median")
        t = t.reindex([(r, a) for r in RAMOS for a in ALGORITMOS])
        print(f"\n[{analise}]")
        print(t.round(2).to_string())
    g = res[(res.recorte == "gt_entre_preprocessamentos")
            & (res.tipo == "concordancia_absoluta")]
    print("\nICC(A,1) entre os 4 pré-processamentos com a máscara padrão-ouro "
          "(mediana por categoria):")
    print(g.groupby("categoria").icc.median().round(2).to_string())
    print("\nSalvo em results/exatidao.csv e results/exatidao_resumo.csv")


if __name__ == "__main__":
    main()