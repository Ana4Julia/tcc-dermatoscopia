"""
Estabilidade dos descritores pelo ICC (modelo de efeitos mistos de duas vias,
medida única), com o Pingouin. Calcula ICC(A,1) e ICC(C,1) com IC de 95% e
classifica pelos limites do intervalo (Koo e Li, 2016).

Recortes: entre os 4 pré-processamentos (algoritmo fixo), entre os 3
algoritmos (pré-processamento fixo) e entre as 12 configurações.
Análises: todas as imagens, sem imagens com mais de 30% de pelos e só
imagens com Jaccard >= 0,5.

Saídas: results/icc.csv e results/icc_resumo.csv

Uso:
    python -m src.stats.icc
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pingouin as pg

from src.descriptors.extracao import NOMES

RAIZ = Path(__file__).resolve().parents[2]
DIR_META = RAIZ / "data" / "metadata"
DIR_RESULTADOS = RAIZ / "results"

RAMOS = ["sem_tratamento", "dullrazor_bilateral", "clahe",
         "dullrazor_bilateral_clahe"]
ALGORITMOS = ["otsu", "kmeans", "snakes"]

LIMITE_PELOS = 30.0     # % de pelos (análise de sensibilidade 1)
LIMITE_JACCARD = 0.5    # Jaccard mínimo (análise de sensibilidade 2)

TIPOS = {  # nomes do Pingouin (novo e antigo) -> forma do ICC
    "ICC(A,1)": "concordancia_absoluta", "ICC2": "concordancia_absoluta",
    "ICC(C,1)": "consistencia", "ICC3": "consistencia",
}


def classe_koo_li(valor: float) -> str:
    if pd.isna(valor):
        return ""
    if valor < 0.5:
        return "fraca"
    if valor < 0.75:
        return "moderada"
    if valor < 0.9:
        return "boa"
    return "excelente"


def categoria(descritor: str) -> str:
    return {"geo": "geométrico", "cor": "colorimétrico",
            "tex": "textura"}[descritor.split("_")[0]]


def _icc(largo: pd.DataFrame) -> dict:
    """ICC(A,1) e ICC(C,1) de uma tabela imagens x configurações (sem NaN)."""
    longo = largo.reset_index().melt(id_vars="image_id", var_name="config",
                                     value_name="valor")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tabela = pg.intraclass_corr(longo, targets="image_id", raters="config",
                                    ratings="valor")
    coluna_ic = "CI95" if "CI95" in tabela.columns else "CI95%"
    saida = {}
    for _, linha in tabela.iterrows():
        tipo = TIPOS.get(linha["Type"])
        if tipo:
            inf, sup = linha[coluna_ic]
            saida[tipo] = (float(linha["ICC"]), float(inf), float(sup))
    return saida


def _recortes():
    """Gera (recorte, grupo_fixo, lista de configurações)."""
    for alg in ALGORITMOS:
        yield "pre_processamento", alg, [(r, alg) for r in RAMOS]
    for ramo in RAMOS:
        yield "segmentacao", ramo, [(ramo, a) for a in ALGORITMOS]
    yield "todas", "-", [(r, a) for r in RAMOS for a in ALGORITMOS]


def main() -> None:
    desc = pd.read_csv(DIR_META / "descritores.csv")
    desc = desc[desc["erro"].fillna("") == ""]
    seg = pd.read_csv(DIR_META / "segmentacao.csv")
    prep = pd.read_csv(DIR_META / "preprocessamento.csv")

    desc["config"] = list(zip(desc["pre_processamento"], desc["algoritmo"]))
    seg["config"] = list(zip(seg["pre_processamento"], seg["algoritmo"]))
    jac = seg.pivot_table(index="image_id", columns="config", values="jaccard")
    pelos_extremos = set(prep.loc[prep["pct_pelos"] > LIMITE_PELOS, "image_id"])

    n_imgs = desc["image_id"].nunique()
    print(f"{n_imgs} imagens | {len(NOMES)} descritores | "
          f"{len(pelos_extremos)} imagens com mais de {LIMITE_PELOS:.0f}% de pelos")

    linhas = []
    for descritor in NOMES:
        largo_total = desc.pivot_table(index="image_id", columns="config",
                                       values=descritor)
        for recorte, grupo, configs in _recortes():
            base = largo_total.reindex(columns=configs)
            analises = {
                "principal": base,
                "sem_pelos_extremos": base[~base.index.isin(pelos_extremos)],
            }
            ok_jac = (jac.reindex(columns=configs) >= LIMITE_JACCARD).all(axis=1)
            analises["jaccard_05"] = base[base.index.isin(ok_jac[ok_jac].index)]

            for analise, largo in analises.items():
                largo = largo.dropna()
                # descritor constante não tem variância: ICC indefinido
                if len(largo) < 30 or np.isclose(largo.values.std(), 0):
                    resultado = {}
                else:
                    largo.columns = [f"{r}|{a}" for r, a in largo.columns]
                    resultado = _icc(largo)
                for tipo in ("concordancia_absoluta", "consistencia"):
                    icc, inf, sup = resultado.get(tipo, (np.nan,) * 3)
                    c_inf, c_sup = classe_koo_li(inf), classe_koo_li(sup)
                    linhas.append({
                        "analise": analise, "recorte": recorte,
                        "grupo_fixo": grupo, "descritor": descritor,
                        "categoria": categoria(descritor),
                        "n_imagens": len(largo), "n_configuracoes": len(configs),
                        "tipo": tipo, "icc": icc, "ic95_inf": inf, "ic95_sup": sup,
                        "classe": (c_inf if c_inf == c_sup
                                   else f"{c_inf} a {c_sup}" if c_inf else ""),
                    })
        print(f"  {descritor}")

    DIR_RESULTADOS.mkdir(exist_ok=True)
    res = pd.DataFrame(linhas)
    res.to_csv(DIR_RESULTADOS / "icc.csv", index=False)

    # Resumo: 12 configurações, análise principal, uma linha por descritor
    sel = res[(res.recorte == "todas") & (res.analise == "principal")]
    resumo = sel.pivot_table(index=["categoria", "descritor"], columns="tipo",
                             values=["icc", "ic95_inf", "ic95_sup"])
    resumo.columns = [f"{t}_{v}" for v, t in resumo.columns]
    classes = sel.pivot_table(index=["categoria", "descritor"], columns="tipo",
                              values="classe", aggfunc="first")
    classes.columns = [f"{t}_classe" for t in classes.columns]
    resumo = resumo.join(classes).reset_index()
    ordem = ["categoria", "descritor"] + [
        f"{t}_{v}" for t in ("concordancia_absoluta", "consistencia")
        for v in ("icc", "ic95_inf", "ic95_sup", "classe")]
    resumo = resumo[ordem]
    resumo.to_csv(DIR_RESULTADOS / "icc_resumo.csv", index=False)

    print(f"\nResultados salvos em results/icc.csv e results/icc_resumo.csv")
    print("\nICC das 12 configurações (análise principal), concordância absoluta:")
    mostrar = resumo[["descritor", "concordancia_absoluta_icc",
                      "concordancia_absoluta_classe", "consistencia_icc"]]
    with pd.option_context("display.max_rows", None, "display.width", 120):
        print(mostrar.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
