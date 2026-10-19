"""
Análises complementares dos resultados de segmentação e dos descritores:
testes de Friedman e Wilcoxon sobre o Jaccard, falhas de segmentação,
área segmentada x padrão-ouro, efeito dos pelos e do tamanho da lesão,
e mediana dos descritores por configuração.

Usa os CSVs de data/metadata/ e salva em results/evidencias/.

Uso:
    python -m src.stats.evidencias
"""

from __future__ import annotations

import io
from contextlib import redirect_stdout
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy import stats

RAIZ = Path(__file__).resolve().parents[2]
DIR_META = RAIZ / "data" / "metadata"
DIR_GT = RAIZ / "data" / "processed" / "masks_gt"
DIR_SAIDA = RAIZ / "results" / "evidencias"

RAMOS = ["sem_tratamento", "dullrazor_bilateral", "clahe",
         "dullrazor_bilateral_clahe"]
ALGORITMOS = ["otsu", "kmeans", "snakes"]
CONFIGS = [f"{r}|{a}" for r in RAMOS for a in ALGORITMOS]
LIMITE_PELOS = 30.0

DESCRITORES_DESLOCAMENTO = [
    "geo_area_px", "geo_circularidade", "geo_assimetria",
    "cor_L_media", "cor_a_media", "cor_b_media", "cor_S_media",
    "tex_contraste", "tex_homogeneidade", "tex_energia", "tex_entropia",
]


# Auxiliares
def holm(p: np.ndarray) -> np.ndarray:
    """Correção de Holm-Bonferroni para comparações múltiplas."""
    p = np.asarray(p, float)
    ordem = np.argsort(p)
    m = len(p)
    ajust = np.empty(m)
    acumulado = 0.0
    for pos, i in enumerate(ordem):
        acumulado = max(acumulado, (m - pos) * p[i])
        ajust[i] = min(1.0, acumulado)
    return ajust


def rank_biserial(x: np.ndarray, y: np.ndarray) -> float:
    """Tamanho de efeito do Wilcoxon pareado (-1 a 1; > 0 indica x > y)."""
    d = x - y
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    postos = stats.rankdata(np.abs(d))
    return float((postos[d > 0].sum() - postos[d < 0].sum()) / postos.sum())


def fmt_p(p: float) -> str:
    return "< 0,001" if p < 0.001 else f"{p:.3f}".replace(".", ",")


# 1. Testes estatísticos sobre o Jaccard
def testes_jaccard(jac: pd.DataFrame) -> pd.DataFrame:
    completo = jac[CONFIGS].dropna()
    n, k = completo.shape
    chi2, p = stats.friedmanchisquare(*[completo[c] for c in CONFIGS])
    w = chi2 / (n * (k - 1))
    print("1. TESTES SOBRE O ÍNDICE DE JACCARD")
    print(f"   Friedman (12 configurações, {n} imagens): chi2 = {chi2:.1f}, "
          f"p {fmt_p(p)}, W de Kendall = {w:.3f}")

    comparacoes = []
    for alg in ALGORITMOS:            # efeito do pré-processamento
        for ramo in RAMOS[1:]:
            comparacoes.append(("pré-processamento", alg,
                                f"{ramo}|{alg}", f"sem_tratamento|{alg}"))
        comparacoes.append(("pré-processamento", alg,
                            f"dullrazor_bilateral_clahe|{alg}",
                            f"dullrazor_bilateral|{alg}"))
    for ramo in RAMOS:                # efeito do algoritmo
        for a, b in [("otsu", "kmeans"), ("otsu", "snakes"), ("kmeans", "snakes")]:
            comparacoes.append(("algoritmo", ramo, f"{ramo}|{a}", f"{ramo}|{b}"))

    linhas = []
    for tipo, fixo, ca, cb in comparacoes:
        x, y = completo[ca].values, completo[cb].values
        _, pv = stats.wilcoxon(x, y)
        linhas.append({
            "comparacao": tipo, "fixo": fixo, "config_a": ca, "config_b": cb,
            "n": len(x), "jaccard_med_a": np.median(x), "jaccard_med_b": np.median(y),
            "dif_media": np.mean(x - y), "dif_mediana": np.median(x - y),
            "pct_a_melhor": 100 * np.mean(x > y), "pct_b_melhor": 100 * np.mean(x < y),
            "rank_biserial": rank_biserial(x, y), "p": pv,
        })
    tab = pd.DataFrame(linhas)
    tab["p_holm"] = holm(tab["p"].values)

    print("   Wilcoxon pareado (p corrigido por Holm; r = rank-biserial; "
          "% = imagens em que A é melhor / B é melhor):")
    for _, l in tab.iterrows():
        print(f"     {l.config_a:<34} x {l.config_b:<30} "
              f"dif. média {l.dif_media:+.3f} | A melhor {l.pct_a_melhor:4.1f}% "
              f"| B melhor {l.pct_b_melhor:4.1f}% | r = {l.rank_biserial:+.2f} "
              f"| p {fmt_p(l.p_holm)}")
    tab.to_csv(DIR_SAIDA / "wilcoxon_jaccard.csv", index=False)
    pd.DataFrame([{"n": n, "k": k, "chi2": chi2, "p": p, "kendall_w": w}]
                 ).to_csv(DIR_SAIDA / "friedman_jaccard.csv", index=False)
    return tab


# 2. Falhas de segmentação
def falhas(jac: pd.DataFrame) -> None:
    print("\n2. FALHAS DE SEGMENTAÇÃO")
    linhas = []
    for c in CONFIGS:
        v = jac[c].dropna()
        linhas.append({"config": c, "n": len(v),
                       "pct_j_menor_05": 100 * (v < 0.5).mean(),
                       "pct_j_menor_01": 100 * (v < 0.1).mean(),
                       "pct_j_maior_08": 100 * (v >= 0.8).mean()})
    tab = pd.DataFrame(linhas)
    tab.to_csv(DIR_SAIDA / "falhas_por_config.csv", index=False)
    for _, l in tab.iterrows():
        print(f"   {l.config:<34} J<0,5: {l.pct_j_menor_05:5.1f}% | "
              f"J<0,1: {l.pct_j_menor_01:5.1f}% | J>=0,8: {l.pct_j_maior_08:5.1f}%")

    completo = jac[CONFIGS].dropna()
    n = len(completo)
    nenhuma = (completo.max(axis=1) < 0.5).sum()
    todas = (completo.min(axis=1) >= 0.5).sum()
    melhor = completo.max(axis=1)
    print(f"   Imagens em que NENHUMA das 12 configurações chega a J>=0,5: "
          f"{nenhuma} ({100 * nenhuma / n:.1f}%)")
    print(f"   Imagens em que TODAS as 12 chegam a J>=0,5: "
          f"{todas} ({100 * todas / n:.1f}%)")
    print(f"   Jaccard médio escolhendo a melhor configuração de cada imagem: "
          f"{melhor.mean():.3f} (mediana {melhor.median():.3f})")
    vencedora = completo.idxmax(axis=1).value_counts()
    vencedora = (100 * vencedora / n).reindex(CONFIGS).fillna(0)
    print("   Configuração com maior Jaccard em cada imagem (% das imagens):")
    for c, v in vencedora.items():
        print(f"     {c:<34} {v:5.1f}%")
    pd.DataFrame({"config": vencedora.index, "pct_imagens_vencedora": vencedora.values}
                 ).to_csv(DIR_SAIDA / "config_vencedora.csv", index=False)
    pd.DataFrame([{"n": n, "nenhuma_j05": nenhuma, "todas_j05": todas,
                   "jaccard_melhor_media": melhor.mean(),
                   "jaccard_melhor_mediana": melhor.median()}]
                 ).to_csv(DIR_SAIDA / "falhas_resumo.csv", index=False)


# 3. Área segmentada x área do padrão-ouro
def area_gt(ids) -> pd.Series:
    areas = {}
    for i in ids:
        gt = cv2.imread(str(DIR_GT / f"{i}_segmentation.png"), cv2.IMREAD_GRAYSCALE)
        if gt is not None:
            areas[i] = int(np.count_nonzero(gt > 127))
            areas[f"{i}__total"] = gt.size
    s = pd.Series(areas)
    return s


def tamanho_mascaras(seg: pd.DataFrame, areas: pd.Series) -> None:
    print("\n3. ÁREA SEGMENTADA / ÁREA DO PADRÃO-OURO")
    seg = seg.copy()
    seg["area_gt"] = seg["image_id"].map(areas)
    seg = seg[seg["area_gt"] > 0]
    seg["razao"] = seg["area_px"] / seg["area_gt"]
    linhas = []
    for (r, a), g in seg.groupby(["pre_processamento", "algoritmo"]):
        linhas.append({"config": f"{r}|{a}", "razao_mediana": g.razao.median(),
                       "pct_sobresegmentada": 100 * (g.razao > 1.25).mean(),
                       "pct_subsegmentada": 100 * (g.razao < 0.8).mean()})
    tab = pd.DataFrame(linhas).set_index("config").reindex(CONFIGS).reset_index()
    tab.to_csv(DIR_SAIDA / "area_vs_gt.csv", index=False)
    print("   (sobressegmentada: área > 1,25 x padrão-ouro; "
          "subsegmentada: área < 0,8 x padrão-ouro)")
    for _, l in tab.iterrows():
        print(f"   {l.config:<34} razão mediana {l.razao_mediana:5.2f} | "
              f"sobre {l.pct_sobresegmentada:5.1f}% | sub {l.pct_subsegmentada:5.1f}%")


# 4 e 5. Pelos e tamanho da lesão x Jaccard
def pelos_e_tamanho(jac: pd.DataFrame, prep: pd.DataFrame, areas: pd.Series) -> None:
    print("\n4. % DE PELOS x JACCARD (Spearman)")
    pelos = prep.set_index("image_id")["pct_pelos"]
    print(f"   % de pelos: mediana {pelos.median():.1f} | P90 {pelos.quantile(.9):.1f} "
          f"| máx {pelos.max():.1f} | > {LIMITE_PELOS:.0f}%: {(pelos > LIMITE_PELOS).sum()} imagens")
    linhas = []
    for c in CONFIGS:
        d = pd.concat([jac[c], pelos], axis=1, join="inner").dropna()
        rho, p = stats.spearmanr(d.iloc[:, 0], d.iloc[:, 1])
        linhas.append({"config": c, "rho": rho, "p": p})
        print(f"   {c:<34} rho = {rho:+.3f} (p {fmt_p(p)})")
    pd.DataFrame(linhas).to_csv(DIR_SAIDA / "pelos_spearman.csv", index=False)

    print("\n5. TAMANHO DA LESÃO x JACCARD")
    ids = [i for i in jac.index if i in areas.index]
    frac = pd.Series({i: 100 * areas[i] / areas[f"{i}__total"] for i in ids})
    print(f"   Lesão ocupa (% da imagem): mediana {frac.median():.1f} | "
          f"P10 {frac.quantile(.1):.1f} | P90 {frac.quantile(.9):.1f}")
    faixas = pd.cut(frac, [0, 5, 15, 30, 50, 100],
                    labels=["< 5%", "5-15%", "15-30%", "30-50%", "> 50%"])
    tab = jac.loc[ids, CONFIGS].groupby(faixas, observed=False).mean()
    tab.insert(0, "n_imagens", faixas.value_counts().reindex(tab.index))
    tab.to_csv(DIR_SAIDA / "jaccard_por_tamanho.csv")
    cols = ["sem_tratamento|otsu", "sem_tratamento|snakes",
            "dullrazor_bilateral|otsu", "dullrazor_bilateral|snakes",
            "clahe|snakes"]
    print("   Jaccard médio por faixa de área da lesão:")
    print(tab[["n_imagens"] + cols].round(3).to_string())


# 6. Mediana dos descritores por configuração (deslocamento sistemático)
def deslocamento(desc: pd.DataFrame) -> None:
    print("\n6. MEDIANA DE DESCRITORES POR CONFIGURAÇÃO (deslocamento sistemático)")
    desc = desc.copy()
    desc["config"] = desc["pre_processamento"] + "|" + desc["algoritmo"]
    cols = [d for d in DESCRITORES_DESLOCAMENTO if d in desc.columns]
    tab = desc.groupby("config")[cols].median().reindex(CONFIGS)
    tab.to_csv(DIR_SAIDA / "descritores_mediana_por_config.csv")
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(tab.round(3).to_string())


# Execução
def main() -> None:
    DIR_SAIDA.mkdir(parents=True, exist_ok=True)
    seg = pd.read_csv(DIR_META / "segmentacao.csv")
    seg = seg[seg["erro"].fillna("") == ""]
    prep = pd.read_csv(DIR_META / "preprocessamento.csv")
    desc = pd.read_csv(DIR_META / "descritores.csv", low_memory=False)
    desc = desc[desc["erro"].fillna("") == ""]

    seg["config"] = seg["pre_processamento"] + "|" + seg["algoritmo"]
    jac = seg.pivot_table(index="image_id", columns="config", values="jaccard")

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        print(f"{jac.shape[0]} imagens com Jaccard\n")
        testes_jaccard(jac)
        falhas(jac)
        areas = area_gt(jac.index)
        tamanho_mascaras(seg, areas)
        pelos_e_tamanho(jac, prep, areas)
        deslocamento(desc)
    texto = buffer.getvalue()
    print(texto)
    (DIR_SAIDA / "relatorio.txt").write_text(texto, encoding="utf-8")
    print(f"Tudo salvo em {DIR_SAIDA.relative_to(RAIZ)}/")


if __name__ == "__main__":
    main()