"""
Figuras do capítulo de resultados:
  - icc_principal e icc_jaccard05: mapas de calor do ICC(A,1) dos 42 descritores;
  - icc_abs_vs_cons: ICC(A,1) x ICC(C,1) entre as 12 configurações;
  - jaccard_boxplot: distribuição do Jaccard nas 12 configurações;
  - exatidao_principal e exatidao_jaccard05: ICC(A,1) com o padrão-ouro.

As cores seguem as faixas de Koo e Li (2016). Saída: results/figuras/

Uso:
    python -m src.stats.figuras
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

RAIZ = Path(__file__).resolve().parents[2]
DIR_RESULTADOS = RAIZ / "results"
DIR_FIGURAS = DIR_RESULTADOS / "figuras"

# Faixas de Koo e Li (2016): fraca, moderada, boa, excelente
CORES_FAIXAS = ["#cde2fb", "#6da7ec", "#256abf", "#0d366b"]
LIMITES = [0, 0.5, 0.75, 0.9, 1.0]
NOMES_FAIXAS = ["Fraca (< 0,50)", "Moderada (0,50–0,75)",
                "Boa (0,75–0,90)", "Excelente (≥ 0,90)"]
COR_ABS, COR_CONS = "#2a78d6", "#eb6834"
CORES_ALG = {"otsu": "#2a78d6", "kmeans": "#eb6834", "snakes": "#1baf7a"}
NOMES_ALG = {"otsu": "Otsu", "kmeans": "K-Means", "snakes": "Snakes"}
RAMOS = ["sem_tratamento", "dullrazor_bilateral", "clahe",
         "dullrazor_bilateral_clahe"]
NOMES_RAMO = ["Sem tratamento", "DullRazor +\nBilateral", "CLAHE",
              "DullRazor +\nBilateral + CLAHE"]
TINTA, TINTA_2, GRADE = "#0b0b0b", "#52514e", "#d9d8d4"

COLUNAS = [  # (recorte, grupo_fixo, rótulo)
    ("pre_processamento", "otsu", "Otsu"),
    ("pre_processamento", "kmeans", "K-Means"),
    ("pre_processamento", "snakes", "Snakes"),
    ("segmentacao", "sem_tratamento", "Sem trat."),
    ("segmentacao", "dullrazor_bilateral", "DR + Bil."),
    ("segmentacao", "clahe", "CLAHE"),
    ("segmentacao", "dullrazor_bilateral_clahe", "DR + Bil.\n+ CLAHE"),
    ("todas", "-", "12 config."),
]
GRUPOS_COLUNAS = [(0, 3, "Entre pré-processamentos\n(algoritmo fixo)"),
                  (3, 7, "Entre algoritmos\n(pré-processamento fixo)"),
                  (7, 8, "Todas")]

CANAIS = ["R", "G", "B", "S", "V", "L", "a", "b"]
ESTATS = {"media": "média", "desvio": "desvio padrão",
          "assimetria": "assimetria", "curtose": "curtose"}
GEO = {"geo_area_px": "Área", "geo_assimetria": "Índice de assimetria",
       "geo_circularidade": "Circularidade",
       "geo_diametro_circulo": "Diâm. circunf. circunscrita",
       "geo_diametro_feret": "Diâmetro de Feret"}
TEX = {"tex_contraste": "Contraste", "tex_correlacao": "Correlação",
       "tex_energia": "Energia", "tex_homogeneidade": "Homogeneidade",
       "tex_entropia": "Entropia"}


def ordem_descritores() -> list[tuple[str, str]]:
    """(nome no CSV, rótulo) na ordem: geométricos, cor por canal, textura."""
    ordem = list(GEO.items())
    for canal in CANAIS:
        rot = f"{canal}*" if canal in ("L", "a", "b") else canal
        for est, nome in ESTATS.items():
            ordem.append((f"cor_{canal}_{est}", f"{rot} – {nome}"))
    ordem += list(TEX.items())
    return ordem


def _estilo(ax) -> None:
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color(GRADE)
    ax.tick_params(colors=TINTA_2, length=0)


def _desenhar_mapa(matriz, rotulos, grupos, separadores_x, arquivo,
                   fonte_coluna=7, fonte_valor=6.3) -> None:
    descritores = ordem_descritores()
    cmap = ListedColormap(CORES_FAIXAS)
    norm = BoundaryNorm(LIMITES, cmap.N)
    fig, ax = plt.subplots(figsize=(6.3, 8.6))
    ax.imshow(np.clip(matriz, 0, 1), cmap=cmap, norm=norm, aspect="auto")

    # linhas brancas entre as células e entre os grupos
    for x in np.arange(-0.5, len(rotulos), 1):
        ax.axvline(x, color="white", lw=1)
    for y in np.arange(-0.5, len(descritores), 1):
        ax.axhline(y, color="white", lw=0.6)
    for x in separadores_x:
        ax.axvline(x, color="white", lw=4)
    for y in (4.5, 36.5):
        ax.axhline(y, color="white", lw=4)

    for i in range(matriz.shape[0]):
        for j in range(matriz.shape[1]):
            v = matriz[i, j]
            if np.isnan(v):
                continue
            v_txt = 0.0 if abs(v) < 0.005 else v
            texto = f"{v_txt:.2f}".replace(".", ",").replace("-", "\u2212")
            ax.text(j, i, texto, ha="center", va="center", fontsize=fonte_valor,
                    color="white" if v >= 0.75 else TINTA)

    ax.set_xticks(range(len(rotulos)), rotulos, fontsize=fonte_coluna)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(descritores)), [d[1] for d in descritores], fontsize=7)
    ax.tick_params(length=0, colors=TINTA)
    for lado in ax.spines.values():
        lado.set_visible(False)

    for ini, fim, rotulo in grupos:
        ax.annotate(rotulo, xy=((ini + fim - 1) / 2, -2.6), xycoords="data",
                    ha="center", va="bottom", fontsize=7, color=TINTA_2,
                    annotation_clip=False)
    for ini, fim, rotulo in [(0, 5, "Geométricos"), (5, 37, "Colorimétricos"),
                             (37, 42, "Textura")]:
        ax.annotate(rotulo, xy=(-0.5, (ini + fim - 1) / 2),
                    xytext=(-118, 0), textcoords="offset points", rotation=90,
                    ha="center", va="center", fontsize=7.5, color=TINTA_2,
                    annotation_clip=False)

    legenda = [Patch(facecolor=c, edgecolor="none", label=n)
               for c, n in zip(CORES_FAIXAS, NOMES_FAIXAS)]
    fig.legend(handles=legenda, loc="lower center", ncol=4, fontsize=7,
               frameon=False, bbox_to_anchor=(0.55, 0.0), handlelength=1.2)
    fig.subplots_adjust(left=0.30, right=0.975, top=0.89, bottom=0.045)
    for ext in ("pdf", "png"):
        fig.savefig(DIR_FIGURAS / f"{arquivo}.{ext}", dpi=300)
    plt.close(fig)


def mapa_de_calor(res: pd.DataFrame, analise: str, arquivo: str) -> None:
    sel = res[(res.analise == analise) & (res.tipo == "concordancia_absoluta")]
    descritores = ordem_descritores()
    matriz = np.full((len(descritores), len(COLUNAS)), np.nan)
    for j, (rec, grupo, _) in enumerate(COLUNAS):
        col = sel[(sel.recorte == rec) & (sel.grupo_fixo == grupo)].set_index("descritor").icc
        matriz[:, j] = [col.get(d, np.nan) for d, _ in descritores]
    _desenhar_mapa(matriz, [c[2] for c in COLUNAS], GRUPOS_COLUNAS, (2.5, 6.5), arquivo)


def mapa_exatidao(ex: pd.DataFrame, analise: str, arquivo: str) -> None:
    """ICC(A,1) de cada configuração com o padrão-ouro (+ padrão-ouro entre
    os pré-processamentos)."""
    sel = ex[(ex.tipo == "concordancia_absoluta")]
    descritores = ordem_descritores()
    colunas = [(r, a) for r in RAMOS for a in NOMES_ALG]
    matriz = np.full((len(descritores), len(colunas) + 1), np.nan)
    for j, (r, a) in enumerate(colunas):
        col = sel[(sel.analise == analise) & (sel.recorte == "config_x_gt")
                  & (sel.pre_processamento == r) & (sel.algoritmo == a)
                  ].set_index("descritor").icc
        matriz[:, j] = [col.get(d, np.nan) for d, _ in descritores]
    col = sel[(sel.analise == "principal")
              & (sel.recorte == "gt_entre_preprocessamentos")].set_index("descritor").icc
    matriz[:, -1] = [col.get(d, np.nan) for d, _ in descritores]
    rotulos = ["Otsu", "K-M.", "Snak."] * len(RAMOS) + ["Entre\npré-proc."]
    grupos = [(0, 3, "Sem\ntratamento"), (3, 6, "DR +\nBil."), (6, 9, "CLAHE"),
              (9, 12, "DR + Bil.\n+ CLAHE"), (12, 13, "Padrão-\nouro")]
    _desenhar_mapa(matriz, rotulos, grupos, (2.5, 5.5, 8.5, 11.5), arquivo,
                   fonte_coluna=6, fonte_valor=5.4)


def halteres(res: pd.DataFrame) -> None:
    sel = res[(res.analise == "principal") & (res.recorte == "todas")]
    tab = sel.pivot_table(index="descritor", columns="tipo", values="icc")
    descritores = ordem_descritores()
    y = np.arange(len(descritores))
    absol = np.array([tab.loc[d, "concordancia_absoluta"] for d, _ in descritores])
    cons = np.array([tab.loc[d, "consistencia"] for d, _ in descritores])

    fig, ax = plt.subplots(figsize=(6.3, 8.6))
    for lim in (0.5, 0.75, 0.9):
        ax.axvline(lim, color=GRADE, lw=0.8, zorder=0)
    ax.hlines(y, absol, cons, color=GRADE, lw=2, zorder=1)
    ax.scatter(absol, y, s=26, color=COR_ABS, edgecolor="white", lw=1,
               zorder=3, label="Concordância absoluta – ICC(A,1)")
    ax.scatter(cons, y, s=26, color=COR_CONS, edgecolor="white", lw=1,
               zorder=3, marker="D", label="Consistência – ICC(C,1)")
    for lim in (4.5, 36.5):
        ax.axhline(lim, color=TINTA_2, lw=0.5)

    ax.set_yticks(y, [d[1] for d in descritores], fontsize=7, color=TINTA)
    ax.set_ylim(len(descritores) - 0.4, -0.6)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 0.9, 1.0],
                  ["0", "0,25", "0,50", "0,75", "0,90", "1"], fontsize=7)
    ax.set_xlabel("ICC entre as 12 configurações", fontsize=8, color=TINTA)
    _estilo(ax)
    ax.legend(loc="lower center", bbox_to_anchor=(0.4, 1.0), ncol=2,
              fontsize=7, frameon=False)
    fig.subplots_adjust(left=0.30, right=0.97, top=0.95, bottom=0.07)
    for ext in ("pdf", "png"):
        fig.savefig(DIR_FIGURAS / f"icc_abs_vs_cons.{ext}", dpi=300)
    plt.close(fig)


def boxplot_jaccard() -> None:
    seg = pd.read_csv(RAIZ / "data" / "metadata" / "segmentacao.csv")
    seg = seg[seg["erro"].fillna("") == ""]
    fig, ax = plt.subplots(figsize=(6.3, 3.4))
    largura = 0.24
    for k, alg in enumerate(NOMES_ALG):
        dados = [seg[(seg.pre_processamento == r) & (seg.algoritmo == alg)].jaccard.dropna()
                 for r in RAMOS]
        pos = np.arange(len(RAMOS)) + (k - 1) * (largura + 0.04)
        bp = ax.boxplot(dados, positions=pos, widths=largura, patch_artist=True,
                        showfliers=False, showmeans=True,
                        medianprops=dict(color="white", lw=1.5),
                        meanprops=dict(marker="o", markerfacecolor="white",
                                       markeredgecolor=TINTA, markersize=3.5),
                        whiskerprops=dict(color=TINTA_2, lw=0.8),
                        capprops=dict(color=TINTA_2, lw=0.8),
                        boxprops=dict(lw=0))
        for caixa in bp["boxes"]:
            caixa.set_facecolor(CORES_ALG[alg])
        ax.plot([], [], "s", color=CORES_ALG[alg], label=NOMES_ALG[alg])
    ax.plot([], [], "o", markerfacecolor="white", markeredgecolor=TINTA,
            markersize=3.5, ls="none", label="Média")
    ax.axhline(0.5, color=GRADE, lw=0.8, ls="--", zorder=0)
    ax.set_xticks(range(len(RAMOS)), NOMES_RAMO, fontsize=7.5, color=TINTA)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1], ["0", "0,25", "0,50", "0,75", "1"], fontsize=7)
    ax.set_ylabel("Índice de Jaccard", fontsize=8, color=TINTA)
    ax.yaxis.grid(True, color=GRADE, lw=0.5)
    ax.set_axisbelow(True)
    _estilo(ax)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4,
              fontsize=7, frameon=False)
    fig.subplots_adjust(left=0.09, right=0.99, top=0.91, bottom=0.14)
    for ext in ("pdf", "png"):
        fig.savefig(DIR_FIGURAS / f"jaccard_boxplot.{ext}", dpi=300)
    plt.close(fig)


def main() -> None:
    res = pd.read_csv(DIR_RESULTADOS / "icc.csv")
    DIR_FIGURAS.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.family"] = "DejaVu Sans"
    mapa_de_calor(res, "principal", "icc_principal")
    mapa_de_calor(res, "jaccard_05", "icc_jaccard05")
    halteres(res)
    boxplot_jaccard()
    arq_ex = DIR_RESULTADOS / "exatidao.csv"
    if arq_ex.exists():
        ex = pd.read_csv(arq_ex)
        mapa_exatidao(ex, "principal", "exatidao_principal")
        mapa_exatidao(ex, "jaccard_05", "exatidao_jaccard05")
    print(f"Figuras salvas em {DIR_FIGURAS.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
