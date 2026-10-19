"""
Figuras de exemplo do capítulo de resultados:
  - ex_preprocessamento: uma lesão nas 4 versões do pré-processamento;
  - ex_mascaras: as 12 máscaras da mesma lesão, com o Jaccard de cada uma;
  - ex_casos: acerto, pior caso, falha do Snakes e lesão com muitos pelos;
  - ex_pelos: imagem com mais de 30% de pelos, antes e depois do DullRazor.

As imagens são escolhidas automaticamente; para trocar, use --id-* (veja --help).
Saída: results/figuras/ex_*.pdf e .png, e ex_escolhidas.txt

Uso:
    python -m src.stats.figuras_exemplos
    python -m src.stats.figuras_exemplos --id-mascaras ISIC_0000123
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

RAIZ = Path(__file__).resolve().parents[2]
DIR_META = RAIZ / "data" / "metadata"
DIR_PROC = RAIZ / "data" / "processed"
DIR_SEG = RAIZ / "data" / "segmentation"
DIR_FIGURAS = RAIZ / "results" / "figuras"

RAMOS = ["sem_tratamento", "dullrazor_bilateral", "clahe",
         "dullrazor_bilateral_clahe"]
ALGORITMOS = ["otsu", "kmeans", "snakes"]
NOME_RAMO = {"sem_tratamento": "Sem tratamento",
             "dullrazor_bilateral": "DullRazor + Bilateral",
             "clahe": "CLAHE",
             "dullrazor_bilateral_clahe": "DullRazor + Bilateral + CLAHE"}
NOME_ALG = {"otsu": "Otsu", "kmeans": "K-Means", "snakes": "Snakes"}

COR_GT = (27, 175, 122)      # padrão-ouro (verde-água)
COR_ALG = (235, 104, 52)     # máscara do algoritmo (laranja)
COR_RECORTE = "#2a78d6"
TINTA, TINTA_2 = "#0b0b0b", "#52514e"
LIMITE_PELOS = 30.0


# Leitura e desenho
def ler_rgb(ramo: str, id_img: str) -> np.ndarray:
    img = cv2.imread(str(DIR_PROC / ramo / f"{id_img}.png"))
    if img is None:
        raise FileNotFoundError(f"{ramo}/{id_img}.png")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def ler_mascara(caminho: Path) -> np.ndarray | None:
    m = cv2.imread(str(caminho), cv2.IMREAD_GRAYSCALE)
    return None if m is None else (m > 127).astype(np.uint8)


def ler_gt(id_img: str) -> np.ndarray | None:
    return ler_mascara(DIR_PROC / "masks_gt" / f"{id_img}_segmentation.png")


def ler_seg(ramo: str, alg: str, id_img: str) -> np.ndarray | None:
    return ler_mascara(DIR_SEG / ramo / alg / f"{id_img}.png")


def contornos(img: np.ndarray, gt: np.ndarray | None,
              mascara: np.ndarray | None) -> np.ndarray:
    saida = img.copy()
    esp = max(2, round(img.shape[1] / 300))
    for m, cor in ((gt, COR_GT), (mascara, COR_ALG)):
        if m is None:
            continue
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(saida, cs, -1, (255, 255, 255), esp + 2)
        cv2.drawContours(saida, cs, -1, cor, esp)
    return saida


def _sem_eixos(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for lado in ax.spines.values():
        lado.set_visible(False)


def _legenda_contornos(fig, y=0.005) -> None:
    itens = [Line2D([], [], color=np.array(COR_GT) / 255, lw=2, label="Padrão-ouro"),
             Line2D([], [], color=np.array(COR_ALG) / 255, lw=2, label="Máscara do algoritmo")]
    fig.legend(handles=itens, loc="lower center", ncol=2, fontsize=7.5,
               frameon=False, bbox_to_anchor=(0.5, y))


def _salvar(fig, nome: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(DIR_FIGURAS / f"{nome}.{ext}", dpi=250)
    plt.close(fig)


def _fmt(v: float) -> str:
    return "–" if pd.isna(v) else f"{v:.2f}".replace(".", ",")


# Escolha automática das imagens
def escolher(jac: pd.DataFrame, pelos: pd.Series, args) -> dict:
    j = jac.dropna()
    esc = {}

    # pré-processamento: bastante pelo (perto de 20%) e bem segmentada
    cand = pelos[(pelos > 12) & (pelos < 28)].index.intersection(j.index)
    cand = cand[j.loc[cand, "dullrazor_bilateral|otsu"] >= 0.7] if len(cand) else cand
    esc["preprocessamento"] = args.id_preproc or (
        (pelos.loc[cand] - 20).abs().idxmin() if len(cand) else pelos.idxmax())

    # 12 máscaras: alguma configuração com J >= 0,8 e grande diferença entre elas
    amplitude = j.max(axis=1) - j.min(axis=1)
    cand = amplitude[j.max(axis=1) >= 0.8]
    esc["mascaras"] = args.id_mascaras or (cand.idxmax() if len(cand) else amplitude.idxmax())

    # casos
    media = j.mean(axis=1)
    usadas = set(esc.values())
    acerto = media[j.min(axis=1) >= 0.8].drop(list(usadas), errors="ignore")
    esc["acerto"] = args.id_acerto or (acerto.idxmax() if len(acerto) else media.idxmax())
    falha = j.max(axis=1).drop(list(usadas | {esc["acerto"]}), errors="ignore")
    esc["falha"] = args.id_falha or falha.idxmin()
    dif = (j["clahe|otsu"] - j["clahe|snakes"]).drop(
        list(usadas | {esc["acerto"], esc["falha"]}), errors="ignore")
    esc["falha_snakes"] = args.id_snakes or dif.idxmax()

    # pelos: > 30% e maior ganho do DullRazor no Otsu
    usadas = set(esc.values())
    muitos = pelos[pelos > LIMITE_PELOS].index.intersection(j.index).difference(list(usadas))
    ganho = (j.loc[muitos, "dullrazor_bilateral|otsu"] - j.loc[muitos, "sem_tratamento|otsu"])
    esc["pelos"] = args.id_pelos or (ganho.idxmax() if len(ganho) else pelos.idxmax())
    return esc


# Figuras
def fig_preprocessamento(id_img: str) -> None:
    imgs = [ler_rgb(r, id_img) for r in RAMOS]
    h, w = imgs[0].shape[:2]
    # recorte ampliado: quadrado de 1/4 da largura, perto da borda da lesão
    gt = ler_gt(id_img)
    if gt is not None and gt.any():
        ys, xs = np.nonzero(gt)
        cy, cx = int(np.median(ys)), int(np.median(xs))
        cx = int(min(w - 1, xs.max() - (xs.max() - xs.min()) * 0.15))
    else:
        cy, cx = h // 2, w // 2
    lado = w // 4
    x0 = int(np.clip(cx - lado // 2, 0, w - lado))
    y0 = int(np.clip(cy - lado // 2, 0, h - lado))

    fig, axs = plt.subplots(2, 4, figsize=(6.3, 3.4),
                            gridspec_kw={"height_ratios": [h / w, 1]})
    for k, (ramo, img) in enumerate(zip(RAMOS, imgs)):
        axs[0, k].imshow(img)
        axs[0, k].add_patch(Rectangle((x0, y0), lado, lado, fill=False,
                                      ec=COR_RECORTE, lw=1.2))
        axs[0, k].set_title(NOME_RAMO[ramo].replace(" + CLAHE", "\n+ CLAHE"),
                            fontsize=7.5, color=TINTA)
        axs[1, k].imshow(img[y0:y0 + lado, x0:x0 + lado])
        for lado_ax in axs[1, k].spines.values():
            lado_ax.set_color(COR_RECORTE)
        axs[1, k].set_xticks([])
        axs[1, k].set_yticks([])
        _sem_eixos(axs[0, k])
    fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.02,
                        wspace=0.04, hspace=0.06)
    _salvar(fig, "ex_preprocessamento")


def fig_mascaras(id_img: str, jac: pd.DataFrame) -> None:
    gt = ler_gt(id_img)
    h, w = gt.shape if gt is not None else ler_rgb(RAMOS[0], id_img).shape[:2]
    alt_celula = 6.3 * 0.86 / 3 * (h / w)
    fig, axs = plt.subplots(4, 3, figsize=(6.3, alt_celula * 4 / 0.91))
    for i, ramo in enumerate(RAMOS):
        img = ler_rgb(ramo, id_img)
        for k, alg in enumerate(ALGORITMOS):
            ax = axs[i, k]
            ax.imshow(contornos(img, gt, ler_seg(ramo, alg, id_img)))
            ax.text(0.02, 0.97, f"J = {_fmt(jac.loc[id_img, f'{ramo}|{alg}'])}",
                    transform=ax.transAxes, ha="left", va="top", fontsize=7,
                    color=TINTA, bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5))
            _sem_eixos(ax)
            if i == 0:
                ax.set_title(NOME_ALG[alg], fontsize=8, color=TINTA)
            if k == 0:
                ax.set_ylabel(NOME_RAMO[ramo].replace(" + CLAHE", "\n+ CLAHE")
                              .replace("DullRazor + Bilateral", "DullRazor +\nBilateral"),
                              fontsize=7, color=TINTA)
    fig.subplots_adjust(left=0.13, right=0.99, top=0.96, bottom=0.05,
                        wspace=0.03, hspace=0.04)
    _legenda_contornos(fig)
    _salvar(fig, "ex_mascaras")


def fig_casos(esc: dict, jac: pd.DataFrame, pelos: pd.Series) -> None:
    # na falha geral, mostra a versão do pré-processamento em que ela foi melhor
    melhor = jac.loc[esc["falha"]].idxmax().split("|")[0]
    casos = [("acerto", "dullrazor_bilateral", "(a) Acerto em todas as configurações"),
             ("falha", melhor, "(b) Pior caso da base"),
             ("falha_snakes", "clahe", "(c) Falha do Snakes"),
             ("pelos", "sem_tratamento", "(d) Muitos pelos")]
    ref = ler_rgb(RAMOS[0], esc["acerto"])
    razao = ref.shape[0] / ref.shape[1]
    fig, axs = plt.subplots(4, 4, figsize=(6.3, 6.3 / 4 * razao * 4 + 1.3))
    for i, (chave, ramo, titulo) in enumerate(casos):
        id_img = esc[chave]
        img = ler_rgb(ramo, id_img)
        gt = ler_gt(id_img)
        axs[i, 0].imshow(contornos(img, gt, None))
        axs[i, 0].set_ylabel(f"{id_img}\n{pelos.get(id_img, np.nan):.0f}% pelos",
                             fontsize=6.5, color=TINTA_2)
        axs[i, 0].text(0, 1.04, titulo + f" — {NOME_RAMO[ramo]}",
                       transform=axs[i, 0].transAxes, fontsize=7.5,
                       color=TINTA, ha="left", va="bottom")
        for k, alg in enumerate(ALGORITMOS, start=1):
            ax = axs[i, k]
            ax.imshow(contornos(img, gt, ler_seg(ramo, alg, id_img)))
            ax.text(0.02, 0.97, f"{NOME_ALG[alg]}  J = {_fmt(jac.loc[id_img, f'{ramo}|{alg}'])}",
                    transform=ax.transAxes, ha="left", va="top", fontsize=6.5,
                    color=TINTA, bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5))
        for ax in axs[i]:
            _sem_eixos(ax)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.96, bottom=0.05,
                        wspace=0.03, hspace=0.22)
    _legenda_contornos(fig)
    _salvar(fig, "ex_casos")


def fig_pelos(id_img: str, jac: pd.DataFrame, pelos: pd.Series) -> None:
    gt = ler_gt(id_img)
    antes, depois = ler_rgb("sem_tratamento", id_img), ler_rgb("dullrazor_bilateral", id_img)
    razao = antes.shape[0] / antes.shape[1]
    fig, axs = plt.subplots(2, 2, figsize=(6.3, 6.3 / 2 * razao * 2 + 0.7))
    paineis = [
        (antes, None, "Sem tratamento"),
        (depois, None, "DullRazor + Bilateral"),
        (antes, ler_seg("sem_tratamento", "otsu", id_img),
         f"Otsu, sem tratamento (J = {_fmt(jac.loc[id_img, 'sem_tratamento|otsu'])})"),
        (depois, ler_seg("dullrazor_bilateral", "otsu", id_img),
         f"Otsu, DullRazor + Bilateral (J = {_fmt(jac.loc[id_img, 'dullrazor_bilateral|otsu'])})"),
    ]
    for ax, (img, m, titulo) in zip(axs.flat, paineis):
        ax.imshow(img if m is None else contornos(img, gt, m))
        ax.set_title(titulo, fontsize=7.5, color=TINTA)
        _sem_eixos(ax)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.07,
                        wspace=0.03, hspace=0.12)
    _legenda_contornos(fig)
    _salvar(fig, "ex_pelos")


# Execução
def main() -> None:
    p = argparse.ArgumentParser(description="Figuras de exemplo do Capítulo 5")
    for nome in ("preproc", "mascaras", "acerto", "falha", "snakes", "pelos"):
        p.add_argument(f"--id-{nome}", default=None, help="ID da imagem (ex.: ISIC_0000123)")
    args = p.parse_args()

    seg = pd.read_csv(DIR_META / "segmentacao.csv")
    seg = seg[seg["erro"].fillna("") == ""]
    seg["config"] = seg["pre_processamento"] + "|" + seg["algoritmo"]
    jac = seg.pivot_table(index="image_id", columns="config", values="jaccard")
    pelos = pd.read_csv(DIR_META / "preprocessamento.csv").set_index("image_id")["pct_pelos"]

    DIR_FIGURAS.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.family"] = "DejaVu Sans"
    esc = escolher(jac, pelos, args)

    fig_preprocessamento(esc["preprocessamento"])
    fig_mascaras(esc["mascaras"], jac)
    fig_casos(esc, jac, pelos)
    fig_pelos(esc["pelos"], jac, pelos)

    linhas = []
    for chave, id_img in esc.items():
        js = "\n".join(f"    {r:<27} " + "  ".join(
            f"{NOME_ALG[a]}={jac.loc[id_img, f'{r}|{a}']:.2f}" for a in ALGORITMOS)
            for r in RAMOS)
        linhas.append(f"{chave:<17} {id_img}  pelos={pelos.get(id_img, np.nan):.1f}%\n{js}")
    texto = "\n".join(linhas)
    (DIR_FIGURAS / "ex_escolhidas.txt").write_text(texto + "\n", encoding="utf-8")
    print(texto)
    print(f"\nFiguras salvas em {DIR_FIGURAS.relative_to(RAIZ)}/ex_*")


if __name__ == "__main__":
    main()