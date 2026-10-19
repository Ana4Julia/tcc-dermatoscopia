"""
Segmentação por contornos ativos (Snakes; Kass et al., 1988), com a
implementação do scikit-image.

A curva parte de uma elipse inscrita no campo de visão (90% dos semieixos),
igual para todas as imagens, e se contrai até ser detida pelas bordas.
A moldura do dermatoscópio é neutralizada antes, para não atrair a curva.

Teste:
    python -m src.segmentation.snakes --ids ISIC_0000039 ISIC_0009944
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from skimage.draw import polygon
from skimage.filters import gaussian
from skimage.segmentation import active_contour

from src.segmentation.fundo import campo_de_visao, neutralizar_fundo
from src.segmentation.otsu import _desenhar
from src.segmentation.pos_processamento import pos_processar
from src.validation.metrics import validar_segmentacao

# Parâmetros. ALFA, BETA e GAMA seguem o exemplo do scikit-image; W_BORDA foi
# elevado de 1 para 10 porque, com peso 1, a curva atravessava a borda da lesão.
SIGMA_GAUSS = 3.0       # suavização gaussiana
N_PONTOS = 400          # nº de pontos do contorno
FRACAO_EIXOS = 0.90     # semieixos iniciais = 90% da meia largura / meia altura
ALFA = 0.015            # elasticidade
BETA = 10.0             # rigidez
GAMA = 0.001            # passo de tempo
W_LINHA = 0.0           # peso da atração pela intensidade
W_BORDA = 10.0          # peso da atração pelas bordas
MAX_ITERACOES = 2500

RAIZ = Path(__file__).resolve().parents[2]
DIR_PROCESSADO = RAIZ / "data" / "processed"
DIR_MASCARAS_GT = DIR_PROCESSADO / "masks_gt"
DIR_TESTE = RAIZ / "data" / "segmentation" / "_test" / "snakes"

RAMOS = ["sem_tratamento", "dullrazor_bilateral", "clahe",
         "dullrazor_bilateral_clahe"]


# Algoritmo
def _caixa(fov: np.ndarray) -> tuple[float, float, float, float]:
    """Retângulo (linha0, linha1, coluna0, coluna1) que envolve o campo de visão."""
    linhas, colunas = np.nonzero(fov)
    return linhas.min(), linhas.max() + 1, colunas.min(), colunas.max() + 1


def contorno_inicial(fov: np.ndarray) -> np.ndarray:
    """Elipse inscrita no campo de visão, em coordenadas (linha, coluna)."""
    r0, r1, c0, c1 = _caixa(fov)
    t = np.linspace(0, 2 * np.pi, N_PONTOS, endpoint=False)
    linhas = (r0 + r1) / 2 + FRACAO_EIXOS * ((r1 - r0) / 2) * np.sin(t)
    colunas = (c0 + c1) / 2 + FRACAO_EIXOS * ((c1 - c0) / 2) * np.cos(t)
    return np.column_stack([linhas, colunas])


def segmentar_snakes(img_bgr: np.ndarray, retornar_contorno: bool = False):
    """Segmenta a lesão por Snakes e retorna a máscara binária (0/255)."""
    if img_bgr is None or img_bgr.ndim != 3:
        raise ValueError("Esperada imagem colorida BGR (H x W x 3).")

    fov = campo_de_visao(img_bgr)
    img_bgr = neutralizar_fundo(img_bgr, fov)
    cinza = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255
    suavizada = gaussian(cinza, sigma=SIGMA_GAUSS, preserve_range=True)
    altura, largura = cinza.shape

    contorno = active_contour(
        suavizada, contorno_inicial(fov),
        alpha=ALFA, beta=BETA, gamma=GAMA, w_line=W_LINHA, w_edge=W_BORDA,
        max_num_iter=MAX_ITERACOES, boundary_condition="periodic")

    mascara = np.zeros((altura, largura), dtype=np.uint8)
    rr, cc = polygon(contorno[:, 0], contorno[:, 1], shape=mascara.shape)
    mascara[rr, cc] = 255
    mascara[~fov] = 0
    mascara = pos_processar(mascara)

    if retornar_contorno:
        return mascara, contorno
    return mascara


# Teste em poucas imagens (salva em data/segmentation/_test/snakes)
def _testar(ids: list[str]) -> None:
    DIR_TESTE.mkdir(parents=True, exist_ok=True)
    for id_img in ids:
        gt = cv2.imread(str(DIR_MASCARAS_GT / f"{id_img}_segmentation.png"),
                        cv2.IMREAD_GRAYSCALE)
        paineis = []
        for ramo in RAMOS:
            img = cv2.imread(str(DIR_PROCESSADO / ramo / f"{id_img}.png"))
            if img is None:
                print(f"[aviso] {ramo}/{id_img}.png não encontrada "
                      "(rode o pipeline de pré-processamento)")
                continue
            mascara = segmentar_snakes(img)
            cv2.imwrite(str(DIR_TESTE / f"{id_img}_{ramo}.png"), mascara)

            texto = f"{ramo}"
            if gt is not None:
                r = validar_segmentacao(mascara, gt)
                texto += f" | J={r['jaccard']:.3f} D={r['dice']:.3f}"
            print(f"{id_img}  {texto}")

            painel = _desenhar(img, mascara, gt, texto)
            h, w = img.shape[:2]                     # azul = contorno inicial
            ini = contorno_inicial(campo_de_visao(img))[:, ::-1].round().astype(np.int32)
            cv2.polylines(painel, [ini], True, (255, 0, 0),
                          max(2, round(max(h, w) / 400)))
            paineis.append(painel)

        if len(paineis) == 4:
            grade = np.vstack([np.hstack(paineis[:2]), np.hstack(paineis[2:])])
            cv2.imwrite(str(DIR_TESTE / f"{id_img}_comparacao.png"), grade)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Teste da segmentação por Snakes")
    parser.add_argument("--ids", nargs="+", required=True,
                        help="IDs ISIC (ex.: ISIC_0000039)")
    _testar(parser.parse_args().ids)