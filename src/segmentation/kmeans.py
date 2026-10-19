"""
Segmentação por K-Means (MacQueen, 1967) no espaço CIE-Lab, com o OpenCV
e inicialização K-Means++.

Os pixels do campo de visão são agrupados em dois grupos pela cor (L*, a*, b*),
e o grupo de menor luminância é tomado como lesão.

Teste:
    python -m src.segmentation.kmeans --ids ISIC_0000039 ISIC_0009944
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from src.segmentation.otsu import _desenhar
from src.segmentation.fundo import campo_de_visao
from src.segmentation.pos_processamento import pos_processar
from src.validation.metrics import validar_segmentacao

# Parâmetros
K = 2                   # nº de grupos: lesão x pele
MAX_ITERACOES = 50      # critério de parada 1
EPSILON = 0.2           # critério de parada 2 (deslocamento mínimo dos centróides)
TENTATIVAS = 10         # nº de inicializações; fica a de menor variância interna
SEMENTE = 42            # semente fixa, para o resultado ser sempre o mesmo

RAIZ = Path(__file__).resolve().parents[2]
DIR_PROCESSADO = RAIZ / "data" / "processed"
DIR_MASCARAS_GT = DIR_PROCESSADO / "masks_gt"
DIR_TESTE = RAIZ / "data" / "segmentation" / "_test" / "kmeans"

RAMOS = ["sem_tratamento", "dullrazor_bilateral", "clahe",
         "dullrazor_bilateral_clahe"]


# Algoritmo
def segmentar_kmeans(img_bgr: np.ndarray) -> np.ndarray:
    """Segmenta a lesão por K-Means (Lab) e retorna a máscara binária (0/255)."""
    if img_bgr is None or img_bgr.ndim != 3:
        raise ValueError("Esperada imagem colorida BGR (H x W x 3).")

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    fov = campo_de_visao(img_bgr)
    amostras = lab[fov].astype(np.float32)          # só o campo de visão

    cv2.setRNGSeed(SEMENTE)
    criterios = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                 MAX_ITERACOES, EPSILON)
    _, rotulos, centros = cv2.kmeans(amostras, K, None, criterios,
                                     TENTATIVAS, cv2.KMEANS_PP_CENTERS)

    # Lesão = grupo com menor luminância média (canal L*)
    grupo_lesao = int(np.argmin(centros[:, 0]))
    mascara = np.zeros(lab.shape[:2], dtype=np.uint8)
    mascara[fov] = np.where(rotulos.ravel() == grupo_lesao, 255, 0)

    return pos_processar(mascara)


# Teste em poucas imagens (salva em data/segmentation/_test/kmeans)
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
            mascara = segmentar_kmeans(img)
            cv2.imwrite(str(DIR_TESTE / f"{id_img}_{ramo}.png"), mascara)

            texto = f"{ramo}"
            if gt is not None:
                r = validar_segmentacao(mascara, gt)
                texto += f" | J={r['jaccard']:.3f} D={r['dice']:.3f}"
            print(f"{id_img}  {texto}")
            paineis.append(_desenhar(img, mascara, gt, texto))

        if len(paineis) == 4:
            grade = np.vstack([np.hstack(paineis[:2]), np.hstack(paineis[2:])])
            cv2.imwrite(str(DIR_TESTE / f"{id_img}_comparacao.png"), grade)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Teste da segmentação por K-Means")
    parser.add_argument("--ids", nargs="+", required=True,
                        help="IDs ISIC (ex.: ISIC_0000039)")
    _testar(parser.parse_args().ids)