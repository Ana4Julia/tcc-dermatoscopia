"""
Segmentação por limiarização de Otsu (Otsu, 1979), com o OpenCV.

O limiar é calculado só com os pixels do campo de visão (sem a moldura do
dermatoscópio). Como a lesão é mais escura que a pele, os pixels abaixo do
limiar formam a máscara.

Teste:
    python -m src.segmentation.otsu --ids ISIC_0000039 ISIC_0009944
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from src.segmentation.fundo import campo_de_visao
from src.segmentation.pos_processamento import pos_processar
from src.validation.metrics import validar_segmentacao

RAIZ = Path(__file__).resolve().parents[2]
DIR_PROCESSADO = RAIZ / "data" / "processed"
DIR_MASCARAS_GT = RAIZ / "data" / "processed" / "masks_gt"
DIR_TESTE = RAIZ / "data" / "segmentation" / "_test" / "otsu"

RAMOS = ["sem_tratamento", "dullrazor_bilateral", "clahe",
         "dullrazor_bilateral_clahe"]


# Algoritmo
def segmentar_otsu(img_bgr: np.ndarray, retornar_limiar: bool = False):
    """Segmenta a lesão por Otsu e retorna a máscara binária (0/255)."""
    if img_bgr is None or img_bgr.ndim != 3:
        raise ValueError("Esperada imagem colorida BGR (H x W x 3).")

    cinza = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    fov = campo_de_visao(img_bgr)

    # limiar de Otsu calculado só com os pixels do campo de visão
    limiar, _ = cv2.threshold(cinza[fov].reshape(-1, 1), 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    mascara = np.where((cinza <= limiar) & fov, 255, 0).astype(np.uint8)
    mascara = pos_processar(mascara)

    if retornar_limiar:
        return mascara, limiar
    return mascara


# Teste em poucas imagens (salva em data/segmentation/_test/otsu)
def _desenhar(img: np.ndarray, mascara: np.ndarray, gt: np.ndarray | None,
              titulo: str) -> np.ndarray:
    saida = img.copy()
    espessura = max(2, round(max(img.shape[:2]) / 400))
    if gt is not None:                                   # amarelo = padrão-ouro
        c, _ = cv2.findContours(gt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(saida, c, -1, (0, 255, 255), espessura)
    c, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(saida, c, -1, (0, 255, 0), espessura)  # verde = segmentação
    escala = max(img.shape[1] / 900, 1.0)
    cv2.putText(saida, titulo, (15, int(40 * escala)), cv2.FONT_HERSHEY_SIMPLEX,
                1.0 * escala, (0, 0, 0), int(4 * escala), cv2.LINE_AA)
    cv2.putText(saida, titulo, (15, int(40 * escala)), cv2.FONT_HERSHEY_SIMPLEX,
                1.0 * escala, (255, 255, 255), int(2 * escala), cv2.LINE_AA)
    return saida


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
            mascara, limiar = segmentar_otsu(img, retornar_limiar=True)
            cv2.imwrite(str(DIR_TESTE / f"{id_img}_{ramo}.png"), mascara)

            texto = f"{ramo} | k={limiar:.0f}"
            if gt is not None:
                r = validar_segmentacao(mascara, gt)
                jac, dice = r["jaccard"], r["dice"]
                texto += f" | J={jac:.3f} D={dice:.3f}"
            print(f"{id_img}  {texto}")
            paineis.append(_desenhar(img, mascara, gt, texto))

        if len(paineis) == 4:
            grade = np.vstack([np.hstack(paineis[:2]), np.hstack(paineis[2:])])
            cv2.imwrite(str(DIR_TESTE / f"{id_img}_comparacao.png"), grade)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Teste da segmentação por Otsu")
    parser.add_argument("--ids", nargs="+", required=True,
                        help="IDs ISIC (ex.: ISIC_0000039)")
    _testar(parser.parse_args().ids)