"""
Índice de Jaccard e Coeficiente de Dice entre uma máscara e a máscara
padrão-ouro. As mesmas funções são usadas para os três algoritmos.

Teste:
    python -m src.validation.metrics ISIC_0000043
"""

from __future__ import annotations

import numpy as np


def _verificar_dimensoes(a: np.ndarray, b: np.ndarray) -> None:
    if a.shape[:2] != b.shape[:2]:
        raise ValueError(f"Máscaras com dimensões diferentes: {a.shape[:2]} x "
                         f"{b.shape[:2]}. Use as máscaras de data/processed/masks_gt.")


def indice_jaccard(mascara_a: np.ndarray, mascara_b: np.ndarray) -> float:
    """
    Calcula o Índice de Jaccard (Intersection over Union) entre duas máscaras binárias.
    J(A,B) = |A ∩ B| / |A ∪ B|
    """
    _verificar_dimensoes(mascara_a, mascara_b)
    a = mascara_a > 0
    b = mascara_b > 0

    intersecao = np.logical_and(a, b).sum()
    uniao = np.logical_or(a, b).sum()

    if uniao == 0:          # as duas máscaras vazias: concordância total
        return 1.0

    return float(intersecao / uniao)


def coeficiente_dice(mascara_a: np.ndarray, mascara_b: np.ndarray) -> float:
    """
    Calcula o Coeficiente de Dice entre duas máscaras binárias.
    D(A,B) = 2|A ∩ B| / (|A| + |B|)
    """
    _verificar_dimensoes(mascara_a, mascara_b)
    a = mascara_a > 0
    b = mascara_b > 0

    intersecao = np.logical_and(a, b).sum()
    soma_areas = a.sum() + b.sum()

    if soma_areas == 0:     # as duas máscaras vazias: concordância total
        return 1.0

    return float(2 * intersecao / soma_areas)


def validar_segmentacao(mascara_predita: np.ndarray, mascara_gt: np.ndarray) -> dict:
    """Calcula Jaccard e Dice de uma vez, retornando um dicionário com os resultados."""
    return {
        "jaccard": indice_jaccard(mascara_predita, mascara_gt),
        "dice": coeficiente_dice(mascara_predita, mascara_gt),
    }


if __name__ == "__main__":
    import sys
    from pathlib import Path

    import cv2

    from src.segmentation.kmeans import segmentar_kmeans
    from src.segmentation.otsu import segmentar_otsu
    from src.segmentation.snakes import segmentar_snakes

    ALGORITMOS = {"Otsu": segmentar_otsu, "K-Means": segmentar_kmeans,
                  "Snakes": segmentar_snakes}
    RAMOS = ["sem_tratamento", "dullrazor_bilateral", "clahe",
             "dullrazor_bilateral_clahe"]

    raiz = Path(__file__).resolve().parents[2]
    id_img = sys.argv[1] if len(sys.argv) > 1 else "ISIC_0000043"

    caminho_gt = raiz / "data" / "processed" / "masks_gt" / f"{id_img}_segmentation.png"
    mascara_gt = cv2.imread(str(caminho_gt), cv2.IMREAD_GRAYSCALE)
    if mascara_gt is None:
        raise FileNotFoundError(f"Máscara padrão-ouro não encontrada: {caminho_gt}")

    print(f"\nValidação para {id_img}:")
    print(f"{'Algoritmo':<10} {'Pré-processamento':<28} {'Jaccard':<9} {'Dice':<9}")
    print("-" * 58)
    for nome_alg, segmentar in ALGORITMOS.items():
        for ramo in RAMOS:
            caminho = raiz / "data" / "processed" / ramo / f"{id_img}.png"
            imagem = cv2.imread(str(caminho))
            if imagem is None:
                print(f"[aviso] não encontrada: {caminho.relative_to(raiz)}")
                continue
            r = validar_segmentacao(segmentar(imagem), mascara_gt)
            print(f"{nome_alg:<10} {ramo:<28} {r['jaccard']:.4f}    {r['dice']:.4f}")