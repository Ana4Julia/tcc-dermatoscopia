"""
Pós-processamento aplicado igualmente às máscaras do Otsu, K-Means e Snakes:
  1. mantém só a região conexa com maior área x centralidade
     (critério semelhante ao de Tamoor et al., 2023);
  2. preenche os buracos internos da lesão.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes


def selecionar_lesao(mascara: np.ndarray) -> np.ndarray:
    """Mantém apenas a região conexa com maior (área x centralidade)."""
    binaria = (mascara > 0).astype(np.uint8)
    n, rotulos, stats, centroides = cv2.connectedComponentsWithStats(
        binaria, connectivity=8)
    if n <= 1:                       # só fundo: máscara vazia
        return np.zeros_like(binaria, dtype=np.uint8)

    h, w = binaria.shape
    centro = np.array([w / 2, h / 2])
    meia_diagonal = np.hypot(w, h) / 2

    melhor, melhor_pontuacao = 0, -1.0
    for i in range(1, n):            # 0 é o fundo
        area = stats[i, cv2.CC_STAT_AREA]
        distancia = np.linalg.norm(centroides[i] - centro) / meia_diagonal
        pontuacao = area * (1.0 - distancia)
        if pontuacao > melhor_pontuacao:
            melhor, melhor_pontuacao = i, pontuacao

    return (rotulos == melhor).astype(np.uint8) * 255


def preencher_buracos(mascara: np.ndarray) -> np.ndarray:
    return binary_fill_holes(mascara > 0).astype(np.uint8) * 255


def pos_processar(mascara: np.ndarray) -> np.ndarray:
    """Aplica as duas etapas e retorna máscara uint8 (0/255)."""
    return preencher_buracos(selecionar_lesao(mascara))
