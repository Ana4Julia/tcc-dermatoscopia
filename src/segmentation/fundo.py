"""
Detecção da moldura escura do dermatoscópio (campo de visão).

A moldura preta que aparece em muitas imagens da ISIC seria confundida com a
lesão por ser a região mais escura. Por isso, antes da segmentação:
  1. detecta-se a moldura (região escura ligada à borda e que ocupa os cantos);
  2. o campo de visão é o restante da imagem, sem a faixa de sombra junto à moldura;
  3. o que fica fora do campo de visão é pintado com a cor da pele.

Nas imagens sem moldura, o campo de visão é a imagem inteira.
"""

from __future__ import annotations

import cv2
import numpy as np

# Parâmetros (frações são relativas ao menor lado da imagem)
LIMIAR_MOLDURA = 60          # nível de cinza abaixo do qual o pixel é "escuro"
AREA_MINIMA_MOLDURA = 0.01   # fração mínima da imagem para haver moldura
CANTOS_MINIMOS = 2           # a moldura precisa ocupar pelo menos 2 cantos
FRACAO_CANTO = 0.03          # tamanho do canto verificado
FRACAO_EROSAO = 0.02         # redução do campo de visão
FRACAO_MARGEM = 0.02         # faixa junto à borda considerada "borda"
LARGURA_MAX_SOMBRA = 0.10    # largura máxima procurada da faixa de sombra
ANEL_PX = 4                  # espessura dos anéis do perfil de brilho (px)
FRACAO_PLATO = 0.90          # a sombra termina quando o brilho atinge 90% do máximo


def campo_de_visao(img_bgr: np.ndarray) -> np.ndarray:
    """Máscara booleana do campo de visão (True = dentro do dermatoscópio)."""
    cinza = cv2.medianBlur(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY), 5)
    h, w = cinza.shape
    escuro = (cinza < LIMIAR_MOLDURA).astype(np.uint8)
    # abertura: remove os pelos e mantém a moldura, que é espessa
    tam = max(3, int(round(FRACAO_EROSAO * min(h, w))) | 1)
    elemento = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tam, tam))
    escuro = cv2.morphologyEx(escuro, cv2.MORPH_OPEN, elemento)

    n, rotulos = cv2.connectedComponents(escuro, connectivity=8)
    # regiões escuras que encostam na borda
    m = max(2, int(round(FRACAO_MARGEM * min(h, w))))
    borda = np.unique(np.concatenate([rotulos[:m].ravel(), rotulos[-m:].ravel(),
                                      rotulos[:, :m].ravel(), rotulos[:, -m:].ravel()]))
    borda = borda[borda != 0]
    moldura = np.isin(rotulos, borda)

    # a moldura ocupa os cantos; uma lesão escura encostada na borda, não
    c = max(2, int(FRACAO_CANTO * min(h, w)))
    cantos = [moldura[m:m + c, m:m + c], moldura[m:m + c, -m - c:-m],
              moldura[-m - c:-m, m:m + c], moldura[-m - c:-m, -m - c:-m]]
    n_cantos = sum(canto.mean() > 0.5 for canto in cantos)

    if moldura.sum() < AREA_MINIMA_MOLDURA * h * w or n_cantos < CANTOS_MINIMOS:
        return np.ones((h, w), dtype=bool)

    fov = cv2.erode((~moldura).astype(np.uint8), elemento)

    # remove a faixa de sombra junto à moldura: mede o brilho em anéis a partir
    # da borda e corta no primeiro anel que atinge 90% do brilho máximo
    distancia = cv2.distanceTransform(fov, cv2.DIST_L2, 5)
    limite = LARGURA_MAX_SOMBRA * min(h, w)
    inicios = np.arange(0, limite, ANEL_PX)
    medianas = np.array([np.median(cinza[(distancia > d) & (distancia <= d + ANEL_PX)])
                         for d in inicios])
    platô = medianas.max()
    largura = inicios[np.argmax(medianas >= FRACAO_PLATO * platô)]
    return distancia > largura


def neutralizar_fundo(img_bgr: np.ndarray, fov: np.ndarray) -> np.ndarray:
    """Pinta o que está fora do campo de visão com a cor da pele junto à borda."""
    if fov.all():
        return img_bgr
    fov_u8 = fov.astype(np.uint8)
    anel = fov_u8 - cv2.erode(fov_u8, np.ones((15, 15), np.uint8))
    referencia = img_bgr[anel.astype(bool)] if anel.any() else img_bgr[fov]
    saida = img_bgr.copy()
    saida[~fov] = np.median(referencia, axis=0).astype(np.uint8)
    return saida