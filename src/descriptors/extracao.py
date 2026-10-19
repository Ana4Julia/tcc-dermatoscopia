"""
Extração dos 42 descritores da região da lesão.

Geométricos (5): área, índice de assimetria (Mendonça et al., 2013),
circularidade (4*pi*A / P^2, perímetro pelo método de Crofton), diâmetro
da circunferência circunscrita e diâmetro de Feret.

Colorimétricos (32): média, desvio padrão, assimetria e curtose dos canais
R, G, B, S, V, L*, a* e b*.

Textura (5): contraste, correlação, energia, homogeneidade e entropia da
GLCM (64 níveis, distância 1, média de 0°, 45°, 90° e 135°), usando apenas
pares de pixels dentro da lesão.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist
from scipy.stats import kurtosis, skew
from skimage.feature import graycomatrix
from skimage.measure import perimeter_crofton

# Parâmetros 
NIVEIS_CINZA = 64                       # quantização da GLCM
DISTANCIA_GLCM = 1                      # distância entre pares de pixels (px)
ANGULOS_GLCM = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
AREA_MINIMA = 10                        # abaixo disso, descritores = NaN

CANAIS = ["R", "G", "B", "S", "V", "L", "a", "b"]
ESTATISTICAS = ["media", "desvio", "assimetria", "curtose"]
GEOMETRICOS = ["area_px", "assimetria", "circularidade",
               "diametro_circulo", "diametro_feret"]
TEXTURA = ["contraste", "correlacao", "energia", "homogeneidade", "entropia"]

NOMES = ([f"geo_{n}" for n in GEOMETRICOS]
         + [f"cor_{c}_{e}" for c in CANAIS for e in ESTATISTICAS]
         + [f"tex_{n}" for n in TEXTURA])



# Descritores Geométricos
def _indice_assimetria(mascara: np.ndarray) -> float:
    m = cv2.moments(mascara, binaryImage=True)
    area = m["m00"]
    cx, cy = m["m10"] / area, m["m01"] / area
    # orientação do eixo principal de inércia (graus)
    theta = 0.5 * np.degrees(np.arctan2(2 * m["mu11"], m["mu20"] - m["mu02"]))

    # tela quadrada de lado ímpar com o centroide no pixel central
    h, w = mascara.shape
    lado = int(np.ceil(np.hypot(h, w))) | 1
    centro = lado // 2
    matriz = cv2.getRotationMatrix2D((cx, cy), theta, 1.0)
    matriz[0, 2] += centro - cx
    matriz[1, 2] += centro - cy
    alinhada = cv2.warpAffine(mascara, matriz, (lado, lado),
                              flags=cv2.INTER_NEAREST) > 0

    area_alinhada = alinhada.sum()
    delta_h = np.logical_xor(alinhada, np.flipud(alinhada)).sum() / 2
    delta_v = np.logical_xor(alinhada, np.fliplr(alinhada)).sum() / 2
    return float((delta_h + delta_v) / area_alinhada)


def descritores_geometricos(mascara: np.ndarray) -> dict:
    binaria = (mascara > 0).astype(np.uint8)
    area = int(binaria.sum())
    if area < AREA_MINIMA:
        return {f"geo_{n}": np.nan for n in GEOMETRICOS} | {"geo_area_px": area}

    contornos, _ = cv2.findContours(binaria, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_NONE)
    contorno = max(contornos, key=cv2.contourArea)
    perimetro = perimeter_crofton(binaria > 0, directions=4)
    (_, _), raio = cv2.minEnclosingCircle(contorno)

    pontos = contorno.reshape(-1, 2).astype(np.float64)
    if len(pontos) >= 3:
        casca = pontos[ConvexHull(pontos).vertices]
        feret = float(pdist(casca).max())
    else:
        feret = float(pdist(pontos).max()) if len(pontos) > 1 else 0.0

    return {
        "geo_area_px": area,
        "geo_assimetria": _indice_assimetria(binaria),
        "geo_circularidade": float(4 * np.pi * area / perimetro ** 2),
        "geo_diametro_circulo": float(2 * raio),
        "geo_diametro_feret": feret,
    }



# Descritores Colorimétricos
def descritores_cor(img_bgr: np.ndarray, mascara: np.ndarray) -> dict:
    regiao = mascara > 0
    if regiao.sum() < AREA_MINIMA:
        return {f"cor_{c}_{e}": np.nan for c in CANAIS for e in ESTATISTICAS}

    flutuante = img_bgr.astype(np.float32) / 255.0
    rgb = img_bgr[..., ::-1].astype(np.float64)                  # 0–255
    hsv = cv2.cvtColor(flutuante, cv2.COLOR_BGR2HSV)            # S, V: 0–1
    lab = cv2.cvtColor(flutuante, cv2.COLOR_BGR2LAB)            # L*: 0–100

    canais = {
        "R": rgb[..., 0], "G": rgb[..., 1], "B": rgb[..., 2],
        "S": hsv[..., 1], "V": hsv[..., 2],
        "L": lab[..., 0], "a": lab[..., 1], "b": lab[..., 2],
    }
    saida = {}
    for nome, canal in canais.items():
        valores = canal[regiao].astype(np.float64)
        constante = np.ptp(valores) == 0
        saida[f"cor_{nome}_media"] = float(valores.mean())
        saida[f"cor_{nome}_desvio"] = float(valores.std())
        saida[f"cor_{nome}_assimetria"] = np.nan if constante else float(skew(valores))
        saida[f"cor_{nome}_curtose"] = np.nan if constante else float(kurtosis(valores))
    return saida


# Descritores de Textura (GLCM)
def descritores_textura(img_bgr: np.ndarray, mascara: np.ndarray) -> dict:
    regiao = mascara > 0
    if regiao.sum() < AREA_MINIMA:
        return {f"tex_{n}": np.nan for n in TEXTURA}

    # recorte no retângulo da lesão (apenas por eficiência)
    linhas, colunas = np.nonzero(regiao)
    r0, r1, c0, c1 = linhas.min(), linhas.max() + 1, colunas.min(), colunas.max() + 1
    cinza = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)[r0:r1, c0:c1]
    dentro = regiao[r0:r1, c0:c1]

    # níveis 1..NIVEIS_CINZA dentro da lesão; 0 fora (descartado depois)
    quantizada = (cinza.astype(np.uint16) * NIVEIS_CINZA // 256 + 1).astype(np.uint8)
    quantizada[~dentro] = 0

    glcm = graycomatrix(quantizada, [DISTANCIA_GLCM], ANGULOS_GLCM,
                        levels=NIVEIS_CINZA + 1, symmetric=True, normed=False)
    glcm = glcm[1:, 1:, 0, :].astype(np.float64)       # remove pares com o fundo

    i, j = np.meshgrid(np.arange(NIVEIS_CINZA), np.arange(NIVEIS_CINZA),
                       indexing="ij")
    resultados = {n: [] for n in TEXTURA}
    for k in range(glcm.shape[-1]):
        total = glcm[..., k].sum()
        if total == 0:
            continue
        p = glcm[..., k] / total
        mu_i, mu_j = (i * p).sum(), (j * p).sum()
        sd_i = np.sqrt(((i - mu_i) ** 2 * p).sum())
        sd_j = np.sqrt(((j - mu_j) ** 2 * p).sum())
        resultados["contraste"].append(((i - j) ** 2 * p).sum())
        resultados["homogeneidade"].append((p / (1 + (i - j) ** 2)).sum())
        resultados["energia"].append(np.sqrt((p ** 2).sum()))
        resultados["correlacao"].append(
            ((i - mu_i) * (j - mu_j) * p).sum() / (sd_i * sd_j)
            if sd_i > 0 and sd_j > 0 else np.nan)
        nz = p[p > 0]
        resultados["entropia"].append(-(nz * np.log2(nz)).sum())

    return {f"tex_{n}": float(np.nanmean(v)) if np.isfinite(v).any() else np.nan
            for n, v in ((n, np.asarray(v, dtype=float)) for n, v in resultados.items())}


# Todos os descritores
def extrair_descritores(img_bgr: np.ndarray, mascara: np.ndarray) -> dict:
    """Retorna os 42 descritores (dicionário na ordem de NOMES)."""
    if img_bgr.shape[:2] != mascara.shape[:2]:
        raise ValueError("Imagem e máscara com dimensões diferentes.")
    d = {}
    d.update(descritores_geometricos(mascara))
    d.update(descritores_cor(img_bgr, mascara))
    d.update(descritores_textura(img_bgr, mascara))
    return {n: d[n] for n in NOMES}
