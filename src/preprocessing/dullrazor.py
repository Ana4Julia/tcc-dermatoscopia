"""
DullRazor: remoção de pelos em imagens dermatoscópicas (Lee et al., 1997).

Baseado na implementação de Javier Velasquez
(https://github.com/BlueDokk/Dullrazor-algorithm).

Etapas: detecção dos pelos por fechamento morfológico (0°, 45° e 90°),
preenchimento por inpainting de Telea e suavização com filtro de mediana.

Teste em poucas imagens:
    python -m src.preprocessing.dullrazor --ids ISIC_0009944
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

# Parâmetros. Os tamanhos em pixels vêm da implementação de referência,
# ajustada para imagens de 600 px no maior lado; por isso são multiplicados
# por max(lado maior / 600, 1). A mediana e o raio do inpainting são fixos.


LADO_REFERENCIA = 600      # lado maior das imagens da implementação de referência
TAMANHO_LINHA = 9          # comprimento (px) dos elementos lineares
KERNEL_GAUSS = (3, 3)      # suavização da resposta morfológica
LIMIAR = 10                # limiar de binarização da máscara de pelos
RAIO_INPAINT = 6           # raio do inpainting (não escalado)
KERNEL_DILATACAO = 3       # dilatação leve da máscara
KERNEL_MEDIANA = 5         # mediana na máscara dilatada (não escalada)

RAIZ = Path(__file__).resolve().parents[2]
DIR_IMAGENS = RAIZ / "data" / "raw" / "images"
DIR_TESTE = RAIZ / "data" / "processed" / "_test" / "dullrazor"
DIR_FIGURAS = Path(__file__).resolve().parent / "figuras"


# Algoritmo
def _escalar(valor: int, img: np.ndarray, impar: bool = False) -> int:
    """Ajusta um tamanho em pixels à resolução da imagem."""
    escala = max(max(img.shape[:2]) / LADO_REFERENCIA, 1.0)
    v = max(int(round(valor * escala)), valor)
    if impar and v % 2 == 0:
        v += 1
    return v


def _elementos_lineares(tamanho: int) -> list[np.ndarray]:
    """Elementos estruturantes lineares em 0°, 45° e 90°."""
    horizontal = np.ones((1, tamanho), dtype=np.uint8)            # 0°
    diagonal = np.fliplr(np.eye(tamanho, dtype=np.uint8))          # 45°
    vertical = np.ones((tamanho, 1), dtype=np.uint8)              # 90°
    return [horizontal, diagonal, vertical]


def detectar_pelos(img_bgr: np.ndarray) -> np.ndarray:
    """Retorna a máscara binária de pelos (0/255)."""
    cinza = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Fechamento nas três direções; o máximo apaga pelos em qualquer orientação
    fechamentos = [
        cv2.morphologyEx(cinza, cv2.MORPH_CLOSE, k)
        for k in _elementos_lineares(_escalar(TAMANHO_LINHA, img_bgr, impar=True))
    ]
    fechamento_max = np.maximum.reduce(fechamentos)

    # A diferença entre o fechamento e a original realça os pelos
    resposta = cv2.subtract(fechamento_max, cinza)

    resposta = cv2.GaussianBlur(resposta, KERNEL_GAUSS, cv2.BORDER_DEFAULT)
    _, mascara = cv2.threshold(resposta, LIMIAR, 255, cv2.THRESH_BINARY)
    return mascara


def aplicar_dullrazor(
    img_bgr: np.ndarray, retornar_mascara: bool = False
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Aplica o DullRazor a uma imagem BGR e retorna a imagem sem pelos."""
    if img_bgr is None or img_bgr.ndim != 3:
        raise ValueError("Esperada imagem colorida BGR (H x W x 3).")

    # 1. Detecção dos pelos
    mascara = detectar_pelos(img_bgr)

    # 2. Preenchimento das regiões de pelo (inpainting)
    limpa = cv2.inpaint(img_bgr, mascara, RAIO_INPAINT, cv2.INPAINT_TELEA)

    # 3. Suavização só dentro da máscara levemente dilatada
    tam_dil = _escalar(KERNEL_DILATACAO, img_bgr, impar=True)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tam_dil, tam_dil))
    mascara_dilatada = cv2.dilate(mascara, k, iterations=1)
    mediana = cv2.medianBlur(limpa, KERNEL_MEDIANA)
    regiao = mascara_dilatada.astype(bool)
    limpa[regiao] = mediana[regiao]

    if retornar_mascara:
        return limpa, mascara
    return limpa


# Teste em poucas imagens (salva em data/processed/_test/dullrazor)
def _rotular(img: np.ndarray, texto: str) -> np.ndarray:
    saida = img.copy()
    cv2.putText(saida, texto, (15, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1.2, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(saida, texto, (15, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1.2, (255, 255, 255), 2, cv2.LINE_AA)
    return saida


def _testar(ids: list[str] | None, n: int) -> None:
    if ids:
        caminhos = [DIR_IMAGENS / f"{i}.jpg" for i in ids]
    else:
        caminhos = sorted(DIR_IMAGENS.glob("*.jpg"))[:n]

    DIR_TESTE.mkdir(parents=True, exist_ok=True)
    DIR_FIGURAS.mkdir(parents=True, exist_ok=True)

    for caminho in caminhos:
        img = cv2.imread(str(caminho), cv2.IMREAD_COLOR)
        if img is None:
            print(f"[aviso] não foi possível ler {caminho}")
            continue

        limpa, mascara = aplicar_dullrazor(img, retornar_mascara=True)
        nome = caminho.stem
        cobertura = 100 * np.count_nonzero(mascara) / mascara.size

        cv2.imwrite(str(DIR_TESTE / f"{nome}.png"), limpa)
        cv2.imwrite(str(DIR_TESTE / f"{nome}_mascara_pelos.png"), mascara)

        comparacao = np.hstack([_rotular(img, "Original"),
                                _rotular(limpa, "DullRazor")])
        cv2.imwrite(str(DIR_FIGURAS / f"dullrazor_{nome}.png"), comparacao)

        print(f"{nome}: {img.shape[1]}x{img.shape[0]} px, "
              f"pelos detectados em {cobertura:.2f}% da imagem")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Teste do DullRazor")
    parser.add_argument("--ids", nargs="*", help="IDs ISIC (ex.: ISIC_0009944)")
    parser.add_argument("--n", type=int, default=5,
                        help="nº de imagens se --ids não for informado")
    args = parser.parse_args()
    _testar(args.ids, args.n)