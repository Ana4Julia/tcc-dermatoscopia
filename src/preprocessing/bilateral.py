"""
Filtragem Bilateral (Tomasi e Manduchi, 1998): suavização que preserva as
bordas, aplicada no espaço CIE-Lab.

Teste em poucas imagens:
    python -m src.preprocessing.bilateral --ids ISIC_0009944
    python -m src.preprocessing.bilateral --ids ISIC_0009944 --com-dullrazor
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

# Parâmetros
DIAMETRO = 9      # vizinhança (px) considerada por pixel
SIGMA_D = 75      # espalhamento geométrico (sigmaSpace no OpenCV)
SIGMA_R = 75      # espalhamento de cor (sigmaColor no OpenCV)

RAIZ = Path(__file__).resolve().parents[2]
DIR_IMAGENS = RAIZ / "data" / "raw" / "images"
DIR_TESTE = RAIZ / "data" / "processed" / "_test"
DIR_FIGURAS = Path(__file__).resolve().parent / "figuras"


# Algoritmo
def aplicar_bilateral(img_bgr: np.ndarray) -> np.ndarray:
    """Aplica a Filtragem Bilateral no espaço CIE-Lab e retorna a imagem em BGR."""
    if img_bgr is None or img_bgr.ndim != 3:
        raise ValueError("Esperada imagem colorida BGR (H x W x 3).")

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    lab_filtrada = cv2.bilateralFilter(lab, DIAMETRO, SIGMA_R, SIGMA_D)
    return cv2.cvtColor(lab_filtrada, cv2.COLOR_LAB2BGR)


# Teste em poucas imagens (salva em data/processed/_test)
def _rotular(img: np.ndarray, texto: str) -> np.ndarray:
    saida = img.copy()
    cv2.putText(saida, texto, (15, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1.2, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(saida, texto, (15, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1.2, (255, 255, 255), 2, cv2.LINE_AA)
    return saida


def _testar(ids: list[str] | None, n: int, com_dullrazor: bool) -> None:
    if ids:
        caminhos = [DIR_IMAGENS / f"{i}.jpg" for i in ids]
    else:
        caminhos = sorted(DIR_IMAGENS.glob("*.jpg"))[:n]

    if com_dullrazor:
        from src.preprocessing.dullrazor import aplicar_dullrazor
        pasta, rotulo = "dullrazor_bilateral", "DullRazor + Bilateral"
    else:
        pasta, rotulo = "bilateral", "Filtro Bilateral"

    dir_saida = DIR_TESTE / pasta
    dir_saida.mkdir(parents=True, exist_ok=True)
    DIR_FIGURAS.mkdir(parents=True, exist_ok=True)

    for caminho in caminhos:
        img = cv2.imread(str(caminho), cv2.IMREAD_COLOR)
        if img is None:
            print(f"[aviso] não foi possível ler {caminho}")
            continue

        entrada = aplicar_dullrazor(img) if com_dullrazor else img
        filtrada = aplicar_bilateral(entrada)
        nome = caminho.stem

        cv2.imwrite(str(dir_saida / f"{nome}.png"), filtrada)
        comparacao = np.hstack([_rotular(img, "Original"),
                                _rotular(filtrada, rotulo)])
        cv2.imwrite(str(DIR_FIGURAS / f"{pasta}_{nome}.png"), comparacao)

        print(f"{nome}: {img.shape[1]}x{img.shape[0]} px -> {dir_saida.name}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Teste da Filtragem Bilateral")
    parser.add_argument("--ids", nargs="*", help="IDs ISIC (ex.: ISIC_0009944)")
    parser.add_argument("--n", type=int, default=5,
                        help="nº de imagens se --ids não for informado")
    parser.add_argument("--com-dullrazor", action="store_true",
                        help="aplica DullRazor antes (ramo DullRazor + Bilateral)")
    args = parser.parse_args()
    _testar(args.ids, args.n, args.com_dullrazor)