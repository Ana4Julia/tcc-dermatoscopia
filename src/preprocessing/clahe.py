"""
CLAHE: realce de contraste local (Pizer et al., 1987), aplicado apenas ao
canal L* do espaço CIE-Lab, preservando a cor (canais a* e b*).

Teste em poucas imagens:
    python -m src.preprocessing.clahe --ids ISIC_0009944
    python -m src.preprocessing.clahe --ids ISIC_0009944 --com-dullrazor-bilateral
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

# Parâmetros. A grade é em nº de ladrilhos, então o efeito é o mesmo em qualquer resolução.
LIMITE_RECORTE = 2.0   # limite de recorte do histograma local
GRADE = (8, 8)         # nº de ladrilhos (linhas x colunas)

RAIZ = Path(__file__).resolve().parents[2]
DIR_IMAGENS = RAIZ / "data" / "raw" / "images"
DIR_TESTE = RAIZ / "data" / "processed" / "_test"
DIR_FIGURAS = Path(__file__).resolve().parent / "figuras"


# Algoritmo
def aplicar_clahe(img_bgr: np.ndarray) -> np.ndarray:
    """Aplica o CLAHE ao canal L* de uma imagem BGR e retorna a imagem realçada."""
    if img_bgr is None or img_bgr.ndim != 3:
        raise ValueError("Esperada imagem colorida BGR (H x W x 3).")

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=LIMITE_RECORTE, tileGridSize=GRADE)
    l_realcado = clahe.apply(l)

    return cv2.cvtColor(cv2.merge([l_realcado, a, b]), cv2.COLOR_LAB2BGR)


# Teste em poucas imagens (salva em data/processed/_test)
def _rotular(img: np.ndarray, texto: str) -> np.ndarray:
    saida = img.copy()
    cv2.putText(saida, texto, (15, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1.2, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(saida, texto, (15, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1.2, (255, 255, 255), 2, cv2.LINE_AA)
    return saida


def _testar(ids: list[str] | None, n: int, combinado: bool) -> None:
    if ids:
        caminhos = [DIR_IMAGENS / f"{i}.jpg" for i in ids]
    else:
        caminhos = sorted(DIR_IMAGENS.glob("*.jpg"))[:n]

    if combinado:
        from src.preprocessing.bilateral import aplicar_bilateral
        from src.preprocessing.dullrazor import aplicar_dullrazor
        pasta, rotulo = "dullrazor_bilateral_clahe", "DullRazor + Bilateral + CLAHE"
    else:
        pasta, rotulo = "clahe", "CLAHE"

    dir_saida = DIR_TESTE / pasta
    dir_saida.mkdir(parents=True, exist_ok=True)
    DIR_FIGURAS.mkdir(parents=True, exist_ok=True)

    for caminho in caminhos:
        img = cv2.imread(str(caminho), cv2.IMREAD_COLOR)
        if img is None:
            print(f"[aviso] não foi possível ler {caminho}")
            continue

        entrada = aplicar_bilateral(aplicar_dullrazor(img)) if combinado else img
        realcada = aplicar_clahe(entrada)
        nome = caminho.stem

        cv2.imwrite(str(dir_saida / f"{nome}.png"), realcada)
        comparacao = np.hstack([_rotular(img, "Original"),
                                _rotular(realcada, rotulo)])
        cv2.imwrite(str(DIR_FIGURAS / f"{pasta}_{nome}.png"), comparacao)

        print(f"{nome}: {img.shape[1]}x{img.shape[0]} px -> {dir_saida.name}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Teste do CLAHE")
    parser.add_argument("--ids", nargs="*", help="IDs ISIC (ex.: ISIC_0009944)")
    parser.add_argument("--n", type=int, default=5,
                        help="nº de imagens se --ids não for informado")
    parser.add_argument("--com-dullrazor-bilateral", action="store_true",
                        help="aplica DullRazor e Bilateral antes (ramo completo)")
    args = parser.parse_args()
    _testar(args.ids, args.n, args.com_dullrazor_bilateral)