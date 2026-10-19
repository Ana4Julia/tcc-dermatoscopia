"""
Extração dos 42 descritores (geométricos, de cor e de textura) de cada
uma das 12 máscaras de cada imagem.

Só processa as imagens que já têm as 12 máscaras prontas.
Saída: data/metadata/descritores.csv

Uso:
    python -m src.pipeline_descritores              # todas as imagens
    python -m src.pipeline_descritores --limite 10  # teste com 10 imagens
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import scipy
import skimage
from tqdm import tqdm

from src.descriptors import extracao
from src.descriptors.extracao import NOMES, extrair_descritores
from src.pipeline import RAMOS
from src.pipeline_segmentacao import ALGORITMOS

RAIZ = Path(__file__).resolve().parents[1]
DIR_PROCESSADO = RAIZ / "data" / "processed"
DIR_SEGMENTACAO = RAIZ / "data" / "segmentation"
CSV_SAIDA = RAIZ / "data" / "metadata" / "descritores.csv"
JSON_PARAMETROS = RAIZ / "data" / "metadata" / "parametros_descritores.json"
SALVAR_A_CADA = 200
CHAVE = ["image_id", "pre_processamento", "algoritmo"]


def _parametros() -> dict:
    return {
        "descritores": NOMES,
        "glcm": {
            "niveis_cinza": extracao.NIVEIS_CINZA,
            "distancia": extracao.DISTANCIA_GLCM,
            "angulos_graus": [0, 45, 90, 135],
            "simetrica": True,
            "agregacao": "média das direções",
        },
        "perimetro": "Crofton (4 direções)",
        "curtose": "excesso de Fisher",
        "area_minima_px": extracao.AREA_MINIMA,
        "versoes": {
            "python": platform.python_version(),
            "opencv": cv2.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit-image": skimage.__version__,
        },
    }


def _verificar_parametros(refazer: bool) -> None:
    atuais = _parametros()
    if JSON_PARAMETROS.exists() and not refazer:
        anteriores = json.loads(JSON_PARAMETROS.read_text(encoding="utf-8"))
        sem_versoes = lambda d: {k: v for k, v in d.items() if k != "versoes"}
        if sem_versoes(anteriores) != sem_versoes(atuais):
            sys.exit("[erro] Os parâmetros dos descritores mudaram desde a última "
                     "execução.\n       Para reprocessar tudo, use: "
                     "python -m src.pipeline_descritores --refazer")
    JSON_PARAMETROS.parent.mkdir(parents=True, exist_ok=True)
    JSON_PARAMETROS.write_text(json.dumps(atuais, indent=2, ensure_ascii=False),
                               encoding="utf-8")


def _mascara(ramo: str, alg: str, id_img: str) -> Path:
    return DIR_SEGMENTACAO / ramo / alg / f"{id_img}.png"


def _segmentacao_completa(id_img: str) -> bool:
    return all(_mascara(r, a, id_img).exists() for r in RAMOS for a in ALGORITMOS)


def _iniciar_processo() -> None:
    cv2.setNumThreads(1)


def extrair_imagem(id_img: str) -> list[dict]:
    linhas = []
    for ramo in RAMOS:
        img = cv2.imread(str(DIR_PROCESSADO / ramo / f"{id_img}.png"))
        for alg in ALGORITMOS:
            linha = {"image_id": id_img, "pre_processamento": ramo,
                     "algoritmo": alg}
            try:
                mascara = cv2.imread(str(_mascara(ramo, alg, id_img)),
                                     cv2.IMREAD_GRAYSCALE)
                if img is None or mascara is None:
                    raise FileNotFoundError("imagem ou máscara não encontrada")
                linha.update(extrair_descritores(img, mascara))
                linha["erro"] = ""
            except Exception as e:
                linha.update({n: np.nan for n in NOMES})
                linha["erro"] = str(e)
            linhas.append(linha)
    return linhas


def _salvar(anteriores: pd.DataFrame, novas: list[dict]) -> None:
    tabela = pd.concat([anteriores, pd.DataFrame(novas)], ignore_index=True)
    tabela = tabela.drop_duplicates(CHAVE, keep="last").sort_values(CHAVE)
    tabela = tabela[CHAVE + NOMES + ["erro"]]
    tabela.to_csv(CSV_SAIDA, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extração de descritores")
    parser.add_argument("--limite", type=int, default=None)
    parser.add_argument("--refazer", action="store_true")
    parser.add_argument("--workers", type=int,
                        default=max(1, (os.cpu_count() or 2) - 2))
    args = parser.parse_args()

    _verificar_parametros(args.refazer)

    ids = sorted(p.stem for p in (DIR_PROCESSADO / RAMOS[0]).glob("*.png"))
    if args.limite:
        ids = ids[:args.limite]
    prontas = [i for i in ids if _segmentacao_completa(i)]

    if CSV_SAIDA.exists() and not args.refazer:
        anteriores = pd.read_csv(CSV_SAIDA)
    else:
        anteriores = pd.DataFrame()
    feitas = set()
    if not anteriores.empty:
        ok = anteriores[anteriores["erro"].fillna("") == ""]
        cont = ok.groupby("image_id").size()
        feitas = set(cont[cont == len(RAMOS) * len(ALGORITMOS)].index)
    pendentes = [i for i in prontas if args.refazer or i not in feitas]

    print(f"{len(ids)} imagens | {len(prontas)} com as 12 máscaras prontas | "
          f"{len(pendentes)} a extrair ({len(NOMES)} descritores) "
          f"com {args.workers} processos.")

    novas: list[dict] = []
    try:
        with ProcessPoolExecutor(max_workers=args.workers,
                                 initializer=_iniciar_processo) as executor:
            futuros = {executor.submit(extrair_imagem, i): i for i in pendentes}
            for n, f in enumerate(tqdm(as_completed(futuros),
                                       total=len(futuros)), start=1):
                try:
                    novas.extend(f.result())
                except Exception as e:
                    novas.append({"image_id": futuros[f], "erro": str(e)})
                if n % SALVAR_A_CADA == 0:
                    _salvar(anteriores, novas)
    finally:
        if novas:
            _salvar(anteriores, novas)
            print(f"Descritores salvos em {CSV_SAIDA.relative_to(RAIZ)}")

    if novas:
        tabela = pd.read_csv(CSV_SAIDA)
        print(f"Linhas: {len(tabela)} | com erro: "
              f"{(tabela['erro'].fillna('') != '').sum()} | "
              f"com algum descritor vazio (NaN): "
              f"{tabela[NOMES].isna().any(axis=1).sum()}")
    faltam = len(ids) - len(prontas)
    if faltam:
        print(f"[aviso] {faltam} imagens ainda sem as 12 máscaras; rode de novo "
              "depois que a segmentação terminar.")


if __name__ == "__main__":
    main()
