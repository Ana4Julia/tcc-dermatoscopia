"""
Extração dos 42 descritores usando a máscara padrão-ouro, nas 4 versões
do pré-processamento. Esses valores são a referência para avaliar a
exatidão dos descritores obtidos com as máscaras dos algoritmos.

Saída: data/metadata/descritores_gt.csv

Uso:
    python -m src.pipeline_descritores_gt
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.descriptors.extracao import NOMES, extrair_descritores
from src.pipeline import RAMOS

RAIZ = Path(__file__).resolve().parents[1]
DIR_PROCESSADO = RAIZ / "data" / "processed"
DIR_GT = DIR_PROCESSADO / "masks_gt"
CSV_SAIDA = RAIZ / "data" / "metadata" / "descritores_gt.csv"
CHAVE = ["image_id", "pre_processamento"]
SALVAR_A_CADA = 200


def _iniciar_processo() -> None:
    cv2.setNumThreads(1)


def extrair_imagem(id_img: str) -> list[dict]:
    linhas = []
    gt = cv2.imread(str(DIR_GT / f"{id_img}_segmentation.png"), cv2.IMREAD_GRAYSCALE)
    for ramo in RAMOS:
        linha = {"image_id": id_img, "pre_processamento": ramo}
        try:
            img = cv2.imread(str(DIR_PROCESSADO / ramo / f"{id_img}.png"))
            if img is None or gt is None:
                raise FileNotFoundError("imagem ou máscara padrão-ouro não encontrada")
            linha.update(extrair_descritores(img, gt))
            linha["erro"] = ""
        except Exception as e:
            linha.update({n: np.nan for n in NOMES})
            linha["erro"] = str(e)
        linhas.append(linha)
    return linhas


def _salvar(anteriores: pd.DataFrame, novas: list[dict]) -> None:
    tabela = pd.concat([anteriores, pd.DataFrame(novas)], ignore_index=True)
    tabela = tabela.drop_duplicates(CHAVE, keep="last").sort_values(CHAVE)
    tabela[CHAVE + NOMES + ["erro"]].to_csv(CSV_SAIDA, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Descritores com a máscara padrão-ouro")
    parser.add_argument("--limite", type=int, default=None)
    parser.add_argument("--refazer", action="store_true")
    parser.add_argument("--workers", type=int,
                        default=max(1, (os.cpu_count() or 2) - 2))
    args = parser.parse_args()

    ids = sorted(p.stem for p in (DIR_PROCESSADO / RAMOS[0]).glob("*.png"))
    if args.limite:
        ids = ids[:args.limite]

    anteriores = (pd.read_csv(CSV_SAIDA) if CSV_SAIDA.exists() and not args.refazer
                  else pd.DataFrame())
    feitas = set()
    if not anteriores.empty:
        ok = anteriores[anteriores["erro"].fillna("") == ""]
        cont = ok.groupby("image_id").size()
        feitas = set(cont[cont == len(RAMOS)].index)
    pendentes = [i for i in ids if args.refazer or i not in feitas]
    print(f"{len(ids)} imagens | {len(pendentes)} a extrair "
          f"({len(RAMOS)} pré-processamentos, máscara padrão-ouro) "
          f"com {args.workers} processos.")

    novas: list[dict] = []
    try:
        with ProcessPoolExecutor(max_workers=args.workers,
                                 initializer=_iniciar_processo) as executor:
            futuros = {executor.submit(extrair_imagem, i): i for i in pendentes}
            for n, f in enumerate(tqdm(as_completed(futuros), total=len(futuros)),
                                  start=1):
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

    if CSV_SAIDA.exists():
        tabela = pd.read_csv(CSV_SAIDA)
        print(f"Linhas: {len(tabela)} | com erro: "
              f"{(tabela['erro'].fillna('') != '').sum()} | "
              f"com algum descritor vazio: {tabela[NOMES].isna().any(axis=1).sum()}")


if __name__ == "__main__":
    main()
