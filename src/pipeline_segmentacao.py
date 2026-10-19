"""
Segmentação das imagens pré-processadas por Otsu, K-Means e Snakes.

Gera as 12 máscaras de cada imagem em data/segmentation/ e calcula o
Jaccard e o Dice em relação ao padrão-ouro (data/metadata/segmentacao.csv).

Uso:
    python -m src.pipeline_segmentacao              # todas as imagens
    python -m src.pipeline_segmentacao --limite 10  # teste com 10 imagens
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
import skimage
from tqdm import tqdm

from src.pipeline import RAMOS, _espaco_livre_gb, _gravar, ESPACO_MINIMO_GB
from src.segmentation import fundo, kmeans, snakes
from src.segmentation.kmeans import segmentar_kmeans
from src.segmentation.otsu import segmentar_otsu
from src.segmentation.snakes import segmentar_snakes
from src.validation.metrics import validar_segmentacao

RAIZ = Path(__file__).resolve().parents[1]
DIR_PROCESSADO = RAIZ / "data" / "processed"
DIR_MASCARAS_GT = DIR_PROCESSADO / "masks_gt"
DIR_SEGMENTACAO = RAIZ / "data" / "segmentation"
CSV_SAIDA = RAIZ / "data" / "metadata" / "segmentacao.csv"
JSON_PARAMETROS = RAIZ / "data" / "metadata" / "parametros_segmentacao.json"

ALGORITMOS = {
    "otsu": segmentar_otsu,
    "kmeans": segmentar_kmeans,
    "snakes": segmentar_snakes,
}
SALVAR_A_CADA = 100        # grava o CSV parcial a cada N imagens


# Parâmetros usados (salvos em JSON para garantir a reprodutibilidade)
def _parametros() -> dict:
    return {
        "moldura_dermatoscopio": {
            "limiar_cinza": fundo.LIMIAR_MOLDURA,
            "area_minima": fundo.AREA_MINIMA_MOLDURA,
            "fracao_erosao": fundo.FRACAO_EROSAO,
            "cantos_minimos": fundo.CANTOS_MINIMOS,
            "fracao_canto": fundo.FRACAO_CANTO,
            "fracao_margem": fundo.FRACAO_MARGEM,
            "sombra": {"largura_max": fundo.LARGURA_MAX_SOMBRA,
                       "anel_px": fundo.ANEL_PX,
                       "fracao_plato": fundo.FRACAO_PLATO},
            "filtro_mediana": 5,
            "tratamento": "excluída do Otsu/K-Means; neutralizada no Snakes",
        },
        "pos_processamento": {
            "selecao_regiao": "maior (área x centralidade)",
            "preenchimento_buracos": True,
        },
        "otsu": {"espaco_cor": "níveis de cinza", "limiarizacao": "invertida"},
        "kmeans": {
            "espaco_cor": "CIE-Lab",
            "k": kmeans.K,
            "inicializacao": "K-Means++",
            "tentativas": kmeans.TENTATIVAS,
            "max_iteracoes": kmeans.MAX_ITERACOES,
            "epsilon": kmeans.EPSILON,
            "semente": kmeans.SEMENTE,
            "grupo_lesao": "menor L*",
        },
        "snakes": {
            "sigma_gauss": snakes.SIGMA_GAUSS,
            "n_pontos": snakes.N_PONTOS,
            "inicializacao": "elipse inscrita no campo de visão",
            "fracao_eixos": snakes.FRACAO_EIXOS,
            "alfa": snakes.ALFA,
            "beta": snakes.BETA,
            "gama": snakes.GAMA,
            "w_linha": snakes.W_LINHA,
            "w_borda": snakes.W_BORDA,
            "max_iteracoes": snakes.MAX_ITERACOES,
        },
        "versoes": {
            "python": platform.python_version(),
            "opencv": cv2.__version__,
            "numpy": np.__version__,
            "scikit-image": skimage.__version__,
        },
    }


def _verificar_parametros(refazer: bool) -> None:
    atuais = _parametros()
    if JSON_PARAMETROS.exists() and not refazer:
        anteriores = json.loads(JSON_PARAMETROS.read_text(encoding="utf-8"))
        sem_versoes = lambda d: {k: v for k, v in d.items() if k != "versoes"}
        if sem_versoes(anteriores) != sem_versoes(atuais):
            sys.exit("[erro] Os parâmetros da segmentação mudaram desde a última "
                     "execução.\n       Para reprocessar tudo, use: "
                     "python -m src.pipeline_segmentacao --refazer")
        if anteriores.get("versoes") != atuais["versoes"]:
            print("[aviso] Versões de bibliotecas diferentes da execução anterior: "
                  f"{anteriores.get('versoes')} -> {atuais['versoes']}")
    JSON_PARAMETROS.parent.mkdir(parents=True, exist_ok=True)
    JSON_PARAMETROS.write_text(json.dumps(atuais, indent=2, ensure_ascii=False),
                               encoding="utf-8")


# Segmentação de uma imagem (4 versões x 3 algoritmos = 12 máscaras)
def _caminho_mascara(ramo: str, algoritmo: str, id_img: str) -> Path:
    return DIR_SEGMENTACAO / ramo / algoritmo / f"{id_img}.png"


def _ja_segmentada(id_img: str) -> bool:
    return all(_caminho_mascara(r, a, id_img).exists()
               for r in RAMOS for a in ALGORITMOS)


def _iniciar_processo() -> None:
    cv2.setNumThreads(1)


def segmentar_imagem(id_img: str) -> list[dict]:
    gt = cv2.imread(str(DIR_MASCARAS_GT / f"{id_img}_segmentation.png"),
                    cv2.IMREAD_GRAYSCALE)
    linhas = []
    for ramo in RAMOS:
        img = cv2.imread(str(DIR_PROCESSADO / ramo / f"{id_img}.png"))
        for nome, segmentar in ALGORITMOS.items():
            linha = {"image_id": id_img, "pre_processamento": ramo,
                     "algoritmo": nome, "jaccard": np.nan, "dice": np.nan,
                     "area_px": np.nan, "erro": ""}
            try:
                if img is None:
                    raise FileNotFoundError(f"{ramo}/{id_img}.png não encontrada")
                mascara = segmentar(img)
                _gravar(_caminho_mascara(ramo, nome, id_img), mascara)
                linha["area_px"] = int(np.count_nonzero(mascara))
                if gt is not None:
                    linha.update(validar_segmentacao(mascara, gt))
            except Exception as e:     # registra e segue para a próxima
                linha["erro"] = str(e)
            linhas.append(linha)
    return linhas



# Execução
def _salvar(anteriores: pd.DataFrame, novas: list[dict]) -> None:
    tabela = pd.concat([anteriores, pd.DataFrame(novas)], ignore_index=True)
    chave = ["image_id", "pre_processamento", "algoritmo"]
    tabela = tabela.drop_duplicates(chave, keep="last").sort_values(chave)
    tabela.to_csv(CSV_SAIDA, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline de segmentação")
    parser.add_argument("--limite", type=int, default=None,
                        help="processa só as N primeiras imagens (teste)")
    parser.add_argument("--refazer", action="store_true",
                        help="reprocessa todas as imagens, sobrescrevendo")
    parser.add_argument("--workers", type=int,
                        default=max(1, (os.cpu_count() or 2) - 2),
                        help="nº de processos em paralelo")
    args = parser.parse_args()

    _verificar_parametros(args.refazer)
    if _espaco_livre_gb() < ESPACO_MINIMO_GB:
        sys.exit(f"[erro] Menos de {ESPACO_MINIMO_GB} GB livres no disco.")

    for ramo in RAMOS:
        for alg in ALGORITMOS:
            pasta = DIR_SEGMENTACAO / ramo / alg
            pasta.mkdir(parents=True, exist_ok=True)
            for resto in pasta.glob("*.tmp.png"):
                resto.unlink()

    ids = sorted(p.stem for p in (DIR_PROCESSADO / RAMOS[0]).glob("*.png"))
    if args.limite:
        ids = ids[:args.limite]

    if CSV_SAIDA.exists() and not args.refazer:
        anteriores = pd.read_csv(CSV_SAIDA)
    else:
        anteriores = pd.DataFrame()
    completas = set()
    if not anteriores.empty:
        ok = anteriores[anteriores["erro"].fillna("") == ""]
        contagem = ok.groupby("image_id").size()
        completas = set(contagem[contagem == len(RAMOS) * len(ALGORITMOS)].index)
    pendentes = [i for i in ids
                 if args.refazer or not (i in completas and _ja_segmentada(i))]

    print(f"{len(ids)} imagens encontradas, {len(pendentes)} a segmentar "
          f"({len(RAMOS)} pré-processamentos x {len(ALGORITMOS)} algoritmos) "
          f"com {args.workers} processos.")

    novas: list[dict] = []
    try:
        with ProcessPoolExecutor(max_workers=args.workers,
                                 initializer=_iniciar_processo) as executor:
            futuros = {executor.submit(segmentar_imagem, i): i for i in pendentes}
            for n, futuro in enumerate(tqdm(as_completed(futuros),
                                            total=len(futuros)), start=1):
                try:
                    novas.extend(futuro.result())
                except Exception as e:
                    novas.append({"image_id": futuros[futuro], "erro": str(e)})
                if n % SALVAR_A_CADA == 0:
                    _salvar(anteriores, novas)
                if _espaco_livre_gb() < ESPACO_MINIMO_GB:
                    print(f"\n[erro] Menos de {ESPACO_MINIMO_GB} GB livres no "
                          "disco. Execução interrompida.")
                    for f in futuros:
                        f.cancel()
                    break
    finally:
        if novas:
            _salvar(anteriores, novas)
            print(f"Resultados salvos em {CSV_SAIDA.relative_to(RAIZ)}")

    if not novas:
        return
    tabela = pd.read_csv(CSV_SAIDA)
    com_erro = (tabela["erro"].fillna("") != "").sum()
    print(f"Linhas com erro: {com_erro}")
    resumo = (tabela.pivot_table(index="pre_processamento", columns="algoritmo",
                                 values="jaccard", aggfunc="mean")
                    .reindex(RAMOS)[list(ALGORITMOS)])
    print("\nJaccard médio (pré-processamento x algoritmo):")
    print(resumo.round(3).to_string())


if __name__ == "__main__":
    main()