"""
Pré-processamento das imagens da ISIC 2018 (Task 1).

Redimensiona cada imagem e sua máscara padrão-ouro (lado maior = 1.024 px)
e gera as quatro versões em data/processed/: sem_tratamento,
dullrazor_bilateral, clahe e dullrazor_bilateral_clahe.

Uso:
    python -m src.pipeline              # todas as imagens
    python -m src.pipeline --limite 10  # teste com 10 imagens
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.preprocessing import bilateral, clahe, dullrazor
from src.preprocessing.bilateral import aplicar_bilateral
from src.preprocessing.clahe import aplicar_clahe
from src.preprocessing.dullrazor import aplicar_dullrazor

RAIZ = Path(__file__).resolve().parents[1]
DIR_IMAGENS = RAIZ / "data" / "raw" / "images"
DIR_MASCARAS = RAIZ / "data" / "raw" / "masks_gt"
DIR_PROCESSADO = RAIZ / "data" / "processed"
CSV_SAIDA = RAIZ / "data" / "metadata" / "preprocessamento.csv"
JSON_PARAMETROS = RAIZ / "data" / "metadata" / "parametros_preprocessamento.json"

RAMOS = ["sem_tratamento", "dullrazor_bilateral", "clahe",
         "dullrazor_bilateral_clahe"]
DIR_MASCARAS_PROC = DIR_PROCESSADO / "masks_gt"

LADO_MAXIMO = 1024        # px — lado maior após o redimensionamento
ESPACO_MINIMO_GB = 2.0    # para a execução se o disco ficar abaixo disso


def _parametros() -> dict:
    """Parâmetros de cada técnica + versões usadas (para reprodutibilidade)."""
    return {
        "redimensionamento": {
            "lado_maximo_px": LADO_MAXIMO,
            "interpolacao_imagem": "INTER_AREA",
            "interpolacao_mascara": "INTER_NEAREST",
        },
        "dullrazor": {
            "lado_referencia_px": dullrazor.LADO_REFERENCIA,
            "tamanho_linha_px": dullrazor.TAMANHO_LINHA,
            "direcoes_graus": [0, 45, 90],
            "kernel_gauss": list(dullrazor.KERNEL_GAUSS),
            "limiar": dullrazor.LIMIAR,
            "raio_inpaint_px": dullrazor.RAIO_INPAINT,
            "metodo_inpaint": "TELEA",
            "kernel_dilatacao_px": dullrazor.KERNEL_DILATACAO,
            "kernel_mediana_px": dullrazor.KERNEL_MEDIANA,
            "escalados_pela_resolucao": ["tamanho_linha_px", "kernel_dilatacao_px"],
        },
        "bilateral": {
            "espaco_cor": "CIE-Lab",
            "diametro_px": bilateral.DIAMETRO,
            "sigma_d": bilateral.SIGMA_D,
            "sigma_r": bilateral.SIGMA_R,
        },
        "clahe": {
            "espaco_cor": "CIE-Lab (apenas canal L*)",
            "limite_recorte": clahe.LIMITE_RECORTE,
            "grade": list(clahe.GRADE),
        },
        "versoes": {
            "python": platform.python_version(),
            "opencv": cv2.__version__,
            "numpy": np.__version__,
        },
    }


def _espaco_livre_gb() -> float:
    return shutil.disk_usage(DIR_PROCESSADO).free / 1024**3


def _iniciar_processo() -> None:
    # Um núcleo por processo: evita que os processos disputem a CPU
    cv2.setNumThreads(1)


def _caminho_saida(ramo: str, id_img: str) -> Path:
    return DIR_PROCESSADO / ramo / f"{id_img}.png"


def _caminho_mascara(id_img: str) -> Path:
    return DIR_MASCARAS_PROC / f"{id_img}_segmentation.png"


def _ja_processada(id_img: str) -> bool:
    tem_gt = (DIR_MASCARAS / f"{id_img}_segmentation.png").exists()
    return (all(_caminho_saida(r, id_img).exists() for r in RAMOS)
            and (not tem_gt or _caminho_mascara(id_img).exists()))


def _redimensionar(img: np.ndarray, interpolacao: int) -> np.ndarray:
    h, w = img.shape[:2]
    escala = LADO_MAXIMO / max(h, w)
    if escala >= 1:
        return img
    return cv2.resize(img, (round(w * escala), round(h * escala)),
                      interpolation=interpolacao)


def _gravar(caminho: Path, imagem: np.ndarray) -> None:
    """Grava via arquivo temporário; só renomeia se a gravação deu certo."""
    temporario = caminho.with_name(caminho.stem + ".tmp.png")
    ok = cv2.imwrite(str(temporario), imagem, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not ok:
        temporario.unlink(missing_ok=True)
        raise OSError(f"falha ao gravar {caminho.name} (disco cheio?)")
    os.replace(temporario, caminho)


def processar_imagem(caminho: Path) -> dict:
    """Gera as quatro versões de uma imagem e retorna seus metadados."""
    id_img = caminho.stem
    original = cv2.imread(str(caminho), cv2.IMREAD_COLOR)
    if original is None:
        return {"image_id": id_img, "erro": "falha na leitura"}
    img = _redimensionar(original, cv2.INTER_AREA)

    gt = cv2.imread(str(DIR_MASCARAS / f"{id_img}_segmentation.png"),
                    cv2.IMREAD_GRAYSCALE)
    if gt is not None:
        gt = cv2.resize(gt, (img.shape[1], img.shape[0]),
                        interpolation=cv2.INTER_NEAREST)
        gt = np.where(gt > 127, 255, 0).astype(np.uint8)

    limpa, mascara_pelos = aplicar_dullrazor(img, retornar_mascara=True)
    db = aplicar_bilateral(limpa)

    saidas = {
        "sem_tratamento": img,
        "dullrazor_bilateral": db,
        "clahe": aplicar_clahe(img),
        "dullrazor_bilateral_clahe": aplicar_clahe(db),
    }
    for ramo, imagem in saidas.items():
        _gravar(_caminho_saida(ramo, id_img), imagem)
    if gt is not None:
        _gravar(_caminho_mascara(id_img), gt)

    return {
        "image_id": id_img,
        "largura_original": original.shape[1],
        "altura_original": original.shape[0],
        "largura": img.shape[1],
        "altura": img.shape[0],
        "tem_mascara_gt": gt is not None,
        "pct_pelos": round(100 * np.count_nonzero(mascara_pelos)
                           / mascara_pelos.size, 3),
        "erro": "",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline de pré-processamento")
    parser.add_argument("--limite", type=int, default=None,
                        help="processa só as N primeiras imagens (teste)")
    parser.add_argument("--refazer", action="store_true",
                        help="reprocessa todas as imagens, sobrescrevendo")
    parser.add_argument("--workers", type=int,
                        default=max(1, (os.cpu_count() or 2) - 2),
                        help="nº de processos em paralelo")
    args = parser.parse_args()

    for pasta in [*RAMOS, "masks_gt"]:
        (DIR_PROCESSADO / pasta).mkdir(parents=True, exist_ok=True)
        for resto in (DIR_PROCESSADO / pasta).glob("*.tmp.png"):
            resto.unlink()               # sobras de execuções interrompidas
    CSV_SAIDA.parent.mkdir(parents=True, exist_ok=True)

    # Trava de parâmetros: impede misturar execuções com valores diferentes
    parametros = _parametros()
    if JSON_PARAMETROS.exists() and not args.refazer:
        anteriores_p = json.loads(JSON_PARAMETROS.read_text(encoding="utf-8"))
        a = {k: v for k, v in anteriores_p.items() if k != "versoes"}
        b = {k: v for k, v in parametros.items() if k != "versoes"}
        if a != b:
            sys.exit("[erro] Os parâmetros mudaram desde a última execução.\n"
                     "       Para reprocessar tudo com os novos valores, use:\n"
                     "       python -m src.pipeline --refazer")
        if anteriores_p.get("versoes") != parametros["versoes"]:
            print("[aviso] Versões de bibliotecas diferentes da execução anterior: "
                  f"{anteriores_p.get('versoes')} -> {parametros['versoes']}")
    JSON_PARAMETROS.write_text(
        json.dumps(parametros, indent=2, ensure_ascii=False), encoding="utf-8")

    caminhos = sorted(DIR_IMAGENS.glob("*.jpg"))
    if args.limite:
        caminhos = caminhos[:args.limite]

    # Retomada: mantém o que já foi registrado e processa só o que falta
    if CSV_SAIDA.exists() and not args.refazer:
        anteriores = pd.read_csv(CSV_SAIDA)
    else:
        anteriores = pd.DataFrame()
    feitas = set(anteriores["image_id"]) if not anteriores.empty else set()
    pendentes = [c for c in caminhos
                 if not (c.stem in feitas and _ja_processada(c.stem))]

    if _espaco_livre_gb() < ESPACO_MINIMO_GB:
        sys.exit(f"[erro] Menos de {ESPACO_MINIMO_GB} GB livres no disco.")

    print(f"{len(caminhos)} imagens encontradas, {len(pendentes)} a processar "
          f"com {args.workers} processos.")

    novos = []
    try:
        with ProcessPoolExecutor(max_workers=args.workers,
                                 initializer=_iniciar_processo) as executor:
            futuros = {executor.submit(processar_imagem, c): c for c in pendentes}
            for futuro in tqdm(as_completed(futuros), total=len(futuros)):
                try:
                    novos.append(futuro.result())
                except Exception as e:  # registra e segue para a próxima
                    novos.append({"image_id": futuros[futuro].stem,
                                  "erro": str(e)})
                if _espaco_livre_gb() < ESPACO_MINIMO_GB:
                    print(f"\n[erro] Menos de {ESPACO_MINIMO_GB} GB livres no disco. "
                          "Execução interrompida; libere espaço e rode de novo.")
                    for f in futuros:
                        f.cancel()
                    break
    finally:
        # salva mesmo se a execução for interrompida
        if novos:
            tabela = pd.concat([anteriores, pd.DataFrame(novos)],
                               ignore_index=True)
            tabela = (tabela.drop_duplicates("image_id", keep="last")
                            .sort_values("image_id"))
            tabela.to_csv(CSV_SAIDA, index=False)
            print(f"Metadados salvos em {CSV_SAIDA.relative_to(RAIZ)}")

    if novos:
        tabela_nova = pd.DataFrame(novos)
        sem_gt = (~tabela_nova["tem_mascara_gt"].fillna(False).astype(bool)).sum()
        com_erro = (tabela_nova["erro"].fillna("") != "").sum()
        print(f"Sem máscara padrão-ouro: {sem_gt} | Com erro: {com_erro}")
        if "pct_pelos" in tabela_nova:
            print(f"% de pelos — mediana: {tabela_nova['pct_pelos'].median():.2f} | "
                  f"máx: {tabela_nova['pct_pelos'].max():.2f}")


if __name__ == "__main__":
    main()