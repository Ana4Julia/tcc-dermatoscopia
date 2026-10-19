# Influência do pré-processamento e da segmentação na precisão de descritores em imagens dermatoscópicas

**Autora:** Ana Júlia Campos Vieira
**Orientadora:** Profa. Dra. Glenda Michele Botelho
**Curso:** Ciência da Computação – UFT, 2026

O trabalho avalia como a escolha das técnicas de pré-processamento e de segmentação altera os valores de descritores extraídos de imagens dermatoscópicas. Para isso, as 2.594 imagens da Task 1 da base **ISIC 2018** passam por 12 configurações de processamento, e a estabilidade (precisão) e a exatidão de 42 descritores são medidas pelo Coeficiente de Correlação Intraclasse (ICC).

## Visão geral do pipeline

```
Imagem ISIC 2018 ──► Redimensionamento (lado maior = 1.024 px)
                       │
                       ├─► Sem tratamento ───────────────┐
                       ├─► DullRazor + Bilateral ────────┤
                       ├─► CLAHE ────────────────────────┤  4 versões
                       └─► DullRazor + Bilateral + CLAHE ┘
                                        │
                       ┌────────────────┼────────────────┐
                     Otsu            K-Means           Snakes     → 12 máscaras por imagem
                       └────────────────┼────────────────┘
                                        │
                  Validação (Jaccard e Dice com a máscara padrão-ouro)
                                        │
             Extração de 42 descritores (5 geométricos, 32 de cor, 5 de textura GLCM)
                                        │
          ICC entre configurações (precisão) e com o padrão-ouro (exatidão)
```

## Principais resultados

- A melhor segmentação foi obtida com **DullRazor + Filtragem Bilateral + Otsu** (Jaccard médio de 0,569). O CLAHE reduziu o desempenho dos três algoritmos.
- Nenhum descritor teve estabilidade boa entre as 12 configurações considerando todas as imagens. Os mais estáveis foram as **médias dos canais a\* e b\*** do espaço CIE L\*a\*b\*, que chegaram a ICC de 0,95 e 0,94 quando apenas segmentações com Jaccard ≥ 0,5 foram consideradas.
- Os descritores de **textura** dependem principalmente do pré-processamento, mesmo com a máscara padrão-ouro; os **geométricos** e **de cor** dependem principalmente da segmentação.
- A **circularidade** não foi exata em nenhuma configuração, por depender do nível de detalhe do contorno.

Os resultados completos estão na pasta [`results/`](results/) e no texto do TCC.

## Estrutura do repositório

```
tcc-dermatoscopia/
├── src/
│   ├── pipeline.py                  # 1. pré-processamento (4 versões de cada imagem)
│   ├── pipeline_segmentacao.py      # 2. segmentação (12 máscaras) + Jaccard e Dice
│   ├── pipeline_descritores.py      # 3. 42 descritores de cada máscara
│   ├── pipeline_descritores_gt.py   # 4. 42 descritores da máscara padrão-ouro
│   ├── preprocessing/               # DullRazor, Filtragem Bilateral, CLAHE
│   ├── segmentation/                # Otsu, K-Means, Snakes, moldura, pós-processamento
│   ├── descriptors/                 # extração dos descritores
│   ├── validation/                  # Jaccard e Dice
│   └── stats/                       # ICC, exatidão, testes estatísticos e figuras
├── data/
│   ├── raw/                         # imagens e máscaras da ISIC 2018 (não versionado)
│   ├── processed/                   # imagens pré-processadas (não versionado)
│   ├── segmentation/                # máscaras geradas (não versionado)
│   └── metadata/                    # CSVs e parâmetros de cada etapa
├── results/                         # tabelas e figuras do Capítulo 5
├── requirements.txt
└── README.md
```

As imagens da ISIC e os arquivos gerados a partir delas não estão no repositório, por causa do tamanho (vários GB) e da licença da base. Os CSVs com os resultados de cada etapa (`data/metadata/` e `results/`) estão incluídos.

## Como reproduzir

### 1. Ambiente

Testado com Python 3.10 no Ubuntu.

```bash
git clone https://github.com/Ana4Julia/tcc-dermatoscopia.git
cd tcc-dermatoscopia
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Dados

Baixe, na página de dados do desafio [ISIC 2018](https://challenge.isic-archive.com/data/#2018), os arquivos de **treinamento da Task 1**:

- `ISIC2018_Task1-2_Training_Input.zip` (2.594 imagens)
- `ISIC2018_Task1_Training_GroundTruth.zip` (máscaras padrão-ouro)

Extraia para:

```
data/raw/images/      ← arquivos ISIC_xxxxxxx.jpg
data/raw/masks_gt/    ← arquivos ISIC_xxxxxxx_segmentation.png
```

### 3. Execução

Rode os comandos a partir da raiz do projeto, com o ambiente ativo, nesta ordem:

```bash
python -m src.pipeline                      # pré-processamento
python -m src.pipeline_segmentacao          # segmentação + Jaccard e Dice
python -m src.pipeline_descritores          # descritores das 12 configurações
python -m src.pipeline_descritores_gt       # descritores da máscara padrão-ouro

python -m src.stats.resumo_segmentacao      # tabelas de Jaccard e Dice
python -m src.stats.evidencias              # testes de Friedman/Wilcoxon, falhas, etc.
python -m src.stats.icc                     # ICC entre configurações (precisão)
python -m src.stats.exatidao                # ICC com o padrão-ouro (exatidão)
python -m src.stats.figuras                 # gráficos e mapas de calor
python -m src.stats.figuras_exemplos        # figuras com imagens de exemplo
```

Para testar rapidamente, os três primeiros pipelines aceitam `--limite 10` (processa só 10 imagens). Todos rodam em paralelo (`--workers N`) e podem ser interrompidos com Ctrl+C e retomados depois.

Os parâmetros de cada etapa são gravados em `data/metadata/parametros_*.json`. Se algum parâmetro for alterado no código, a retomada é bloqueada para não misturar resultados; nesse caso, use `--refazer`.

## Parâmetros principais

| Etapa | Parâmetros |
|---|---|
| Redimensionamento | lado maior = 1.024 px; interpolação por área (imagens) e vizinho mais próximo (máscaras) |
| DullRazor | fechamento morfológico a 0°, 45° e 90°; *inpainting* de Telea; mediana 5×5; tamanhos escalonados pela resolução (referência 600 px) |
| Filtragem Bilateral | espaço CIE L\*a\*b\*; d = 9; σ<sub>cor</sub> = σ<sub>espaço</sub> = 75 |
| CLAHE | só no canal L\*; limite de recorte 2,0; grade 8×8 |
| Otsu | níveis de cinza; limiar sobre o campo de visão |
| K-Means | CIE L\*a\*b\*; k = 2; K-Means++; semente 42 |
| Snakes | elipse inicial com semieixos de 90% do campo de visão; α = 0,015; β = 10; γ = 0,001; peso de borda = 10 |
| Pós-processamento | maior região (área × centralidade) + preenchimento de buracos |
| GLCM | 64 níveis; distância 1; 0°, 45°, 90° e 135°; só pares de pixels da lesão |
| ICC | efeitos mistos de duas vias, medida única; concordância absoluta e consistência (Pingouin) |

## Créditos

- Base de dados: Codella *et al.*, *Skin Lesion Analysis Toward Melanoma Detection 2018: A Challenge Hosted by the International Skin Imaging Collaboration (ISIC)*, arXiv:1902.03368, 2019.
- DullRazor: Lee *et al.*, *DullRazor: A software approach to hair removal from images*, Computers in Biology and Medicine, 1997. A implementação tomou como referência o repositório [BlueDokk/Dullrazor-algorithm](https://github.com/BlueDokk/Dullrazor-algorithm).
- Bibliotecas: OpenCV, scikit-image, NumPy, SciPy, pandas, Pingouin e Matplotlib.

## Licença

O código está sob a licença MIT (veja [`LICENSE`](LICENSE)). As imagens da ISIC 2018 seguem a licença da própria base e não são redistribuídas aqui.
