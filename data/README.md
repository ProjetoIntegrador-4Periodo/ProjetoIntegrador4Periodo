# Dados

**O conteúdo destas pastas não vai para o Git.** Só a estrutura é versionada.
Mamografias são arquivos grandes e dados de saúde — eles ficam na sua máquina.

| Pasta | O que guarda |
|-------|--------------|
| `raw/` | Download original, intocado (DICOMs e CSVs do RSNA como vieram do Kaggle) |
| `interim/` | Etapas intermediárias (ex.: DICOM convertido para PNG 16-bit) |
| `processed/` | Dataset final pronto para o treino |
| `external/` | Outras fontes (CBIS-DDSM, VinDr-Mammo para validação externa) |

## Baixar o dataset RSNA

1. Aceite as regras da competição em
   <https://www.kaggle.com/competitions/rsna-breast-cancer-detection>
2. Gere o token da API em <https://www.kaggle.com/settings> → *Create New Token*
3. Preencha `KAGGLE_USERNAME` e `KAGGLE_KEY` no seu `.env`
4. Rode:

```bash
kaggle competitions download -c rsna-breast-cancer-detection -p data/raw
unzip data/raw/rsna-breast-cancer-detection.zip -d data/raw
```

São ~314 GB no download completo. Para desenvolver, use a amostra reduzida
combinada pelo time em vez do dataset inteiro.

## Disco externo

Se os dados não couberem no seu disco principal, aponte `DATA_ROOT` no `.env`
para onde eles estiverem. O código em `src/utils/paths.py` respeita essa
variável — nada mais precisa mudar.

## Pré-processamento

Dois passos, sempre a partir da raiz do repositório:

```bash
# 1. Lê os cabeçalhos, remove duplicatas e monta data/manifest.csv
python -m scripts.inventariar_dicoms --annotations data/raw/breast-level_annotations.csv

# 2. Gera os PNGs de treino em data/processed/
python -m scripts.preprocessar_mamografia
```

O passo 2 aplica a cadeia radiométrica do DICOM (LUT de modalidade, VOI LUT ou
janelamento, correção de MONOCHROME1), recorta a mama e grava, por imagem:

| Pasta | Conteúdo |
|-------|----------|
| `processed/png_anatomia/` | Imagem corrigida e normalizada, 1 canal |
| `processed/png_realce_tophat/` | 3 canais: anatomia, top-hat (calcificação), CLAHE (massa) |
| `processed/png_mascara_mama/` | Silhueta da mama, separa tecido de padding |
| `processed/relatorio_preprocessamento.csv` | Caminho radiométrico, crop e redução de cada imagem |

As mamas direitas saem espelhadas e todas vão para um canvas de 768x1920 com a
parede torácica à esquerda. A lógica fica em `src/data/preprocessamento.py`.
