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
