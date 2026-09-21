# Modelos

Pesos treinados **não vão para o Git** (arquivos de centenas de MB estouram o
limite do GitHub e poluem o histórico).

Onde colocar cada coisa:

- **Pesos e checkpoints** → ficam aqui localmente, ou em MLflow / GitHub Release.
- **Métricas e hiperparâmetros** → registrados no MLflow e resumidos no PR.
- **Config do experimento** → versionada em `configs/<experimento>.yaml`.

Convenção de nome: `<backbone>_<dataset>_<data>_<iniciais>.pt`
Exemplo: `effnetb0_rsna_20260315_dsp.pt`
