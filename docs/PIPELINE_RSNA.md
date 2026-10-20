# Pipeline RSNA: divisão e pré-processamento (Etapa 1)

Gera a divisão por paciente e as imagens processadas para o treino do
ConvNeXt-Tiny. É reproduzível: com os mesmos DICOMs, a mesma config e as
versões do `requirements-pipeline.txt`, os arquivos saem idênticos (confira
pelos `SHA256SUMS`).

| Arquivo | O que faz |
|---------|-----------|
| `configs/pipeline_rsna.yaml` | Caminhos, semente, proporções, pool, resolução, limiares, workers |
| `scripts/rsna_dividir.py` | Script 1: limpeza, divisão por paciente, seleção do fold 0 (só CSVs) |
| `scripts/rsna_preprocessar.py` | Script 2: DICOM -> anatomia 8 bits + máscara (PNGs) |
| `scripts/verificar_checksums.py` | Confere a integridade de uma pasta pelo `SHA256SUMS` |
| `src/data/rsna_divisao.py`, `src/data/rsna_imagem.py` | A lógica, testada em `tests/test_rsna_*.py` |
| `splits/rsna_fold0/` | `splits.csv`, `manifest_fold0.csv`, relatório e hashes (versionados) |

## Passo a passo

### 1. Ambiente (Python 3.12)

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements-pipeline.txt
```

### 2. Baixar o RSNA (~290 GB zipado, ~315 GB extraído)

1. Aceite as regras em <https://www.kaggle.com/competitions/rsna-breast-cancer-detection/rules>.
2. Em <https://www.kaggle.com/settings>, seção **API**, clique em **Generate New Token**.
   O token começa com `KGAT_`. Salve-o (o cliente lê sozinho):

   ```bash
   mkdir -p ~/.kaggle
   read -s -p "Token: " T && printf '%s' "$T" > ~/.kaggle/access_token && chmod 600 ~/.kaggle/access_token
   ```

   Alternativa: `export KAGGLE_API_TOKEN=KGAT_...`. O antigo `kaggle.json`
   (usuário + chave) também funciona, mas não é mais o padrão.
3. Baixe e extraia (o download aceita retomada com `-C -`):

   ```bash
   curl -L -C - --fail -H "Authorization: Bearer $(cat ~/.kaggle/access_token)" \
     -o rsna.zip https://www.kaggle.com/api/v1/competitions/data/download-all/rsna-breast-cancer-detection
   unzip -q rsna.zip -d rsna-breast-cancer-detection
   ```

A estrutura esperada é `rsna-breast-cancer-detection/train.csv` e
`rsna-breast-cancer-detection/train_images/<patient_id>/<image_id>.dcm`.

### 3. Ajustar a config

Crie o `configs/local.yaml` (fora do git) a partir do `configs/local.exemplo.yaml`
e preencha a seção `pipeline_rsna` (`rsna_dir` e `saida_dir`). O
`configs/pipeline_rsna.yaml` não tem caminhos da máquina; qualquer valor dele
muda os dados gerados.

### 4. Divisão (segundos)

```bash
.venv/bin/python -m scripts.rsna_dividir
```

Imprime a limpeza, as fusões de estratos raros, a checagem de vazamento e as
distribuições. Os CSVs em `splits/rsna_fold0/` devem bater com o `SHA256SUMS`
versionado: `python -m scripts.verificar_checksums splits/rsna_fold0`.

### 5. Pré-processamento

```bash
# teste: amostra de 50 imagens em <saida>/amostra, com grade antes/depois e estimativas
.venv/bin/python -m scripts.rsna_preprocessar --amostra

# lote completo (~6 h com 8 workers lendo de HD; ~21 GB)
.venv/bin/python -m scripts.rsna_preprocessar
```

O lote é retomável: se parar, rode o mesmo comando e ele continua (a imagem
conta como pronta quando está no `progresso.jsonl` e os dois PNGs existem).
Falhas vão para `erros.log` e são tentadas de novo na próxima execução.

Saída em `saida_dir`:

| Item | Conteúdo |
|------|----------|
| `anatomia/<image_id>.png` | 8 bits, parede torácica à esquerda, lado maior até 1536 px |
| `mascara/<image_id>.png` | Silhueta da mama (0/255), mesmo tamanho da anatomia |
| `manifest_processado.csv` | Manifest do fold 0 + caminhos (relativos) e medidas de cada imagem |
| `suspeitas.csv` | Máscara com cobertura fora de 25-92%, positivas primeiro: revisar à mão |
| `SHA256SUMS` | Hash de todos os PNGs e CSVs de saída |

### 6. Entrega por HD

Copie a pasta `saida_dir` inteira. Quem recebe confere com:

```bash
python -m scripts.verificar_checksums /caminho/no/hd/rsna_fold0_etapa1
```

## Decisões que valem saber

- **Divisão por paciente.** 15% das pacientes no teste; o resto em 5 folds
  (`StratifiedGroupKFold`), estratificados por câncer da paciente + site +
  faixa etária. Faixas com menos de 10 pacientes são fundidas com a vizinha
  (o relatório lista quais). `machine_id` não entra na estratificação.
- **Exclusão manual.** Erros de rótulo achados na revisão visual vão em
  `divisao.excluir_imagens` na config, um `image_id` por linha com o motivo em
  comentário. A exclusão é aplicada **depois** do sorteio: aplicada antes, uma
  paciente a menos moveu 28% das pacientes de split. Assim, cada exclusão nova
  só remove as próprias linhas (e o pool puxa um substituto da fila); os PNGs
  já gerados e os treinos já feitos continuam valendo.
- **Pool do treino.** Todos os positivos + 22.000 negativos fixos, incluindo
  todos os `difficult_negative_case`/`biopsy`. A subamostragem por época é do
  DataLoader. Teste e validação ficam na proporção real.
- **Orientação.** A checagem de fundo usa a saturação do ar (o janelamento o
  joga num extremo da escala), não os cantos: no RSNA a mama pode ocupar os
  dois cantos da parede e há marcadores brancos em cantos.
- **Espelhamento.** Pelo lado com mais tecido, não pela tag: no site 1 a tag
  diverge do lado real da parede em ~30% das imagens (`diverge_tag`).
- **Ar ruidoso (aparelho 210).** Nesse aparelho o ar não satura em zero (fica
  entre ~0,1% e 2% da escala). A máscara sobe o limiar para o dobro do pico do
  ar quando ele não está no primeiro bin; nos outros aparelhos nada muda. Sem
  isso, a máscara cobria a chapa e ~60% das imagens saíam com a parede à
  direita. As 812 imagens do 210 da primeira entrega foram refeitas; a lista
  está em `reprocessamento.csv` na pasta de saída.
- **Chapa e letras (aparelho 210).** Com o limiar elevado pelo ar ruidoso, a
  borda clara da chapa e as letras do marcador ainda passavam, encostavam na
  mama e o preenchimento por linha levava a máscara até a borda da chapa. Só
  nesse caso, uma abertura por disco apaga o que é mais fino que ~2,5% do lado
  menor (a faixa da chapa) antes do maior componente. Fora da imagem conta como
  fundo, porque a chapa costuma ser a última linha do DICOM. As 811 imagens do
  210 foram refeitas de novo (segunda linha por imagem no `reprocessamento.csv`);
  as que ainda mostravam letras foram excluídas pela config.
- **Spacing.** 6 aparelhos não gravam o espaçamento; usa 0,065238 mm e marca
  `spacing_estimado`. `spacing_resultante_mm` já considera o redimensionamento
  (é o que a Etapa 2 usa para o top-hat de 0,6 mm).
- **Rótulos auxiliares** (BIRADS, biopsy, invasive, difficult_negative_case)
  servem para limpeza, pool e análise. Nunca são entrada do modelo.
