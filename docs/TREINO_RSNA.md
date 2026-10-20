# Treino RSNA: referência técnica

Treina e avalia a detecção de câncer por mama no fold 0, para qualquer modelo do
timm e qualquer representação da Etapa 2 (`cinza` ou `realce_3canais`). O mesmo
código roda nas duas máquinas; muda só a config. O guia passo a passo para
quem recebe o HD é o `docs/GUIA_DAVI.md`.

| Arquivo | O que faz |
|---------|-----------|
| `configs/treino.yaml` | Treino, ajuste, MLflow e uma seção de hiperparâmetros por modelo |
| `configs/local.yaml` | Caminhos, MLflow e nome desta máquina (fora do git; modelo em `local.exemplo.yaml`) |
| `src/utils/config_local.py` | Junta o `local.yaml` às configs versionadas |
| `scripts/rsna_treinar.py` | Ajuste (`--modo ajuste`) e treino final (`--modo final`), retomáveis |
| `scripts/rsna_treino_verificar.py` | `maior-batch`, `smoke` e `sanidade`, antes de qualquer treino completo |
| `scripts/rsna_comparar.py` | Tabela dos treinos finais e, com `--ajuste`, das tentativas de ajuste |
| `scripts/rsna_avaliar_teste.py` | Avaliação final no teste hold-out (uma vez, no fim), em dois pontos de operação |
| `src/training/rsna_config.py` | Config de treino; `carregar_manifest` recusa o teste |
| `src/training/rsna_amostragem.py` | Positivos + negativos do pool por época; segmentos de tamanho fixo |
| `src/training/rsna_treino.py` | Orçamento, laço, acumulação, EMA, validação por mama, checkpoints, retomada |
| `src/training/rsna_execucao.py` | Monta um run a partir das configs (nome, pastas, params, tags) |
| `src/training/rsna_registro.py` | Arquivos de `runs/`, MLflow, informações da máquina |
| `src/models/rsna_modelo.py` | Modelo timm com 1 logit, otimizador e scheduler |
| `src/models/rsna_ema.py` | Média móvel exponencial (EMA) dos pesos |
| `src/evaluation/rsna_metricas.py` | Agregação por mama, F-beta, métricas, bootstrap, recortes |
| `requirements-treino.txt` | Etapa 2 + timm e MLflow, versões fixas |

## Comandos

```bash
pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements-treino.txt

# antes de qualquer treino completo
python -m scripts.rsna_treino_verificar maior-batch
python -m scripts.rsna_treino_verificar smoke
python -m scripts.rsna_treino_verificar sanidade

# ajuste: uma tentativa por comando (12 épocas de referência em passos, só validação)
python -m scripts.rsna_treinar --modo ajuste --modelo convnext_tiny --representacao cinza --seed 0 --set lr=5e-5
python -m scripts.rsna_comparar --ajuste              # runs/ajuste/ajuste.{csv,md}

# treino final: grave a config escolhida em configs/treino.yaml e rode as 3 seeds
for rep in cinza realce_3canais; do for seed in 1 2 3; do
  python -m scripts.rsna_treinar --modo final --modelo convnext_tiny --representacao $rep --seed $seed
done; done

# caiu: o mesmo comando com --retomar (no ajuste, também --tentativa N)
python -m scripts.rsna_treinar --modo final --modelo convnext_tiny --representacao cinza --seed 1 --retomar

python -m scripts.rsna_comparar                       # runs/comparacao.{csv,md}
mlflow ui --backend-store-uri sqlite:////mnt/SSD2/rsna_treino/mlflow.db --port 5000

# só no fim, uma vez por configuração escolhida
python -m scripts.rsna_avaliar_teste --run runs/convnext_tiny_cinza_seed1 --uso-unico-confirmado
```

**Sanidade:**
- **O que faz:** decora 64 imagens fixas (32 positivas), sem augmentation, drop path, weight decay nem EMA, sem acumulação e com warmup curto de 5 épocas (30 passos).
- **Quando passa:** a loss fica abaixo de 0,02 em até 100 épocas. O teste só confirma que o laço aprende.
- **Diagnóstico:** o `--set` troca hiperparâmetros (ex.: `--set lr=2e-5`).
- **Warmup:** sem warmup, ou com só 1 época (6 passos), o ConvNeXt com AdamW a 1e-4 colapsou para uma resposta constante, com a loss presa em ln 2 por 100 épocas; com 5 épocas, decorou em 12.

## Métrica de seleção

Neste projeto, **um falso negativo (câncer não detectado) é muito mais grave que
um falso positivo** (uma paciente chamada de volta sem câncer). Por isso a
seleção usa o **F2 por mama**: no F-beta, a revocação (sensibilidade) pesa
beta² vezes a precisão, então no F2 ela pesa 4 vezes.

- A métrica de seleção decide o early stopping, o `melhor.pt` e a escolha entre tentativas do ajuste.
- **Opções** (`treino.metrica_selecao`): `f1_melhor`, `f1_5_melhor`, `f2_melhor` (padrão), `esp_sens90` e `roc_auc`. Com `betas_f`, outros F-betas viram opção (ex.: 3 dá `f3_melhor`).
- **Cada F-beta tem o seu limiar:** `f2_melhor` é o maior F2 entre todos os limiares, e `limiar_f2` é o limiar que o atinge. O limiar do F2 fica abaixo do limiar do F1: ele aceita mais falsos positivos para perder menos positivas.
- **Sempre calculadas e reportadas**, qualquer que seja a seleção: `pf1`, F1, F1.5 e F2 (com os limiares), `roc_auc`, `pr_auc`, `sens_esp90` (sensibilidade com especificidade de 90%) e `esp_sens90` (especificidade com sensibilidade de 90%, com o limiar em `limiar_sens90`).
- **Ponto de operação da seleção** (`limiar_selecao` no `metricas_finais.json`): o limiar da métrica de seleção. O ROC AUC não tem limiar; com ele, vale o limiar do maior beta (F2).

## Orçamento por passos

A proporção de amostragem muda o tamanho da época: com todos os 780 positivos
do treino, 1:10 dá 8.580 imagens por época, 1:4 dá 3.900 e 1:1 dá 1.560 (5,5
vezes menos que 1:10). Com orçamento por épocas, uma tentativa em 1:1 treinaria
5,5 vezes menos passos que uma em 1:10, e a comparação mediria o tempo de treino,
não a proporção. Por isso o orçamento conta **passos do otimizador**.

- **Época de referência:** os passos de uma época na `proporcao_referencia` (1:10). Com batch efetivo 36, são 8.580 imagens em **239 passos**.
- **Ajuste:** 12 épocas de referência = **2.868 passos**, com validação a cada 239 passos (12 validações), em qualquer proporção.
- **Treino final:** até 30 épocas de referência = **7.170 passos**, com early stopping depois de 5 validações sem melhora.
- **Segmentos:** o treino é um fluxo contínuo de épocas de amostragem da proporção do modelo, cortado em segmentos de 8.580 imagens; cada segmento termina numa validação. Em 1:10, o segmento é exatamente uma época de amostragem; em 1:1, são 5,5 épocas (cada positivo aparece 5,5 vezes por segmento).
- **Warmup e scheduler** também contam em épocas de referência: `warmup_epocas: 1` são 239 passos de warmup, e o cosine vai até o último passo do orçamento.
- **Outro batch efetivo:** a época de referência continua com 8.580 imagens; o número de passos é recalculado para o batch do modelo (ex.: batch 32 dá 269 passos por validação).
- **Por épocas, se preferir:** `orcamento: epocas` volta ao modo antigo, em que cada validação é uma época de amostragem da proporção do modelo. O ajuste do projeto usa passos.
- `max_passos` e `passos_por_validacao` fixam os números diretamente, em vez de derivá-los da época de referência. O total é arredondado para cima até uma validação inteira.

## EMA dos pesos

Com `ema: true`, o laço mantém uma média móvel exponencial dos pesos, atualizada
a cada passo do otimizador:

- `ema = d * ema + (1 - d) * modelo`, com `d = ema_decaimento` a cada passo.
- **Aquecimento:** durante o warmup do lr (239 passos com `warmup_epocas: 1`), `d = 0`: a EMA é só uma cópia do modelo, para não carregar os pesos do ImageNet nem os do warmup na média. O aquecimento clássico do TensorFlow e do timm, `min(d, (1 + t) / (10 + t))`, só chega a 0,999 no passo ~9.000, depois do fim do ajuste (2.868 passos): o decaimento da config nunca valeria.
- **Padrão: `ema_decaimento: 0.999`.** A janela efetiva é cerca de 1 / (1 - d) = 1.000 passos, cerca de 4 épocas de referência: um terço do ajuste e um sétimo do treino final. A receita do ConvNeXt usa 0,9999 (janela de 10.000 passos), mas para 300 épocas do ImageNet com ~1.250 passos cada; aqui, com 0,9999, a EMA ainda guardaria 77% dos pesos do fim do warmup no fim do ajuste (com 0,999, 7%).
- **Validação e checkpoint:** com a EMA ligada, a validação, o early stopping, o `melhor.pt` e as predições usam os pesos da EMA (o `melhor.pt` marca `pesos: ema`). O `ultimo.pt` guarda os pesos crus e a EMA, para retomar.
- **BatchNorm (ResNet):** as estatísticas do BatchNorm entram na média como os pesos.
- Custo: uma cópia do modelo na GPU (~110 MB no ConvNeXt-Tiny) e uma média por passo.

## Plano do ajuste (ConvNeXt-Tiny, representação `cinza`)

Cada tentativa tem o mesmo orçamento (2.868 passos) e é avaliada só na validação.
A escolha é pela métrica de seleção (F2). Ao fim de cada fase, a tabela de
`python -m scripts.rsna_comparar --ajuste` vai para o registro
(`docs/registro_experimentos_convnext_tiny.md`) e para o relatório, e a fase
seguinte só começa com aprovação.

| Fase | O que varia | Fixo | Seeds | Tentativas |
|------|-------------|------|-------|------------|
| 1. learning rate | `lr` 1e-4, 5e-5 e 2e-5 | 1:10, EMA desligada | 2 cada (0 e 1) | 6 |
| 2. proporção | `negativos_por_positivo` 1 e 4 (1:10 vem da fase 1) | o melhor `lr`; em 1:1, `pos_weight` 1 | 1 cada | 2 |
| 3. EMA | `ema: true` | a melhor combinação até ali | 1 | 1 |

- **Empates nas fases 2 e 3:** se a diferença para a melhor opção for pequena (dentro da diferença entre seeds da fase 1), roda-se a segunda seed só nas opções empatadas antes de decidir.
- **Ainda melhorando?** A tabela do ajuste marca se a melhor validação foi uma das 2 últimas do orçamento sem early stopping. Nesse caso, o vencedor pode render mais no treino final (que vai até 30 épocas de referência).
- **Comandos** (fase 1; nas fases 2 e 3, os `--set` com o lr escolhido):
  ```bash
  for lr in 1e-4 5e-5 2e-5; do for seed in 0 1; do
    python -m scripts.rsna_treinar --modo ajuste --modelo convnext_tiny --representacao cinza --seed $seed --set lr=$lr
  done; done
  python -m scripts.rsna_treinar --modo ajuste ... --seed 0 --set lr=<melhor> negativos_por_positivo=1 pos_weight=1
  python -m scripts.rsna_treinar --modo ajuste ... --seed 0 --set lr=<melhor> negativos_por_positivo=4
  python -m scripts.rsna_treinar --modo ajuste ... --seed 0 --set lr=<melhor> negativos_por_positivo=<melhor> ema=true
  ```
- **Tempo:** cerca de 5 min por época de referência na RTX 4060 (treino + validação), ~1 h por tentativa, ~6 h na fase 1.
- **Depois do ajuste:** a config vencedora vai para a seção `convnext_tiny` do `configs/treino.yaml` e roda no treino final com 3 seeds (1, 2 e 3) em cada representação.

## Config (`configs/treino.yaml`)

**Caminhos da máquina (`configs/local.yaml`, fora do git)**

As configs versionadas não têm caminhos absolutos. Cada máquina cria o `configs/local.yaml` a partir do `configs/local.exemplo.yaml`; cada seção completa a config de mesmo nome (`src/utils/config_local.py`), e um valor na própria config vale sobre o `local.yaml`. Sem um caminho necessário, o script para com uma mensagem que aponta para o exemplo.
- `maquina`: o nome da máquina que vai nas tags (vazio = hostname).
- `etapa2`: `etapa1_dir`, `realce_dir` e `verificacao_dir` dos dados da Etapa 2.
- `treino`: `checkpoints_dir` (`ultimo.pt` e `melhor.pt`) e `verificacao_dir` (saídas de `rsna_treino_verificar`).
- `mlflow`: `tracking_uri` (SQLite local no SSD, ou `http://servidor:5000` para um servidor compartilhado) e `artefatos_dir` (onde os artefatos ficam quando o experimento é criado localmente).
- `pipeline_rsna`: `rsna_dir` e `saida_dir`, só para a divisão e a Etapa 1.

**`caminhos`**
- `etapa2_config`: a config da Etapa 2, de onde vem a representação.
- `runs_dir`: os arquivos pequenos de cada run, versionados.

**`mlflow`**
- `ativo`: liga e desliga o MLflow.
- `experimento`: nome do experimento, `rsna_fold0`.

**Raiz da config**
- `modelo`: o modelo padrão quando não há `--modelo`.

**`treino`**
- `orcamento`: `passos` (padrão) ou `epocas` (ver "Orçamento por passos").
- `proporcao_referencia`: negativos por positivo da época de referência (10).
- `max_epocas`: limite do treino final, 30, em épocas de referência (ou em épocas, com `orcamento: epocas`).
- `max_passos` e `passos_por_validacao`: vazios, derivados da época de referência; preenchidos, fixam os números.
- `paciencia`: quantas validações sem melhora até o early stopping.
- `metrica_selecao`: `f2_melhor` (ver "Métrica de seleção"). Não mude sem combinar: ela vale para os dois modelos.
- `betas_f`: os F-betas calculados (1, 1,5 e 2 sempre entram).
- `precisao`: `auto` usa bf16 se a GPU suportar e, senão, fp16 com GradScaler.
- `channels_last`: formato de memória das imagens na GPU.
- `bootstrap_amostras` e `semente_bootstrap`: o IC95 por mama no fim do treino.

**`ajuste`**
- `epocas`: orçamento de cada tentativa, 12 épocas de referência (2.868 passos com batch 36).
- `max_passos`: vazio, derivado de `epocas`.
- `paciencia`: early stopping no ajuste, em validações.

**`modelos.<nome>`**: uma seção por modelo. Cada pessoa ajusta a do próprio modelo.

| Campo | Efeito |
|-------|--------|
| `timm_nome` | Arquitetura e pesos do timm (nome livre; ex.: `convnext_tiny.fb_in1k`). Modelo novo = seção nova |
| `batch_real` | Imagens por passada na GPU (o que cabe na memória) |
| `acumulacao` | Passadas por passo do otimizador. **Batch efetivo = batch_real x acumulacao** |
| `otimizador` | `adamw` (usa `betas`) ou `sgd` (Nesterov, usa `momento`) |
| `lr`, `lr_minimo` | Learning rate de pico e piso do cosine |
| `weight_decay` | Só em pesos com 2+ dimensões; normas e bias ficam sem decay |
| `drop_path` | Stochastic depth (0 desliga) |
| `warmup_epocas` | Warmup linear, em épocas de referência (fração aceita) |
| `scheduler` | `cosine` (por passo, até o fim do orçamento) ou `constante` |
| `pos_weight` | Peso da classe positiva na BCEWithLogitsLoss |
| `negativos_por_positivo` | Negativos por positivo em cada época de amostragem (padrão 10) |
| `peso_negativo_dificil` | Peso de sorteio dos negativos `difficult_negative_case` ou `biopsy` (padrão 2) |
| `ema`, `ema_decaimento` | EMA dos pesos (padrão desligada; decaimento 0,999) |

O `--set chave=valor` (só no ajuste) troca campos dessa seção na hora, e o que
mudou fica gravado no run. O treino final recusa `--set`: a config escolhida
tem de estar no arquivo, para o repositório refletir o que foi treinado.

**Batch efetivo:** a recomendação é cerca de 36 nos dois modelos, para facilitar a comparação. Se usar outro valor, ele fica em `metricas_finais.json`, nas tags e na tabela de comparação.

### Valores iniciais e origem

- **ConvNeXt-Tiny** (Liu et al., 2022, *A ConvNet for the 2020s*, receita do ImageNet-1k):
  - AdamW, weight decay 0,05, drop path 0,1 (T), warmup e cosine.
  - O `lr` da receita é 4e-3 para batch 4096, o que dá cerca de 3,5e-5 para batch 36 por escala linear. O ajuste testa 1e-4, 5e-5 e 2e-5, em volta desse valor; 1e-4 é o padrão da config por ser um fine-tuning em outro domínio.
  - Pesos: `convnext_tiny.in12k_ft_in1k`, pré-treinado no ImageNet-22k e ajustado no 1k.
- **ResNet-50** (Goyal et al., 2017, *Accurate, Large Minibatch SGD*):
  - SGD com momento 0,9 (Nesterov), weight decay 1e-4, warmup e cosine.
  - O `lr` da receita é 0,1 por 256 imagens, o que dá cerca de 0,014 para batch 36; o padrão é 0,01. Essa é a escala do treino do zero; o fine-tuning costuma ficar nela ou abaixo. A fase 1 recomendada, com SGD (Nesterov), testa **1e-2, 5e-3 e 2e-3** (o mesmo fator ~5 entre o maior e o menor do ConvNeXt).
  - Pesos: `resnet50.tv2_in1k`, da receita nova do torchvision.
  - Essa faixa de `lr` só vale para SGD. Alternativa: AdamW com `lr` 3e-4, 1e-4 e 5e-5 e weight decay 0,01 (cerca de 100 vezes menor que o SGD).
- **EMA:** decaimento 0,999 nos dois (ver "EMA dos pesos").

## Como funciona

**Dados**
- Treino e validação do fold 0, saídos de `manifest_processado.csv`.
- Treino, ajuste, smoke e sanidade usam só `carregar_manifest`, que recusa o teste. O laço também chama `recusar_teste` antes de montar os Datasets.
- Só `rsna_avaliar_teste.py` passa `permitir_teste=True`, e um teste unitário garante que os scripts de treino nunca fazem isso.

**Amostragem**
- A cada época de amostragem entram todos os positivos do treino, mais `negativos_por_positivo` vezes esse número de negativos do pool (`in_negative_pool`), sem reposição.
- O sorteio é ponderado: negativo difícil tem peso `peso_negativo_dificil`. Como o sorteio é sem reposição, a frequência real fica abaixo do peso. No pool do fold 0 (7.800 de 22.000 negativos por época em 1:10), o peso 2 dá 1,69 vez mais aparições: os difíceis passam de 23% do pool para 34% dos negativos de cada época.
- O gerador é `SeedSequence([seed, época])`: a mesma época sai igual em qualquer máquina e depois de retomar. Somar seed + época faria a seed 1 na época 2 coincidir com a seed 2 na época 1.
- Com orçamento por passos, o segmento k são as posições [k x 8.580, (k + 1) x 8.580) do fluxo dessas épocas (`indices_do_segmento`), então também depende só de (seed, k).
- A validação usa as 9.033 imagens, sem balanceamento e sem augmentation.

**Treino**
- O loop usa `MamografiaRSNA.da_config` da Etapa 2, com augmentation só no treino.
- A loss é BCEWithLogitsLoss (soma) dividida pelo tamanho do grupo de acumulação. Assim, `batch_real x acumulacao` dá exatamente o gradiente do batch efetivo, há um teste para isso, e o último grupo do segmento pode ser menor.
- Com BatchNorm (ResNet), as estatísticas são as do micro-batch.
- Precisão bf16 (ou fp16 com scaler) e `channels_last`.

**Validação a cada segmento**
- A probabilidade de cada imagem é agregada por mama (`patient_id` + `laterality`) pelo máximo.
- Métricas por mama: as da seção "Métrica de seleção".
- No fim do treino, um IC95 por bootstrap sobre as mamas, na melhor validação.

**Checkpoints**
- Ficam em `checkpoints_dir/<run>/`.
- `ultimo.pt` é gravado a cada validação, com modelo, EMA, otimizador, scheduler, scaler, estado do early stopping, histórico e id do run no MLflow.
- `melhor.pt` guarda os pesos avaliados (os da EMA, se ligada), a validação, as métricas, as configs (`params`) e as tags.
- Retomar continua do `ultimo.pt` com o mesmo run do MLflow. O resultado é idêntico ao de não ter parado: byte a byte na CPU, e na GPU a menos do não determinismo do cuDNN.

**Reprodutibilidade:** `set_seed(seed)` no início. Depois, cada segmento tem sua própria semente, que vale para a amostragem, a augmentation e o dropout.

## Saídas padronizadas

`runs/<modelo>_<rep>_seed<N>/` para o treino final e `runs/ajuste/<modelo>_<rep>_tentativa<N>/` para o ajuste. Tudo é pequeno e versionado:

| Arquivo | Conteúdo |
|---------|----------|
| `config_treino.yaml` | Hiperparâmetros do modelo usado (com `batch_efetivo`), treino, ajuste, modo, seed, `alteracoes` |
| `config_etapa2.yaml` | `entrada` (representação e realce), `augmentation` e `carregamento` |
| `maquina.json` | Máquina, GPU, VRAM, driver, Python, torch, timm, CUDA, cuDNN |
| `execucao.json` | Commit, alterações locais, `checkpoint_dir`, MLflow, início, comando, retomadas |
| `historico.csv` | Uma linha por validação (ver abaixo) |
| `predicoes_val_imagem.csv` | Melhor validação, por imagem: `image_id, patient_id, laterality, view, cancer, prob` |
| `predicoes_val_mama.csv` | Melhor validação, por mama: `prediction_id, patient_id, laterality, cancer, prob, n_imagens` |
| `metricas_finais.json` | Melhor validação e passo, métricas, `limiar_selecao`, `ic95_validacao`, batch, passos por validação, EMA, tempo, `mlflow_run_id` |
| `curvas_roc_pr.png` | ROC e precisão-revocação por mama da melhor validação |

Colunas do `historico.csv`:
- `epoca` (o número da validação), `passo` (passos do otimizador até ali), `epocas_amostragem` (épocas da proporção do modelo até ali), `lr` (no início do segmento), `loss_treino` e `imagens_s`;
- `segundos_treino` e `segundos_validacao`;
- `val_pf1`, `val_f1_melhor`, `val_f1_5_melhor`, `val_f2_melhor`, os limiares `val_limiar_f1`, `val_limiar_f1_5`, `val_limiar_f2` e `val_limiar_sens90`, `val_roc_auc`, `val_pr_auc`, `val_sens_esp90`, `val_esp_sens90`, `val_n_mamas` e `val_n_positivas`;
- `vram_pico_gb` e `melhorou`.

No `prob`, a probabilidade é a sigmoide do logit. No `cancer`, 1 numa mama quer dizer que alguma das imagens dela tem câncer.

**MLflow** (experimento `rsna_fold0`, um run por treino):
- **Params:** todos os hiperparâmetros e a config da Etapa 2, achatados (ex.: `treino.hiperparametros.lr`).
- **Métricas:** as do `historico.csv` por validação, com step = passo do otimizador (as curvas de proporções diferentes ficam no mesmo eixo), mais `melhor_*`.
- **Tags:** `modelo`, `timm_nome`, `representacao`, `seed`, `modo`, `maquina`, `gpu`, `commit`, `alteracoes_locais`, `batch_efetivo`, `alteracoes` e `checkpoint_dir`.
- **Artefatos:** a pasta do run (configs, predições, curvas). Os checkpoints não sobem.
- O backend de arquivos (`mlruns/`) do MLflow 3 está em modo de manutenção e recusa gravar, por isso o padrão é SQLite.

**Comparação dos treinos finais:** `runs/comparacao.{csv,md}`, uma linha por modelo e representação, com:
- média ± desvio das métricas entre as seeds;
- IC95 médio entre as seeds;
- batch efetivo usado;
- número de tentativas de ajuste concluídas do modelo;
- épocas, horas e GPU.

**Tabela do ajuste:** `runs/ajuste/ajuste.{csv,md}`, com:
- uma linha por configuração (as alterações do `--set`), com a média e a diferença (maior - menor) entre seeds de cada métrica, ordenada pela métrica de seleção;
- a melhor validação de cada seed sobre as validações treinadas, e se ainda estava melhorando no fim do orçamento;
- uma linha por tentativa, com as métricas de cada seed.

**Teste hold-out:** `runs/<run>/teste/` com:
- `metricas_teste.json`: as métricas sem limiar, os **dois pontos de operação** escolhidos na validação e o IC95 de cada um:
  - `selecao`: o limiar da métrica de seleção (`limiar_selecao`, o do F2);
  - `sens90`: o maior limiar com sensibilidade de pelo menos 90% na validação (`limiar_sens90`);
  - em cada um, F1, F1.5, F2, precisão, sensibilidade, especificidade e a matriz de confusão;
- `recortes_teste.csv`, com a coluna `ponto_de_operacao`: os dois pontos, por site, faixa etária, aparelho e spacing estimado, e por negativos difíceis contra comuns;
- as predições por imagem e por mama.

O script recusa rodar de novo no mesmo run.

## Testes

`tests/test_rsna_amostragem.py`, `test_rsna_metricas.py`, `test_rsna_treino.py` e `test_rsna_comparar.py` usam dados sintéticos e um modelo minúsculo na CPU. Eles cobrem:
- a proporção e o determinismo do sampler, o peso dos negativos difíceis e os segmentos de tamanho fixo;
- a agregação pelo máximo, as métricas (F1, F1.5 e F2 com os seus limiares, especificidade com sensibilidade de 90%, calculados à mão), o limiar da métrica de seleção, o bootstrap e os recortes;
- a recusa do teste e de uma métrica de seleção inválida;
- a montagem de cada modelo e otimizador pela config;
- a equivalência entre a acumulação e o batch efetivo;
- o orçamento por passos: o mesmo número de passos e de validações em 1:1, 1:3 e 1:10, e o mesmo scheduler;
- a EMA: a fórmula com aquecimento, o BatchNorm, e que a validação e o `melhor.pt` usam os pesos dela;
- a retomada idêntica a não parar, por épocas e por passos com EMA;
- o formato das saídas e as tabelas de comparação e do ajuste.
