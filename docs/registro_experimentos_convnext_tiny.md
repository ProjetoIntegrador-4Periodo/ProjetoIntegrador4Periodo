# Registro de experimentos: convnext_tiny

Preencha durante o trabalho, não só no fim. Commite este arquivo junto com os
resultados de `runs/` (ver `docs/GUIA_DAVI.md`, passo 11). Os números vêm de
`runs/.../metricas_finais.json`, da tabela do `python -m scripts.rsna_comparar --ajuste`
ou do MLflow; copie-os, não arredonde de cabeça.

- Responsável: Jhonatan
- Período: início em 19/10/2026

**Regras comuns aos dois modelos** (`configs/treino.yaml`, não mudam sem combinar):
- **Métrica de seleção:** F2 por mama (`f2_melhor`), porque um falso negativo é mais grave que um falso positivo. Ela decide o early stopping, o `melhor.pt` e a escolha entre tentativas.
- **Orçamento por passos:** cada tentativa de ajuste tem 12 épocas de referência (1 época de referência = os passos de uma época em 1:10; 239 passos com batch efetivo 36), com validação a cada época de referência e paciência de 5 validações. O treino final vai até 30 épocas de referência.
- **Só a validação do fold 0.** O teste hold-out não é usado aqui.

## 1. Máquina

Saída de `nvidia-smi` e do `runs/<qualquer run>/maquina.json`.

| Item | Valor |
|------|-------|
| GPU | NVIDIA GeForce RTX 4060 |
| VRAM (GB) | 8 (7,62 visíveis ao torch) |
| Driver NVIDIA | 595.91.07 |
| CUDA (torch.version.cuda) | 13.0 (cuDNN 9.24) |
| Python | 3.12.3 |
| torch | 2.14.0+cu130 |
| timm | 1.0.29 |
| Sistema | Linux Mint 22.3 (kernel 6.17) |
| Maior `batch_real` que cabe (`rsna_treino_verificar maior-batch`) | 12 (5,94 GB); 16 não coube |
| `batch_real` x `acumulacao` usados (batch efetivo) | 12 x 3 = 36 |
| Passos por época de referência (`passos_por_validacao` no `metricas_finais.json`) | 239 (8.580 imagens) |
| Smoke: imagens/s no treino e min por época de referência estimados | cinza 39,3 img/s, 5,0 min; realce 40,3 img/s, 4,9 min (validação ~110 img/s) |
| Sanidade: épocas até loss < 0,02 | 12 (warmup de 5 épocas; com 0 ou 1 época de warmup, colapsou em loss ln 2) |

## 2. Ajuste

Representação `cinza`. Uma linha por pasta `runs/ajuste/<modelo>_<rep>_tentativaN/`.
"Alterado" é o que foi passado no `--set` em relação à config inicial. As
métricas são por mama, na validação, na melhor validação da tentativa. "Melhor
val." é a melhor validação sobre as validações treinadas (ex.: `11/12`), e
"ainda melhorando" é a coluna da tabela do ajuste (melhor validação numa das 2
últimas do orçamento, sem early stopping).

Ao fim de cada fase, cole a tabela do `runs/ajuste/ajuste.md` (média e
diferença entre seeds) e escreva a decisão: o vencedor, a margem para o segundo
e se ele ainda estava melhorando no fim do orçamento.

### Fase 1: learning rate

Plano: `lr` 1e-4, 5e-5 e 2e-5 (AdamW), proporção 1:10, EMA desligada, seeds 0 e 1 (6 tentativas).

| Nº | Data | Seed | Alterado | Melhor val. | Ainda melhorando | f2_melhor | f1_melhor | f1_5_melhor | pf1 | roc_auc | pr_auc | sens_esp90 | esp_sens90 | Observações |
|----|------|------|----------|-------------|------------------|-----------|-----------|-------------|-----|---------|--------|------------|------------|-------------|
| | | | | | | | | | | | | | | |

Tabela da fase (média e diferença entre seeds):

Decisão:

### Fase 2: proporção

Plano: o melhor `lr` da fase 1, com `negativos_por_positivo` 1 (e `pos_weight` 1) e 4, 1 seed cada; o 1:10 vem da fase 1. Se a diferença para a melhor opção for pequena, a segunda seed roda só nas opções empatadas.

| Nº | Data | Seed | Alterado | Melhor val. | Ainda melhorando | f2_melhor | f1_melhor | f1_5_melhor | pf1 | roc_auc | pr_auc | sens_esp90 | esp_sens90 | Observações |
|----|------|------|----------|-------------|------------------|-----------|-----------|-------------|-----|---------|--------|------------|------------|-------------|
| | | | | | | | | | | | | | | |

Tabela da fase:

Decisão:

### Fase 3: EMA

Plano: a melhor combinação até aqui com `ema=true` (decaimento 0,999), 1 seed; segunda seed se empatar.

| Nº | Data | Seed | Alterado | Melhor val. | Ainda melhorando | f2_melhor | f1_melhor | f1_5_melhor | pf1 | roc_auc | pr_auc | sens_esp90 | esp_sens90 | Observações |
|----|------|------|----------|-------------|------------------|-----------|-----------|-------------|-----|---------|--------|------------|------------|-------------|
| | | | | | | | | | | | | | | |

Tabela da fase:

Decisão:

### Outras tentativas (fora do plano)

Qualquer tentativa extra, com o motivo.

| Nº | Data | Seed | Alterado | Melhor val. | Ainda melhorando | f2_melhor | f1_melhor | f1_5_melhor | pf1 | roc_auc | pr_auc | sens_esp90 | esp_sens90 | Motivo e observações |
|----|------|------|----------|-------------|------------------|-----------|-----------|-------------|-----|---------|--------|------------|------------|----------------------|
| | | | | | | | | | | | | | | |

## 3. Configuração final escolhida

Copie a seção do modelo que ficou no `configs/treino.yaml` (a mesma para as
duas representações) e explique a escolha.

```yaml
convnext_tiny:
  timm_nome:
  batch_real:
  acumulacao:        # batch efetivo =
  otimizador:
  lr:
  weight_decay:
  drop_path:
  warmup_epocas:
  scheduler:
  lr_minimo:
  pos_weight:
  negativos_por_positivo:
  peso_negativo_dificil:
  ema:
  ema_decaimento:
```

Motivo da escolha (qual tentativa, por que ela e não as outras, se o ganho
passou da diferença entre seeds ou foi dentro do ruído):

## 4. Treinos finais

Uma linha por pasta `runs/<modelo>_<rep>_seed<N>/`. "Melhor val." é a
validação do `melhor.pt` sobre as treinadas (até 30). Tempo = `segundos_total` / 3600.

| Representação | Seed | Melhor val. / treinadas | f2_melhor | limiar_f2 | f1_melhor | f1_5_melhor | pf1 | roc_auc | pr_auc | sens_esp90 | esp_sens90 | limiar_sens90 | Tempo (h) | Retomado? |
|---------------|------|-------------------------|-----------|-----------|-----------|-------------|-----|---------|--------|------------|------------|---------------|-----------|-----------|
| cinza | 1 | | | | | | | | | | | | | |
| cinza | 2 | | | | | | | | | | | | | |
| cinza | 3 | | | | | | | | | | | | | |
| realce_3canais | 1 | | | | | | | | | | | | | |
| realce_3canais | 2 | | | | | | | | | | | | | |
| realce_3canais | 3 | | | | | | | | | | | | | |

## 5. Observações e problemas encontrados

Erros, quedas e retomadas, mudanças de batch, avisos estranhos, curvas que
divergiram, qualquer coisa fora do roteiro. Com data.

- **19/10/2026 ~21:50, queda da tentativa 1 da fase 1** (lr 1e-4, seed 0):
  - O laço da fase 1 rodava preso à sessão do terminal. Quando a sessão foi encerrada, o processo morreu, depois da validação 6 de 12 (passo 1.673, F2 0,229). As outras 5 tentativas não chegaram a começar, e a GPU ficou parada de ~21:50 a 08:30.
  - **Retomada em 20/10, 08:30**, pelo `ultimo.pt` gravado às 21:50, com `--retomar --tentativa 1`, mesma seed e mesmo `--set lr=1e-4`. O log confirmou `retomando de .../ultimo.pt: validação 7 de 12`, e a validação 7 saiu no passo 1.912 (= 8 x 239), no mesmo run do MLflow, com o pico de VRAM de 6,05 GB de antes.
  - **Efeito no resultado:** nenhum esperado. O `ultimo.pt` guarda modelo, otimizador, scheduler, scaler, early stopping e histórico, e a amostragem de cada segmento depende só de (seed, segmento); o teste `test_retomar_da_o_mesmo_treino_que_nao_parar` confere, na CPU, que retomar dá o mesmo treino que não parar. Na GPU, a diferença possível é só a do não determinismo do cuDNN, a mesma entre duas execuções sem queda.
  - **Mudança no procedimento:** o restante da fase (retomada da 1 e as tentativas 2 a 6) roda desacoplado da sessão, com `setsid nohup` e log em `/mnt/SSD2/rsna_treino/logs/fase1.log`. Como fazer foi documentado no GUIA_DAVI (seções 9 e 10) e no TREINO_RSNA ("Rodar desacoplado do terminal").

-
