# Registro de experimentos: <MODELO>

Preencha durante o trabalho, não só no fim. Commite este arquivo junto com os
resultados de `runs/` (ver `docs/GUIA_DAVI.md`, passo 11). Os números vêm de
`runs/.../metricas_finais.json`, da tabela do `python -m scripts.rsna_comparar --ajuste`
ou do MLflow; copie-os, não arredonde de cabeça.

- Responsável:
- Período:

**Regras comuns aos dois modelos** (`configs/treino.yaml`, não mudam sem combinar):
- **Métrica de seleção:** F2 por mama (`f2_melhor`), porque um falso negativo é mais grave que um falso positivo. Ela decide o early stopping, o `melhor.pt` e a escolha entre tentativas.
- **Orçamento por passos:** cada tentativa de ajuste tem 12 épocas de referência (1 época de referência = os passos de uma época em 1:10; 239 passos com batch efetivo 36), com validação a cada época de referência e paciência de 5 validações. O treino final vai até 30 épocas de referência.
- **Só a validação do fold 0.** O teste hold-out não é usado aqui.

## 1. Máquina

Saída de `nvidia-smi` e do `runs/<qualquer run>/maquina.json`.

| Item | Valor |
|------|-------|
| GPU | |
| VRAM (GB) | |
| Driver NVIDIA | |
| CUDA (torch.version.cuda) | |
| Python | |
| torch | |
| timm | |
| Sistema | |
| Maior `batch_real` que cabe (`rsna_treino_verificar maior-batch`) | |
| `batch_real` x `acumulacao` usados (batch efetivo) | |
| Passos por época de referência (`passos_por_validacao` no `metricas_finais.json`) | |
| Smoke: imagens/s no treino e min por época de referência estimados | |
| Sanidade: épocas até loss < 0,02 | |

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

Plano: `lr` A, B e C (os três valores do seu plano), proporção 1:10, EMA desligada, seeds 0 e 1 (6 tentativas).

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
<modelo>:
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

-
