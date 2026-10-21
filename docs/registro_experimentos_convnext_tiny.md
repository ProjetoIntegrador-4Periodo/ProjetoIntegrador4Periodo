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
| 1 | 19-20/10 | 0 | lr=1e-4 | 9/12 | não | 0.386 | 0.308 | 0.354 | 0.117 | 0.826 | 0.243 | 0.560 | 0.473 | caiu na validação 6 com a sessão; retomada pelo `ultimo.pt` (seção 5) |
| 2 | 20/10 | 1 | lr=1e-4 | 11/12 | sim | 0.367 | 0.333 | 0.324 | 0.118 | 0.832 | 0.210 | 0.595 | 0.484 | |
| 3 | 20/10 | 0 | lr=5e-5 | 8/12 | não | 0.358 | 0.308 | 0.332 | 0.120 | 0.850 | 0.222 | 0.583 | 0.579 | |
| 4 | 20/10 | 1 | lr=5e-5 | 11/12 | sim | 0.347 | 0.365 | 0.336 | 0.137 | 0.836 | 0.268 | 0.583 | 0.531 | |
| 5 | 20/10 | 0 | lr=2e-5 | 11/12 | sim | 0.392 | 0.388 | 0.370 | 0.169 | 0.846 | 0.294 | 0.560 | 0.577 | |
| 6 | 20/10 | 1 | lr=2e-5 | 11/12 | sim | 0.373 | 0.354 | 0.343 | 0.156 | 0.859 | 0.291 | 0.631 | 0.554 | |

"Melhor val." conta as validações a partir de 0: `11/12` é a última das 12 (passo 2.868), `9/12` é a décima (passo 2.390).

Tabela da fase (média e diferença entre seeds):

| configuração | seeds | pf1 | f1_melhor | f1_5_melhor | f2_melhor | roc_auc | pr_auc | sens_esp90 | esp_sens90 | melhor val./treinadas | ainda melhorando |
|---|---|---|---|---|---|---|---|---|---|---|---|
| lr=2e-05 (cinza) | 2 (0,1) | 0.163 (Δ 0.013) | 0.371 (Δ 0.034) | 0.357 (Δ 0.027) | 0.382 (Δ 0.019) | 0.853 (Δ 0.013) | 0.293 (Δ 0.003) | 0.595 (Δ 0.071) | 0.566 (Δ 0.023) | 11/12,11/12 | sim,sim |
| lr=0.0001 (cinza) | 2 (0,1) | 0.118 (Δ 0.001) | 0.321 (Δ 0.025) | 0.339 (Δ 0.030) | 0.376 (Δ 0.019) | 0.829 (Δ 0.005) | 0.226 (Δ 0.033) | 0.577 (Δ 0.036) | 0.478 (Δ 0.011) | 9/12,11/12 | não,sim |
| lr=5e-05 (cinza) | 2 (0,1) | 0.128 (Δ 0.017) | 0.336 (Δ 0.057) | 0.334 (Δ 0.003) | 0.353 (Δ 0.011) | 0.843 (Δ 0.013) | 0.245 (Δ 0.046) | 0.583 (Δ 0.000) | 0.555 (Δ 0.048) | 8/12,11/12 | não,sim |

Decisão (20/10, aprovada): **lr 2e-5**.
- **Motivo:** o 2e-5 venceu na média em todas as métricas. No F2 (0,382 contra 0,376 do 1e-4), a margem de 0,006 é menor que a diferença entre seeds (0,019), um empate. O maior ganho, e com a menor variação, foi no PR AUC: 0,293 (diferença entre seeds 0,003) contra 0,245 do 5e-5 e 0,226 do 1e-4, +0,067 sobre o 1e-4. Ganhou também em especificidade com sensibilidade de 90% (+0,088), pF1 (+0,045) e ROC AUC (+0,024) sobre o 1e-4.
- **Ainda melhorando:** sim, nas duas seeds, com a melhor validação na última do orçamento (passo 2.868). F2 nas 4 últimas validações: 0,380, 0,370, 0,384 e 0,392 (seed 0); 0,360, 0,363, 0,371 e 0,373 (seed 1). O ROC AUC ficou estável (0,846 a 0,861).
- **Borda da faixa:** como o vencedor é o menor lr testado, roda uma tentativa extra com lr 1e-5 (seed 0, mesmo orçamento) antes da fase 2, em "Outras tentativas". Regra combinada: o 1e-5 só substitui o 2e-5 se o F2 dele passar a média do 2e-5 por mais que a diferença entre as seeds do 2e-5 (0,382 + 0,019 = 0,401).

### Fase 2: proporção

Plano: `lr` 2e-5 (fase 1), com `negativos_por_positivo` 1 e 4, seed 0; o 1:10 vem da fase 1 (tentativas 5 e 6).

**`pos_weight`:** 1,0 nas três proporções. É o padrão da seção `convnext_tiny` (a amostragem já reequilibra as classes), conferido no `config_treino.yaml` das tentativas 5 e 6; o 1:1 passa `pos_weight=1` explícito e o 1:4 usa o padrão. Só a proporção de amostragem muda entre as linhas.

**Empate:** se a diferença de F2 entre a melhor proporção e a segunda for menor que a variação entre as seeds do 2e-5 na fase 1 (0,019), a seed 1 roda nas empatadas que ainda não a têm (o 1:10 já tem) antes de decidir. O 1:10 entra na comparação pela seed 0 (tentativa 5, F2 0,392), a mesma seed das novas (decidido em 20/10; a média das duas seeds, 0,382, aparece só como referência). **Desempate pelo PR AUC** (20/10): se as proporções empatarem no F2, mas uma ficar acima das outras no PR AUC por mais de 0,033, ela vence sem a seed 1. O 0,033 é a mediana das variações de PR AUC entre seeds da fase 1 (0,003, 0,033 e 0,048); com 2 seeds, o 0,003 do 2e-5 sozinho subestimaria o ruído.

| Nº | Data | Seed | Proporção | pos_weight | Alterado | Melhor val. | Ainda melhorando | f2_melhor | f1_melhor | f1_5_melhor | pf1 | roc_auc | pr_auc | sens_esp90 | esp_sens90 | Observações |
|----|------|------|-----------|------------|----------|-------------|------------------|-----------|-----------|-------------|-----|---------|--------|------------|------------|-------------|
| 5 | 20/10 | 0 | 1:10 | 1,0 | lr=2e-5 | 11/12 | sim | 0.392 | 0.388 | 0.370 | 0.169 | 0.846 | 0.294 | 0.560 | 0.577 | da fase 1 |
| 6 | 20/10 | 1 | 1:10 | 1,0 | lr=2e-5 | 11/12 | sim | 0.373 | 0.354 | 0.343 | 0.156 | 0.859 | 0.291 | 0.631 | 0.554 | da fase 1 |
| 8 | 20/10 | 0 | 1:1 | 1,0 | lr=2e-5, negativos_por_positivo=1, pos_weight=1 | 7/12 | não | 0.353 | 0.298 | 0.330 | 0.177 | 0.834 | 0.224 | 0.536 | 0.469 | F2 oscilando entre 0,32 e 0,35 da validação 5 em diante, com a loss caindo de 0,29 para 0,09 |
| 9 | 20/10 | 0 | 1:4 | 1,0 | lr=2e-5, negativos_por_positivo=4 | 9/12 | não | 0.372 | 0.323 | 0.352 | 0.180 | 0.835 | 0.213 | 0.560 | 0.458 | F2 0,359 e 0,357 nas duas últimas validações |

Tabela da fase (seed 0; o 1:10 também com a média e a diferença entre seeds):

| Proporção | seeds | pf1 | f1_melhor | f1_5_melhor | f2_melhor | roc_auc | pr_auc | sens_esp90 | esp_sens90 | melhor val./treinadas | ainda melhorando |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1:10 (seed 0) | 0 | 0.169 | 0.388 | 0.370 | 0.392 | 0.846 | 0.294 | 0.560 | 0.577 | 11/12 | sim |
| 1:10 (média) | 2 (0,1) | 0.163 (Δ 0.013) | 0.371 (Δ 0.034) | 0.357 (Δ 0.027) | 0.382 (Δ 0.019) | 0.853 (Δ 0.013) | 0.293 (Δ 0.003) | 0.595 (Δ 0.071) | 0.566 (Δ 0.023) | 11/12,11/12 | sim,sim |
| 1:4 | 0 | 0.180 | 0.323 | 0.352 | 0.372 | 0.835 | 0.213 | 0.560 | 0.458 | 9/12 | não |
| 1:1 | 0 | 0.177 | 0.298 | 0.330 | 0.353 | 0.834 | 0.224 | 0.536 | 0.469 | 7/12 | não |

Decisão (20/10, aprovada): **1:10** (`negativos_por_positivo` 10, `pos_weight` 1,0), sem seed 1 nas outras.
- **F2:** 1:10 0,3919 contra 0,3723 do 1:4, diferença de 0,0196, acima do limite de empate (0,0194, a diferença entre as seeds do 2e-5). A margem sobre o limite é só 0,0002, então o F2 sozinho não seria conclusivo.
- **PR AUC:** mesmo tratado como empate, o 1:10 vence pelo desempate: 0,294 contra 0,213 do 1:4 e 0,224 do 1:1, +0,070 a +0,081, acima do limite de 0,033. Também vence em esp_sens90 (0,577 contra 0,458 e 0,469), F1 e F1,5.
- **pF1:** foi a única métrica que favoreceu levemente as proporções balanceadas (0,180 no 1:4 e 0,177 no 1:1 contra 0,169).
- **Ainda melhorando:** o 1:10 sim, nas duas seeds (melhor validação na última). O 1:4 e o 1:1 não: o melhor veio antes do fim (validações 9 e 7).
- **Achado:** mais negativos generalizaram melhor que o balanceamento forte (1:1 a 1:4) usado nas referências da competição. Hipótese: sobreajuste nos positivos. Com o mesmo orçamento de passos, cada positivo foi visto 66 vezes no 1:1, 26,4 no 1:4 e 12 no 1:10 (`epocas_amostragem` no `historico.csv`), o que combina com a melhor validação mais cedo no 1:1 e no 1:4. A loss de treino não serve de evidência aqui, porque muda com a mistura de classes. A hipótese não foi testada diretamente.
- **Borda da faixa:** como o 1:10 é a maior proporção testada e a tendência foi de melhora com mais negativos, roda uma tentativa extra com 1:20 (seed 0, `pos_weight` 1, mesmo orçamento) antes da fase 3, em "Outras tentativas", com as mesmas regras comparando com a seed 0 do 1:10.

### Fase 3: EMA

Plano: a melhor combinação até aqui (lr 2e-5, 1:10, `pos_weight` 1,0) com `ema=true` (decaimento 0,999), seed 0 (tentativa 11, iniciada em 20/10 às 19:27). Compara com a seed 0 do 1:10 sem EMA (tentativa 5, F2 0,392, PR AUC 0,294) pelas mesmas regras da fase 2: diferença de F2 acima de 0,019 decide; no empate, PR AUC acima por mais de 0,033 decide; senão, seed 1 antes de decidir.

| Nº | Data | Seed | Alterado | Melhor val. | Ainda melhorando | f2_melhor | f1_melhor | f1_5_melhor | pf1 | roc_auc | pr_auc | sens_esp90 | esp_sens90 | Observações |
|----|------|------|----------|-------------|------------------|-----------|-----------|-------------|-----|---------|--------|------------|------------|-------------|
| 5 | 20/10 | 0 | lr=2e-5 (sem EMA) | 11/12 | sim | 0.392 | 0.388 | 0.370 | 0.169 | 0.846 | 0.294 | 0.560 | 0.577 | da fase 1, referência |
| 11 | 20/10 | 0 | lr=2e-5, ema=true (0,999) | 11/12 | sim | 0.356 | 0.339 | 0.342 | 0.145 | 0.851 | 0.275 | 0.571 | 0.603 | F2 subiu em todas as 12 validações, sem queda: 0,126, 0,135, 0,141, 0,168, 0,215, 0,260, 0,285, 0,305, 0,313, 0,331, 0,349 e 0,356 |

Tabela da fase (seed 0; a referência para o ruído é a diferença entre as seeds do 2e-5 sem EMA):

| configuração | seeds | pf1 | f1_melhor | f1_5_melhor | f2_melhor | roc_auc | pr_auc | sens_esp90 | esp_sens90 | melhor val./treinadas | ainda melhorando |
|---|---|---|---|---|---|---|---|---|---|---|---|
| sem EMA | 0 | 0.169 | 0.388 | 0.370 | 0.392 | 0.846 | 0.294 | 0.560 | 0.577 | 11/12 | sim |
| EMA 0,999 | 0 | 0.145 | 0.339 | 0.342 | 0.356 | 0.851 | 0.275 | 0.571 | 0.603 | 11/12 | sim |
| diferença entre seeds (2e-5 sem EMA) | | 0.013 | 0.034 | 0.027 | 0.019 | 0.013 | 0.003 | 0.071 | 0.023 | | |

Decisão (20/10, aprovada): **sem EMA**.
- **F2:** 0,356 contra 0,392 (-0,036), quase o dobro do limite de empate (0,019); não precisou do desempate nem da seed 1. O PR AUC também ficou abaixo (0,275 contra 0,294).
- **Onde o EMA ganhou:** esp_sens90 0,603 contra 0,577 (+0,026, pouco acima da diferença entre seeds, 0,023) e ROC AUC 0,851 contra 0,846 (dentro do ruído).
- **Ainda melhorando:** sim, com a curva mais estável do ajuste (subiu em todas as validações), enquanto a sem EMA oscilou entre 0,350 e 0,392 nas 7 últimas. Com decaimento 0,999, a média cobre os últimos cerca de 1.000 passos, um terço do orçamento, numa fase em que o modelo ainda melhorava; por isso chega atrasada. Ressalva na seção 6.
- **Extra (tentativa 12, EMA 0,995):** perdeu só no F2 (0,365 contra 0,392) e empatou ou ficou levemente à frente nas outras métricas, dentro da variação entre seeds, com o maior PR AUC do ajuste (0,300). Empate técnico decidido pelo F2: o treino final fica sem EMA. Detalhes em "Outras tentativas".

### Outras tentativas (fora do plano)

Qualquer tentativa extra, com o motivo.

| Nº | Data | Seed | Alterado | Melhor val. | Ainda melhorando | f2_melhor | f1_melhor | f1_5_melhor | pf1 | roc_auc | pr_auc | sens_esp90 | esp_sens90 | Motivo e observações |
|----|------|------|----------|-------------|------------------|-----------|-----------|-------------|-----|---------|--------|------------|------------|----------------------|
| 7 | 20/10 | 0 | lr=1e-5 | 11/12 | sim | 0.355 | 0.320 | 0.313 | 0.148 | 0.855 | 0.234 | 0.583 | 0.581 | Borda da faixa da fase 1 (o vencedor 2e-5 era o menor lr). Não superou o 2e-5: F2 0,355 abaixo da média 0,382 (e longe do limite de 0,401); PR AUC 0,234 contra 0,293 (-0,059, cerca de 20 vezes a diferença entre as seeds do 2e-5). ROC AUC (0,855 contra 0,853) e esp_sens90 (0,581 contra 0,566) empatados, dentro da variação entre seeds. Segue o 2e-5. |
| 10 | 20/10 | 0 | lr=2e-5, negativos_por_positivo=20, pos_weight=1 | 9/12 | não | 0.344 | 0.309 | 0.321 | 0.152 | 0.849 | 0.228 | 0.619 | 0.577 | Borda da faixa da fase 2 (o vencedor 1:10 era a maior proporção). Mesmas regras da fase 2 contra a seed 0 do 1:10: F2 0,344 contra 0,392 (-0,048, mais que o dobro do limite de empate de 0,019) e PR AUC 0,228 contra 0,294 (-0,066). ROC AUC (0,849 contra 0,846) e esp_sens90 (0,577 nos dois) empatados; sens_esp90 0,619 contra 0,560, dentro da variação entre seeds dessa métrica (0,071). Curva lenta e instável: F2 0,118 na validação 3, só passou de 0,30 na 6; ROC AUC e esp_sens90 ainda subiam no fim. Cada positivo foi visto 6,3 vezes (contra 12 no 1:10). **Segue o 1:10.** Ressalva na seção 6, "Para explorar no futuro". |
| 12 | 20/10 | 0 | lr=2e-5, ema=true, ema_decaimento=0.995 | 11/12 | sim | 0.365 | 0.378 | 0.362 | 0.165 | 0.860 | 0.300 | 0.607 | 0.568 | A EMA 0,999 perdeu na fase 3 por chegar atrasada; com 0,995 a janela cai de cerca de 1.000 para cerca de 200 passos. Contra a seed 0 sem EMA: **perdeu só no F2** (0,365 contra 0,392, -0,027, acima do limite de empate de 0,019). Nas outras métricas ficou empatada ou levemente à frente, todas dentro da variação entre seeds do 2e-5: ROC AUC 0,860 contra 0,846 (+0,014; variação 0,013), PR AUC 0,300 contra 0,294 (+0,006, o maior PR AUC do ajuste), sens_esp90 0,607 contra 0,560 (+0,047; variação 0,071), F1 0,378 contra 0,388, F1,5 0,362 contra 0,370, pF1 0,165 contra 0,169, esp_sens90 0,568 contra 0,577. Ainda melhorava no fim (melhor validação na última; F2 0,350, 0,360, 0,360 e 0,365 nas 4 últimas). **Empate técnico decidido pelo F2**, a métrica de seleção combinada: pela regra, perdeu, e o treino final fica sem EMA. Ressalva na seção 6. |

## 3. Configuração final escolhida

Seção `convnext_tiny` do `configs/treino.yaml` usada no treino final (a mesma para as duas representações):

```yaml
convnext_tiny:
  timm_nome: convnext_tiny.in12k_ft_in1k
  batch_real: 12
  acumulacao: 3                # batch efetivo = 36
  otimizador: adamw
  lr: 2.0e-5                   # fase 1
  weight_decay: 0.05
  drop_path: 0.1
  warmup_epocas: 1
  scheduler: cosine
  lr_minimo: 1.0e-6
  pos_weight: 1.0
  negativos_por_positivo: 10   # fase 2
  peso_negativo_dificil: 2.0
  ema: false                   # fase 3 e tentativa 12
  ema_decaimento: 0.999
```

Treino: até 30 épocas de referência (cerca de 7.170 passos), paciência 5, seleção por `f2_melhor`.

Motivo da escolha (21/10, aprovada): tentativa 5 (seed 0; seed 1 na tentativa 6).
- **lr 2e-5 (fase 1):** venceu na média de 2 seeds em todas as métricas; no F2, empate com o 1e-4 (0,382 contra 0,376, dentro da diferença entre seeds de 0,019), decidido pelo PR AUC (0,293 contra 0,226, com variação de 0,003). O 1e-5 (tentativa 7) ficou abaixo.
- **1:10 (fase 2):** à frente do 1:4 por 0,0196 de F2 (limite 0,0194) e por 0,081 de PR AUC; do 1:1 e do 1:20 (tentativa 10) com folga.
- **Sem EMA (fase 3):** a EMA 0,999 perdeu por 0,036 de F2. A EMA 0,995 (tentativa 12) perdeu por 0,027 de F2, mas empatou nas outras métricas: empate técnico decidido pelo F2.
- **Orçamento:** o vencedor ainda melhorava no fim das 12 épocas de referência do ajuste, nas duas seeds; o treino final usa até 30.

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

## 6. Para explorar no futuro

Perguntas que o ajuste levantou e não respondeu, por falta de tempo de GPU. Nenhuma muda as decisões já tomadas.

- **1:20 contra 1:10: proporção ou exposição aos positivos?** (20/10, tentativa 10). Com o mesmo orçamento de passos (2.868), o 1:20 viu cada positivo 6,3 vezes, contra 12 no 1:10. Então não dá para separar o efeito da proporção do efeito de menos exposição aos positivos. A curva do 1:20 foi lenta e o ROC AUC ainda subia no fim, o que deixa as duas explicações abertas. Com mais tempo de GPU, valeria testar o 1:20 com um orçamento maior, ou com o mesmo número de exposições por positivo do 1:10 (cerca de 5.475 passos, 1,9 vez o orçamento), para responder se a proporção alta é pior de fato ou só aprende mais devagar. **Isso não muda a decisão atual:** com o orçamento do ajuste, o 1:10 é o vencedor.
- **EMA com orçamento maior** (20/10, tentativa 11). Com decaimento 0,999, a média dos pesos cobre os últimos cerca de 1.000 passos, um terço do orçamento do ajuste (2.868), num trecho em que o modelo ainda melhorava; a média fica atrás do modelo atual. A curva com EMA subiu em todas as validações e ainda subia no fim, então não dá para separar "EMA é pior" de "EMA chega mais tarde". Com mais tempo de GPU, valeria testar a EMA 0,999 com o orçamento do treino final (até 30 épocas de referência, cerca de 7.170 passos), onde o atraso pesa menos. **Isso não muda a decisão da fase 3:** com o orçamento do ajuste, sem EMA venceu.
- **EMA 0,995 no treino final** (21/10, tentativa 12). Perdeu só no F2 (0,365 contra 0,392) e ficou empatada ou levemente à frente nas outras métricas, todas dentro da variação entre seeds, com o maior PR AUC do ajuste (0,300), e ainda melhorava no fim. Foi um empate técnico decidido pelo F2, a métrica de seleção. Com mais tempo de GPU, valeria rodar a EMA 0,995 no orçamento do treino final, com as mesmas seeds, para ver se a vantagem do F2 sem EMA se mantém e se o ganho em ROC AUC e PR AUC se confirma. **Isso não muda a decisão:** pela regra combinada, o treino final fica sem EMA.

