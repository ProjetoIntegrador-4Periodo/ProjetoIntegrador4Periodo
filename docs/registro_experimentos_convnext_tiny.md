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
| 7 | 20/10 | 0 | lr=1e-5 | 11/12 | sim | 0.355 | 0.320 | 0.313 | 0.148 | 0.855 | 0.234 | 0.583 | 0.581 | Borda da faixa da fase 1 (o vencedor 2e-5 era o menor lr). Não superou o 2e-5: F2 0,355 abaixo da média 0,382 (e longe do limite de 0,401); PR AUC 0,234 contra 0,293 (-0,059, cerca de 20 vezes a diferença entre as seeds do 2e-5). ROC AUC (0,855 contra 0,853) e esp_sens90 (0,581 contra 0,566) empatados, dentro da variação entre seeds. Segue o 2e-5. |
| 10 | 20/10 | 0 | lr=2e-5, negativos_por_positivo=20, pos_weight=1 | | | | | | | | | | | Em andamento. Borda da faixa da fase 2 (o vencedor 1:10 era a maior proporção). Mesmas regras da fase 2 contra a seed 0 do 1:10 (F2 0,392, PR AUC 0,294). Com o mesmo orçamento de passos, cada positivo aparece cerca de 6,3 vezes em vez de 12 (cerca de 782 positivos; 15.640 negativos por época, dentro do pool de 22.000). |

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
