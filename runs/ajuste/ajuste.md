| configuração | seeds | pf1 | f1_melhor | f1_5_melhor | f2_melhor | roc_auc | pr_auc | sens_esp90 | esp_sens90 | melhor val./treinadas | ainda melhorando |
|---|---|---|---|---|---|---|---|---|---|---|---|
| lr=2e-05 (cinza) | 2 (0,1) | 0.163 (Δ 0.013) | 0.371 (Δ 0.034) | 0.357 (Δ 0.027) | 0.382 (Δ 0.019) | 0.853 (Δ 0.013) | 0.293 (Δ 0.003) | 0.595 (Δ 0.071) | 0.566 (Δ 0.023) | 11/12,11/12 | sim,sim |
| lr=0.0001 (cinza) | 2 (0,1) | 0.118 (Δ 0.001) | 0.321 (Δ 0.025) | 0.339 (Δ 0.030) | 0.376 (Δ 0.019) | 0.829 (Δ 0.005) | 0.226 (Δ 0.033) | 0.577 (Δ 0.036) | 0.478 (Δ 0.011) | 9/12,11/12 | não,sim |
| lr=2e-05, negativos_por_positivo=4 (cinza) | 1 (0) | 0.180 | 0.323 | 0.352 | 0.372 | 0.835 | 0.213 | 0.560 | 0.458 | 9/12 | não |
| ema=True, lr=2e-05 (cinza) | 1 (0) | 0.144 | 0.339 | 0.342 | 0.356 | 0.851 | 0.275 | 0.571 | 0.603 | 11/12 | sim |
| lr=1e-05 (cinza) | 1 (0) | 0.148 | 0.320 | 0.313 | 0.355 | 0.855 | 0.234 | 0.583 | 0.581 | 11/12 | sim |
| lr=5e-05 (cinza) | 2 (0,1) | 0.128 (Δ 0.017) | 0.336 (Δ 0.057) | 0.334 (Δ 0.003) | 0.353 (Δ 0.011) | 0.843 (Δ 0.013) | 0.245 (Δ 0.046) | 0.583 (Δ 0.000) | 0.555 (Δ 0.048) | 8/12,11/12 | não,sim |
| lr=2e-05, negativos_por_positivo=1, pos_weight=1 (cinza) | 1 (0) | 0.177 | 0.298 | 0.330 | 0.353 | 0.834 | 0.224 | 0.536 | 0.469 | 7/12 | não |
| lr=2e-05, negativos_por_positivo=20, pos_weight=1 (cinza) | 1 (0) | 0.152 | 0.309 | 0.321 | 0.344 | 0.849 | 0.228 | 0.619 | 0.577 | 9/12 | não |

Métricas por mama na validação (fold 0), na melhor validação de cada tentativa: média entre seeds e, entre parênteses, a diferença entre a maior e a menor. Ordenado pela métrica de seleção. "ainda melhorando": a melhor validação foi uma das 2 últimas do orçamento, sem early stopping.

| tentativa | configuração | seed | pf1 | f1_melhor | f1_5_melhor | f2_melhor | roc_auc | pr_auc | sens_esp90 | esp_sens90 | melhor val. |
|---|---|---|---|---|---|---|---|---|---|---|---|
| convnext_tiny_cinza_tentativa1 | lr=0.0001 | 0 | 0.117 | 0.308 | 0.354 | 0.386 | 0.826 | 0.243 | 0.560 | 0.473 | 9/12 |
| convnext_tiny_cinza_tentativa10 | lr=2e-05, negativos_por_positivo=20, pos_weight=1 | 0 | 0.152 | 0.309 | 0.321 | 0.344 | 0.849 | 0.228 | 0.619 | 0.577 | 9/12 |
| convnext_tiny_cinza_tentativa11 | ema=True, lr=2e-05 | 0 | 0.144 | 0.339 | 0.342 | 0.356 | 0.851 | 0.275 | 0.571 | 0.603 | 11/12 |
| convnext_tiny_cinza_tentativa2 | lr=0.0001 | 1 | 0.118 | 0.333 | 0.324 | 0.367 | 0.832 | 0.210 | 0.595 | 0.484 | 11/12 |
| convnext_tiny_cinza_tentativa3 | lr=5e-05 | 0 | 0.120 | 0.308 | 0.332 | 0.358 | 0.850 | 0.222 | 0.583 | 0.579 | 8/12 |
| convnext_tiny_cinza_tentativa4 | lr=5e-05 | 1 | 0.137 | 0.365 | 0.336 | 0.347 | 0.836 | 0.268 | 0.583 | 0.531 | 11/12 |
| convnext_tiny_cinza_tentativa5 | lr=2e-05 | 0 | 0.169 | 0.388 | 0.370 | 0.392 | 0.846 | 0.294 | 0.560 | 0.577 | 11/12 |
| convnext_tiny_cinza_tentativa6 | lr=2e-05 | 1 | 0.156 | 0.354 | 0.343 | 0.373 | 0.859 | 0.291 | 0.631 | 0.554 | 11/12 |
| convnext_tiny_cinza_tentativa7 | lr=1e-05 | 0 | 0.148 | 0.320 | 0.313 | 0.355 | 0.855 | 0.234 | 0.583 | 0.581 | 11/12 |
| convnext_tiny_cinza_tentativa8 | lr=2e-05, negativos_por_positivo=1, pos_weight=1 | 0 | 0.177 | 0.298 | 0.330 | 0.353 | 0.834 | 0.224 | 0.536 | 0.469 | 7/12 |
| convnext_tiny_cinza_tentativa9 | lr=2e-05, negativos_por_positivo=4 | 0 | 0.180 | 0.323 | 0.352 | 0.372 | 0.835 | 0.213 | 0.560 | 0.458 | 9/12 |
