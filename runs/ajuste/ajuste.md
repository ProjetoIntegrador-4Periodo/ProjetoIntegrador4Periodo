| configuração | seeds | pf1 | f1_melhor | f1_5_melhor | f2_melhor | roc_auc | pr_auc | sens_esp90 | esp_sens90 | melhor val./treinadas | ainda melhorando |
|---|---|---|---|---|---|---|---|---|---|---|---|
| lr=2e-05 (cinza) | 2 (0,1) | 0.163 (Δ 0.013) | 0.371 (Δ 0.034) | 0.357 (Δ 0.027) | 0.382 (Δ 0.019) | 0.853 (Δ 0.013) | 0.293 (Δ 0.003) | 0.595 (Δ 0.071) | 0.566 (Δ 0.023) | 11/12,11/12 | sim,sim |
| lr=0.0001 (cinza) | 2 (0,1) | 0.118 (Δ 0.001) | 0.321 (Δ 0.025) | 0.339 (Δ 0.030) | 0.376 (Δ 0.019) | 0.829 (Δ 0.005) | 0.226 (Δ 0.033) | 0.577 (Δ 0.036) | 0.478 (Δ 0.011) | 9/12,11/12 | não,sim |
| lr=5e-05 (cinza) | 2 (0,1) | 0.128 (Δ 0.017) | 0.336 (Δ 0.057) | 0.334 (Δ 0.003) | 0.353 (Δ 0.011) | 0.843 (Δ 0.013) | 0.245 (Δ 0.046) | 0.583 (Δ 0.000) | 0.555 (Δ 0.048) | 8/12,11/12 | não,sim |

Métricas por mama na validação (fold 0), na melhor validação de cada tentativa: média entre seeds e, entre parênteses, a diferença entre a maior e a menor. Ordenado pela métrica de seleção. "ainda melhorando": a melhor validação foi uma das 2 últimas do orçamento, sem early stopping.

| tentativa | configuração | seed | pf1 | f1_melhor | f1_5_melhor | f2_melhor | roc_auc | pr_auc | sens_esp90 | esp_sens90 | melhor val. |
|---|---|---|---|---|---|---|---|---|---|---|---|
| convnext_tiny_cinza_tentativa1 | lr=0.0001 | 0 | 0.117 | 0.308 | 0.354 | 0.386 | 0.826 | 0.243 | 0.560 | 0.473 | 9/12 |
| convnext_tiny_cinza_tentativa2 | lr=0.0001 | 1 | 0.118 | 0.333 | 0.324 | 0.367 | 0.832 | 0.210 | 0.595 | 0.484 | 11/12 |
| convnext_tiny_cinza_tentativa3 | lr=5e-05 | 0 | 0.120 | 0.308 | 0.332 | 0.358 | 0.850 | 0.222 | 0.583 | 0.579 | 8/12 |
| convnext_tiny_cinza_tentativa4 | lr=5e-05 | 1 | 0.137 | 0.365 | 0.336 | 0.347 | 0.836 | 0.268 | 0.583 | 0.531 | 11/12 |
| convnext_tiny_cinza_tentativa5 | lr=2e-05 | 0 | 0.169 | 0.388 | 0.370 | 0.392 | 0.846 | 0.294 | 0.560 | 0.577 | 11/12 |
| convnext_tiny_cinza_tentativa6 | lr=2e-05 | 1 | 0.156 | 0.354 | 0.343 | 0.373 | 0.859 | 0.291 | 0.631 | 0.554 | 11/12 |
