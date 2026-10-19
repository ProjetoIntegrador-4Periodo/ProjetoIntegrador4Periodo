# Etapa 2: PNG da Etapa 1 -> entrada do modelo

Transforma a anatomia e a máscara da Etapa 1 (`docs/PIPELINE_RSNA.md`) no
tensor que entra no ConvNeXt-Tiny. Roda em tempo de carregamento, sem gravar
nada em disco. Treino e inferência chamam a mesma função, `preparar_entrada`.

| Arquivo | O que faz |
|---------|-----------|
| `configs/etapa2.yaml` | Caminhos, canvas, representação, augmentation, workers |
| `src/data/rsna_entrada.py` | Config, canvas, augmentation e representação (sem torch) |
| `src/data/rsna_dataset.py` | `MamografiaRSNA`: Dataset PyTorch sobre o `manifest_processado.csv` |
| `scripts/rsna_etapa2_verificar.py` | Grades da entrada final e medição de tempos |
| `tests/test_rsna_entrada.py`, `tests/test_rsna_dataset.py` | Testes com imagens sintéticas |

## Uso

```bash
pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements-etapa2.txt
# em configs/etapa2.yaml, troque só a seção `caminhos`
python -m scripts.rsna_etapa2_verificar
```

```python
cfg = carregar_config("configs/etapa2.yaml")
treino = MamografiaRSNA(manifest[manifest.split == "train"], etapa1_dir, cfg.entrada,
                        replace(cfg.augmentation, ativa=True))
validacao = MamografiaRSNA(manifest[manifest.split == "val"], etapa1_dir, cfg.entrada)
torch.save({"modelo": modelo.state_dict(), "etapa2": cfg.para_checkpoint()}, caminho)
# na inferência: ConfigEtapa2.de_dict(checkpoint["etapa2"]).entrada
```

Use `worker_init_fn=iniciar_worker` no DataLoader.

## Decisões

- **Ordem.** Primeiro o canvas, depois a augmentation opcional e por fim a
  representação e a normalização. A augmentation trabalha no canvas 8 bits com
  a máscara, então o top-hat do realce vai ser calculado depois dela.
- **Canvas.** A anatomia é zerada fora da máscara antes de reduzir, para um
  marcador não vazar pela média do INTER_AREA, e de novo depois, com a máscara
  reduzida por vizinho mais próximo. A mama fica encostada à esquerda e
  centrada na vertical, e o padding é 0.
- **Preto depois da normalização.** O fundo vira `-média/desvio` em cada canal,
  que é o preto do ImageNet.
- **Spacing.** `spacing_canvas = spacing_resultante_mm / escala_do_canvas`,
  dividido também pela escala da augmentation. É esse valor que a representação
  recebe, e o top-hat de 0,6 mm deve usá-lo.
- **Augmentation.**
  - Implementada com OpenCV, sem dependência nova, e desligada por padrão.
  - Não tem flip horizontal.
  - A rotação e a escala giram em torno do meio da parede, em x = 0. Depois,
    a imagem é deslocada para o tecido voltar a encostar na borda esquerda,
    sem o vão que a rotação e a translação abrem.
  - Brilho e contraste mexem só no tecido. Tudo fora da máscara transformada
    volta a 0.
  - O sorteio usa o gerador do torch de cada worker, então `set_seed` deixa as
    épocas reproduzíveis.
- **Representação.**
  - O registro `REPRESENTACOES` liga cada nome a uma função
    `(canvas, máscara, spacing_mm, cfg) -> (H, W, 3) em [0, 1]`.
  - `realce_3canais` já está registrado, mas ainda levanta `NotImplementedError`.
- **Checkpoint.** `para_checkpoint()` salva só as seções `entrada` e
  `augmentation`; os caminhos ficam de fora.
