# Etapa 2: PNG da Etapa 1 -> entrada do modelo

Transforma a anatomia e a máscara da Etapa 1 (`docs/PIPELINE_RSNA.md`) no
tensor que entra no ConvNeXt-Tiny. Treino, geração da pasta de realce e
inferência chamam as mesmas funções (`representar` e `preparar_entrada`).

| Arquivo | O que faz |
|---------|-----------|
| `configs/etapa2.yaml` | Canvas, representação, realce, augmentation, workers |
| `configs/local.yaml` | Caminhos desta máquina, seção `etapa2` (fora do git; modelo em `local.exemplo.yaml`) |
| `src/data/rsna_entrada.py` | Config, representação, canvas, augmentation e normalização (sem torch) |
| `src/data/rsna_dataset.py` | `MamografiaRSNA`: Dataset PyTorch; `da_config` escolhe pasta e representação |
| `scripts/rsna_realce_gerar.py` | Gera a pasta de realce (amostra com grade, equivalência e tempos) |
| `scripts/rsna_etapa2_verificar.py` | Grades da entrada final e medição de tempos |
| `tests/test_rsna_entrada.py`, `test_rsna_realce.py`, `test_rsna_dataset.py` | Testes sintéticos |

## Uso

```bash
pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements-etapa2.txt
cp configs/local.exemplo.yaml configs/local.yaml   # e preencha a seção etapa2
python -m scripts.rsna_realce_gerar --amostra   # 50 imagens, grade em <realce_dir>/amostra
python -m scripts.rsna_realce_gerar             # pasta completa (retomável)
python -m scripts.verificar_checksums /mnt/SSD2/rsna_fold0_realce
```

```python
cfg = carregar_config("configs/etapa2.yaml")   # entrada.representacao: cinza | realce_3canais
treino = MamografiaRSNA.da_config(cfg, manifest[manifest.split == "train"], treino=True)
validacao = MamografiaRSNA.da_config(cfg, manifest[manifest.split == "val"])
torch.save({"modelo": modelo.state_dict(), "etapa2": cfg.para_checkpoint()}, caminho)
# na inferência: ConfigEtapa2.de_dict(checkpoint["etapa2"]).entrada
```

Use `worker_init_fn=iniciar_worker` no DataLoader. Os dois treinos (cinza e
realce) diferem só em `entrada.representacao`: canvas e augmentation são os
mesmos.

## Decisões

- **Ordem.** Representação no tamanho da Etapa 1 (lado maior 1536), depois o
  canvas, a augmentation opcional e a normalização. Assim o top-hat em mm usa o
  `spacing_resultante_mm` do manifest direto.
- **Realce (`realce_3canais`).** Fundo zerado fora da máscara nos três canais.
  - R: a anatomia da Etapa 1.
  - G: mediana 3x3, top-hat branco com disco de 0,6 mm de diâmetro (lado ímpar,
    mínimo 3 px). Normalização: mediana do tecido -> 0, percentil 99,9 do tecido
    -> 255 (o top-hat é ruído baixo com picos raros; ver o Davi). Na borda da
    imagem, a extensão é por reflexão ímpar: o tecido que clareia em direção à
    parede (peitoral na CC) é uma rampa, e repetindo o último pixel o top-hat
    deixava uma linha de 1 px no G ao longo da parede.
  - B: CLAHE (`clahe_clip`, `clahe_tiles` na config).
  - Antes do top-hat e do CLAHE o fundo recebe o valor do tecido mais próximo,
    para a pele não virar aresta. O Davi usava a mediana do tecido, que ainda
    deixa um degrau na pele.
- **Pasta de realce.** `realce/<image_id>.png` em RGB, compressão 9 (bytes
  reproduzíveis), `manifest_realce.csv`, `config_usada.yaml` (parâmetros, commit
  e o hash do `SHA256SUMS` da Etapa 1) e `SHA256SUMS`. O Dataset recusa a pasta
  se os parâmetros de realce forem outros. Calcular na hora dá o mesmo resultado
  que o PNG, byte a byte.
- **Canvas.** Zerada fora da máscara antes de reduzir, para um marcador não vazar
  pela média do INTER_AREA, e de novo depois, com a máscara reduzida por vizinho
  mais próximo. A mama fica encostada à esquerda e centrada na vertical, e o
  padding é 0. No realce, o G passa antes por um filtro de máximo de lado
  `ceil(2 / escala)` (ímpar): esse bloco sempre cobre um pixel de saída inteiro,
  então um ponto de 1 px chega ao canvas com o valor cheio. R e B usam só
  INTER_AREA.
- **Preto depois da normalização.** O fundo vira `-média/desvio` em cada canal,
  que é o preto do ImageNet.
- **Augmentation.**
  - Implementada com OpenCV, sem dependência nova, e desligada por padrão.
  - Geometria igual em todos os canais. Não tem flip horizontal.
  - A rotação e a escala giram em torno do meio da parede, em x = 0. Depois,
    a imagem é deslocada para o tecido voltar a encostar na borda esquerda,
    sem o vão que a rotação e a translação abrem.
  - Brilho e contraste só nos `canais_fotometricos` (padrão: só o R) e só no
    tecido. Tudo fora da máscara transformada volta a 0.
  - O sorteio usa o gerador do torch de cada worker, então `set_seed` deixa as
    épocas reproduzíveis.
- **Representação nova.** Registre em `REPRESENTACOES` uma função
  `(anatomia, máscara, spacing_mm, cfg) -> (H, W, C) uint8` e os canais de pico.
- **Checkpoint.** `para_checkpoint()` salva `entrada` (com os parâmetros do
  realce) e `augmentation`; os caminhos ficam de fora.
