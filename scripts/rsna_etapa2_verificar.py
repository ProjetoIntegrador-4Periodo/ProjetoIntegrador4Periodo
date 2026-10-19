"""Verificação da Etapa 2: grades da entrada final do modelo e tempos de carregamento.

Não treina nada nem grava nada na pasta da Etapa 1. Rode a partir da raiz:

    python -m scripts.rsna_etapa2_verificar                # grades + tempos
    python -m scripts.rsna_etapa2_verificar --sem-tempos   # só as grades

Saídas em ``caminhos.verificacao_dir``: grade_*.png e tempos.txt.

As grades mostram o tensor que entra no modelo, desnormalizado (canal 0), no
tamanho do canvas e com uma moldura cinza marcando a borda do canvas.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from src.data.rsna_dataset import MamografiaRSNA, iniciar_worker  # noqa: E402
from src.data.rsna_entrada import (  # noqa: E402
    ConfigEtapa2,
    carregar_config,
    ler_pngs,
    preparar_entrada,
)
from src.utils.paths import CONFIGS_DIR  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402

SEMENTE = 42
COLUNAS_GRADE = 10


def para_exibir(tensor: torch.Tensor, cfg: ConfigEtapa2) -> np.ndarray:
    """Desfaz a normalização do canal 0 e volta para uint8, com moldura cinza."""
    canal = tensor[0].numpy() * cfg.entrada.desvio[0] + cfg.entrada.media[0]
    imagem = np.clip(canal * 255, 0, 255).astype(np.uint8)
    return cv2.copyMakeBorder(imagem, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=110)


def grade(itens: list[tuple[np.ndarray, str]], titulo: str, destino: Path) -> None:
    linhas = int(np.ceil(len(itens) / COLUNAS_GRADE))
    fig, eixos = plt.subplots(
        linhas, COLUNAS_GRADE, figsize=(COLUNAS_GRADE * 2.1, linhas * 3.8), squeeze=False
    )
    for ax in eixos.ravel():
        ax.axis("off")
    for ax, (imagem, legenda) in zip(eixos.ravel(), itens, strict=False):
        ax.imshow(imagem, cmap="gray", vmin=0, vmax=255)
        ax.set_title(legenda, fontsize=6.5)
    fig.suptitle(titulo, fontsize=11)
    fig.tight_layout()
    fig.savefig(destino, dpi=120)
    plt.close(fig)
    print(destino)


def legenda(linha: pd.Series) -> str:
    rotulo = "POSITIVA" if linha.cancer else "negativa"
    return (
        f"{linha.image_id} | {linha.split}\n"
        f"maq {linha.machine_id} {linha.laterality} {linha.view} | {rotulo}"
    )


def gerar_grades(manifest: pd.DataFrame, cfg: ConfigEtapa2, etapa1: Path, saida: Path) -> None:
    sem_aug = MamografiaRSNA(manifest, etapa1, cfg.entrada)
    posicao = {image_id: i for i, image_id in enumerate(manifest.image_id)}

    def itens(selecao: pd.DataFrame) -> list[tuple[np.ndarray, str]]:
        return [
            (para_exibir(sem_aug[posicao[r.image_id]][0], cfg), legenda(r))
            for _, r in selecao.iterrows()
        ]

    suspeitas = pd.read_csv(etapa1 / "suspeitas.csv")
    s210 = manifest[manifest.image_id.isin(suspeitas[suspeitas.machine_id == 210].image_id)]
    grade(
        itens(s210), f"Suspeitas do aparelho 210 ({len(s210)})", saida / "grade_1_suspeitas_210.png"
    )

    treino = manifest[manifest.split == "train"].sample(20, random_state=SEMENTE)
    grade(itens(treino), "20 imagens aleatórias do treino", saida / "grade_2_treino.png")

    positivas = manifest[manifest.cancer == 1].sample(10, random_state=SEMENTE)
    grade(itens(positivas), "10 positivas (todos os splits)", saida / "grade_3_positivas.png")

    aug = replace(cfg.augmentation, ativa=True)
    com_aug = MamografiaRSNA(manifest, etapa1, cfg.entrada, aug)
    torch.manual_seed(SEMENTE)
    pares = []
    for _, r in manifest[manifest.split == "train"].sample(10, random_state=SEMENTE + 1).iterrows():
        i = posicao[r.image_id]
        pares.append((para_exibir(sem_aug[i][0], cfg), f"{r.image_id}\nsem augmentation"))
        pares.append((para_exibir(com_aug[i][0], cfg), f"{r.image_id}\ncom augmentation"))
    grade(pares, "10 imagens do treino: sem | com augmentation", saida / "grade_4_augmentation.png")


def medir_por_imagem(manifest: pd.DataFrame, cfg: ConfigEtapa2, etapa1: Path, n: int) -> str:
    """Um processo só: quanto da conta é ler o PNG e quanto é preparar a entrada."""
    cv2.setNumThreads(1)
    amostra = manifest.sample(n, random_state=SEMENTE + 2)
    leitura, preparo = [], []
    for _, r in amostra.iterrows():
        t0 = time.perf_counter()
        anatomia, mascara = ler_pngs(etapa1, r.png_anatomia, r.png_mascara)
        t1 = time.perf_counter()
        preparar_entrada(anatomia, mascara, r.spacing_resultante_mm, cfg.entrada)
        preparo.append(time.perf_counter() - t1)
        leitura.append(t1 - t0)
    return (
        f"1 processo, {n} imagens: leitura dos 2 PNGs {np.mean(leitura) * 1000:.1f} ms, "
        f"preparar_entrada {np.mean(preparo) * 1000:.1f} ms, "
        f"total {(np.mean(leitura) + np.mean(preparo)) * 1000:.1f} ms por imagem"
    )


def medir_dataloader(dataset: MamografiaRSNA, workers: int, batch: int, lotes: int) -> str:
    """Vazão com shuffle; o 1º lote (subida dos workers) fica fora da conta."""
    loader = DataLoader(
        dataset,
        batch_size=batch,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        worker_init_fn=iniciar_worker,
        persistent_workers=False,
        prefetch_factor=4,
    )
    iterador = iter(loader)
    inicio = time.perf_counter()
    next(iterador)
    primeiro = time.perf_counter() - inicio
    inicio = time.perf_counter()
    for _ in range(lotes):
        next(iterador)
    duracao = time.perf_counter() - inicio
    del iterador
    imagens = lotes * batch
    return (
        f"{imagens / duracao:.0f} imagens/s ({duracao / imagens * 1000:.1f} ms por imagem; "
        f"1º lote {primeiro:.1f} s)"
    )


def medir_gpu(config: Path, batch: int) -> str:
    """
    Passo de treino da ConvNeXt-Tiny; se o batch não couber na GPU, tenta a metade.

    Cada tentativa roda num processo novo: depois de um estouro de memória, o
    mesmo processo não devolve a GPU inteira e a tentativa menor falharia também.
    """
    if not torch.cuda.is_available():
        return "GPU indisponível"
    while batch >= 1:
        comando = [sys.executable, "-m", "scripts.rsna_etapa2_verificar", "--config", str(config)]
        resultado = subprocess.run(
            [*comando, "--so-gpu", str(batch)], capture_output=True, text=True, check=False
        )
        if resultado.returncode == 0:
            return resultado.stdout.strip().splitlines()[-1]
        if "out of memory" not in resultado.stderr and "ALLOC_FAILED" not in resultado.stderr:
            raise RuntimeError(resultado.stderr[-2000:])
        print(f"batch {batch} não coube na GPU; tentando {batch // 2}", flush=True)
        batch //= 2
    return "nem batch 1 coube na GPU"


def passo_de_treino(cfg: ConfigEtapa2, batch: int, passos: int = 20) -> str:
    """Passo de treino de uma ConvNeXt-Tiny aleatória com AMP: quanto a GPU consome."""
    from torchvision.models import convnext_tiny

    modelo = convnext_tiny(num_classes=1).cuda().to(memory_format=torch.channels_last)
    otimizador = torch.optim.AdamW(modelo.parameters(), lr=1e-4)
    x = torch.randn(batch, 3, cfg.entrada.altura, cfg.entrada.largura, device="cuda")
    x = x.to(memory_format=torch.channels_last)
    y = torch.randint(0, 2, (batch, 1), device="cuda").float()
    torch.cuda.reset_peak_memory_stats()

    def passo() -> None:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            perda = torch.nn.functional.binary_cross_entropy_with_logits(modelo(x), y)
        otimizador.zero_grad(set_to_none=True)
        perda.backward()
        otimizador.step()

    for _ in range(3):
        passo()
    torch.cuda.synchronize()
    inicio = time.perf_counter()
    for _ in range(passos):
        passo()
    torch.cuda.synchronize()
    duracao = time.perf_counter() - inicio
    pico = torch.cuda.max_memory_allocated() / 2**30
    return (
        f"ConvNeXt-Tiny, batch {batch}, bf16, channels_last: {passos * batch / duracao:.0f} "
        f"imagens/s ({torch.cuda.get_device_name(0)}, pico {pico:.1f} GB)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=CONFIGS_DIR / "etapa2.yaml")
    parser.add_argument("--sem-tempos", action="store_true", help="gera só as grades")
    parser.add_argument("--lotes", type=int, default=150, help="lotes medidos no DataLoader")
    parser.add_argument("--so-gpu", type=int, metavar="BATCH", help=argparse.SUPPRESS)
    args = parser.parse_args()

    set_seed(SEMENTE, deterministic=False)
    cfg = carregar_config(args.config)
    if args.so_gpu:
        print(passo_de_treino(cfg, args.so_gpu))
        return
    etapa1 = Path(cfg.caminhos["etapa1_dir"])
    saida = Path(cfg.caminhos["verificacao_dir"])
    saida.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(etapa1 / "manifest_processado.csv")

    gerar_grades(manifest, cfg, etapa1, saida)
    if args.sem_tempos:
        return

    workers = int(cfg.carregamento.get("workers", 8))
    batch = int(cfg.carregamento.get("batch_size", 16))
    treino = manifest[manifest.split == "train"]
    # Recortes disjuntos: a 2ª medição não aproveita o cache de disco da 1ª.
    metade = treino.sample(frac=1, random_state=SEMENTE + 3)
    n = args.lotes * batch + batch
    a, b = metade.iloc[:n], metade.iloc[n : 2 * n]
    aug = replace(cfg.augmentation, ativa=True)

    medicoes = [
        lambda: f"canvas {cfg.entrada.altura}x{cfg.entrada.largura}, "
        f"representação {cfg.entrada.representacao}",
        lambda: medir_por_imagem(manifest, cfg, etapa1, 200),
        lambda: f"DataLoader {workers} workers, batch {batch}, sem augmentation: "
        + medir_dataloader(MamografiaRSNA(a, etapa1, cfg.entrada), workers, batch, args.lotes),
        lambda: f"DataLoader {workers} workers, batch {batch}, com augmentation: "
        + medir_dataloader(MamografiaRSNA(b, etapa1, cfg.entrada, aug), workers, batch, args.lotes),
        lambda: medir_gpu(args.config, batch),
    ]
    with open(saida / "tempos.txt", "w", encoding="utf-8") as arquivo:
        for medir in medicoes:
            linha = medir()
            print(linha, flush=True)
            arquivo.write(linha + "\n")


if __name__ == "__main__":
    main()
