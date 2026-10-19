"""Gera a pasta de realce (representação "realce_3canais") a partir da Etapa 1.

Mesmos image_ids e mesmo tamanho da Etapa 1 (antes do canvas), em PNG RGB 8
bits: R = anatomia, G = top-hat, B = CLAHE, fundo 0. O cálculo é o mesmo
``representar`` que o Dataset e a inferência usam. Rode a partir da raiz:

    # teste: amostra (padrão 50) em <realce_dir>/amostra, com grade, equivalência e tempos
    python -m scripts.rsna_realce_gerar --amostra

    # lote completo (retomável: rodar de novo pula os PNGs que já existem)
    python -m scripts.rsna_realce_gerar

Saídas em ``caminhos.realce_dir``: realce/<image_id>.png, manifest_realce.csv
(o manifest da Etapa 1 mais ``png_realce``), config_usada.yaml (parâmetros do
realce e commit), erros.log e SHA256SUMS (PNGs e manifest).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import time
import traceback
from dataclasses import asdict, replace
from multiprocessing import Pool
from pathlib import Path

for _variavel in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_variavel, "1")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from tqdm import tqdm  # noqa: E402

from scripts.rsna_preprocessar import gravar_png, selecionar_amostra  # noqa: E402
from src.data.rsna_entrada import (  # noqa: E402
    REPRESENTACOES,
    ConfigEntrada,
    caminho_realce,
    carregar_config,
    ler_pngs,
    ler_realce,
    montar_canvas,
    para_png,
    preparar_entrada,
    representar,
)
from src.utils.checksums import gravar_sha256sums, sha256  # noqa: E402
from src.utils.paths import CONFIGS_DIR, PROJECT_ROOT  # noqa: E402

cv2.setNumThreads(1)

SEMENTE = 42
POR_PAGINA = 8


def gerar_uma(tarefa: dict) -> dict:
    """Roda num processo separado: recebe e devolve só tipos simples."""
    linha = {"image_id": tarefa["image_id"], "status": "ok"}
    try:
        inicio = time.perf_counter()
        anatomia, mascara = ler_pngs(
            Path(tarefa["etapa1_dir"]), tarefa["png_anatomia"], tarefa["png_mascara"]
        )
        realce = representar(anatomia, mascara, tarefa["spacing_mm"], tarefa["entrada"])
        meio = time.perf_counter()
        destino = Path(tarefa["saida_dir"]) / caminho_realce(tarefa["image_id"])
        destino.parent.mkdir(parents=True, exist_ok=True)
        gravar_png(destino, para_png(realce))
        linha["seg_calculo"] = round(meio - inicio, 4)
        linha["seg_gravacao"] = round(time.perf_counter() - meio, 4)
        linha["bytes"] = destino.stat().st_size
    except Exception as erro:
        linha["status"] = "erro"
        linha["erro"] = f"{type(erro).__name__}: {erro}"
        linha["traceback"] = traceback.format_exc()
    return linha


def commit_atual() -> dict[str, object]:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), *args], capture_output=True, text=True, check=True
        ).stdout.strip()

    return {
        "commit": git("rev-parse", "HEAD"),
        "alteracoes_locais": bool(git("status", "--porcelain", "-uno")),
    }


def gravar_config_usada(saida: Path, entrada: ConfigEntrada, etapa1: Path) -> None:
    dados = {
        "representacao": "realce_3canais",
        "realce": {
            k: list(v) if isinstance(v, tuple) else v for k, v in asdict(entrada.realce).items()
        },
        **commit_atual(),
        # Amarra a pasta à versão da Etapa 1 de onde ela saiu.
        "etapa1_sha256sums": sha256(etapa1 / "SHA256SUMS"),
    }
    with open(saida / "config_usada.yaml", "w", encoding="utf-8") as arquivo:
        yaml.safe_dump(dados, arquivo, allow_unicode=True, sort_keys=False)


def gerar(
    manifest: pd.DataFrame, entrada: ConfigEntrada, etapa1: Path, saida: Path, workers: int
) -> list[dict]:
    (saida / "realce").mkdir(parents=True, exist_ok=True)
    pendentes = manifest[
        [not (saida / caminho_realce(i)).is_file() for i in manifest.image_id]
    ].sort_values(["patient_id", "image_id"])
    print(
        f"{len(manifest)} imagens, {len(manifest) - len(pendentes)} já prontas, {len(pendentes)} a gerar"
    )

    tarefas = [
        {
            "image_id": int(r.image_id),
            "png_anatomia": r.png_anatomia,
            "png_mascara": r.png_mascara,
            "spacing_mm": float(r.spacing_resultante_mm),
            "etapa1_dir": str(etapa1),
            "saida_dir": str(saida),
            "entrada": entrada,
        }
        for r in pendentes.itertuples()
    ]
    resultados = []
    with Pool(workers) as pool:
        for linha in tqdm(pool.imap_unordered(gerar_uma, tarefas, chunksize=4), total=len(tarefas)):
            resultados.append(linha)

    erros = [r for r in resultados if r["status"] != "ok"]
    with open(saida / "erros.log", "w", encoding="utf-8") as log:
        for r in erros:
            log.write(f"{r['image_id']}: {r['erro']}\n{r['traceback']}\n")

    prontas = manifest[[(saida / caminho_realce(i)).is_file() for i in manifest.image_id]].copy()
    prontas["png_realce"] = [caminho_realce(i) for i in prontas.image_id]
    prontas.to_csv(saida / "manifest_realce.csv", index=False)
    gravar_sha256sums(saida, [*prontas.png_realce, "manifest_realce.csv"])
    print(f"ok: {len(prontas)} | erros nesta execução: {len(erros)} (ver erros.log)")
    return resultados


def grade(amostra: pd.DataFrame, entrada: ConfigEntrada, etapa1: Path, saida: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    picos = REPRESENTACOES["realce_3canais"].canais_de_pico
    linhas = list(amostra.itertuples())
    for pagina in range(0, len(linhas), POR_PAGINA):
        bloco = linhas[pagina : pagina + POR_PAGINA]
        fig, eixos = plt.subplots(4, 8, figsize=(16, 13.5))
        for ax in eixos.ravel():
            ax.axis("off")
        for k, r in enumerate(bloco):
            _, mascara = ler_pngs(etapa1, r.png_anatomia, r.png_mascara)
            canvas, _, _ = montar_canvas(
                ler_realce(saida, r.image_id), mascara, entrada.altura, entrada.largura, picos
            )
            lin, col = divmod(k, 2)
            paineis = (
                (canvas[:, :, 0], "R: anatomia"),
                (canvas[:, :, 1], "G: top-hat"),
                (canvas[:, :, 2], "B: CLAHE"),
                (canvas, "RGB"),
            )
            rotulo = "POSITIVA" if r.cancer else "negativa"
            estimado = ", spacing estimado" if r.spacing_estimado else ""
            for j, (img, nome) in enumerate(paineis):
                ax = eixos[lin, col * 4 + j]
                ax.imshow(img, cmap="gray" if img.ndim == 2 else None, vmin=0, vmax=255)
                ax.set_title(nome, fontsize=9)
            eixos[lin, col * 4].text(
                0, 1.12,
                f"{r.image_id} | site {r.site_id}, aparelho {r.machine_id}{estimado} | "
                f"{r.laterality} {r.view} | {rotulo}",
                transform=eixos[lin, col * 4].transAxes, fontsize=9,
            )  # fmt: skip
        fig.suptitle("Realce no canvas 768x512 (máximo no G antes de reduzir)", fontsize=12)
        fig.tight_layout()
        destino = saida / f"grade_realce_{pagina // POR_PAGINA + 1:02d}.png"
        fig.savefig(destino, dpi=80)
        plt.close(fig)
        print(destino)


def conferir_equivalencia(
    amostra: pd.DataFrame, entrada: ConfigEntrada, etapa1: Path, saida: Path
) -> None:
    """Realce calculado na hora == PNG salvo, e o tensor final também."""
    pngs_iguais = tensores_iguais = 0
    for r in amostra.itertuples():
        anatomia, mascara = ler_pngs(etapa1, r.png_anatomia, r.png_mascara)
        na_hora = representar(anatomia, mascara, r.spacing_resultante_mm, entrada)
        salvo = ler_realce(saida, r.image_id)
        pngs_iguais += bool(np.array_equal(na_hora, salvo))
        a = preparar_entrada(anatomia, mascara, r.spacing_resultante_mm, entrada)
        b = preparar_entrada(
            anatomia, mascara, r.spacing_resultante_mm, entrada, representada=salvo
        )
        tensores_iguais += bool(np.array_equal(a, b))
    print(
        f"equivalência: realce idêntico ao PNG em {pngs_iguais}/{len(amostra)}; "
        f"tensor final idêntico em {tensores_iguais}/{len(amostra)}"
    )


def tempos(resultados: list[dict], duracao: float, workers: int, total: int) -> None:
    ok = [r for r in resultados if r["status"] == "ok"]
    if not ok:
        return
    calculo = np.mean([r["seg_calculo"] for r in ok])
    gravacao = np.mean([r["seg_gravacao"] for r in ok])
    tamanho = np.mean([r["bytes"] for r in ok])
    vazao = len(ok) / duracao
    print(
        f"por imagem (1 processo): cálculo {calculo * 1000:.0f} ms, gravação {gravacao * 1000:.0f} ms | "
        f"{workers} workers: {vazao:.1f} imagens/s | PNG médio {tamanho / 2**20:.2f} MB\n"
        f"estimativa para {total} imagens: {total / vazao / 60:.0f} min e {total * tamanho / 2**30:.1f} GB"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=CONFIGS_DIR / "etapa2.yaml")
    parser.add_argument("--amostra", action="store_true", help="só a amostra, com grade e tempos")
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    cfg = carregar_config(args.config)
    entrada = replace(cfg.entrada, representacao="realce_3canais")
    etapa1 = Path(cfg.caminhos["etapa1_dir"])
    saida = Path(cfg.caminhos["realce_dir"])
    workers = args.workers or int(cfg.geracao_realce.get("workers", 8))

    manifest = pd.read_csv(etapa1 / "manifest_processado.csv")
    manifest = manifest[manifest.status == "ok"]
    total = len(manifest)
    if args.amostra:
        saida = saida / "amostra"
        manifest = selecionar_amostra(
            manifest, int(cfg.geracao_realce.get("amostra_n", 50)), SEMENTE
        )

    saida.mkdir(parents=True, exist_ok=True)
    gravar_config_usada(saida, entrada, etapa1)
    inicio = time.perf_counter()
    resultados = gerar(manifest, entrada, etapa1, saida, workers)
    duracao = time.perf_counter() - inicio

    if args.amostra:
        conferir_equivalencia(manifest, entrada, etapa1, saida)
        tempos(resultados, duracao, workers, total)
        grade(manifest, entrada, etapa1, saida)


if __name__ == "__main__":
    main()
