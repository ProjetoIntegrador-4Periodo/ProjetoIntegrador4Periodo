"""Script 2: pré-processamento (Etapa 1) das imagens do manifest_fold0.csv.

Lê os DICOMs (só leitura) e grava, por imagem, a anatomia em PNG 8 bits e a
máscara binária, mais o manifest processado. Rode a partir da raiz:

    # teste com amostra (padrão 50 imagens) em <saida>/amostra, com grade antes/depois
    python -m scripts.rsna_preprocessar --amostra

    # lote completo (retomável: rodar de novo continua de onde parou)
    python -m scripts.rsna_preprocessar

Saídas em ``caminhos.saida_dir``: anatomia/, mascara/, manifest_processado.csv,
suspeitas.csv, erros.log, progresso.jsonl, config_usada.yaml e SHA256SUMS.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
import traceback
from multiprocessing import Pool
from pathlib import Path

# Antes do numpy: com vários processos, cada um abrindo seu pool de threads
# faz os núcleos brigarem entre si.
for _variavel in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_variavel, "1")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pydicom  # noqa: E402
import yaml  # noqa: E402
from tqdm import tqdm  # noqa: E402

from src.data.rsna_imagem import Parametros, processar  # noqa: E402
from src.utils.checksums import gravar_sha256sums  # noqa: E402
from src.utils.paths import PROJECT_ROOT  # noqa: E402

cv2.setNumThreads(1)

MINIATURA_ALTURA = 384
# Explícito para os bytes do PNG (e os SHA256SUMS) não dependerem do padrão do
# OpenCV. Nível 9 economiza ~4% sobre o padrão (1) sem custo de tempo relevante.
PNG_COMPRESSAO = [cv2.IMWRITE_PNG_COMPRESSION, 9]


def caminho_dicom(rsna_dir: Path, patient_id: int, image_id: int) -> Path:
    return rsna_dir / "train_images" / str(patient_id) / f"{image_id}.dcm"


def gravar_png(destino: Path, array: np.ndarray) -> None:
    """Grava num temporário e renomeia: um PNG pela metade nunca conta como pronto."""
    temporario = destino.with_name(destino.stem + ".tmp.png")
    if not cv2.imwrite(str(temporario), array, PNG_COMPRESSAO):
        raise OSError(f"Falha ao gravar {temporario}")
    os.replace(temporario, destino)


def miniatura(array: np.ndarray) -> np.ndarray:
    escala = MINIATURA_ALTURA / array.shape[0]
    tamanho = (max(1, round(array.shape[1] * escala)), MINIATURA_ALTURA)
    return cv2.resize(array, tamanho, interpolation=cv2.INTER_AREA)


def processar_uma(tarefa: dict) -> dict:
    """Roda num processo separado: recebe e devolve só tipos simples."""
    inicio = time.perf_counter()
    linha = {"image_id": tarefa["image_id"]}

    try:
        # dcmread abre o arquivo em modo leitura; nada é escrito na origem.
        ds = pydicom.dcmread(tarefa["dicom"])
        anatomia, mascara, meta = processar(ds, tarefa["laterality"], tarefa["parametros"])

        saida = Path(tarefa["saida_dir"])
        gravar_png(saida / tarefa["png_anatomia"], anatomia)
        gravar_png(saida / tarefa["png_mascara"], mascara)

        linha.update(meta)
        linha["status"] = "ok"

        if tarefa["miniaturas"]:
            bruto = ds.pixel_array.astype(np.float32)
            bruto = (bruto - bruto.min()) / max(float(np.ptp(bruto)), 1e-6)
            linha["_antes"] = miniatura((bruto * 255).astype(np.uint8))
            linha["_depois"] = miniatura(anatomia)
            linha["_mascara"] = miniatura(mascara)

    except Exception as erro:
        linha["status"] = "erro"
        linha["erro"] = f"{type(erro).__name__}: {erro}"
        linha["traceback"] = traceback.format_exc()

    linha["segundos"] = round(time.perf_counter() - inicio, 3)
    return linha


def selecionar_amostra(manifest: pd.DataFrame, n: int, semente: int) -> pd.DataFrame:
    """
    Amostra pequena que cobre os casos difíceis do RSNA.

    Garante: um de cada aparelho (inclui os sem spacing), positivos, e
    depois completa rodando por (site, lateralidade, view), o que cobre CC
    e MLO, os dois sites e L/R do site 1.
    """
    rng = np.random.default_rng(semente)
    escolhidos: list[int] = []

    def sortear(subconjunto: pd.DataFrame, quantos: int) -> None:
        livres = subconjunto.index.difference(escolhidos)
        if len(livres):
            escolhidos.extend(rng.choice(livres, size=min(quantos, len(livres)), replace=False))

    for _, grupo in manifest.groupby("machine_id"):
        sortear(grupo, 1)
    for _, grupo in manifest[manifest.cancer == 1].groupby("site_id"):
        sortear(grupo, max(1, n // 10))

    grupos = [g for _, g in manifest.groupby(["site_id", "laterality", "view"])]
    while len(escolhidos) < min(n, len(manifest)):
        for grupo in grupos:
            if len(escolhidos) >= n:
                break
            sortear(grupo, 1)

    return manifest.loc[sorted(escolhidos)].sort_values(["patient_id", "image_id"])


def carregar_progresso(arquivo: Path) -> dict[int, dict]:
    """Última linha de cada imagem no progresso.jsonl."""
    progresso: dict[int, dict] = {}
    if arquivo.is_file():
        with open(arquivo, encoding="utf-8") as entrada:
            for texto in entrada:
                if texto.strip():
                    linha = json.loads(texto)
                    progresso[int(linha["image_id"])] = linha
    return progresso


def montar_grade(resultados: list[dict], manifest: pd.DataFrame, destino: Path) -> None:
    """Grade com antes (DICOM cru) e depois (anatomia + contorno da máscara)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    info = manifest.set_index("image_id")
    validos = [r for r in resultados if r.get("status") == "ok"]
    colunas_pares = 5
    linhas = -(-len(validos) // colunas_pares)
    fig, eixos = plt.subplots(
        linhas, colunas_pares * 2, figsize=(colunas_pares * 4.2, linhas * 3.6), squeeze=False
    )

    for eixo in eixos.ravel():
        eixo.axis("off")

    for i, r in enumerate(validos):
        linha = info.loc[r["image_id"]]
        antes, depois = eixos[
            i // colunas_pares, (i % colunas_pares) * 2 : (i % colunas_pares) * 2 + 2
        ]
        antes.imshow(r["_antes"], cmap="gray")
        antes.set_title(
            f"{r['image_id']}\nsite {linha.site_id} {linha.laterality} {linha.view} "
            f"maq {linha.machine_id}{' CANCER' if linha.cancer else ''}",
            fontsize=7,
            color="red" if linha.cancer else "black",
        )
        depois.imshow(r["_depois"], cmap="gray")
        depois.contour(r["_mascara"] > 0, levels=[0.5], colors="cyan", linewidths=0.6)
        flags = [
            "espelhada" if r["espelhada"] else "",
            "DIVERGE" if r["diverge_tag"] else "",
            "spacing est." if r["spacing_estimado"] else "",
            "SUSPEITA" if r["suspeita"] else "",
        ]
        depois.set_title(
            f"{r['altura_saida']}x{r['largura_saida']} cob {r['cobertura_mascara']:.2f}\n"
            + " ".join(f for f in flags if f),
            fontsize=7,
            color="red" if r["suspeita"] else "black",
        )

    fig.tight_layout()
    fig.savefig(destino, dpi=90)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/pipeline_rsna.yaml")
    parser.add_argument(
        "--amostra", action="store_true", help="processa só a amostra de teste em <saida>/amostra"
    )
    parser.add_argument("--workers", type=int, default=None, help="sobrescreve a config")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    pp = cfg["preprocessamento"]
    rsna_dir = Path(cfg["caminhos"]["rsna_dir"])
    splits_dir = PROJECT_ROOT / cfg["caminhos"]["splits_dir"]
    saida = Path(cfg["caminhos"]["saida_dir"])
    workers = args.workers or pp["workers"]
    parametros = Parametros(
        lado_maior=pp["lado_maior"],
        margem_crop=pp["margem_crop"],
        percentil_baixo=pp["percentil_baixo"],
        percentil_alto=pp["percentil_alto"],
        cobertura_min=pp["cobertura_min"],
        cobertura_max=pp["cobertura_max"],
        spacing_padrao_mm=pp["spacing_padrao_mm"],
        fator_mascara=pp["fator_mascara"],
    )

    manifest = pd.read_csv(splits_dir / "manifest_fold0.csv")
    # Ordem das pastas no HD: leitura quase sequencial.
    manifest = manifest.sort_values(["patient_id", "image_id"]).reset_index(drop=True)

    if args.amostra:
        saida = saida / "amostra"
        manifest = selecionar_amostra(manifest, pp["amostra_n"], cfg["semente"])

    for pasta in ("anatomia", "mascara"):
        (saida / pasta).mkdir(parents=True, exist_ok=True)
        # Sobra de uma queda no meio de gravar_png: nunca é PNG válido.
        for temporario in (saida / pasta).glob("*.tmp.png"):
            temporario.unlink()
    shutil.copyfile(args.config, saida / "config_usada.yaml")

    arquivo_progresso = saida / "progresso.jsonl"
    progresso = carregar_progresso(arquivo_progresso)

    tarefas = []
    for linha in manifest.itertuples():
        png_anatomia = f"anatomia/{linha.image_id}.png"
        png_mascara = f"mascara/{linha.image_id}.png"
        feito = progresso.get(linha.image_id, {}).get("status") == "ok"
        if (
            not args.amostra
            and feito
            and (saida / png_anatomia).is_file()
            and (saida / png_mascara).is_file()
        ):
            continue
        tarefas.append(
            {
                "image_id": linha.image_id,
                "dicom": str(caminho_dicom(rsna_dir, linha.patient_id, linha.image_id)),
                "laterality": linha.laterality,
                "png_anatomia": png_anatomia,
                "png_mascara": png_mascara,
                "saida_dir": str(saida),
                "parametros": parametros,
                "miniaturas": args.amostra,
            }
        )

    print(f"{len(manifest)} imagens no manifest | {len(manifest) - len(tarefas)} já prontas")
    print(f"processando {len(tarefas)} com {workers} workers -> {saida}")

    resultados = []
    inicio = time.perf_counter()
    modo = "w" if args.amostra else "a"
    with (
        open(arquivo_progresso, modo, encoding="utf-8") as jsonl,
        open(saida / "erros.log", modo, encoding="utf-8") as log_erros,
        Pool(workers) as pool,
    ):
        for r in tqdm(pool.imap(processar_uma, tarefas, chunksize=2), total=len(tarefas)):
            registro = {k: v for k, v in r.items() if not k.startswith("_") and k != "traceback"}
            jsonl.write(json.dumps(registro, default=str) + "\n")
            jsonl.flush()
            if r["status"] == "erro":
                log_erros.write(f"{r['image_id']}: {r['erro']}\n{r['traceback']}\n")
                log_erros.flush()
            if args.amostra:
                resultados.append(r)
    parede = time.perf_counter() - inicio

    # Manifest processado a partir do progresso: inclui as imagens puladas.
    progresso = carregar_progresso(arquivo_progresso)
    processado = pd.DataFrame([progresso[i] for i in manifest.image_id if i in progresso])
    processado = manifest.merge(processado, on="image_id", how="left")
    processado.insert(2, "png_anatomia", "anatomia/" + processado.image_id.astype(str) + ".png")
    processado.insert(3, "png_mascara", "mascara/" + processado.image_id.astype(str) + ".png")
    ok = processado[processado.status == "ok"].copy()
    erros = processado[processado.status != "ok"]
    # O tempo por imagem fica só no progresso.jsonl: no manifest ele mudaria o
    # hash a cada execução e quebraria a conferência de reprodutibilidade.
    processado.drop(columns="segundos").to_csv(
        saida / "manifest_processado.csv", index=False, lineterminator="\n"
    )

    suspeitas = ok[ok.suspeita.astype(bool)].sort_values(
        ["cancer", "cobertura_mascara"], ascending=[False, True]
    )
    suspeitas.drop(columns="segundos").to_csv(
        saida / "suspeitas.csv", index=False, lineterminator="\n"
    )

    pngs = [*ok.png_anatomia, *ok.png_mascara]
    gravar_sha256sums(saida, [*pngs, "manifest_processado.csv", "suspeitas.csv"])

    print("\n" + "=" * 70)
    print(f"ok: {len(ok)} | erros: {len(erros)} (ver erros.log) | suspeitas: {len(suspeitas)}")
    if len(suspeitas):
        print(f"  suspeitas positivas: {int(suspeitas.cancer.sum())} (topo de suspeitas.csv)")
    print(
        f"espelhadas: {int(ok.espelhada.astype(bool).sum())} | divergem da tag: "
        f"{int(ok.diverge_tag.astype(bool).sum())} | spacing estimado: "
        f"{int(ok.spacing_estimado.astype(bool).sum())}"
    )
    print("divergência da tag por site:")
    print(
        (ok.groupby("site_id").diverge_tag.apply(lambda s: s.astype(bool).mean()) * 100)
        .round(1)
        .astype(str)
        .add("%")
        .to_string()
    )

    if args.amostra and tarefas:
        tam_anat = np.mean([(saida / p).stat().st_size for p in ok.png_anatomia]) / 1e6
        tam_masc = np.mean([(saida / p).stat().st_size for p in ok.png_mascara]) / 1e6
        por_imagem = parede / len(tarefas)
        total = len(pd.read_csv(splits_dir / "manifest_fold0.csv"))
        print("\nMEDIDAS DA AMOSTRA")
        print(f"  tempo por imagem num worker: {ok.segundos.mean():.2f} s (média)")
        print(
            f"  vazão com {workers} workers : {por_imagem:.2f} s/imagem ({parede:.0f} s no total)"
        )
        print(f"  PNG anatomia: {tam_anat:.2f} MB | máscara: {tam_masc:.3f} MB (média)")
        print(
            f"  estimativa para {total} imagens: {total * por_imagem / 3600:.1f} h, "
            f"{total * (tam_anat + tam_masc) / 1e3:.1f} GB"
        )
        print("  cobertura da amostra:")
        print(
            f"    views: {sorted(manifest.view.unique())} | sites: {sorted(manifest.site_id.unique().tolist())}"
            f" | L/R site 1: {sorted(manifest[manifest.site_id == 1].laterality.unique())}"
        )
        print(
            f"    positivas: {int(manifest.cancer.sum())} | aparelhos: "
            f"{sorted(manifest.machine_id.unique().tolist())}"
        )
        montar_grade(resultados, manifest, saida / "grade_antes_depois.png")
        print(f"  grade: {saida / 'grade_antes_depois.png'}")


if __name__ == "__main__":
    main()
