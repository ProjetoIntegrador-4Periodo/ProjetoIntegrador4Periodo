"""Inventaria os DICOMs de mamografia, remove duplicatas e monta o manifesto.

Lê só o cabeçalho de cada arquivo (sem tocar nos pixels), deduplica por
``SOPInstanceUID`` e por hash de conteúdo, junta BI-RADS e densidade a partir
do CSV de anotações do VinDr e grava o manifesto usado por
``scripts/preprocessar_mamografia.py``.

Rode a partir da raiz do repositório:

    python -m scripts.inventariar_dicoms \\
        --annotations data/raw/breast-level_annotations.csv
"""

from __future__ import annotations

import argparse
import hashlib
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import pandas as pd
import pydicom

from src.utils.paths import DATA_DIR, RAW_DIR

# Colunas do manifesto final, na ordem em que são gravadas.
COLUMNS = [
    "study_id",
    "series_id",
    "image_id",
    "laterality",
    "view_position",
    "height",
    "width",
    "pixel_spacing_mm",
    "manufacturer",
    "manufacturer_model",
    "photometric",
    "breast_birads",
    "breast_density",
    "split",
    "birads_number",
    "exam_birads_min",
    "exam_birads_max",
    "clinical_group",
    "birads_incerto",
    "filename",
    "local_path",
]


def read_header(
    path_text: str,
) -> dict[str, Any]:
    """
    Lê só o cabeçalho do DICOM, sem tocar nos pixels.

    O `SOPInstanceUID` é o identificador canônico da imagem — no VinDr
    ele coincide com o `image_id`. É por ele que a deduplicação
    acontece, não pelo nome do arquivo, que pode ter sido renomeado.
    """
    path = Path(path_text)

    row: dict[str, Any] = {
        "local_path": path_text,
        "filename": path.name,
        "error": "",
    }

    try:
        ds = pydicom.dcmread(
            path,
            stop_before_pixels=True,
        )

        spacing = getattr(
            ds,
            "ImagerPixelSpacing",
            getattr(
                ds,
                "PixelSpacing",
                [0.0],
            ),
        )

        row.update(
            {
                "study_id": str(getattr(ds, "StudyInstanceUID", "")),
                "series_id": str(getattr(ds, "SeriesInstanceUID", "")),
                "image_id": str(getattr(ds, "SOPInstanceUID", "")),
                "laterality": str(
                    getattr(
                        ds,
                        "ImageLaterality",
                        getattr(ds, "Laterality", ""),
                    )
                )
                .strip()
                .upper(),
                "view_position": str(getattr(ds, "ViewPosition", "")).strip().upper(),
                "height": int(getattr(ds, "Rows", 0)),
                "width": int(getattr(ds, "Columns", 0)),
                "pixel_spacing_mm": float(spacing[0]),
                "manufacturer": str(getattr(ds, "Manufacturer", "")),
                "manufacturer_model": str(
                    getattr(
                        ds,
                        "ManufacturerModelName",
                        "",
                    )
                ),
                "photometric": str(
                    getattr(
                        ds,
                        "PhotometricInterpretation",
                        "",
                    )
                ),
                "transfer_syntax": str(ds.file_meta.TransferSyntaxUID),
                "compressed": bool(ds.file_meta.TransferSyntaxUID.is_compressed),
                "size_bytes": path.stat().st_size,
            }
        )

        # Hash do conteúdo, para pegar cópia byte a byte que tenha
        # recebido UID diferente. Só os primeiros megabytes: o
        # cabeçalho mais o início dos pixels já distingue.
        digest = hashlib.blake2b(
            digest_size=16,
        )

        with open(path, "rb") as handle:
            digest.update(handle.read(4 * 1024 * 1024))

        row["content_hash"] = digest.hexdigest()

    except Exception as exception:
        row["error"] = repr(exception)

    return row


def extract_birads_number(value: Any) -> int | None:
    """Extrai o número de valores como 'BI-RADS 4'."""
    found = (
        pd.Series([str(value)])
        .str.extract(
            r"([1-5])",
            expand=False,
        )
        .iloc[0]
    )

    if pd.isna(found):
        return None

    return int(found)


def attach_labels(
    frame: pd.DataFrame,
    annotations_path: Path | None,
    manifest_path: Path | None,
    birads_suspeito: int = 3,
) -> pd.DataFrame:
    """
    Traz BI-RADS e densidade de fora, porque o DICOM não os carrega.

    Prioridade: o CSV de anotação do VinDr, que cobre a base inteira;
    depois o manifesto atual, que cobre só o que já foi processado.
    """
    label_columns = [
        "breast_birads",
        "breast_density",
        "split",
    ]

    for column in label_columns:
        if column not in frame.columns:
            frame[column] = ""

    sources = []

    if annotations_path is not None:
        sources.append(annotations_path)

    if manifest_path is not None:
        sources.append(manifest_path)

    for source in sources:
        if not source.is_file():
            print(f"  aviso: {source} não existe, ignorado")
            continue

        outside = pd.read_csv(
            source,
            dtype=str,
        )

        faltando = [c for c in ["study_id", "image_id"] if c not in outside.columns]

        if faltando:
            print(f"  aviso: {source.name} não tem {faltando}, ignorado")
            continue

        outside["image_id"] = (
            outside["image_id"]
            .astype(str)
            .str.replace(
                ".dicom",
                "",
                regex=False,
            )
        )

        presentes = [c for c in label_columns if c in outside.columns]

        if not presentes:
            continue

        lookup = outside.drop_duplicates(subset=["study_id", "image_id"]).set_index(
            ["study_id", "image_id"]
        )[presentes]

        chaves = pd.MultiIndex.from_frame(frame[["study_id", "image_id"]])

        for column in presentes:
            valores = lookup[column].reindex(chaves)
            valores.index = frame.index

            vazio = frame[column].isna() | (frame[column].astype(str) == "")

            frame.loc[vazio, column] = valores[vazio]

        cobertos = int((frame["breast_birads"].astype(str).str.len() > 0).sum())

        print(f"  {source.name}: rótulos em " f"{cobertos}/{len(frame)} imagens")

    frame["birads_number"] = frame["breast_birads"].map(extract_birads_number)

    # O min/max por exame precisa vir das anotações completas, não das
    # imagens que estão em disco. Num download parcial, um exame com
    # 2 das 4 vistas baixadas teria o máximo calculado só sobre essas
    # duas — e se a vista faltante fosse a do achado, o exame inteiro
    # receberia rótulo de benigno.
    agregado_completo = False

    if annotations_path is not None and annotations_path.is_file():
        completo = pd.read_csv(annotations_path, dtype=str)

        if {"study_id", "breast_birads"} <= set(completo.columns):
            completo["n"] = completo["breast_birads"].map(extract_birads_number)

            por_exame_completo = completo.groupby("study_id")["n"]

            minimos = por_exame_completo.min()
            maximos = por_exame_completo.max()

            frame["exam_birads_min"] = frame["study_id"].map(minimos)
            frame["exam_birads_max"] = frame["study_id"].map(maximos)

            cobertos = int(frame["exam_birads_max"].notna().sum())

            print(
                f"  agregado por exame vindo das anotações completas: "
                f"{cobertos}/{len(frame)} imagens"
            )

            agregado_completo = True

    if not agregado_completo:
        por_exame = frame.groupby("study_id")["birads_number"]

        frame["exam_birads_min"] = por_exame.transform("min")
        frame["exam_birads_max"] = por_exame.transform("max")

        print(
            "  AVISO: agregado por exame calculado só sobre as imagens "
            "em disco. Se o download estiver incompleto, exames com "
            "vistas faltando podem receber rótulo errado."
        )

    # Preenche buracos com o que der para calcular localmente.
    local = frame.groupby("study_id")["birads_number"]

    frame["exam_birads_min"] = frame["exam_birads_min"].fillna(local.transform("min"))
    frame["exam_birads_max"] = frame["exam_birads_max"].fillna(local.transform("max"))

    # BI-RADS 3 é "provavelmente benigno, controle em 6 meses" — não é
    # nem claramente normal nem claramente suspeito. O limiar é
    # parâmetro justamente porque essa fronteira é uma escolha de
    # projeto, não um fato clínico. Com limiar 3 a base do VinDr passa
    # de 9,6% para 18,3% de exames positivos.
    frame["clinical_group"] = frame["exam_birads_max"].map(
        lambda v: "" if pd.isna(v) else ("suspicious" if v >= birads_suspeito else "non_suspicious")
    )

    # Preserva a distinção mesmo com o rótulo binário achatado, para
    # dar para reagrupar depois sem reprocessar nada.
    frame["birads_incerto"] = frame["birads_number"].map(
        lambda v: "" if pd.isna(v) else str(int(v) == 3)
    )

    return frame


def main() -> None:
    parser = argparse.ArgumentParser(
        description=("Inventaria DICOMs, remove duplicatas e monta o manifesto.")
    )

    parser.add_argument(
        "--dicom-dir",
        type=Path,
        default=RAW_DIR,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DATA_DIR / "manifest.csv",
    )

    parser.add_argument(
        "--annotations",
        type=Path,
        default=None,
        help=("breast-level_annotations.csv do VinDr, " "que traz BI-RADS e densidade"),
    )

    parser.add_argument(
        "--manifest-anterior",
        type=Path,
        default=None,
        help="manifesto atual, para reaproveitar rótulos já obtidos",
    )

    parser.add_argument(
        "--birads-suspeito",
        type=int,
        default=3,
        help=(
            "menor BI-RADS que conta como suspeito. Padrão 3, que trata "
            "'provavelmente benigno' como positivo; use 4 para exigir "
            "achado claramente suspeito."
        ),
    )

    parser.add_argument(
        "--apagar-duplicatas",
        action="store_true",
        help=("apaga de fato os arquivos duplicados; " "sem esta flag apenas relata"),
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) // 2),
    )

    args = parser.parse_args()

    caminhos = sorted(str(p) for p in args.dicom_dir.rglob("*.dicom"))

    caminhos += sorted(str(p) for p in args.dicom_dir.rglob("*.dcm"))

    if not caminhos:
        raise RuntimeError(f"Nenhum DICOM encontrado em {args.dicom_dir}")

    print(f"Lendo cabeçalho de {len(caminhos)} arquivos...")

    with ProcessPoolExecutor(
        max_workers=args.workers,
    ) as pool:
        linhas = list(
            pool.map(
                read_header,
                caminhos,
                chunksize=32,
            )
        )

    frame = pd.DataFrame(linhas)

    ilegiveis = frame[frame["error"] != ""]

    if len(ilegiveis) > 0:
        print(f"\n{len(ilegiveis)} arquivos ilegíveis:")

        for _, linha in ilegiveis.head(10).iterrows():
            print(f"  {linha['filename']}: {linha['error'][:90]}")

        frame = frame[frame["error"] == ""].copy()

    print("\n" + "=" * 70)
    print("DUPLICATAS")
    print("=" * 70)

    # Qual cópia manter não pode ser por ordem alfabética: durante um
    # download em andamento existem arquivos truncados com o mesmo UID
    # do arquivo íntegro, e a ordem alfabética descartaria o bom na
    # metade dos casos. Critérios, em ordem:
    #   1. maior arquivo — o truncado é sempre menor que o completo
    #   2. dentro de uma pasta de exame — arquivo solto na raiz de
    #      images/ faria o study_id virar o nome da pasta errado
    frame["_dentro_de_exame"] = frame["local_path"].map(
        lambda p: Path(p).parent.name.lower() != "images"
    )

    frame = frame.sort_values(
        ["size_bytes", "_dentro_de_exame", "local_path"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    por_uid = frame.duplicated(
        subset=["image_id"],
        keep="first",
    )

    por_hash = (
        frame.duplicated(
            subset=["content_hash"],
            keep="first",
        )
        & ~por_uid
    )

    print(f"  arquivos lidos            : {len(frame)}")
    print(f"  duplicados por SOPInstanceUID: {int(por_uid.sum())}")
    print(f"  duplicados só por conteúdo   : {int(por_hash.sum())}")

    duplicados = frame[por_uid | por_hash]

    if len(duplicados) > 0:
        print("\n  exemplos:")

        for _, linha in duplicados.head(8).iterrows():
            original = frame[(frame["image_id"] == linha["image_id"]) & (frame.index != linha.name)]

            manteve = Path(original.iloc[0]["local_path"]).parent.name if len(original) > 0 else "?"

            tamanho_mb = linha["size_bytes"] / 1e6

            mantido_mb = original.iloc[0]["size_bytes"] / 1e6 if len(original) > 0 else 0.0

            print(
                f"    {linha['filename'][:26]}  "
                f"descarta {tamanho_mb:5.1f} MB de "
                f"'{Path(linha['local_path']).parent.name[:12]}'  ->  "
                f"mantém {mantido_mb:5.1f} MB de '{manteve[:12]}'"
            )

        # Se algum descartado for maior que o mantido, a heurística
        # falhou e apagar seria perda de dado.
        perigosos = 0

        for _, linha in duplicados.iterrows():
            gemeos = frame[(frame["image_id"] == linha["image_id"]) & (frame.index != linha.name)]

            if len(gemeos) > 0 and (linha["size_bytes"] > gemeos.iloc[0]["size_bytes"]):
                perigosos += 1

        if perigosos:
            print(
                f"\n  ATENÇÃO: {perigosos} descartados são MAIORES que "
                "a cópia mantida. Isso não deveria acontecer — verifique "
                "antes de apagar."
            )

        truncados = int((duplicados["size_bytes"] < 8e6).sum())

        if truncados:
            print(
                f"\n  {truncados} dos duplicados têm menos de 8 MB, "
                "tamanho típico de arquivo ainda em transferência. "
                "Se o download ainda estiver rodando, espere terminar "
                "antes de apagar."
            )

        if args.apagar_duplicatas:
            for caminho in duplicados["local_path"]:
                Path(caminho).unlink()

            print(f"\n  {len(duplicados)} arquivos APAGADOS")
        else:
            print("\n  nada foi apagado. Use --apagar-duplicatas " "para remover de fato.")

    frame = frame[~(por_uid | por_hash)].copy()

    print("\n" + "=" * 70)
    print("CARACTERÍSTICAS DE AQUISIÇÃO")
    print("=" * 70)

    print(f"  imagens únicas : {len(frame)}")
    print(f"  exames         : {frame['study_id'].nunique()}")

    for coluna in [
        "manufacturer_model",
        "pixel_spacing_mm",
        "photometric",
        "view_position",
        "laterality",
    ]:
        contagem = frame[coluna].value_counts()
        resumo = ", ".join(f"{k}={v}" for k, v in contagem.head(6).items())
        print(f"  {coluna:<20}: {resumo}")

    comprimidos = int(frame["compressed"].sum())

    if comprimidos:
        print(
            f"\n  ATENÇÃO: {comprimidos} arquivos comprimidos. "
            "Instale pylibjpeg para conseguir ler os pixels."
        )

    incompletos = frame.groupby("study_id").size().pipe(lambda s: s[s != 4])

    if len(incompletos) > 0:
        print(
            f"\n  {len(incompletos)} exames sem as 4 vistas "
            f"(distribuição: {dict(incompletos.value_counts())})"
        )

    print("\n" + "=" * 70)
    print("RÓTULOS")
    print("=" * 70)

    frame = attach_labels(
        frame,
        args.annotations,
        args.manifest_anterior,
        args.birads_suspeito,
    )

    sem_rotulo = int(frame["birads_number"].isna().sum())

    print(f"  imagens sem BI-RADS: {sem_rotulo}/{len(frame)}")

    por_exame = frame.drop_duplicates("study_id")
    incertos = int((por_exame["exam_birads_max"] == 3).sum())

    print(f"  limiar de suspeição: BI-RADS >= {args.birads_suspeito}")

    contagem = por_exame["clinical_group"].value_counts()

    for grupo, quantos in contagem.items():
        if grupo:
            print(f"    {grupo:<16} {quantos} exames")

    if incertos:
        lado = "suspeitos" if args.birads_suspeito <= 3 else "não suspeitos"
        print(
            f"    dos quais {incertos} exames têm BI-RADS 3 como " f"máximo e entraram como {lado}"
        )

    if sem_rotulo > 0:
        print(
            "  Essas imagens dá para pré-processar, mas não para "
            "treinar supervisionado. O BI-RADS vem do "
            "breast-level_annotations.csv do VinDr."
        )

    for coluna in COLUMNS:
        if coluna not in frame.columns:
            frame[coluna] = ""

    frame = frame[COLUMNS].sort_values(["study_id", "laterality", "view_position"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)

    print(f"\nManifesto escrito: {args.output} ({len(frame)} linhas)")


if __name__ == "__main__":
    main()
