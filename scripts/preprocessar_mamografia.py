"""Pré-processa em lote os DICOMs de mamografia e grava os PNGs de treino.

Para cada DICOM gera três PNGs, já no canvas padronizado:

- ``png_anatomia/``: canal único, imagem corrigida e normalizada.
- ``png_realce_tophat/``: 3 canais, ``[anatomia, top-hat em mm, CLAHE]``.
- ``png_mascara_mama/``: silhueta da mama, separa tecido de padding.

e um ``relatorio_preprocessamento.csv`` com o caminho radiométrico, o crop,
a fração de tecido e a redução aplicada em cada imagem.

Rode a partir da raiz do repositório:

    python -m scripts.preprocessar_mamografia --manifest data/manifest.csv

O manifesto sai de ``python -m scripts.inventariar_dicoms``.
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

# Precisa vir antes de numpy/scipy: com vários processos, deixar cada
# um abrir seu próprio pool de threads faz os núcleos brigarem entre si.
for _variavel in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_variavel, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402
from tqdm import tqdm  # noqa: E402

from src.data.preprocessamento import (  # noqa: E402
    create_representation,
    mm_to_pixels,
    read_dicom,
    standardize_geometry,
    to_uint8,
)
from src.utils.paths import DATA_DIR, PROCESSED_DIR, RAW_DIR  # noqa: E402


def load_manifest(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    """Indexa o manifesto por (study_id, image_id)."""
    frame = pd.read_csv(path, dtype=str)

    frame["image_id_clean"] = frame["image_id"].astype(str).str.replace(".dicom", "", regex=False)

    return {
        (str(row["study_id"]), str(row["image_id_clean"])): row.to_dict()
        for _, row in frame.iterrows()
    }


def process_one(task: dict[str, Any]) -> dict[str, Any]:
    """
    Processa uma imagem. Roda num processo separado, então recebe e
    devolve apenas tipos simples — nada de handles de arquivo.
    """
    dicom_path = Path(task["dicom_path"])

    row = dict(task["row"])

    gray_path = Path(task["gray_path"])
    purple_path = Path(task["purple_path"])
    mask_path = Path(task["mask_path"])

    try:
        image, mask, dicom_metadata = read_dicom(dicom_path)

        representation = create_representation(
            image,
            mask,
            "D",
            task["tophat_mm"],
            task["clahe_clip"],
            dicom_metadata["pixel_spacing_mm"],
            percentile=task["percentile"],
            denoise=task["denoise"],
        )

        row["crop_height"] = image.shape[0]
        row["crop_width"] = image.shape[1]
        row["aspect_ratio"] = round(image.shape[0] / image.shape[1], 4)
        row["tissue_fraction"] = round(float(mask.mean()), 4)
        row["radiometric_path"] = dicom_metadata["radiometric_path"]
        row["pixel_spacing_mm"] = dicom_metadata["pixel_spacing_mm"]
        row["intensity_sign"] = dicom_metadata["intensity_sign"]
        row["tophat_px"] = mm_to_pixels(task["tophat_mm"], dicom_metadata["pixel_spacing_mm"])

        if task["native_resolution"]:
            final = representation
            final_mask = mask

            row["flipped"] = False
            row["scale"] = 1.0
            row["reduction"] = 1.0
            row["canvas_usage"] = 1.0

        else:
            laterality = dicom_metadata["laterality"] or row["laterality"]

            if laterality not in ("L", "R"):
                raise ValueError(
                    "Lateralidade ausente no DICOM e no "
                    "manifesto. Sem ela a mama não pode ser "
                    "orientada e o lote sairia com mamas "
                    "espelhadas misturadas."
                )

            row["laterality"] = laterality

            if not row["view_position"]:
                row["view_position"] = dicom_metadata["view_position"]

            final, final_mask, geometry = standardize_geometry(
                representation,
                mask,
                laterality,
                task["canvas_height"],
                task["canvas_width"],
            )

            row.update(geometry)

        row["output_height"] = final.shape[0]
        row["output_width"] = final.shape[1]

        if task["dry_run"]:
            row["status"] = "dry_run"
            return row

        for path in (gray_path, purple_path, mask_path):
            path.parent.mkdir(parents=True, exist_ok=True)

        Image.fromarray(to_uint8(final[:, :, 0])).save(gray_path)
        Image.fromarray(to_uint8(final)).save(purple_path)
        Image.fromarray((final_mask * 255).astype(np.uint8)).save(mask_path)

        row["status"] = "sucesso"

    except Exception as exception:
        row["error"] = repr(exception)

    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])

    parser.add_argument("--dicom-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--manifest", type=Path, default=DATA_DIR / "manifest.csv")
    parser.add_argument("--output-dir", type=Path, default=PROCESSED_DIR)

    # Em milímetros de tecido, não em pixels. Calcificações suspeitas
    # ficam na faixa de 0,1 a 0,5 mm, então um SE de ~0,6 mm as isola
    # sem deixar passar estrutura maior.
    parser.add_argument("--tophat-mm", type=float, default=0.6)
    parser.add_argument("--clahe-clip", type=float, default=0.01)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")

    # Canvas escolhido a partir das 100 imagens: a proporção mediana
    # do crop é 2,5, então um canvas 768x1920 aproveita 82% da área e
    # nenhuma imagem precisa encolher mais que 1,94x.
    parser.add_argument("--canvas-width", type=int, default=768)
    parser.add_argument("--canvas-height", type=int, default=1920)
    parser.add_argument("--percentile", type=float, default=99.9)
    parser.add_argument("--no-denoise", action="store_true")
    parser.add_argument("--native-resolution", action="store_true")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))

    args = parser.parse_args()

    dicom_paths = sorted([*args.dicom_dir.rglob("*.dicom"), *args.dicom_dir.rglob("*.dcm")])

    if not dicom_paths:
        raise RuntimeError(f"Nenhum DICOM encontrado em {args.dicom_dir}")

    manifest_lookup = load_manifest(args.manifest)

    gray_root = args.output_dir / "png_anatomia"
    purple_root = args.output_dir / "png_realce_tophat"
    mask_root = args.output_dir / "png_mascara_mama"
    report_path = args.output_dir / "relatorio_preprocessamento.csv"

    for root in (gray_root, purple_root, mask_root):
        root.mkdir(parents=True, exist_ok=True)

    # Um run que pula imagens já processadas não tem como recalcular
    # crop, proporção e fração de tecido — esses números só existem
    # durante o processamento. Sem carregar o relatório anterior, cada
    # rerun apagaria as medidas das imagens puladas.
    previous: dict[tuple[str, str], dict[str, Any]] = {}

    if report_path.is_file():
        anterior = pd.read_csv(report_path, dtype={"study_id": str, "image_id": str})

        previous = {
            (str(linha["study_id"]), str(linha["image_id"])): linha.to_dict()
            for _, linha in anterior.iterrows()
        }

        print(f"Relatório anterior: {len(previous)} imagens já medidas.")

    report_rows = []
    tasks = []

    for dicom_path in dicom_paths:
        study_id = dicom_path.parent.name
        image_id = dicom_path.stem

        metadata = manifest_lookup.get((study_id, image_id), {})

        gray_path = gray_root / study_id / f"{image_id}.png"
        purple_path = purple_root / study_id / f"{image_id}.png"
        mask_path = mask_root / study_id / f"{image_id}.png"

        row = {
            "study_id": study_id,
            "image_id": image_id,
            "laterality": metadata.get("laterality", ""),
            "view_position": metadata.get("view_position", ""),
            "breast_birads": metadata.get("breast_birads", ""),
            "clinical_group": metadata.get("clinical_group", ""),
            "status": "erro",
            "crop_height": 0,
            "crop_width": 0,
            "aspect_ratio": 0.0,
            "tissue_fraction": 0.0,
            "radiometric_path": "",
            "pixel_spacing_mm": 0.0,
            "intensity_sign": "",
            "tophat_px": 0,
            "flipped": False,
            "scale": 0.0,
            "reduction": 0.0,
            "canvas_usage": 0.0,
            "output_height": 0,
            "output_width": 0,
            "png_anatomia": str(gray_path),
            "png_realce_tophat": str(purple_path),
            "png_mascara_mama": str(mask_path),
            "error": "",
        }

        if (
            gray_path.exists()
            and purple_path.exists()
            and mask_path.exists()
            and not args.overwrite
            and not args.dry_run
        ):
            anterior = previous.get((study_id, image_id))

            if anterior is not None:
                row.update(
                    {
                        chave: valor
                        for chave, valor in anterior.items()
                        if chave in row and pd.notna(valor)
                    }
                )

            row["status"] = "já_existia"
            report_rows.append(row)
            continue

        tasks.append(
            {
                "dicom_path": str(dicom_path),
                "row": row,
                "gray_path": str(gray_path),
                "purple_path": str(purple_path),
                "mask_path": str(mask_path),
                "tophat_mm": args.tophat_mm,
                "clahe_clip": args.clahe_clip,
                "percentile": args.percentile,
                "denoise": not args.no_denoise,
                "canvas_height": args.canvas_height,
                "canvas_width": args.canvas_width,
                "native_resolution": args.native_resolution,
                "dry_run": args.dry_run,
            }
        )

    if tasks:
        if args.workers == 1:
            for task in tqdm(tasks, desc="Pré-processamento"):
                report_rows.append(process_one(task))

        else:
            # Cada imagem é independente, então o paralelismo é entre
            # imagens. As bibliotecas numéricas já usam threads por
            # dentro; limitá-las a 1 evita que os processos briguem
            # pelos mesmos núcleos e percam para troca de contexto.
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                futures = [pool.submit(process_one, task) for task in tasks]

                for future in tqdm(
                    as_completed(futures), total=len(futures), desc="Pré-processamento"
                ):
                    report_rows.append(future.result())

    report_rows.sort(key=lambda item: (item["study_id"], item["image_id"]))

    report = pd.DataFrame(report_rows)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(report_path, index=False)

    print()
    print("=" * 70)
    print(report["status"].value_counts().to_string())
    print()
    print(
        "Proporção média (alt/larg):",
        round(float(report["aspect_ratio"].replace(0.0, np.nan).mean()), 4),
    )
    print("Relatório:", report_path)
    print("=" * 70)


if __name__ == "__main__":
    main()
