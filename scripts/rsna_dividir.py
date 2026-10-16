"""Script 1: limpeza, divisão por paciente e seleção do fold 0 do RSNA.

Só lê o train.csv e grava CSVs (não toca nos DICOMs). Rode a partir da raiz:

    python -m scripts.rsna_dividir --config configs/pipeline_rsna.yaml

Saídas em ``caminhos.splits_dir``: splits.csv, manifest_fold0.csv,
relatorio_divisao.txt e SHA256SUMS.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from src.data.rsna_divisao import (
    distribuicao,
    dividir,
    excluir_manual,
    fundir_estratos_raros,
    limpar,
    ordenar,
    selecionar_fold0,
    tabela_pacientes,
    verificar_vazamento,
)
from src.utils.checksums import gravar_sha256sums
from src.utils.paths import PROJECT_ROOT


class Relatorio:
    """Imprime e guarda o texto, para o relatório gravado ser igual ao da tela."""

    def __init__(self):
        self.linhas: list[str] = []

    def __call__(self, texto: object = "") -> None:
        print(texto)
        self.linhas.append(str(texto))

    def secao(self, titulo: str) -> None:
        self(f"\n{'=' * 70}\n{titulo}\n{'=' * 70}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/pipeline_rsna.yaml")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    div = cfg["divisao"]
    semente = cfg["semente"]
    rsna_dir = Path(cfg["caminhos"]["rsna_dir"])
    splits_dir = PROJECT_ROOT / cfg["caminhos"]["splits_dir"]

    rel = Relatorio()
    df = ordenar(pd.read_csv(rsna_dir / "train.csv"))
    rel(f"train.csv: {len(df)} imagens, {df.patient_id.nunique()} pacientes | semente {semente}")

    rel.secao("LIMPEZA")
    df, removidos = limpar(df, div["views"], div["remover_implante"])
    rel(pd.DataFrame(removidos).set_index("regra").to_string())
    rel(
        f"\nrestam: {len(df)} imagens, "
        f"{df[['patient_id', 'laterality']].drop_duplicates().shape[0]} mamas, "
        f"{df.patient_id.nunique()} pacientes, {int(df.cancer.sum())} imagens com câncer"
    )

    rel.secao("ESTRATOS (câncer da paciente + site + faixa etária)")
    pacientes = tabela_pacientes(df, div["faixas_etarias"])
    estrato, fusoes = fundir_estratos_raros(
        pacientes, div["faixas_etarias"], div["min_pacientes_estrato"]
    )
    rel(f"faixas fundidas (estratos com < {div['min_pacientes_estrato']} pacientes):")
    for fusao in fusoes or ["nenhuma"]:
        rel(f"  {fusao}")
    rel(f"\n{estrato.nunique()} estratos; menores:")
    rel(estrato.value_counts().tail(6).to_string())

    df["faixa_etaria"] = df["patient_id"].map(pacientes["faixa_etaria"])
    df = dividir(df, estrato, div["frac_teste"], div["n_folds"], div["fold_validacao"], semente)

    # A exclusão manual vem depois do sorteio, para não reembaralhar a divisão
    # a cada imagem excluída. O pool precisa da divisão completa (ver
    # selecionar_fold0), então é calculado antes de tirar as linhas.
    excluir = div.get("excluir_imagens") or []
    manifest = selecionar_fold0(df, div["pool_negativos"], semente, excluir)
    df, exclusao = excluir_manual(df, excluir)

    rel.secao("EXCLUSÃO MANUAL (depois da divisão, lista na config)")
    rel(pd.DataFrame([exclusao]).set_index("regra").to_string())
    rel(f"restam: {len(df)} imagens, {df.patient_id.nunique()} pacientes")

    rel.secao("VALIDAÇÃO")
    verificar_vazamento(df)
    rel("sem vazamento: nenhuma paciente em mais de um split ou fold")
    rel("\npor split:")
    rel(distribuicao(df, "split").to_string())
    rel("\npor fold (-1 = teste):")
    rel(distribuicao(df, "fold").to_string())

    rel.secao("SELEÇÃO DO FOLD 0")
    treino = manifest[manifest.split == "train"]
    obrigatorios = treino[
        (treino.cancer == 0) & (treino.difficult_negative_case | (treino.biopsy == 1))
    ]
    rel(
        f"treino: {int(treino.cancer.sum())} positivos + {int(treino.in_negative_pool.sum())} "
        f"negativos no pool ({len(obrigatorios)} obrigatórios por difficult/biopsy)"
    )
    resumo = manifest.groupby("split").agg(
        imagens=("image_id", "size"),
        pacientes=("patient_id", "nunique"),
        positivas=("cancer", "sum"),
        pct_cancer=("cancer", "mean"),
    )
    resumo["pct_cancer"] = (resumo["pct_cancer"] * 100).round(2)
    rel(resumo.to_string())
    rel(f"total no manifest_fold0: {len(manifest)} imagens")

    splits_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(splits_dir / "splits.csv", index=False, lineterminator="\n")
    manifest.to_csv(splits_dir / "manifest_fold0.csv", index=False, lineterminator="\n")
    (splits_dir / "relatorio_divisao.txt").write_text(
        "\n".join(rel.linhas) + "\n", encoding="utf-8", newline="\n"
    )
    destino = gravar_sha256sums(
        splits_dir, ["splits.csv", "manifest_fold0.csv", "relatorio_divisao.txt"]
    )

    print(f"\nGravado em {splits_dir}:")
    print(destino.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
