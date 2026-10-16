"""Confere o SHA256SUMS de uma pasta (CSVs da divisão ou PNGs entregues por HD).

Funciona em qualquer sistema, inclusive Windows (onde não há ``sha256sum -c``):

    python -m scripts.verificar_checksums splits/rsna_fold0
    python -m scripts.verificar_checksums /caminho/do/hd/rsna_fold0_etapa1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.utils.checksums import NOME_ARQUIVO, verificar_sha256sums


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("pasta", type=Path, help=f"pasta que contém o {NOME_ARQUIVO}")
    args = parser.parse_args()

    total = len((args.pasta / NOME_ARQUIVO).read_text(encoding="utf-8").splitlines())
    problemas = verificar_sha256sums(args.pasta)

    for problema in problemas[:50]:
        print(problema)
    if problemas:
        print(f"\nFALHOU: {len(problemas)} de {total} arquivos com problema.")
        sys.exit(1)
    print(f"OK: {total} arquivos conferidos, todos íntegros.")


if __name__ == "__main__":
    main()
