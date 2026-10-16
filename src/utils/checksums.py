"""SHA-256 de arquivos, no formato do ``sha256sum``.

O ``SHA256SUMS`` gerado aqui pode ser conferido com ``sha256sum -c`` (Linux)
ou com ``python -m scripts.verificar_checksums <pasta>`` (qualquer sistema).
"""

from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

NOME_ARQUIVO = "SHA256SUMS"


def sha256(caminho: Path, bloco: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(caminho, "rb") as arquivo:
        while dados := arquivo.read(bloco):
            digest.update(dados)
    return digest.hexdigest()


def gravar_sha256sums(raiz: Path, relativos: list[str], workers: int = 8) -> Path:
    """Grava ``raiz/SHA256SUMS`` com os arquivos dados (caminhos relativos a ``raiz``)."""
    relativos = sorted(relativos)
    with ThreadPoolExecutor(workers) as pool:
        hashes = list(pool.map(lambda r: sha256(raiz / r), relativos))

    destino = raiz / NOME_ARQUIVO
    # Barra normal mesmo no Windows, para o arquivo ser idêntico em qualquer máquina.
    linhas = [f"{h}  {Path(r).as_posix()}\n" for h, r in zip(hashes, relativos, strict=True)]
    destino.write_text("".join(linhas), encoding="utf-8", newline="\n")
    return destino


def verificar_sha256sums(raiz: Path, workers: int = 8) -> list[str]:
    """Confere ``raiz/SHA256SUMS``. Devolve a lista de problemas (vazia se íntegro)."""
    entradas = []
    for linha in (raiz / NOME_ARQUIVO).read_text(encoding="utf-8").splitlines():
        esperado, relativo = linha.split("  ", 1)
        entradas.append((esperado, relativo))

    def conferir(entrada):
        esperado, relativo = entrada
        caminho = raiz / relativo
        if not caminho.is_file():
            return f"AUSENTE  {relativo}"
        if sha256(caminho) != esperado:
            return f"DIFERENTE  {relativo}"
        return None

    with ThreadPoolExecutor(workers) as pool:
        return [p for p in pool.map(conferir, entradas) if p]
