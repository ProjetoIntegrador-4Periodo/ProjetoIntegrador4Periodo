"""Caminhos desta máquina: ``configs/local.yaml``, fora do git.

As configs versionadas (``etapa2.yaml``, ``treino.yaml``, ``pipeline_rsna.yaml``)
guardam só caminhos relativos ao repositório. Os da máquina (dados, checkpoints,
MLflow) ficam na seção de mesmo nome do ``local.yaml``, criado a partir de
``configs/local.exemplo.yaml``. Um valor na própria config vale sobre o
``local.yaml`` (é o que uma config de teste fora do repositório usa).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.utils.paths import CONFIGS_DIR

LOCAL = CONFIGS_DIR / "local.yaml"


class DaMaquina(dict):
    """Dicionário que, sem a chave, diz onde defini-la."""

    def __missing__(self, chave: str) -> Any:
        raise KeyError(
            f"{chave!r} não está definido: crie configs/local.yaml a partir de "
            "configs/local.exemplo.yaml e preencha os caminhos desta máquina"
        )


def ler_local(arquivo: Path | str = LOCAL) -> dict[str, Any]:
    """O ``local.yaml`` inteiro; vazio se ele não existir (os testes não precisam dele)."""
    arquivo = Path(arquivo)
    if not arquivo.is_file():
        return {}
    with open(arquivo, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def da_maquina(secao: str, da_config: dict[str, Any], arquivo: Path | str = LOCAL) -> DaMaquina:
    """``local.yaml[secao]`` com os valores da config por cima; vazios são ignorados."""
    local = ler_local(arquivo).get(secao) or {}
    return DaMaquina(
        {
            **{k: v for k, v in local.items() if v not in (None, "")},
            **{k: v for k, v in da_config.items() if v not in (None, "")},
        }
    )
