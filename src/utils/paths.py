"""Caminhos canônicos do projeto.

Centralizar os caminhos aqui evita `../../data` espalhado pelos notebooks e
garante que o código funcione igual na máquina de qualquer integrante.

Uso:
    from src.utils.paths import RAW_DIR, PROCESSED_DIR
"""

from __future__ import annotations

import os
from pathlib import Path

# Raiz do repositório: .../src/utils/paths.py -> sobe 3 níveis
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

# DATA_ROOT permite apontar para um disco externo sem mexer no código (ver .env.example)
DATA_DIR: Path = Path(os.getenv("DATA_ROOT", PROJECT_ROOT / "data")).resolve()

RAW_DIR: Path = DATA_DIR / "raw"
INTERIM_DIR: Path = DATA_DIR / "interim"
PROCESSED_DIR: Path = DATA_DIR / "processed"
EXTERNAL_DIR: Path = DATA_DIR / "external"

CONFIGS_DIR: Path = PROJECT_ROOT / "configs"
MODELS_DIR: Path = PROJECT_ROOT / "models"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"

ALL_DATA_DIRS = (RAW_DIR, INTERIM_DIR, PROCESSED_DIR, EXTERNAL_DIR)


def ensure_dirs() -> None:
    """Cria as pastas de dados e saídas caso ainda não existam."""
    for directory in (*ALL_DATA_DIRS, MODELS_DIR, FIGURES_DIR):
        directory.mkdir(parents=True, exist_ok=True)
