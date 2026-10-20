"""Monta um treino a partir das configs: nome do run, pastas, Datasets, modelo, params e tags.

Usado pelos scripts de treino e de verificação, para que o treino de verdade e
o smoke test passem pelo mesmo caminho.
"""

from __future__ import annotations

import contextlib
import json
import re
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.data.rsna_dataset import MamografiaRSNA
from src.data.rsna_entrada import ConfigEtapa2, carregar_config
from src.training.rsna_config import (
    ConfigTreino,
    caminho_do_repo,
    carregar_config_treino,
    carregar_manifest,
)
from src.training.rsna_registro import (
    commit_atual,
    gravar_json,
    gravar_yaml,
    informacoes_da_maquina,
)
from src.utils.paths import CONFIGS_DIR


def ler_alteracoes(pares: list[str]) -> dict[str, Any]:
    """``["lr=2e-4", "betas=[0.9,0.99]"]`` -> dict, com os valores lidos como YAML."""
    alteracoes = {}
    for par in pares:
        chave, sinal, valor = par.partition("=")
        if not sinal:
            raise ValueError(f"--set espera chave=valor, veio {par!r}")
        lido = yaml.safe_load(valor)
        if isinstance(lido, str):
            # O YAML 1.1 lê "2e-5" (sem ponto) como texto.
            with contextlib.suppress(ValueError):
                lido = float(lido)
        alteracoes[chave.strip()] = lido
    return alteracoes


def carregar_configs(
    treino: Path | str = CONFIGS_DIR / "treino.yaml",
    modelo: str | None = None,
    representacao: str | None = None,
    alteracoes: dict[str, Any] | None = None,
) -> tuple[ConfigTreino, ConfigEtapa2]:
    cfg = carregar_config_treino(treino)
    if modelo:
        cfg = cfg.com_modelo(modelo)
    if alteracoes:
        cfg = cfg.com_hiperparametros(alteracoes)
    cfg2 = carregar_config(
        caminho_do_repo(cfg.caminhos.get("etapa2_config", "configs/etapa2.yaml"))
    )
    if representacao:
        cfg2 = replace(cfg2, entrada=replace(cfg2.entrada, representacao=representacao))
    return cfg, cfg2


def proxima_tentativa(pasta_ajuste: Path, prefixo: str) -> int:
    numeros = [
        int(m.group(1))
        for p in pasta_ajuste.glob(f"{prefixo}_tentativa*")
        if (m := re.fullmatch(rf"{re.escape(prefixo)}_tentativa(\d+)", p.name))
    ]
    return max(numeros, default=0) + 1


def caminho_relativo(
    modo: str, cfg: ConfigTreino, cfg2: ConfigEtapa2, semente: int, tentativa: int | None
) -> Path:
    """final -> <modelo>_<rep>_seed<N>; ajuste -> ajuste/<modelo>_<rep>_tentativa<N>."""
    base = f"{cfg.modelo}_{cfg2.entrada.representacao}"
    if modo == "final":
        return Path(f"{base}_seed{semente}")
    if modo == "ajuste":
        return Path("ajuste") / f"{base}_tentativa{tentativa}"
    return Path(modo) / f"{base}_seed{semente}"


def fabrica_de_datasets(cfg2: ConfigEtapa2):
    def fabrica(manifest: pd.DataFrame, treino: bool):
        return MamografiaRSNA.da_config(cfg2, manifest, treino=treino)

    return fabrica


def dados_do_treino(cfg2: ConfigEtapa2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Treino e validação do fold 0; o teste nunca é lido aqui."""
    manifest = carregar_manifest(cfg2.caminhos["etapa1_dir"])
    return manifest[manifest.split == "train"], manifest[manifest.split == "val"]


def registrar_inicio(
    run_dir: Path,
    checkpoint_dir: Path,
    cfg: ConfigTreino,
    cfg2: ConfigEtapa2,
    modo: str,
    semente: int,
    alteracoes: dict[str, Any] | None,
    retomar: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Grava as configs usadas, a máquina e a execução; devolve (params, tags) do MLflow.

    Ao retomar, as configs gravadas no início ficam como estão e só a retomada
    é anotada em execucao.json.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    treino = {**cfg.resolvida(), "modo": modo, "semente": semente, "alteracoes": alteracoes or {}}
    etapa2 = {
        **cfg2.para_checkpoint(),
        "carregamento": cfg2.carregamento,
    }
    maquina = informacoes_da_maquina(cfg.maquina)
    versao = commit_atual()
    agora = time.strftime("%Y-%m-%d %H:%M:%S")
    if retomar and (run_dir / "execucao.json").is_file():
        execucao = json.loads((run_dir / "execucao.json").read_text(encoding="utf-8"))
        execucao.setdefault("retomadas", []).append({"quando": agora, **versao})
        gravar_json(run_dir / "execucao.json", execucao)
    else:
        gravar_yaml(run_dir / "config_treino.yaml", treino)
        gravar_yaml(run_dir / "config_etapa2.yaml", etapa2)
        gravar_json(run_dir / "maquina.json", maquina)
        gravar_json(
            run_dir / "execucao.json",
            {
                **versao,
                "checkpoint_dir": str(checkpoint_dir),
                "mlflow_tracking_uri": (
                    cfg.mlflow.get("tracking_uri") if cfg.mlflow.get("ativo") else None
                ),
                "mlflow_experimento": cfg.mlflow.get("experimento"),
                "inicio": agora,
                "comando": " ".join(sys.argv),
            },
        )
    params = {"treino": treino, "etapa2": etapa2}
    tags = {
        "modelo": cfg.modelo,
        "timm_nome": cfg.do_modelo.timm_nome,
        "representacao": cfg2.entrada.representacao,
        "seed": semente,
        "modo": modo,
        "maquina": maquina["maquina"],
        "gpu": maquina["gpu"],
        "commit": versao["commit"],
        "alteracoes_locais": versao["alteracoes_locais"],
        "batch_efetivo": cfg.do_modelo.batch_efetivo,
        "alteracoes": ", ".join(f"{k}={v}" for k, v in (alteracoes or {}).items()) or "-",
    }
    return params, tags
