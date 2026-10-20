"""Registro de cada treino: arquivos padronizados em runs/ e o MLflow.

Os arquivos de runs/<nome>/ são a fonte que vai para o git e que o script de
comparação lê; o MLflow é a interface para navegar e comparar runs. Os dois
recebem as mesmas métricas. Checkpoints nunca sobem como artefato: o caminho
deles fica na tag ``checkpoint_dir``.
"""

from __future__ import annotations

import contextlib
import json
import os
import platform
import socket
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.utils.paths import PROJECT_ROOT

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")


def commit_atual() -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), *args], capture_output=True, text=True, check=True
        ).stdout.strip()

    try:
        return {
            "commit": git("rev-parse", "HEAD"),
            "alteracoes_locais": bool(git("status", "--porcelain", "-uno")),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"commit": "desconhecido", "alteracoes_locais": True}


def informacoes_da_maquina(nome: str = "") -> dict[str, Any]:
    import timm
    import torch

    info: dict[str, Any] = {
        "maquina": nome or socket.gethostname(),
        "sistema": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "timm": timm.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": None,
        "vram_gb": None,
        "driver": None,
    }
    if torch.cuda.is_available():
        propriedades = torch.cuda.get_device_properties(0)
        info["gpu"] = propriedades.name
        info["vram_gb"] = round(propriedades.total_memory / 2**30, 2)
        with contextlib.suppress(OSError, subprocess.CalledProcessError, IndexError):
            info["driver"] = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True, text=True, check=True,
            ).stdout.strip().splitlines()[0]  # fmt: skip
    return info


def achatar(dados: dict[str, Any], prefixo: str = "") -> dict[str, Any]:
    """{'a': {'b': 1}} -> {'a.b': 1}, para os params do MLflow."""
    plano: dict[str, Any] = {}
    for chave, valor in dados.items():
        nome = f"{prefixo}{chave}"
        if isinstance(valor, dict):
            plano.update(achatar(valor, f"{nome}."))
        else:
            plano[nome] = list(valor) if isinstance(valor, tuple) else valor
    return plano


def gravar_yaml(caminho: Path, dados: dict[str, Any]) -> None:
    with open(caminho, "w", encoding="utf-8") as arquivo:
        yaml.safe_dump(_simples(dados), arquivo, allow_unicode=True, sort_keys=False)


def gravar_json(caminho: Path, dados: dict[str, Any]) -> None:
    with open(caminho, "w", encoding="utf-8") as arquivo:
        json.dump(_simples(dados), arquivo, ensure_ascii=False, indent=2)


def _simples(valor: Any) -> Any:
    if isinstance(valor, dict):
        return {k: _simples(v) for k, v in valor.items()}
    if isinstance(valor, list | tuple):
        return [_simples(v) for v in valor]
    if hasattr(valor, "item") and not isinstance(valor, list | tuple | dict):
        valor = valor.item()  # escalares do numpy
    if isinstance(valor, float) and valor != valor:
        return None  # NaN não é JSON/YAML portável
    return valor


class Registro:
    """
    MLflow opcional (``mlflow.ativo``) por trás de uma interface mínima.

    ``run_id`` vai para o checkpoint: ao retomar um treino, as métricas
    continuam no mesmo run.
    """

    def __init__(self, cfg_mlflow: dict[str, Any]) -> None:
        self.ativo = bool(cfg_mlflow.get("ativo", False))
        self.run_id: str | None = None
        if not self.ativo:
            return
        import mlflow

        self.mlflow = mlflow
        mlflow.set_tracking_uri(cfg_mlflow["tracking_uri"])
        nome = cfg_mlflow.get("experimento", "rsna_fold0")
        if mlflow.get_experiment_by_name(nome) is None:
            artefatos = cfg_mlflow.get("artefatos_dir")
            local = Path(artefatos).resolve().as_uri() if artefatos else None
            mlflow.create_experiment(nome, artifact_location=local)
        mlflow.set_experiment(nome)

    def iniciar(
        self, nome: str, params: dict[str, Any], tags: dict[str, Any], run_id: str | None
    ) -> None:
        if not self.ativo:
            return
        if run_id:
            self.mlflow.start_run(run_id=run_id)
        else:
            self.mlflow.start_run(run_name=nome)
            planos = achatar(params)
            itens = [(k, str(v)[:6000]) for k, v in planos.items()]
            for inicio in range(0, len(itens), 100):
                self.mlflow.log_params(dict(itens[inicio : inicio + 100]))
        self.mlflow.set_tags({k: str(v) for k, v in tags.items()})
        self.run_id = self.mlflow.active_run().info.run_id

    def epoca(self, metricas: dict[str, float], passo: int) -> None:
        """Métricas de uma validação, no passo do otimizador (o eixo x no MLflow)."""
        if self.ativo:
            validas = {k: float(v) for k, v in metricas.items() if v is not None and v == v}
            self.mlflow.log_metrics(validas, step=passo)

    def resumo(self, metricas: dict[str, float]) -> None:
        if self.ativo:
            self.mlflow.log_metrics(
                {k: float(v) for k, v in metricas.items() if v is not None and v == v}
            )

    def artefatos(self, pasta: Path, subpasta: str | None = None) -> None:
        if self.ativo:
            self.mlflow.log_artifacts(str(pasta), artifact_path=subpasta)

    def tag(self, chave: str, valor: Any) -> None:
        if self.ativo:
            self.mlflow.set_tag(chave, str(valor))

    def encerrar(self, status: str = "FINISHED") -> None:
        if self.ativo and self.mlflow.active_run() is not None:
            self.mlflow.end_run(status=status)


def curvas_roc_pr(mamas: pd.DataFrame, destino: Path, titulo: str) -> None:
    """ROC e precisão-revocação por mama, num PNG para o MLflow."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, roc_curve

    fpr, tpr, _ = roc_curve(mamas.cancer, mamas.prob)
    precisao, revocacao, _ = precision_recall_curve(mamas.cancer, mamas.prob)
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4.5))
    a.plot(fpr, tpr)
    a.plot([0, 1], [0, 1], "--", color="gray")
    a.axvline(0.1, ls=":", color="gray")
    a.set(xlabel="1 - especificidade", ylabel="sensibilidade", title="ROC por mama")
    b.plot(revocacao, precisao)
    b.axhline(mamas.cancer.mean(), ls="--", color="gray")
    b.set(xlabel="revocação", ylabel="precisão", title="Precisão-revocação por mama")
    fig.suptitle(titulo)
    fig.tight_layout()
    fig.savefig(destino, dpi=100)
    plt.close(fig)
