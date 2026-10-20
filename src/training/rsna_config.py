"""Config de treino do RSNA (configs/treino.yaml) e a seleção dos dados sem o teste.

O treino e o ajuste só enxergam o treino e a validação do fold 0: ``carregar_manifest``
recusa o split de teste, a menos que quem chama declare ``permitir_teste=True``
(só o script de avaliação final faz isso).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.evaluation.rsna_metricas import BETAS, metricas_de_selecao, todos_os_betas
from src.utils.config_local import da_maquina, ler_local
from src.utils.paths import PROJECT_ROOT

SPLITS_DO_TREINO = ("train", "val")


@dataclass(frozen=True)
class ConfigModelo:
    timm_nome: str
    batch_real: int = 12
    acumulacao: int = 3
    otimizador: str = "adamw"
    lr: float = 1e-4
    weight_decay: float = 0.05
    betas: tuple[float, float] = (0.9, 0.999)
    momento: float = 0.9
    drop_path: float = 0.0
    warmup_epocas: float = 1.0
    scheduler: str = "cosine"
    lr_minimo: float = 1e-6
    pos_weight: float = 1.0
    negativos_por_positivo: float = 10.0
    peso_negativo_dificil: float = 2.0
    ema: bool = False
    ema_decaimento: float = 0.999

    def __post_init__(self) -> None:
        # O YAML lê "3e-4" (sem ponto) como texto: converte os campos numéricos.
        for campo in fields(self):
            valor = getattr(self, campo.name)
            if campo.type in ("int", "float") and not isinstance(valor, bool):
                tipo = int if campo.type == "int" else float
                try:
                    object.__setattr__(self, campo.name, tipo(valor))
                except (TypeError, ValueError) as erro:
                    raise ValueError(f"{campo.name}: {valor!r} não é um número") from erro
            elif campo.name == "betas":
                object.__setattr__(self, "betas", tuple(float(b) for b in valor))
        if self.otimizador not in ("adamw", "sgd"):
            raise ValueError(f"Otimizador {self.otimizador!r}; opções: adamw, sgd")
        if self.scheduler not in ("cosine", "constante"):
            raise ValueError(f"Scheduler {self.scheduler!r}; opções: cosine, constante")
        if self.batch_real < 1 or self.acumulacao < 1:
            raise ValueError("batch_real e acumulacao precisam ser >= 1")
        if not isinstance(self.ema, bool):
            raise ValueError(f"ema: {self.ema!r}; use true ou false")
        if not 0 < self.ema_decaimento <= 1:
            raise ValueError("ema_decaimento precisa estar em (0, 1]")

    @property
    def batch_efetivo(self) -> int:
        return self.batch_real * self.acumulacao


@dataclass(frozen=True)
class ConfigTreino:
    modelo: str
    modelos: dict[str, ConfigModelo]
    caminhos: dict[str, str] = field(default_factory=dict)
    mlflow: dict[str, Any] = field(default_factory=dict)
    maquina: str = ""
    orcamento: str = "passos"
    proporcao_referencia: float = 10.0
    max_epocas: int = 30
    max_passos: int | None = None
    passos_por_validacao: int | None = None
    paciencia: int = 5
    metrica_selecao: str = "f2_melhor"
    betas_f: tuple[float, ...] = BETAS
    precisao: str = "auto"
    channels_last: bool = True
    bootstrap_amostras: int = 2000
    semente_bootstrap: int = 0
    epocas_ajuste: int = 12
    max_passos_ajuste: int | None = None
    paciencia_ajuste: int = 5

    def __post_init__(self) -> None:
        if self.modelo not in self.modelos:
            raise ValueError(
                f"Modelo {self.modelo!r} sem seção em `modelos`: {sorted(self.modelos)}"
            )
        if self.precisao not in ("auto", "bf16", "fp16", "fp32"):
            raise ValueError(f"precisao {self.precisao!r}; opções: auto, bf16, fp16, fp32")
        if self.orcamento not in ("passos", "epocas"):
            raise ValueError(f"orcamento {self.orcamento!r}; opções: passos, epocas")
        object.__setattr__(self, "betas_f", todos_os_betas(self.betas_f))
        opcoes = metricas_de_selecao(self.betas_f)
        if self.metrica_selecao not in opcoes:
            raise ValueError(f"metrica_selecao {self.metrica_selecao!r}; opções: {opcoes}")
        for nome in ("max_passos", "passos_por_validacao", "max_passos_ajuste"):
            valor = getattr(self, nome)
            if valor is not None and int(valor) < 1:
                raise ValueError(f"{nome} precisa ser >= 1 (ou vazio)")

    @property
    def do_modelo(self) -> ConfigModelo:
        return self.modelos[self.modelo]

    def com_modelo(self, nome: str) -> ConfigTreino:
        return replace(self, modelo=nome)

    def com_hiperparametros(self, alteracoes: dict[str, Any]) -> ConfigTreino:
        """Troca campos da seção do modelo atual (o ``--set`` da linha de comando)."""
        validos = {f.name for f in fields(ConfigModelo)}
        desconhecidos = set(alteracoes) - validos
        if desconhecidos:
            raise ValueError(f"Hiperparâmetros desconhecidos: {sorted(desconhecidos)}")
        novo = replace(self.do_modelo, **_tuplas(alteracoes))
        return replace(self, modelos={**self.modelos, self.modelo: novo})

    def resolvida(self) -> dict[str, Any]:
        """O que este treino usou: só o modelo atual, com o batch efetivo explícito."""
        return {
            "modelo": self.modelo,
            "hiperparametros": {
                **asdict(self.do_modelo),
                "batch_efetivo": self.do_modelo.batch_efetivo,
            },
            "treino": {
                "orcamento": self.orcamento,
                "proporcao_referencia": self.proporcao_referencia,
                "max_epocas": self.max_epocas,
                "max_passos": self.max_passos,
                "passos_por_validacao": self.passos_por_validacao,
                "paciencia": self.paciencia,
                "metrica_selecao": self.metrica_selecao,
                "betas_f": list(self.betas_f),
                "precisao": self.precisao,
                "channels_last": self.channels_last,
                "bootstrap_amostras": self.bootstrap_amostras,
                "semente_bootstrap": self.semente_bootstrap,
            },
            "ajuste": {
                "epocas": self.epocas_ajuste,
                "max_passos": self.max_passos_ajuste,
                "paciencia": self.paciencia_ajuste,
            },
        }

    @classmethod
    def de_dict(cls, dados: dict[str, Any]) -> ConfigTreino:
        treino = dict(dados.get("treino", {}))
        if "betas_f" in treino:
            treino["betas_f"] = tuple(float(b) for b in treino["betas_f"])
        ajuste = dados.get("ajuste", {})
        return cls(
            modelo=dados["modelo"],
            modelos={k: ConfigModelo(**_tuplas(v)) for k, v in dados["modelos"].items()},
            caminhos=dict(dados.get("caminhos", {})),
            mlflow=dict(dados.get("mlflow", {})),
            maquina=dados.get("maquina") or "",
            epocas_ajuste=int(ajuste.get("epocas", 12)),
            max_passos_ajuste=ajuste.get("max_passos"),
            paciencia_ajuste=int(ajuste.get("paciencia", 5)),
            **treino,
        )


def _tuplas(dados: dict[str, Any]) -> dict[str, Any]:
    return {k: tuple(v) if isinstance(v, list) else v for k, v in dados.items()}


def carregar_config_treino(caminho: Path | str) -> ConfigTreino:
    """A config com os caminhos e o MLflow da máquina (configs/local.yaml)."""
    with open(caminho, encoding="utf-8") as arquivo:
        cfg = ConfigTreino.de_dict(yaml.safe_load(arquivo))
    return replace(
        cfg,
        caminhos=da_maquina("treino", cfg.caminhos),
        mlflow=da_maquina("mlflow", cfg.mlflow),
        maquina=cfg.maquina or ler_local().get("maquina") or "",
    )


def caminho_do_repo(caminho: str | Path) -> Path:
    """Caminho relativo vale a partir da raiz do repositório."""
    caminho = Path(caminho)
    return caminho if caminho.is_absolute() else PROJECT_ROOT / caminho


def carregar_manifest(
    etapa1_dir: Path | str,
    splits: tuple[str, ...] = SPLITS_DO_TREINO,
    permitir_teste: bool = False,
) -> pd.DataFrame:
    """
    ``manifest_processado.csv`` só com os splits pedidos.

    O teste hold-out só sai daqui com ``permitir_teste=True``: o treino e o
    ajuste nunca passam esse argumento.
    """
    if "test" in splits and not permitir_teste:
        raise ValueError("O teste hold-out não pode ser carregado no treino nem no ajuste")
    manifest = pd.read_csv(Path(etapa1_dir) / "manifest_processado.csv")
    manifest = manifest[(manifest.status == "ok") & manifest.split.isin(splits)]
    return manifest.reset_index(drop=True)


def recusar_teste(manifest: pd.DataFrame) -> None:
    """Última barreira antes de montar um Dataset no treino ou no ajuste."""
    if (manifest.split == "test").any():
        raise ValueError("Imagens do teste hold-out chegaram ao treino; abortando")
