"""Modelo do RSNA: qualquer arquitetura do timm com 1 saída (logit) e o otimizador da config.

Para um modelo novo, basta uma seção nova em ``modelos`` do configs/treino.yaml
com o ``timm_nome`` (ex.: efficientnet_b2.ra_in1k); nada aqui muda.
"""

from __future__ import annotations

import math

import timm
import torch
from torch import nn

from src.training.rsna_config import ConfigModelo


def criar_modelo(cfg: ConfigModelo, pretrained: bool = True) -> nn.Module:
    """``timm_nome`` com pesos do ImageNet (``pretrained``) e cabeça de 1 logit."""
    extras = {"drop_path_rate": cfg.drop_path} if cfg.drop_path > 0 else {}
    return timm.create_model(cfg.timm_nome, pretrained=pretrained, num_classes=1, **extras)


def grupos_de_parametros(modelo: nn.Module, weight_decay: float) -> list[dict]:
    """Sem weight decay em normas e bias (parâmetros de 1 dimensão), como nas duas receitas."""
    com, sem = [], []
    for parametro in modelo.parameters():
        if parametro.requires_grad:
            (sem if parametro.ndim <= 1 else com).append(parametro)
    return [
        {"params": com, "weight_decay": weight_decay},
        {"params": sem, "weight_decay": 0.0},
    ]


def criar_otimizador(modelo: nn.Module, cfg: ConfigModelo) -> torch.optim.Optimizer:
    grupos = grupos_de_parametros(modelo, cfg.weight_decay)
    if cfg.otimizador == "adamw":
        return torch.optim.AdamW(grupos, lr=cfg.lr, betas=cfg.betas)
    return torch.optim.SGD(grupos, lr=cfg.lr, momentum=cfg.momento, nesterov=True)


def passos_de_aquecimento(cfg: ConfigModelo, passos_por_epoca: int) -> int:
    """Passos de warmup do lr; a EMA usa os mesmos."""
    return max(1, round(cfg.warmup_epocas * passos_por_epoca)) if cfg.warmup_epocas else 0


def criar_scheduler(
    otimizador: torch.optim.Optimizer, cfg: ConfigModelo, passos_por_epoca: int, epocas: int
) -> torch.optim.lr_scheduler.LambdaLR:
    """Warmup linear de ``warmup_epocas`` e depois cosine até ``lr_minimo`` (por passo)."""
    aquecimento = passos_de_aquecimento(cfg, passos_por_epoca)
    total = max(passos_por_epoca * epocas, aquecimento + 1)
    piso = cfg.lr_minimo / cfg.lr if cfg.lr > 0 else 0.0

    def fator(passo: int) -> float:
        if passo < aquecimento:
            return (passo + 1) / aquecimento
        if cfg.scheduler == "constante":
            return 1.0
        progresso = min(1.0, (passo - aquecimento) / (total - aquecimento))
        return piso + (1 - piso) * 0.5 * (1 + math.cos(math.pi * progresso))

    return torch.optim.lr_scheduler.LambdaLR(otimizador, fator)
