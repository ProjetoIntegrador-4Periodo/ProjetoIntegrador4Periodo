"""Média móvel exponencial (EMA) dos pesos do modelo, atualizada a cada passo do otimizador.

``ema = d * ema + (1 - d) * modelo`` a cada passo. Durante os ``aquecimento``
primeiros passos (o warmup do learning rate), ``d = 0``: a EMA é só uma cópia
do modelo, para não guardar os pesos do ImageNet nem os do warmup. Depois, ``d``
é o ``decaimento`` pedido, e a janela efetiva é ~1 / (1 - decaimento) passos
(1.000 com 0,999, cerca de 4 épocas de referência com batch 36).

O aquecimento do TensorFlow e do timm, ``min(decaimento, (1 + t) / (10 + t))``,
só chega a 0,999 no passo ~9.000, depois do fim do ajuste (2.868 passos): o
decaimento da config não valeria. Por isso o aquecimento aqui é o do lr.

Os buffers de ponto flutuante (as estatísticas do BatchNorm da ResNet) entram
na média como os pesos; os inteiros (``num_batches_tracked``) são copiados.
"""

from __future__ import annotations

import copy
from typing import Any

import torch
from torch import nn


class ModeloEMA:
    def __init__(self, modelo: nn.Module, decaimento: float = 0.999, aquecimento: int = 0) -> None:
        self.decaimento = decaimento
        self.aquecimento = aquecimento
        self.passos = 0
        self.modelo = copy.deepcopy(modelo).eval()
        for parametro in self.modelo.parameters():
            parametro.requires_grad_(False)

    def decaimento_atual(self) -> float:
        return 0.0 if self.passos < self.aquecimento else self.decaimento

    @torch.no_grad()
    def atualizar(self, modelo: nn.Module) -> None:
        d = self.decaimento_atual()
        atual = modelo.state_dict()
        for nome, media in self.modelo.state_dict().items():
            novo = atual[nome].detach()
            if media.is_floating_point():
                media.lerp_(novo.to(media.dtype), 1 - d)
            else:
                media.copy_(novo)
        self.passos += 1

    def state_dict(self) -> dict[str, Any]:
        return {
            "modelo": self.modelo.state_dict(),
            "passos": self.passos,
            "decaimento": self.decaimento,
            "aquecimento": self.aquecimento,
        }

    def load_state_dict(self, estado: dict[str, Any]) -> None:
        self.modelo.load_state_dict(estado["modelo"])
        self.passos = int(estado["passos"])
        self.decaimento = float(estado["decaimento"])
        self.aquecimento = int(estado["aquecimento"])
