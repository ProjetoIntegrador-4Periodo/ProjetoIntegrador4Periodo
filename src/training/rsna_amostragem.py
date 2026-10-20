"""Amostragem do treino por época: todos os positivos + negativos sorteados do pool.

O sorteio depende só de (semente do treino, época), por ``np.random.SeedSequence``:
a mesma época dá a mesma lista em qualquer máquina e depois de retomar o treino,
e sementes diferentes nunca coincidem (o que somar semente + época faria).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd


def gerador_da_epoca(semente: int, epoca: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([semente, epoca]))


def pesos_dos_negativos(negativos: pd.DataFrame, peso_dificil: float) -> np.ndarray:
    """Negativo difícil (difficult_negative_case ou biopsy) pesa ``peso_dificil``."""
    dificil = (negativos.difficult_negative_case == 1) | (negativos.biopsy == 1)
    pesos = np.where(dificil.to_numpy(), peso_dificil, 1.0)
    return pesos / pesos.sum()


def indices_da_epoca(
    treino: pd.DataFrame,
    semente: int,
    epoca: int,
    negativos_por_positivo: float,
    peso_dificil: float,
) -> np.ndarray:
    """
    Posições (em ``treino``) das imagens desta época, já embaralhadas.

    Os negativos vêm só do pool (``in_negative_pool``), sem reposição; se o pool
    for menor que o pedido, entram todos.
    """
    positivos = np.flatnonzero(treino.cancer.to_numpy() == 1)
    pool = np.flatnonzero((treino.cancer.to_numpy() == 0) & treino.in_negative_pool.to_numpy())
    rng = gerador_da_epoca(semente, epoca)
    n_negativos = min(len(pool), int(round(negativos_por_positivo * len(positivos))))
    pesos = pesos_dos_negativos(treino.iloc[pool], peso_dificil)
    negativos = rng.choice(pool, size=n_negativos, replace=False, p=pesos)
    return rng.permutation(np.concatenate([positivos, negativos]))


def indices_do_segmento(
    indices_da_epoca_n: Callable[[int], np.ndarray], segmento: int, tamanho: int
) -> np.ndarray:
    """
    As ``tamanho`` imagens do segmento ``segmento`` (o intervalo entre duas validações).

    Com o orçamento por passos, o treino é um fluxo contínuo de épocas de
    amostragem (cada uma de ``indices_da_epoca_n``), cortado em segmentos de
    tamanho fixo: o segmento k são as posições [k * tamanho, (k + 1) * tamanho)
    do fluxo. Assim, proporções diferentes dão o mesmo número de passos entre
    validações, e o segmento depende só de (semente, k), como a época.
    """
    n = len(indices_da_epoca_n(0))
    inicio = segmento * tamanho
    primeira, ultima = inicio // n, (inicio + tamanho - 1) // n
    fluxo = np.concatenate([indices_da_epoca_n(e) for e in range(primeira, ultima + 1)])
    deslocamento = inicio - primeira * n
    return fluxo[deslocamento : deslocamento + tamanho]
