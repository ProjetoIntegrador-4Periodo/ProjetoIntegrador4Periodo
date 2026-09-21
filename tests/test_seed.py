"""Verifica que fixar a semente torna a geração aleatória reproduzível."""

import random

from src.utils.seed import set_seed


def test_mesma_semente_gera_mesma_sequencia():
    set_seed(123)
    primeira = [random.random() for _ in range(5)]

    set_seed(123)
    segunda = [random.random() for _ in range(5)]

    assert primeira == segunda


def test_set_seed_retorna_a_semente_aplicada():
    assert set_seed(7) == 7
