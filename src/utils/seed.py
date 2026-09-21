"""Reprodutibilidade: fixa as sementes aleatórias usadas no projeto."""

from __future__ import annotations

import os
import random

DEFAULT_SEED = int(os.getenv("RANDOM_SEED", "42"))


def set_seed(seed: int = DEFAULT_SEED, *, deterministic: bool = True) -> int:
    """Fixa a semente de `random`, `numpy` e `torch` (quando disponíveis).

    Args:
        seed: valor da semente.
        deterministic: força algoritmos determinísticos no cuDNN. Deixa o
            treino mais lento, porém reproduzível entre execuções.

    Returns:
        A semente efetivamente aplicada.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    return seed
