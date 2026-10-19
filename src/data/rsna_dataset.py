"""Dataset PyTorch do RSNA sobre os PNGs da Etapa 1.

Cada item passa por ``preparar_entrada`` na hora (nada é gravado em disco).
A augmentation, quando ativa, sorteia pelo gerador do torch: o DataLoader
semeia cada worker a cada época a partir da semente do processo principal,
então ``set_seed`` antes do treino deixa as épocas reproduzíveis.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.rsna_entrada import ConfigAugmentation, ConfigEntrada, ler_pngs, preparar_entrada

COLUNAS_NECESSARIAS = ["image_id", "png_anatomia", "png_mascara", "spacing_resultante_mm", "cancer"]


class MamografiaRSNA(Dataset):
    """
    Itens: (imagem float32 (3, altura, largura), rótulo float32 0/1).

    ``manifest`` é o ``manifest_processado.csv`` (ou um recorte dele, por
    exemplo só o split de treino); ``etapa1_dir`` é a pasta que o contém.
    """

    def __init__(
        self,
        manifest: pd.DataFrame,
        etapa1_dir: Path | str,
        entrada: ConfigEntrada,
        augmentation: ConfigAugmentation | None = None,
    ) -> None:
        faltando = set(COLUNAS_NECESSARIAS) - set(manifest.columns)
        if faltando:
            raise ValueError(f"Manifest sem as colunas {sorted(faltando)}")
        self.manifest = manifest.reset_index(drop=True)
        self.etapa1_dir = Path(etapa1_dir)
        self.entrada = entrada
        self.augmentation = augmentation if augmentation and augmentation.ativa else None

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, indice: int) -> tuple[torch.Tensor, torch.Tensor]:
        linha = self.manifest.iloc[indice]
        anatomia, mascara = ler_pngs(self.etapa1_dir, linha.png_anatomia, linha.png_mascara)
        rng = None
        if self.augmentation is not None:
            rng = np.random.default_rng(int(torch.randint(0, 2**62, (1,)).item()))
        imagem = preparar_entrada(
            anatomia,
            mascara,
            float(linha.spacing_resultante_mm),
            self.entrada,
            self.augmentation,
            rng,
        )
        return torch.from_numpy(imagem), torch.tensor(float(linha.cancer))


def iniciar_worker(_: int) -> None:
    """``worker_init_fn``: um thread de OpenCV por worker, sem disputa de núcleos."""
    cv2.setNumThreads(1)
