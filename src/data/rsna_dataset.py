"""Dataset PyTorch do RSNA sobre os PNGs da Etapa 1 (e, no realce, a pasta de realce).

Cada item passa por ``preparar_entrada`` na hora; nada é gravado em disco. Com
``representacao: realce_3canais`` e uma pasta de realce, o PNG RGB pré-calculado
substitui o cálculo (mesmo resultado, byte a byte); a máscara vem da Etapa 1.
A augmentation, quando ativa, sorteia pelo gerador do torch: o DataLoader
semeia cada worker a cada época a partir da semente do processo principal,
então ``set_seed`` antes do treino deixa as épocas reproduzíveis.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.rsna_entrada import (
    ConfigAugmentation,
    ConfigEntrada,
    ConfigEtapa2,
    conferir_pasta_realce,
    ler_pngs,
    ler_realce,
    preparar_entrada,
)

COLUNAS_NECESSARIAS = ["image_id", "png_anatomia", "png_mascara", "spacing_resultante_mm", "cancer"]


class MamografiaRSNA(Dataset):
    """
    Itens: (imagem float32 (3, altura, largura), rótulo float32 0/1).

    ``manifest`` é o ``manifest_processado.csv`` (ou um recorte dele, por
    exemplo só o split de treino); ``etapa1_dir`` é a pasta que o contém.
    ``realce_dir`` só vale para ``realce_3canais``; sem ela, o realce é
    calculado na hora.
    """

    def __init__(
        self,
        manifest: pd.DataFrame,
        etapa1_dir: Path | str,
        entrada: ConfigEntrada,
        augmentation: ConfigAugmentation | None = None,
        realce_dir: Path | str | None = None,
    ) -> None:
        faltando = set(COLUNAS_NECESSARIAS) - set(manifest.columns)
        if faltando:
            raise ValueError(f"Manifest sem as colunas {sorted(faltando)}")
        self.manifest = manifest.reset_index(drop=True)
        self.etapa1_dir = Path(etapa1_dir)
        self.entrada = entrada
        self.augmentation = augmentation if augmentation and augmentation.ativa else None
        self.realce_dir = None
        if realce_dir is not None and entrada.representacao == "realce_3canais":
            self.realce_dir = Path(realce_dir)
            conferir_pasta_realce(self.realce_dir, entrada)

    @classmethod
    def da_config(
        cls, cfg: ConfigEtapa2, manifest: pd.DataFrame, treino: bool = False
    ) -> MamografiaRSNA:
        """Pasta e representação pela config; augmentation só com ``treino=True``."""
        realce_dir = None
        if cfg.entrada.representacao == "realce_3canais":
            realce_dir = cfg.caminhos["realce_dir"]
        augmentation = replace(cfg.augmentation, ativa=True) if treino else None
        return cls(manifest, cfg.caminhos["etapa1_dir"], cfg.entrada, augmentation, realce_dir)

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, indice: int) -> tuple[torch.Tensor, torch.Tensor]:
        linha = self.manifest.iloc[indice]
        anatomia, mascara = ler_pngs(self.etapa1_dir, linha.png_anatomia, linha.png_mascara)
        representada = None
        if self.realce_dir is not None:
            representada = ler_realce(self.realce_dir, linha.image_id)
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
            representada,
        )
        return torch.from_numpy(imagem), torch.tensor(float(linha.cancer))


def iniciar_worker(_: int) -> None:
    """``worker_init_fn``: um thread de OpenCV por worker, sem disputa de núcleos."""
    cv2.setNumThreads(1)
