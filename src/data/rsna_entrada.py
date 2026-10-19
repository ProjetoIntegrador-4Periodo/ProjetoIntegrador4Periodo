"""Etapa 2 do pré-processamento do RSNA: PNG da Etapa 1 -> entrada do modelo.

Roda em tempo de carregamento, sem gravar nada em disco. ``preparar_entrada`` é
a única porta: o Dataset do treino e a inferência chamam a mesma função.

1. **Canvas** — zera tudo fora da máscara (some com marcadores como "L MLO"),
   reduz mantendo a proporção até caber em altura x largura e completa com
   preto. A mama fica encostada à esquerda (parede) e centrada na vertical.
2. **Augmentation** (opcional, separada) — no canvas 8 bits, antes da
   representação: flip vertical, rotação, escala, translação, brilho e
   contraste. Nunca flip horizontal; o fundo continua exatamente 0.
3. **Representação** — 3 canais em [0, 1] e normalização do ImageNet. Hoje só
   "cinza"; "realce_3canais" é o ponto de extensão (precisa do spacing do canvas
   para o top-hat em mm).

Nada aqui importa torch: o Dataset fica em ``src/data/rsna_dataset.py``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConfigEntrada:
    altura: int = 768
    largura: int = 512
    representacao: str = "cinza"
    media: tuple[float, float, float] = (0.485, 0.456, 0.406)
    desvio: tuple[float, float, float] = (0.229, 0.224, 0.225)
    tophat_mm: float = 0.6

    def __post_init__(self) -> None:
        if self.representacao not in REPRESENTACOES:
            raise ValueError(
                f"Representação {self.representacao!r} desconhecida; "
                f"opções: {sorted(REPRESENTACOES)}"
            )


@dataclass(frozen=True)
class ConfigAugmentation:
    ativa: bool = False
    prob_flip_vertical: float = 0.5
    rotacao_graus: float = 10.0
    escala: tuple[float, float] = (0.9, 1.1)
    translacao: float = 0.05
    brilho: float = 0.1
    contraste: tuple[float, float] = (0.9, 1.1)


@dataclass(frozen=True)
class ConfigEtapa2:
    entrada: ConfigEntrada = field(default_factory=ConfigEntrada)
    augmentation: ConfigAugmentation = field(default_factory=ConfigAugmentation)
    # Específicos da máquina: não vão para o checkpoint.
    caminhos: dict[str, str] = field(default_factory=dict)
    carregamento: dict[str, int] = field(default_factory=dict)

    def para_checkpoint(self) -> dict[str, Any]:
        """O que define a entrada do modelo; salve junto com os pesos."""
        return {"entrada": asdict(self.entrada), "augmentation": asdict(self.augmentation)}

    @classmethod
    def de_dict(cls, dados: dict[str, Any]) -> ConfigEtapa2:
        """Aceita o YAML inteiro ou o dicionário salvo no checkpoint."""
        entrada = {
            k: tuple(v) if isinstance(v, list) else v for k, v in dados.get("entrada", {}).items()
        }
        aug = {
            k: tuple(v) if isinstance(v, list) else v
            for k, v in dados.get("augmentation", {}).items()
        }
        return cls(
            entrada=ConfigEntrada(**entrada),
            augmentation=ConfigAugmentation(**aug),
            caminhos=dict(dados.get("caminhos", {})),
            carregamento=dict(dados.get("carregamento", {})),
        )


def carregar_config(caminho: Path | str) -> ConfigEtapa2:
    with open(caminho, encoding="utf-8") as arquivo:
        return ConfigEtapa2.de_dict(yaml.safe_load(arquivo))


# ---------------------------------------------------------------------------
# 1. Canvas
# ---------------------------------------------------------------------------


def montar_canvas(
    anatomia: np.ndarray, mascara: np.ndarray, altura: int, largura: int
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    (anatomia uint8, máscara) da Etapa 1 -> (canvas uint8, máscara bool, escala).

    Zera fora da máscara antes de reduzir (um marcador não vaza pela média do
    INTER_AREA) e de novo depois, com a máscara reduzida por vizinho mais
    próximo, para o fundo ficar exatamente 0.
    """
    alto, largo = anatomia.shape
    escala = min(altura / alto, largura / largo)
    tamanho = (
        min(largura, max(1, round(largo * escala))),
        min(altura, max(1, round(alto * escala))),
    )

    dentro = mascara > 0
    limpa = np.where(dentro, anatomia, 0).astype(np.uint8)
    interpolacao = cv2.INTER_AREA if escala < 1 else cv2.INTER_LINEAR
    reduzida = cv2.resize(limpa, tamanho, interpolation=interpolacao)
    mascara_reduzida = cv2.resize(dentro.astype(np.uint8), tamanho, interpolation=cv2.INTER_NEAREST)

    canvas = np.zeros((altura, largura), np.uint8)
    mascara_canvas = np.zeros((altura, largura), bool)
    y0 = (altura - tamanho[1]) // 2
    janela = (slice(y0, y0 + tamanho[1]), slice(0, tamanho[0]))
    mascara_canvas[janela] = mascara_reduzida > 0
    canvas[janela] = np.where(mascara_canvas[janela], reduzida, 0)
    return canvas, mascara_canvas, escala


# ---------------------------------------------------------------------------
# 2. Augmentation
# ---------------------------------------------------------------------------


def encostar_a_esquerda(canvas: np.ndarray, mascara: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Desloca para a esquerda até o tecido tocar a borda (fecha o vão da rotação)."""
    colunas = np.flatnonzero(mascara.any(axis=0))
    if len(colunas) == 0 or colunas[0] == 0:
        return canvas, mascara
    vao = int(colunas[0])
    deslocado, mascara_deslocada = np.zeros_like(canvas), np.zeros_like(mascara)
    deslocado[:, :-vao], mascara_deslocada[:, :-vao] = canvas[:, vao:], mascara[:, vao:]
    return deslocado, mascara_deslocada


def aumentar(
    canvas: np.ndarray, mascara: np.ndarray, cfg: ConfigAugmentation, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Augmentation no canvas 8 bits. Devolve (canvas, máscara, fator de escala).

    A geometria gira e escala em torno do meio da parede (x = 0); depois, a
    imagem é deslocada para o tecido voltar a encostar na borda esquerda, sem o
    vão que a rotação e a translação abrem. Pixels que saem da máscara transformada
    voltam a 0, então o fundo segue preto mesmo com brilho positivo.
    """
    altura, largura = canvas.shape
    if rng.random() < cfg.prob_flip_vertical:
        canvas, mascara = canvas[::-1], mascara[::-1]

    linhas = np.flatnonzero(mascara.any(axis=1))
    centro_y = float(linhas.mean()) if len(linhas) else altura / 2
    angulo = rng.uniform(-cfg.rotacao_graus, cfg.rotacao_graus)
    fator = rng.uniform(*cfg.escala)
    matriz = cv2.getRotationMatrix2D((0.0, centro_y), angulo, fator)
    matriz[0, 2] += rng.uniform(-cfg.translacao, cfg.translacao) * largura
    matriz[1, 2] += rng.uniform(-cfg.translacao, cfg.translacao) * altura

    tamanho = (largura, altura)
    canvas = cv2.warpAffine(
        np.ascontiguousarray(canvas), matriz, tamanho, flags=cv2.INTER_LINEAR, borderValue=0
    )
    mascara = (
        cv2.warpAffine(
            np.ascontiguousarray(mascara, dtype=np.uint8),
            matriz,
            tamanho,
            flags=cv2.INTER_NEAREST,
            borderValue=0,
        )
        > 0
    )
    canvas, mascara = encostar_a_esquerda(canvas, mascara)

    if mascara.any():
        tecido = canvas[mascara].astype(np.float32)
        media = float(tecido.mean())
        contraste = rng.uniform(*cfg.contraste)
        brilho = rng.uniform(-cfg.brilho, cfg.brilho) * 255
        canvas = np.zeros_like(canvas)
        canvas[mascara] = np.clip((tecido - media) * contraste + media + brilho, 0, 255).round()
    else:
        canvas = np.zeros_like(canvas)
    return canvas, mascara, fator


# ---------------------------------------------------------------------------
# 3. Representação
# ---------------------------------------------------------------------------

# canvas uint8, máscara bool, spacing do canvas em mm, config -> (H, W, 3) float32 em [0, 1]
Representacao = Callable[[np.ndarray, np.ndarray, float, ConfigEntrada], np.ndarray]


def representar_cinza(
    canvas: np.ndarray, mascara: np.ndarray, spacing_mm: float, cfg: ConfigEntrada
) -> np.ndarray:
    cinza = canvas.astype(np.float32) / 255
    return np.repeat(cinza[:, :, None], 3, axis=2)


def representar_realce_3canais(
    canvas: np.ndarray, mascara: np.ndarray, spacing_mm: float, cfg: ConfigEntrada
) -> np.ndarray:
    # Ponto de extensão: canal 0 = anatomia; canal 1 = top-hat branco com raio
    # cfg.tophat_mm / spacing_mm pixels; canal 2 = CLAHE só dentro da máscara.
    raise NotImplementedError("Representação 'realce_3canais' ainda não implementada")


REPRESENTACOES: dict[str, Representacao] = {
    "cinza": representar_cinza,
    "realce_3canais": representar_realce_3canais,
}


def normalizar(imagem: np.ndarray, cfg: ConfigEntrada) -> np.ndarray:
    """(H, W, 3) em [0, 1] -> (3, H, W) float32 com média e desvio por canal."""
    media = np.asarray(cfg.media, np.float32)
    desvio = np.asarray(cfg.desvio, np.float32)
    return np.ascontiguousarray(((imagem - media) / desvio).transpose(2, 0, 1), dtype=np.float32)


# ---------------------------------------------------------------------------
# Imagem inteira
# ---------------------------------------------------------------------------


def preparar_entrada(
    anatomia: np.ndarray,
    mascara: np.ndarray,
    spacing_mm: float,
    cfg: ConfigEntrada,
    augmentation: ConfigAugmentation | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    PNGs da Etapa 1 -> tensor de entrada (3, altura, largura) float32.

    ``spacing_mm`` é o ``spacing_resultante_mm`` do manifest; aqui ele é
    convertido para o canvas (e para a escala da augmentation), que é o que o
    top-hat do realce vai precisar. Sem ``augmentation`` ativa, é determinística.
    """
    canvas, mascara_canvas, escala = montar_canvas(anatomia, mascara, cfg.altura, cfg.largura)
    spacing_canvas = spacing_mm / escala
    if augmentation is not None and augmentation.ativa:
        if rng is None:
            raise ValueError("Augmentation ativa precisa de um rng")
        canvas, mascara_canvas, fator = aumentar(canvas, mascara_canvas, augmentation, rng)
        spacing_canvas /= fator
    imagem = REPRESENTACOES[cfg.representacao](canvas, mascara_canvas, spacing_canvas, cfg)
    return normalizar(imagem, cfg)


def ler_pngs(
    etapa1_dir: Path, png_anatomia: str, png_mascara: str
) -> tuple[np.ndarray, np.ndarray]:
    """Lê o par de PNGs da Etapa 1 pelos caminhos relativos do manifest."""
    anatomia = cv2.imread(str(etapa1_dir / png_anatomia), cv2.IMREAD_GRAYSCALE)
    mascara = cv2.imread(str(etapa1_dir / png_mascara), cv2.IMREAD_GRAYSCALE)
    if anatomia is None or mascara is None:
        raise FileNotFoundError(f"PNG ilegível ou ausente: {png_anatomia} / {png_mascara}")
    return anatomia, mascara
