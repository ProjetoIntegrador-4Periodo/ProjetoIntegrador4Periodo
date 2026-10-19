"""Etapa 2 do pré-processamento do RSNA: PNG da Etapa 1 -> entrada do modelo.

``preparar_entrada`` é a única porta: o Dataset do treino, a geração da pasta de
realce e a inferência chamam as mesmas funções.

1. **Representação** — no tamanho da Etapa 1 (lado maior 1536), com o fundo
   zerado fora da máscara. "cinza": só a anatomia. "realce_3canais": R =
   anatomia, G = top-hat de ``tophat_mm`` (calcificação), B = CLAHE dentro da
   máscara (massa). O realce pode vir pré-calculado da pasta gerada por
   ``scripts/rsna_realce_gerar.py``; o resultado é o mesmo, byte a byte.
2. **Canvas** — reduz mantendo a proporção até caber em altura x largura e
   completa com preto. A mama fica encostada à esquerda (parede) e centrada na
   vertical. Os canais de pico (G do realce) passam por um filtro de máximo
   antes do INTER_AREA, para um ponto de calcificação não sumir na redução.
3. **Augmentation** (opcional, separada) — geometria igual em todos os canais;
   brilho e contraste só nos canais fotométricos (por padrão, só o primeiro).
   Nunca flip horizontal; o fundo continua exatamente 0.
4. **Normalização** — 3 canais em [0, 1] (o cinza é repetido) e média/desvio do
   ImageNet.

Nada aqui importa torch: o Dataset fica em ``src/data/rsna_dataset.py``.
Top-hat e CLAHE adaptados de ``src/data/preprocessamento.py`` (Davi), copiados
em vez de importados para que mudanças lá não alterem a pasta gerada.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from scipy import ndimage

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConfigRealce:
    # Diâmetro do disco do top-hat; o que for maior que isso some do canal G.
    tophat_mm: float = 0.6
    # Mediana antes do top-hat (lado em px; 1 desliga): apaga pixel isolado de ruído.
    mediana: int = 3
    # Normalização do G: mediana do tecido -> 0, este percentil do tecido -> 255.
    percentil_tophat: float = 99.9
    clahe_clip: float = 2.0
    # Grade de tiles do CLAHE (colunas, linhas), sobre a imagem da Etapa 1.
    clahe_tiles: tuple[int, int] = (8, 8)


@dataclass(frozen=True)
class ConfigEntrada:
    altura: int = 768
    largura: int = 512
    representacao: str = "cinza"
    media: tuple[float, float, float] = (0.485, 0.456, 0.406)
    desvio: tuple[float, float, float] = (0.229, 0.224, 0.225)
    realce: ConfigRealce = field(default_factory=ConfigRealce)

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
    # Canais que recebem brilho e contraste. No realce, só R: mexer em G e B
    # distorceria o que o top-hat e o CLAHE medem.
    canais_fotometricos: tuple[int, ...] = (0,)


def _tuplas(dados: dict[str, Any]) -> dict[str, Any]:
    return {k: tuple(v) if isinstance(v, list) else v for k, v in dados.items()}


@dataclass(frozen=True)
class ConfigEtapa2:
    entrada: ConfigEntrada = field(default_factory=ConfigEntrada)
    augmentation: ConfigAugmentation = field(default_factory=ConfigAugmentation)
    # Específicos da máquina: não vão para o checkpoint.
    caminhos: dict[str, str] = field(default_factory=dict)
    carregamento: dict[str, int] = field(default_factory=dict)
    geracao_realce: dict[str, int] = field(default_factory=dict)

    def para_checkpoint(self) -> dict[str, Any]:
        """O que define a entrada do modelo; salve junto com os pesos."""
        return {"entrada": asdict(self.entrada), "augmentation": asdict(self.augmentation)}

    @classmethod
    def de_dict(cls, dados: dict[str, Any]) -> ConfigEtapa2:
        """Aceita o YAML inteiro ou o dicionário salvo no checkpoint."""
        entrada = _tuplas(dados.get("entrada", {}))
        entrada["realce"] = ConfigRealce(**_tuplas(entrada.get("realce", {})))
        return cls(
            entrada=ConfigEntrada(**entrada),
            augmentation=ConfigAugmentation(**_tuplas(dados.get("augmentation", {}))),
            caminhos=dict(dados.get("caminhos", {})),
            carregamento=dict(dados.get("carregamento", {})),
            geracao_realce=dict(dados.get("geracao_realce", {})),
        )


def carregar_config(caminho: Path | str) -> ConfigEtapa2:
    with open(caminho, encoding="utf-8") as arquivo:
        return ConfigEtapa2.de_dict(yaml.safe_load(arquivo))


# ---------------------------------------------------------------------------
# 1. Representação (no tamanho da Etapa 1)
# ---------------------------------------------------------------------------


def diametro_em_pixels(milimetros: float, spacing_mm: float) -> int:
    """Tamanho físico -> lado ímpar em pixels (mínimo 3), centrado no pixel."""
    if spacing_mm <= 0:
        raise ValueError("Spacing ausente ou inválido: não dá para medir o top-hat em mm")
    lado = max(3, int(round(milimetros / spacing_mm)))
    return lado + 1 if lado % 2 == 0 else lado


def preencher_fundo(imagem: np.ndarray, dentro: np.ndarray) -> np.ndarray:
    """
    Fundo recebe o valor do pixel de tecido mais próximo.

    Assim a transição mama/fundo não vira aresta para o top-hat nem para o
    CLAHE. O Davi preenchia com a mediana do tecido, que ainda deixa um degrau
    na pele (gordura escura x mediana mais clara).
    """
    _, (linhas, colunas) = ndimage.distance_transform_edt(~dentro, return_indices=True)
    return imagem[linhas, colunas]


def normalizar_esparso(canal: np.ndarray, dentro: np.ndarray, percentil: float) -> np.ndarray:
    """
    Canal de sinal raro -> uint8: mediana do tecido vira 0, o percentil vira 255.

    O top-hat é quase todo ruído baixo com picos raros; esticar entre percentis
    baixos e altos poria o ruído no topo da escala (ver o Davi). Fora da
    máscara, 0.
    """
    valores = canal[dentro].astype(np.float32)
    saida = np.zeros(canal.shape, np.uint8)
    if valores.size == 0:
        return saida
    base = float(np.median(valores))
    alto = float(np.percentile(valores, percentil))
    if alto <= base:
        return saida
    esticado = np.clip((canal.astype(np.float32) - base) / (alto - base), 0, 1)
    saida[dentro] = np.round(esticado[dentro] * 255).astype(np.uint8)
    return saida


def canal_tophat(
    preenchida: np.ndarray, dentro: np.ndarray, spacing_mm: float, cfg: ConfigRealce
) -> np.ndarray:
    """
    Top-hat branco com disco de ``tophat_mm``: realça o que é menor que o disco.

    Na parede o tecido costuma clarear em direção à borda da imagem. Repetindo
    o último pixel (padrão do OpenCV), a abertura não reconstrói essa rampa e
    sobrava uma linha de 1 px no G ao longo da parede. A imagem é estendida por
    reflexão ímpar (``2 * borda - espelho``), que continua a rampa como rampa.
    """
    suave = cv2.medianBlur(preenchida, cfg.mediana) if cfg.mediana > 1 else preenchida
    lado = diametro_em_pixels(cfg.tophat_mm, spacing_mm)
    disco = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (lado, lado))
    margem = lado
    estendida = np.pad(suave.astype(np.float32), margem, mode="reflect", reflect_type="odd")
    topo = cv2.morphologyEx(estendida, cv2.MORPH_TOPHAT, disco, borderType=cv2.BORDER_REPLICATE)
    return normalizar_esparso(topo[margem:-margem, margem:-margem], dentro, cfg.percentil_tophat)


def canal_clahe(preenchida: np.ndarray, dentro: np.ndarray, cfg: ConfigRealce) -> np.ndarray:
    """CLAHE sobre a imagem com o fundo preenchido; fora da máscara, 0."""
    clahe = cv2.createCLAHE(clipLimit=cfg.clahe_clip, tileGridSize=tuple(cfg.clahe_tiles))
    equalizada = clahe.apply(preenchida)
    equalizada[~dentro] = 0
    return equalizada


def representar_cinza(
    anatomia: np.ndarray, dentro: np.ndarray, spacing_mm: float, cfg: ConfigEntrada
) -> np.ndarray:
    return np.where(dentro, anatomia, 0).astype(np.uint8)[:, :, None]


def representar_realce_3canais(
    anatomia: np.ndarray, dentro: np.ndarray, spacing_mm: float, cfg: ConfigEntrada
) -> np.ndarray:
    """(H, W, 3) uint8: R = anatomia, G = top-hat, B = CLAHE; fundo 0 nos três."""
    anatomia_limpa = np.where(dentro, anatomia, 0).astype(np.uint8)
    if not dentro.any():
        return np.zeros((*anatomia.shape, 3), np.uint8)
    preenchida = preencher_fundo(anatomia, dentro)
    return np.dstack(
        [
            anatomia_limpa,
            canal_tophat(preenchida, dentro, spacing_mm, cfg.realce),
            canal_clahe(preenchida, dentro, cfg.realce),
        ]
    )


@dataclass(frozen=True)
class Representacao:
    # (anatomia uint8, máscara bool, spacing_resultante_mm, config) -> (H, W, C) uint8, fundo 0
    gerar: Callable[[np.ndarray, np.ndarray, float, ConfigEntrada], np.ndarray]
    # Canais de sinal pontual: reduzidos no canvas com filtro de máximo
    canais_de_pico: tuple[int, ...] = ()


REPRESENTACOES: dict[str, Representacao] = {
    "cinza": Representacao(representar_cinza),
    "realce_3canais": Representacao(representar_realce_3canais, canais_de_pico=(1,)),
}


def representar(
    anatomia: np.ndarray, mascara: np.ndarray, spacing_mm: float, cfg: ConfigEntrada
) -> np.ndarray:
    """PNGs da Etapa 1 -> representação (H, W, C) uint8 no tamanho da Etapa 1."""
    return REPRESENTACOES[cfg.representacao].gerar(anatomia, mascara > 0, spacing_mm, cfg)


# ---------------------------------------------------------------------------
# 2. Canvas
# ---------------------------------------------------------------------------


def _com_canais(imagem: np.ndarray) -> np.ndarray:
    """O OpenCV devolve 2D quando o array tem um canal só."""
    return imagem[:, :, None] if imagem.ndim == 2 else imagem


def lado_do_maximo(escala: float) -> int:
    """
    Lado ímpar do filtro de máximo para reduzir por ``escala`` sem perder pico.

    Um bloco de ``2 / escala`` pixels cobre ao menos um pixel de saída inteiro,
    seja qual for o alinhamento; o INTER_AREA desse pixel é então o próprio pico.
    """
    lado = max(1, math.ceil(2 / escala))
    return lado + 1 if lado % 2 == 0 else lado


def montar_canvas(
    imagem: np.ndarray,
    mascara: np.ndarray,
    altura: int,
    largura: int,
    canais_de_pico: tuple[int, ...] = (),
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    (imagem uint8 2D ou (H, W, C), máscara) -> (canvas, máscara bool, escala).

    O canvas tem a mesma forma de canais da entrada. Zera fora da máscara antes
    de reduzir (um marcador não vaza pela média do INTER_AREA) e de novo
    depois, com a máscara reduzida por vizinho mais próximo, para o fundo ficar
    exatamente 0.
    """
    plana = imagem.ndim == 2
    imagem = _com_canais(imagem)
    alto, largo, _ = imagem.shape
    escala = min(altura / alto, largura / largo)
    tamanho = (
        min(largura, max(1, round(largo * escala))),
        min(altura, max(1, round(alto * escala))),
    )

    dentro = mascara > 0
    limpa = np.where(dentro[:, :, None], imagem, 0).astype(np.uint8)
    if escala < 1 and canais_de_pico:
        bloco = np.ones((lado_do_maximo(escala),) * 2, np.uint8)
        for canal in canais_de_pico:
            limpa[:, :, canal] = cv2.dilate(limpa[:, :, canal], bloco)
    interpolacao = cv2.INTER_AREA if escala < 1 else cv2.INTER_LINEAR
    reduzida = _com_canais(cv2.resize(limpa, tamanho, interpolation=interpolacao))
    mascara_reduzida = cv2.resize(dentro.astype(np.uint8), tamanho, interpolation=cv2.INTER_NEAREST)

    canvas = np.zeros((altura, largura, imagem.shape[2]), np.uint8)
    mascara_canvas = np.zeros((altura, largura), bool)
    y0 = (altura - tamanho[1]) // 2
    janela = (slice(y0, y0 + tamanho[1]), slice(0, tamanho[0]))
    mascara_canvas[janela] = mascara_reduzida > 0
    canvas[janela] = np.where(mascara_canvas[janela][:, :, None], reduzida, 0)
    return (canvas[:, :, 0] if plana else canvas), mascara_canvas, escala


# ---------------------------------------------------------------------------
# 3. Augmentation
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
) -> tuple[np.ndarray, np.ndarray]:
    """
    Augmentation no canvas 8 bits (2D ou (H, W, C)). Devolve (canvas, máscara).

    A geometria é a mesma em todos os canais e gira e escala em torno do meio
    da parede (x = 0); depois, a imagem é deslocada para o tecido voltar a
    encostar na borda esquerda, sem o vão que a rotação e a translação abrem.
    Brilho e contraste só nos ``canais_fotometricos``. Pixels que saem da
    máscara transformada voltam a 0, então o fundo segue preto mesmo com brilho
    positivo.
    """
    plana = canvas.ndim == 2
    canvas = _com_canais(canvas)
    altura, largura = mascara.shape
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
    canvas = _com_canais(
        cv2.warpAffine(
            np.ascontiguousarray(canvas), matriz, tamanho, flags=cv2.INTER_LINEAR, borderValue=0
        )
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
    canvas = canvas.copy()

    if mascara.any():
        contraste = rng.uniform(*cfg.contraste)
        brilho = rng.uniform(-cfg.brilho, cfg.brilho) * 255
        for canal in cfg.canais_fotometricos:
            if canal >= canvas.shape[2]:
                continue
            tecido = canvas[:, :, canal][mascara].astype(np.float32)
            media = float(tecido.mean())
            ajustado = np.clip((tecido - media) * contraste + media + brilho, 0, 255).round()
            canvas[:, :, canal][mascara] = ajustado.astype(np.uint8)
    canvas[~mascara] = 0
    return (canvas[:, :, 0] if plana else canvas), mascara


# ---------------------------------------------------------------------------
# 4. Normalização e imagem inteira
# ---------------------------------------------------------------------------


def normalizar(imagem: np.ndarray, cfg: ConfigEntrada) -> np.ndarray:
    """(H, W, 3) em [0, 1] -> (3, H, W) float32 com média e desvio por canal."""
    media = np.asarray(cfg.media, np.float32)
    desvio = np.asarray(cfg.desvio, np.float32)
    return np.ascontiguousarray(((imagem - media) / desvio).transpose(2, 0, 1), dtype=np.float32)


def preparar_entrada(
    anatomia: np.ndarray,
    mascara: np.ndarray,
    spacing_mm: float,
    cfg: ConfigEntrada,
    augmentation: ConfigAugmentation | None = None,
    rng: np.random.Generator | None = None,
    representada: np.ndarray | None = None,
) -> np.ndarray:
    """
    PNGs da Etapa 1 -> tensor de entrada (3, altura, largura) float32.

    ``spacing_mm`` é o ``spacing_resultante_mm`` do manifest (o top-hat é medido
    no tamanho da Etapa 1). ``representada`` é a representação já calculada (a
    pasta de realce); sem ela, é calculada aqui com ``representar``. Sem
    ``augmentation`` ativa, é determinística.
    """
    if representada is None:
        representada = representar(anatomia, mascara, spacing_mm, cfg)
    picos = REPRESENTACOES[cfg.representacao].canais_de_pico
    canvas, mascara_canvas, _ = montar_canvas(
        representada, mascara, cfg.altura, cfg.largura, picos
    )
    if augmentation is not None and augmentation.ativa:
        if rng is None:
            raise ValueError("Augmentation ativa precisa de um rng")
        canvas, mascara_canvas = aumentar(canvas, mascara_canvas, augmentation, rng)
    imagem = _com_canais(canvas).astype(np.float32) / 255
    if imagem.shape[2] == 1:
        imagem = np.repeat(imagem, 3, axis=2)
    return normalizar(imagem, cfg)


# ---------------------------------------------------------------------------
# Arquivos
# ---------------------------------------------------------------------------


def caminho_realce(image_id: int) -> str:
    """Caminho do PNG de realce, relativo à pasta de realce."""
    return f"realce/{int(image_id)}.png"


def ler_pngs(
    etapa1_dir: Path, png_anatomia: str, png_mascara: str
) -> tuple[np.ndarray, np.ndarray]:
    """Lê o par de PNGs da Etapa 1 pelos caminhos relativos do manifest."""
    anatomia = cv2.imread(str(etapa1_dir / png_anatomia), cv2.IMREAD_GRAYSCALE)
    mascara = cv2.imread(str(etapa1_dir / png_mascara), cv2.IMREAD_GRAYSCALE)
    if anatomia is None or mascara is None:
        raise FileNotFoundError(f"PNG ilegível ou ausente: {png_anatomia} / {png_mascara}")
    return anatomia, mascara


def ler_realce(realce_dir: Path, image_id: int) -> np.ndarray:
    """PNG RGB da pasta de realce -> (H, W, 3) uint8 em ordem RGB."""
    bgr = cv2.imread(str(Path(realce_dir) / caminho_realce(image_id)), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"PNG de realce ilegível ou ausente: {image_id}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def para_png(representada: np.ndarray) -> np.ndarray:
    """(H, W, 3) RGB -> o array que o ``cv2.imwrite`` grava como RGB."""
    return cv2.cvtColor(representada, cv2.COLOR_RGB2BGR)


def conferir_pasta_realce(realce_dir: Path, cfg: ConfigEntrada) -> None:
    """Falha se a pasta foi gerada com parâmetros de realce diferentes da config."""
    arquivo = Path(realce_dir) / "config_usada.yaml"
    with open(arquivo, encoding="utf-8") as f:
        usada = yaml.safe_load(f)
    esperado = asdict(cfg.realce)
    gerado = {k: list(v) if isinstance(v, tuple) else v for k, v in usada["realce"].items()}
    esperado = {k: list(v) if isinstance(v, tuple) else v for k, v in esperado.items()}
    if gerado != esperado:
        raise ValueError(
            f"A pasta {realce_dir} foi gerada com outro realce: {gerado} (config: {esperado})"
        )
