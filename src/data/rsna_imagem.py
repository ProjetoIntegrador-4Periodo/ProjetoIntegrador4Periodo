"""Etapa 1 do pré-processamento do RSNA: DICOM -> anatomia 8 bits + máscara.

Para cada imagem:

1. **Radiometria** — LUT de modalidade, depois VOI LUT ou janelamento, sem
   indexar LUT com float; inverte MONOCHROME1 e ``PixelIntensityRelationshipSign
   = -1``; confere pela saturação do ar que ele ficou escuro e, se não, inverte de novo.
2. **Máscara + crop** — silhueta contínua da mama; recorte retangular com margem,
   sem apagar pixels. Cobertura fora da faixa só marca a imagem como suspeita.
3. **Normalização** — percentis dentro da máscara e só então 8 bits.
4. **Espelhamento** — decidido pela massa de tecido em cada metade, para a
   parede torácica ficar sempre à esquerda; a tag de lateralidade só é comparada.
5. **Redimensionamento** — lado maior limitado, proporção mantida, INTER_AREA.

Trechos de radiometria e máscara adaptados de ``src/data/preprocessamento.py``
(Davi, commit 4998046), feito para o VinDr-Mammo. Copiados em vez de importados
para que mudanças lá não alterem dados já gerados por este pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from pydicom.pixels import apply_modality_lut, apply_voi_lut
from scipy import ndimage


@dataclass(frozen=True)
class Parametros:
    lado_maior: int = 1536
    margem_crop: int = 12
    percentil_baixo: float = 0.1
    percentil_alto: float = 99.9
    cobertura_min: float = 0.25
    cobertura_max: float = 0.92
    spacing_padrao_mm: float = 0.065238
    fator_mascara: int = 4


# ---------------------------------------------------------------------------
# 1. Radiometria
# ---------------------------------------------------------------------------


def aplicar_radiometria(ds: Any) -> tuple[np.ndarray, str]:
    """
    Modality LUT e VOI LUT/janelamento, mais as inversões pelas tags.

    Quando há VOI LUT e o rescale é identidade, a LUT é aplicada direto nos
    pixels inteiros: passar pela modalidade antes transformaria o array em
    float e a LUT indexada quebraria. Devolve a imagem e o caminho aplicado.
    """
    bruto = ds.pixel_array

    tem_voi_lut = "VOILUTSequence" in ds and len(ds.VOILUTSequence) > 0
    tem_janela = "WindowCenter" in ds and "WindowWidth" in ds
    tem_modality_lut = "ModalityLUTSequence" in ds
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    rescale_identidade = abs(slope - 1.0) < 1e-12 and abs(intercept) < 1e-12

    if tem_voi_lut and not tem_modality_lut and rescale_identidade:
        imagem = apply_voi_lut(bruto, ds, prefer_lut=True)
        caminho = "voi_lut"
    else:
        modalidade = apply_modality_lut(bruto, ds)
        if tem_voi_lut and not tem_janela:
            if np.issubdtype(modalidade.dtype, np.floating):
                raise ValueError("VOI LUT sobre modalidade em float e sem janelamento.")
            imagem = apply_voi_lut(modalidade, ds, prefer_lut=True)
            caminho = "modalidade+voi_lut"
        elif tem_janela:
            imagem = apply_voi_lut(modalidade, ds, prefer_lut=False)
            caminho = "modalidade+janela"
        else:
            imagem = modalidade
            caminho = "modalidade"

    imagem = np.asarray(imagem, dtype=np.float32)

    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        imagem = imagem.max() - imagem
        caminho += "+inverte_mono1"

    sinal = getattr(ds, "PixelIntensityRelationshipSign", None)
    if sinal is not None and int(sinal) == -1:
        imagem = imagem.max() - imagem
        caminho += "+inverte_sinal"

    return imagem, caminho


def fundo_escuro(imagem: np.ndarray, faixa: float = 0.02) -> bool:
    """
    Confere "ar escuro, tecido claro" pela medida, não pelas tags.

    O ar é a maior região plana da chapa e o janelamento o satura num dos
    extremos da escala. Basta ver em qual extremo há mais pixels (dentro de
    ``faixa`` da amplitude): se no escuro, a orientação está certa.

    Os cantos não servem no RSNA: a mama pode encostar na borda de cima a
    baixo (dois cantos viram tecido) e há marcadores brancos ("R MLO") num
    canto. Em 400 imagens dos dois sites, a regra dos cantos errou 5 das 10
    divergências conferidas visualmente; esta acertou as 10, com a fração
    dominante sempre >= 8,8x a outra (mediana 5000x).
    """
    baixo, alto = float(imagem.min()), float(imagem.max())
    margem = (alto - baixo) * faixa
    no_escuro = np.count_nonzero(imagem <= baixo + margem)
    no_claro = np.count_nonzero(imagem >= alto - margem)
    return no_escuro > no_claro


def garantir_fundo_escuro(imagem: np.ndarray, caminho: str) -> tuple[np.ndarray, str]:
    """
    Inverte se a checagem dos cantos falhar.

    No site 1 do RSNA, imagens MONOCHROME2 com Sign = -1 já estão na
    orientação certa: a inversão pela tag as deixa ao contrário e é esta
    checagem que desfaz.
    """
    if fundo_escuro(imagem):
        return imagem, caminho

    imagem = imagem.max() - imagem
    if not fundo_escuro(imagem):
        raise ValueError("Fundo continua claro depois de inverter; verificar o arquivo.")
    return imagem, caminho + "+corrige_fundo"


def spacing_mm(ds: Any, padrao: float) -> tuple[float, bool]:
    """Espaçamento do detector; ``(padrao, True)`` quando o DICOM não traz."""
    valor = getattr(ds, "ImagerPixelSpacing", None) or getattr(ds, "PixelSpacing", None)
    if valor:
        spacing = float(valor[0])
        if spacing > 0:
            return spacing, False
    return padrao, True


# ---------------------------------------------------------------------------
# 2. Máscara e recorte
# ---------------------------------------------------------------------------


def normalizar_percentis(
    imagem: np.ndarray, mascara: np.ndarray | None, baixo: float, alto: float
) -> np.ndarray:
    """Estica entre dois percentis (medidos na máscara, se houver) para [0, 1]."""
    valores = imagem[mascara] if mascara is not None and mascara.any() else imagem.ravel()
    lo, hi = np.percentile(valores, [baixo, alto])
    if hi <= lo:
        return np.zeros_like(imagem, dtype=np.float32)
    return np.clip((imagem - lo) / (hi - lo), 0, 1).astype(np.float32)


def fechamento_disco(mascara: np.ndarray, raio: int) -> np.ndarray:
    """
    Fechamento morfológico por disco em O(N), via transformada de distância.

    Diferente do ``ndimage.binary_closing`` (e do original do Davi), a
    erosão replica a borda em vez de tratar o lado de fora como fundo: com
    fundo lá fora, a faixa de ``raio`` pixels onde a mama encosta na borda
    (a parede torácica) era erodida e saía do crop.
    """
    dilatada = ndimage.distance_transform_edt(~mascara) <= raio
    pad = raio + 1
    preenchida = np.pad(dilatada, pad, mode="edge")
    erodida = ndimage.distance_transform_edt(preenchida) > raio
    return erodida[pad:-pad, pad:-pad]


def maior_componente(mascara: np.ndarray) -> np.ndarray:
    rotulada, quantos = ndimage.label(mascara)
    if quantos == 0:
        return mascara
    tamanhos = np.bincount(rotulada.ravel())
    tamanhos[0] = 0
    return rotulada == int(np.argmax(tamanhos))


def criar_mascara(imagem: np.ndarray, fator: int = 1) -> np.ndarray:
    """
    Silhueta inteira da mama (tecido x fundo, não denso x adiposo).

    Limiar baixo para manter a gordura, maior componente para tirar
    marcadores como "LCC", fechamento e preenchimento por linha para a
    silhueta ficar contínua.

    Com ``fator > 1`` a silhueta é calculada numa cópia reduzida e ampliada
    de volta: ela só precisa ser grossa, e a borda erra no máximo ``fator``
    pixels, menos que a margem do crop. A transformada de distância em
    resolução cheia era ~75% do tempo por imagem.
    """
    if fator > 1:
        altura, largura = imagem.shape
        reduzida = cv2.resize(
            imagem,
            (max(1, largura // fator), max(1, altura // fator)),
            interpolation=cv2.INTER_AREA,
        )
        mascara = criar_mascara(reduzida, 1).astype(np.uint8)
        return cv2.resize(mascara, (largura, altura), interpolation=cv2.INTER_NEAREST) > 0

    preliminar = normalizar_percentis(imagem, None, 1.0, 99.0)
    positivos = preliminar[preliminar > 0]
    if positivos.size < 100:
        return preliminar > 0

    limiar = float(np.clip(np.percentile(positivos, 0.5), 0.003, 0.02))
    mascara = maior_componente(preliminar > limiar)

    raio = max(5, int(round(min(mascara.shape) * 0.006)))
    mascara = fechamento_disco(mascara, raio)

    # Preenche cada linha entre o primeiro e o último pixel de tecido.
    linhas = mascara.any(axis=1)
    primeira = np.argmax(mascara, axis=1)
    ultima = mascara.shape[1] - 1 - np.argmax(mascara[:, ::-1], axis=1)
    colunas = np.arange(mascara.shape[1])
    envelope = linhas[:, None] & (colunas >= primeira[:, None]) & (colunas <= ultima[:, None])

    mascara = fechamento_disco(envelope, raio)
    mascara = ndimage.binary_fill_holes(mascara)
    return maior_componente(mascara)


def caixa_com_margem(mascara: np.ndarray, margem: int) -> tuple[int, int, int, int]:
    """Bounding box ``(y0, y1, x0, x1)`` da máscara, com margem e limitada à imagem."""
    linhas = np.flatnonzero(mascara.any(axis=1))
    colunas = np.flatnonzero(mascara.any(axis=0))
    if linhas.size == 0:
        return 0, mascara.shape[0], 0, mascara.shape[1]
    return (
        max(int(linhas[0]) - margem, 0),
        min(int(linhas[-1]) + margem + 1, mascara.shape[0]),
        max(int(colunas[0]) - margem, 0),
        min(int(colunas[-1]) + margem + 1, mascara.shape[1]),
    )


# ---------------------------------------------------------------------------
# 4 e 5. Espelhamento e redimensionamento
# ---------------------------------------------------------------------------


def parede_a_direita(mascara: np.ndarray) -> bool:
    """
    A parede torácica fica do lado com mais tecido: é onde a mama é larga.

    Compara a massa da máscara em cada metade do recorte.
    """
    meio = mascara.shape[1] // 2
    return int(mascara[:, meio:].sum()) > int(mascara[:, :meio].sum())


def redimensionar(
    imagem: np.ndarray, mascara: np.ndarray, lado_maior: int
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Limita o lado maior mantendo a proporção. Nunca amplia.

    A anatomia usa INTER_AREA; a máscara, vizinho mais próximo, para
    continuar binária.
    """
    altura, largura = imagem.shape
    escala = min(1.0, lado_maior / max(altura, largura))
    if escala == 1.0:
        return imagem, mascara, 1.0

    tamanho = (max(1, round(largura * escala)), max(1, round(altura * escala)))
    imagem = cv2.resize(imagem, tamanho, interpolation=cv2.INTER_AREA)
    mascara = cv2.resize(mascara.astype(np.uint8), tamanho, interpolation=cv2.INTER_NEAREST)
    return imagem, mascara.astype(bool), escala


# ---------------------------------------------------------------------------
# Imagem inteira
# ---------------------------------------------------------------------------


def processar(
    ds: Any, lateralidade_tag: str, p: Parametros
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """
    DICOM carregado -> (anatomia uint8, máscara uint8 0/255, metadados).

    ``lateralidade_tag`` vem do train.csv e só é usada para registrar se a
    decisão de espelhar divergiu do que a tag sugeria (R espelhada).
    """
    imagem, caminho = aplicar_radiometria(ds)
    imagem, caminho = garantir_fundo_escuro(imagem, caminho)
    spacing, estimado = spacing_mm(ds, p.spacing_padrao_mm)

    mascara = criar_mascara(imagem, p.fator_mascara)
    y0, y1, x0, x1 = caixa_com_margem(mascara, p.margem_crop)
    imagem, mascara = imagem[y0:y1, x0:x1], mascara[y0:y1, x0:x1]

    cobertura = float(mascara.mean())
    suspeita = not p.cobertura_min <= cobertura <= p.cobertura_max

    normalizada = normalizar_percentis(imagem, mascara, p.percentil_baixo, p.percentil_alto)
    anatomia = np.round(normalizada * 255).astype(np.uint8)

    espelhada = parede_a_direita(mascara)
    if espelhada:
        anatomia, mascara = anatomia[:, ::-1], mascara[:, ::-1]

    altura_crop, largura_crop = anatomia.shape
    anatomia, mascara, escala = redimensionar(
        np.ascontiguousarray(anatomia), np.ascontiguousarray(mascara), p.lado_maior
    )

    meta = {
        "caminho_radiometrico": caminho,
        "bbox_y0": y0,
        "bbox_y1": y1,
        "bbox_x0": x0,
        "bbox_x1": x1,
        "altura_original": int(ds.Rows),
        "largura_original": int(ds.Columns),
        "altura_saida": anatomia.shape[0],
        "largura_saida": anatomia.shape[1],
        "proporcao_crop": round(altura_crop / largura_crop, 4),
        "escala": round(escala, 6),
        "spacing_original_mm": spacing,
        "spacing_estimado": estimado,
        "spacing_resultante_mm": round(spacing / escala, 6),
        "cobertura_mascara": round(cobertura, 4),
        "suspeita": suspeita,
        "espelhada": espelhada,
        "diverge_tag": espelhada != (lateralidade_tag.strip().upper() == "R"),
    }
    return anatomia, (mascara.astype(np.uint8) * 255), meta
