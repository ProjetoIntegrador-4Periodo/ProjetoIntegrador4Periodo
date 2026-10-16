"""Pré-processamento de mamografias: DICOM → imagem normalizada → realce.

O caminho de cada imagem é:

1. **Radiometria** — LUT de modalidade, VOI LUT ou janelamento, na ordem certa
   para não indexar uma LUT com float; correção de MONOCHROME1 e do sinal de
   intensidade; e, por último, uma checagem medida de que o fundo ficou escuro.
2. **Máscara da mama + crop** — silhueta contínua do tecido, recorte com margem.
3. **Normalização** — percentis 0,1 e 99,9 dentro da máscara, fundo zerado.
4. **Realce** — 3 canais: ``[anatomia, top-hat em mm, CLAHE na máscara]``.
5. **Geometria** — espelha as mamas direitas e coloca tudo num canvas fixo,
   sem esticar nem cortar tecido.

O lote (leitura do manifesto, PNGs de saída e relatório) fica em
``scripts/preprocessar_mamografia.py``.

Uso:
    from src.data.preprocessamento import read_dicom, create_representation

    image, mask, meta = read_dicom(path)
    rep = create_representation(image, mask, "D", 0.6, 0.01, meta["pixel_spacing_mm"])
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pydicom
from PIL import Image
from pydicom.pixels import apply_modality_lut, apply_voi_lut
from scipy import ndimage
from skimage import exposure


def mm_to_pixels(millimeters: float, pixel_spacing: float) -> int:
    """
    Converte um tamanho físico para pixels naquela imagem.

    O elemento estruturante precisa medir milímetros de tecido, não
    pixels: `ImagerPixelSpacing` varia de ~0,05 mm (Hologic) a ~0,1 mm
    (algumas Siemens), então um SE fixo em pixels significaria uma
    coisa diferente em cada aparelho.
    """
    if pixel_spacing <= 0:
        raise ValueError(
            "Espaçamento de pixel ausente ou inválido: não dá para "
            "dimensionar o elemento estruturante em milímetros."
        )

    size = max(3, int(round(millimeters / pixel_spacing)))

    # Ímpar mantém o elemento centrado no pixel.
    if size % 2 == 0:
        size += 1

    return size


def robust_normalize(
    image: np.ndarray,
    mask: np.ndarray | None = None,
    low_percentile: float = 1.0,
    high_percentile: float = 99.0,
) -> np.ndarray:
    """Estica a imagem entre dois percentis (calculados na máscara) para [0, 1]."""
    image = np.asarray(image, dtype=np.float32)

    if mask is None:
        values = image[np.isfinite(image)]
    else:
        values = image[mask & np.isfinite(image)]

    if values.size == 0:
        return np.zeros_like(image, dtype=np.float32)

    low, high = np.percentile(values, [low_percentile, high_percentile])

    if high <= low:
        return np.zeros_like(image, dtype=np.float32)

    normalized = (image - low) / (high - low)

    return np.clip(normalized, 0, 1).astype(np.float32)


def disk_closing(mask: np.ndarray, radius: int) -> np.ndarray:
    """
    Fechamento morfológico por disco, via transformada de distância.

    `binary_closing` com um disco de raio 17 custa O(N x K), com K os
    1225 pixels do elemento — 92% do tempo do pipeline ficava aqui.
    A transformada de distância euclidiana resolve o mesmo em O(N),
    independente do raio:

        dilatação por disco de raio r  ==  edt(~mask) <= r
        erosão   por disco de raio r  ==  edt(mask)  >  r

    A dilatação roda sem padding, porque o `border_value=0` do scipy
    corta na borda do array. A erosão roda com padding False, que é o
    que replica o mesmo `border_value=0` tratando o lado de fora como
    fundo. Verificado bit a bit contra `ndimage.binary_closing`.
    """
    dilated = ndimage.distance_transform_edt(~mask) <= radius

    pad = radius + 1
    padded = np.pad(dilated, pad, mode="constant", constant_values=False)

    eroded = ndimage.distance_transform_edt(padded) > radius

    return eroded[pad:-pad, pad:-pad]


def largest_component(mask: np.ndarray) -> np.ndarray:
    """Mantém só o maior componente conexo da máscara."""
    labeled, count = ndimage.label(mask)

    if count == 0:
        return mask

    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0

    return labeled == int(np.argmax(sizes))


def create_breast_mask(image: np.ndarray) -> np.ndarray:
    """
    Cria uma máscara da silhueta inteira da mama.

    O objetivo aqui é separar tecido de fundo, não separar
    tecido denso de tecido adiposo. Por isso, usa-se um
    threshold baixo após normalização robusta.
    """
    preliminary = robust_normalize(image)

    positive = preliminary[preliminary > 0]

    if positive.size < 100:
        return (preliminary > 0).astype(bool)

    # Threshold propositalmente baixo para preservar
    # regiões adiposas e pouco intensas.
    threshold = float(np.percentile(positive, 0.5))
    threshold = float(np.clip(threshold, 0.003, 0.02))

    mask = preliminary > threshold

    # Remove ruídos e marcadores isolados.
    mask = largest_component(mask)

    radius = max(5, int(round(min(mask.shape) * 0.006)))

    mask = disk_closing(mask, radius)

    # Cria uma silhueta contínua por linha. Isso evita
    # excluir tecido adiposo ligado ao fundo por canais
    # estreitos.
    envelope = mask.copy()

    for row_index in range(envelope.shape[0]):
        columns = np.flatnonzero(envelope[row_index])

        if columns.size >= 2:
            envelope[row_index, columns[0] : columns[-1] + 1] = True

    mask = disk_closing(envelope, radius)
    mask = ndimage.binary_fill_holes(mask)
    mask = largest_component(mask)

    return mask.astype(bool)


def crop_to_mask(
    image: np.ndarray,
    mask: np.ndarray,
    margin: int = 12,
) -> tuple[np.ndarray, np.ndarray]:
    """Recorta imagem e máscara na caixa envolvente do tecido, com margem."""
    rows, columns = np.where(mask)

    if rows.size == 0:
        return image, mask

    y0 = max(int(rows.min()) - margin, 0)
    y1 = min(int(rows.max()) + margin + 1, image.shape[0])
    x0 = max(int(columns.min()) - margin, 0)
    x1 = min(int(columns.max()) + margin + 1, image.shape[1])

    return image[y0:y1, x0:x1], mask[y0:y1, x0:x1]


def apply_radiometric_transform(ds: Any) -> tuple[np.ndarray, str]:
    """
    Aplica o pipeline radiométrico DICOM sem enviar
    arrays float para uma VOI LUT indexada.

    Para o VinDr atualmente auditado:
    - VOI LUT + rescale identidade:
      LUT diretamente sobre os pixels armazenados.
    - Sem VOI LUT:
      modalidade seguida de windowing.

    Devolve a imagem em float32 e o caminho radiométrico escolhido, que vai
    para o relatório para dar para auditar imagem a imagem.
    """
    raw = ds.pixel_array

    has_voi_lut = "VOILUTSequence" in ds and len(ds.VOILUTSequence) > 0
    has_window = "WindowCenter" in ds and "WindowWidth" in ds
    has_modality_lut = "ModalityLUTSequence" in ds

    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))

    identity_rescale = abs(slope - 1.0) < 1e-12 and abs(intercept) < 1e-12

    if has_voi_lut and not has_modality_lut and identity_rescale:
        # O rescale não modificaria os valores.
        # Mantemos o dtype inteiro para indexar a LUT.
        image = apply_voi_lut(raw, ds, prefer_lut=True)
        radiometric_path = "voi_lut_on_raw_identity_rescale"

    else:
        modality = apply_modality_lut(raw, ds)

        if has_voi_lut and not has_window:
            if np.issubdtype(modality.dtype, np.floating):
                raise ValueError(
                    "VOI LUT com modalidade não identidade "
                    "e sem WindowCenter/WindowWidth. "
                    "O caso exige tratamento específico."
                )

            image = apply_voi_lut(modality, ds, prefer_lut=True)
            radiometric_path = "voi_lut_after_modality"

        elif has_window:
            # prefer_lut=False força WindowCenter/Width.
            image = apply_voi_lut(modality, ds, prefer_lut=False)
            radiometric_path = "windowing_after_modality"

        else:
            image = modality
            radiometric_path = "modality_only"

    image = np.asarray(image, dtype=np.float32)

    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        image = image.max() - image
        radiometric_path += "+inverte_mono1"

    # PixelIntensityRelationshipSign = -1 significa que valor menor
    # corresponde a maior atenuação, ou seja, a escala está invertida
    # em relação à convenção que o resto do pipeline assume.
    sign = getattr(ds, "PixelIntensityRelationshipSign", None)

    if sign is not None and int(sign) == -1:
        image = image.max() - image
        radiometric_path += "+inverte_sinal"

    return image, radiometric_path


def background_is_dark(image: np.ndarray) -> bool:
    """
    Confere a convenção "tecido claro, ar escuro".

    Vale como rede de segurança independente das tags: se alguma
    combinação de MONOCHROME1, LUT e sinal de intensidade inverter a
    escala sem que as tags digam, os cantos da chapa (que são ar)
    ficariam mais claros que a mediana da imagem.
    """
    height, width = image.shape

    janela_y = max(1, height // 50)
    janela_x = max(1, width // 50)

    cantos = np.concatenate(
        [
            image[:janela_y, :janela_x].ravel(),
            image[:janela_y, -janela_x:].ravel(),
            image[-janela_y:, :janela_x].ravel(),
            image[-janela_y:, -janela_x:].ravel(),
        ]
    )

    # Comparar com a mediana da imagem não serve: a mama costuma
    # ocupar só um quarto da chapa, então a mediana global já é fundo
    # e o teste viraria zero contra zero. O p90 é sempre tecido, e o
    # ar tem que ser bem mais escuro que ele.
    #
    # Medido em 1500+ imagens: nas corretas essa razão vai de 0,00 a
    # 0,29; nas invertidas, de 0,95 a 1,00. O limiar de 0,5 fica no
    # meio de um vão enorme.
    tecido = float(np.percentile(image, 90))

    if tecido <= 0:
        return True

    return float(np.median(cantos)) / tecido < 0.5


def dicom_metadata(ds: Any, radiometric_path: str) -> dict[str, Any]:
    """Extrai do cabeçalho o que o pipeline e o relatório precisam."""
    return {
        "manufacturer": str(getattr(ds, "Manufacturer", "")),
        "manufacturer_model": str(getattr(ds, "ManufacturerModelName", "")),
        # ImagerPixelSpacing é o espaçamento no detector, que é o que
        # dimensiona o elemento estruturante em milímetros.
        "pixel_spacing_mm": float(
            getattr(ds, "ImagerPixelSpacing", getattr(ds, "PixelSpacing", [0.0]))[0]
        ),
        "intensity_sign": str(getattr(ds, "PixelIntensityRelationshipSign", "")),
        "intensity_relationship": str(getattr(ds, "PixelIntensityRelationship", "")),
        "original_rows": int(getattr(ds, "Rows", 0)),
        "original_columns": int(getattr(ds, "Columns", 0)),
        # Vindo do próprio DICOM, o pipeline não depende do manifesto
        # para saber se precisa espelhar a mama.
        "laterality": str(getattr(ds, "ImageLaterality", getattr(ds, "Laterality", "")))
        .strip()
        .upper(),
        "view_position": str(getattr(ds, "ViewPosition", "")).strip().upper(),
        "radiometric_path": radiometric_path,
        "has_voi_lut": bool("VOILUTSequence" in ds),
        "has_window": bool("WindowCenter" in ds and "WindowWidth" in ds),
        "rescale_slope": float(getattr(ds, "RescaleSlope", 1.0)),
        "rescale_intercept": float(getattr(ds, "RescaleIntercept", 0.0)),
    }


def preprocess_dataset(ds: Any) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """
    Leva um DICOM já carregado até a imagem cropada e normalizada.

    Devolve a imagem em float32 no intervalo [0, 1], com o fundo zerado,
    a máscara da mama no mesmo recorte e os metadados do cabeçalho.
    """
    image, radiometric_path = apply_radiometric_transform(ds)

    # As tags nem sempre bastam: há Planmed MONOCHROME1 em que a VOI
    # LUT já entrega a imagem na orientação certa, e a inversão por
    # PhotometricInterpretation acaba invertendo duas vezes. Em vez de
    # enumerar combinações de fabricante, decide-se pela medida.
    if not background_is_dark(image):
        image = image.max() - image
        radiometric_path += "+corrige_orientacao"

        if not background_is_dark(image):
            raise ValueError(
                "O fundo continua claro mesmo depois de inverter. "
                "A imagem não segue a convenção de ar escuro e "
                "tecido claro; verifique o arquivo."
            )

    mask = create_breast_mask(image)
    image, mask = crop_to_mask(image, mask)

    # Uma máscara sã cobre entre 25% e 92% do recorte: medido em 1500+
    # imagens, a faixa real vai de 49% a 90%. Fora disso ela virou um
    # retângulo (pegou a chapa inteira) ou colapsou. Isso acontece em
    # imagens cuja escala de intensidade não segue a convenção, e é
    # melhor recusar do que emitir tensor errado sem avisar.
    tissue_fraction = float(mask.mean())

    if not 0.25 <= tissue_fraction <= 0.92:
        raise ValueError(
            "Máscara implausível: cobre "
            f"{100 * tissue_fraction:.0f}% do recorte, fora da faixa "
            "de 25% a 92% observada em imagens sãs. Provavelmente "
            "virou um retângulo em vez da silhueta da mama."
        )

    # Percentis 0,1 e 99,9 em vez de 1 e 99: a cauda alta é onde moram
    # as calcificações. Medido, cortar em 99 achatava 4,3% dos pixels
    # de calcificação contra 1% de todo o tecido.
    image = robust_normalize(image, mask, low_percentile=0.1, high_percentile=99.9)
    image[~mask] = 0

    return image, mask, dicom_metadata(ds, radiometric_path)


def read_dicom(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Lê o arquivo DICOM e aplica `preprocess_dataset`."""
    return preprocess_dataset(pydicom.dcmread(path))


def normalize_sparse_channel(
    channel: np.ndarray,
    mask: np.ndarray,
    percentile: float = 99.9,
) -> np.ndarray:
    """
    Normaliza um canal cujo sinal é esparso.

    A saída do top-hat é quase toda zero, com picos raros. Esticar
    entre os percentis 1 e 99 coloca o topo da escala dentro do
    ruído: medido nas 100 imagens, isso deixava ~15% do tecido
    saturado em 1,0, misturando ruído do detector com calcificação
    verdadeira.

    Aqui o zero fica na mediana (o nível do ruído) e a escala vem de
    um percentil alto, onde o ruído já acabou. O tecido ativo cai
    para ~1,5%, que é a ordem de grandeza de calcificação real.
    """
    values = channel[mask]

    if values.size == 0:
        return np.zeros_like(channel, dtype=np.float32)

    base = float(np.median(values))
    high = float(np.percentile(values, percentile))

    if high <= base:
        return np.zeros_like(channel, dtype=np.float32)

    normalized = np.clip((channel - base) / (high - base), 0, 1).astype(np.float32)
    normalized[~mask] = 0

    return normalized


def enhance_large_scale(
    image: np.ndarray,
    mask: np.ndarray,
    clip_limit: float,
) -> np.ndarray:
    """
    Canal de escala grande, sensível a massa e espiculação.

    O top-hat é cego a massa por construção: uma massa de centímetros
    é ordens de grandeza maior que o elemento estruturante. Medido na
    base, o antigo canal de 1,3 mm respondia 0,019 na massa e tinha
    correlação 0,753 com o canal de calcificação — quase redundante.

    O CLAHE equaliza contraste local e responde 0,871 na mesma massa,
    com correlação 0,275. É aplicado **dentro da máscara**: se rodasse
    na imagem inteira, a transição mama/fundo viraria uma borda de
    contraste altíssimo, que o modelo aprenderia a usar como atalho.
    """
    inside = image.copy()

    if mask.any():
        # Preenche o fundo com a mediana do tecido para o CLAHE não
        # enxergar a borda da mama como uma aresta.
        inside[~mask] = float(np.median(image[mask]))

    equalized = exposure.equalize_adapthist(
        np.clip(inside, 0, 1),
        kernel_size=(max(8, image.shape[0] // 8), max(8, image.shape[1] // 8)),
        clip_limit=clip_limit,
    )

    equalized = np.asarray(equalized, dtype=np.float32)
    equalized[~mask] = 0

    return equalized


def create_representation(
    image: np.ndarray,
    mask: np.ndarray,
    representation: str,
    tophat_mm: float,
    clahe_clip: float,
    pixel_spacing: float,
    percentile: float = 99.9,
    denoise: bool = True,
) -> np.ndarray:
    """
    Monta a entrada de 3 canais do modelo.

    - ``"A1"``: a anatomia repetida nos três canais.
    - ``"D"``: ``[anatomia, top-hat em mm (calcificação), CLAHE (massa)]``.
    """
    if representation == "A1":
        return np.stack([image, image, image], axis=-1)

    if representation == "D":
        source = image

        if denoise:
            # Mediana 3x3 apaga pixel isolado de ruído do detector.
            # Uma microcalcificação tem 3 a 6 px no espaçamento de
            # 0,085 mm e atravessa o filtro intacta.
            source = ndimage.median_filter(image, size=3)

        # O elemento estruturante é dimensionado em milímetros de
        # tecido e convertido para pixels nesta imagem.
        tophat_size = mm_to_pixels(tophat_mm, pixel_spacing)

        calcium = ndimage.white_tophat(source, size=(tophat_size, tophat_size))
        calcium = normalize_sparse_channel(calcium, mask, percentile)

        large_scale = enhance_large_scale(image, mask, clahe_clip)

        return np.stack([image, calcium, large_scale], axis=-1)

    raise ValueError(f"Representação desconhecida: {representation}")


def to_uint8(array: np.ndarray) -> np.ndarray:
    """Converte um array em [0, 1] para uint8."""
    return (np.clip(array, 0, 1) * 255).astype(np.uint8)


def resize_channel(
    channel: np.ndarray,
    target_height: int,
    target_width: int,
    preserve_peaks: bool,
) -> np.ndarray:
    """
    Redimensiona um canal para o tamanho pedido.

    Nos canais de realce o sinal é um ponto brilhante de poucos
    pixels. Uma média bilinear dilui esse pico na razão da redução,
    então aplica-se antes um filtro de máximo: cada pixel de saída
    passa a enxergar o maior valor da sua vizinhança de origem e a
    microcalcificação sobrevive à redução.
    """
    height, width = channel.shape

    if preserve_peaks:
        factor = max(1, int(round(max(height / target_height, width / target_width))))

        if factor > 1:
            channel = ndimage.maximum_filter(channel, size=factor)

    resized = Image.fromarray(np.asarray(channel, dtype=np.float32)).resize(
        (target_width, target_height),
        Image.BILINEAR,
    )

    return np.asarray(resized, dtype=np.float32)


def standardize_geometry(
    representation: np.ndarray,
    mask: np.ndarray,
    laterality: str,
    canvas_height: int,
    canvas_width: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """
    Coloca a mama num canvas de tamanho fixo sem perder tecido.

    A escala é a mesma nos dois eixos e sai do lado mais apertado,
    então a mama inteira cabe e a proporção original é mantida.
    O que sobra vira padding preto.

    As mamas direitas são espelhadas: medido nos 100 crops, a parede
    torácica fica à esquerda nas mamas L e à direita nas R. Espelhar
    as R deixa todas com a parede torácica à esquerda, que é onde a
    mama é ancorada no canvas.
    """
    flipped = laterality.strip().upper() == "R"

    if flipped:
        representation = representation[:, ::-1]
        mask = mask[:, ::-1]

    height, width = mask.shape

    scale = min(canvas_height / height, canvas_width / width)

    new_height = max(1, min(canvas_height, int(round(height * scale))))
    new_width = max(1, min(canvas_width, int(round(width * scale))))

    channels = [
        resize_channel(
            representation[:, :, index],
            new_height,
            new_width,
            # O canal 0 é a anatomia: interpolação suave.
            # Os canais de realce preservam o pico.
            preserve_peaks=index > 0,
        )
        for index in range(representation.shape[2])
    ]

    resized = np.stack(channels, axis=-1)

    resized_mask = (
        resize_channel(mask.astype(np.float32), new_height, new_width, preserve_peaks=True) >= 0.5
    )

    canvas = np.zeros((canvas_height, canvas_width, representation.shape[2]), dtype=np.float32)
    canvas_mask = np.zeros((canvas_height, canvas_width), dtype=bool)

    # Ancorada à esquerda, onde fica a parede torácica, e centrada
    # na vertical. Assim o padding é sempre previsível.
    top = (canvas_height - new_height) // 2

    canvas[top : top + new_height, 0:new_width] = resized
    canvas_mask[top : top + new_height, 0:new_width] = resized_mask

    info = {
        "flipped": flipped,
        "scale": round(float(scale), 5),
        "reduction": round(float(1 / scale), 4),
        "placed_height": new_height,
        "placed_width": new_width,
        "canvas_usage": round(float(new_height * new_width / (canvas_height * canvas_width)), 4),
    }

    return canvas, canvas_mask, info
