"""Testa a Etapa 1 do pré-processamento do RSNA com DICOMs sintéticos.

Nenhuma imagem de paciente: a "mama" é uma meia-elipse com textura,
encostada numa borda como a parede torácica, sobre fundo de ar saturado.
"""

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")
pytest.importorskip("cv2")
pytest.importorskip("pydicom")

from pydicom.dataset import Dataset, FileMetaDataset  # noqa: E402
from pydicom.uid import ExplicitVRLittleEndian  # noqa: E402

from src.data.rsna_imagem import (  # noqa: E402
    Parametros,
    aplicar_radiometria,
    fundo_escuro,
    garantir_fundo_escuro,
    parede_a_direita,
    processar,
    redimensionar,
    spacing_mm,
)

PARAMETROS = Parametros(lado_maior=256, fator_mascara=1)


def mama_sintetica(altura=400, largura=240, raio=170, parede="esquerda", semente=0):
    """Tecido claro (12 bits) com ar em 0, parede torácica na borda pedida."""
    rng = np.random.default_rng(semente)
    yy, xx = np.mgrid[:altura, :largura]
    x = xx if parede == "esquerda" else (largura - 1 - xx)
    dentro = ((yy - altura / 2) / raio) ** 2 + (x / raio) ** 2 <= 1
    imagem = np.where(dentro, 2500 + rng.normal(0, 150, (altura, largura)), 0.0)
    return np.clip(imagem, 0, 4095).astype(np.uint16)


def dicom_sintetico(pixels, photometric="MONOCHROME2", spacing=0.07, **tags):
    ds = Dataset()
    ds.file_meta = FileMetaDataset()
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.Rows, ds.Columns = pixels.shape
    ds.SamplesPerPixel = 1
    ds.BitsAllocated = 16
    ds.BitsStored = 12
    ds.HighBit = 11
    ds.PixelRepresentation = 0
    ds.PhotometricInterpretation = photometric
    if spacing is not None:
        ds.ImagerPixelSpacing = [spacing, spacing]
    for nome, valor in tags.items():
        setattr(ds, nome, valor)
    ds.PixelData = pixels.astype(np.uint16).tobytes()
    return ds


def invertida(pixels):
    return (4095 - pixels.astype(np.int32)).astype(np.uint16)


# --- Radiometria -----------------------------------------------------------


def test_monochrome1_e_invertido():
    ds = dicom_sintetico(invertida(mama_sintetica()), photometric="MONOCHROME1")
    imagem, caminho = aplicar_radiometria(ds)

    assert caminho == "modalidade+inverte_mono1"
    assert fundo_escuro(imagem)


def test_sign_menos_1_e_invertido():
    ds = dicom_sintetico(invertida(mama_sintetica()), PixelIntensityRelationshipSign=-1)
    imagem, caminho = aplicar_radiometria(ds)

    assert caminho.endswith("+inverte_sinal")
    assert fundo_escuro(imagem)


def test_sign_menos_1_em_imagem_ja_certa_e_desfeito_pela_checagem_de_fundo():
    # Caso do site 1: MONOCHROME2 na orientação certa, mas com Sign = -1.
    ds = dicom_sintetico(mama_sintetica(), PixelIntensityRelationshipSign=-1)
    imagem, caminho = aplicar_radiometria(ds)
    assert not fundo_escuro(imagem)

    imagem, caminho = garantir_fundo_escuro(imagem, caminho)

    assert caminho == "modalidade+inverte_sinal+corrige_fundo"
    assert fundo_escuro(imagem)


def test_checagem_de_fundo_corrige_quando_as_tags_mentem():
    ds = dicom_sintetico(invertida(mama_sintetica()))
    _, _, meta = processar(ds, "L", PARAMETROS)

    assert meta["caminho_radiometrico"] == "modalidade+corrige_fundo"


def test_checagem_de_fundo_aguenta_mama_encostada_nos_dois_cantos():
    # A mama ocupa a borda inteira: os dois cantos da parede são tecido.
    pixels = mama_sintetica(raio=260)
    assert pixels[0, 0] > 0 and pixels[-1, 0] > 0

    assert fundo_escuro(pixels.astype(np.float32))
    assert not fundo_escuro(invertida(pixels).astype(np.float32))


def test_checagem_de_fundo_ignora_marcador_branco_no_canto():
    pixels = mama_sintetica()
    pixels[:20, -40:] = 4095  # marcador "R MLO"

    assert fundo_escuro(pixels.astype(np.float32))


def test_janelamento_e_aplicado_quando_ha_window():
    ds = dicom_sintetico(mama_sintetica(), WindowCenter=2000, WindowWidth=4000)
    _, caminho = aplicar_radiometria(ds)

    assert caminho == "modalidade+janela"


# --- Spacing ---------------------------------------------------------------


def test_spacing_ausente_usa_padrao_e_marca_estimado():
    assert spacing_mm(dicom_sintetico(mama_sintetica(), spacing=None), 0.065238) == (
        0.065238,
        True,
    )
    assert spacing_mm(dicom_sintetico(mama_sintetica(), spacing=0.05), 0.065238) == (0.05, False)


def test_spacing_resultante_acompanha_a_escala():
    ds = dicom_sintetico(mama_sintetica(altura=800, largura=480, raio=340), spacing=None)
    _, _, meta = processar(ds, "L", PARAMETROS)

    assert meta["spacing_estimado"]
    assert meta["spacing_original_mm"] == pytest.approx(0.065238)
    assert meta["spacing_resultante_mm"] == pytest.approx(0.065238 / meta["escala"], rel=1e-4)


# --- Espelhamento ----------------------------------------------------------


@pytest.mark.parametrize(
    ("parede", "tag", "espelha", "diverge"),
    [
        ("esquerda", "L", False, False),
        ("direita", "R", True, False),
        # Site 1: tag não bate com o lado real da parede.
        ("direita", "L", True, True),
        ("esquerda", "R", False, True),
    ],
)
def test_espelhamento_pela_medida_e_nao_pela_tag(parede, tag, espelha, diverge):
    ds = dicom_sintetico(mama_sintetica(parede=parede))
    anatomia, mascara, meta = processar(ds, tag, PARAMETROS)

    assert meta["espelhada"] is espelha
    assert meta["diverge_tag"] is diverge
    # Depois de espelhar, a parede torácica fica sempre à esquerda.
    assert not parede_a_direita(mascara > 0)


# --- Cobertura, crop e saída -----------------------------------------------


def test_cobertura_fora_da_faixa_marca_suspeita_mas_nao_descarta():
    ds = dicom_sintetico(mama_sintetica())
    restrito = Parametros(lado_maior=256, fator_mascara=1, cobertura_min=0.99)

    anatomia, mascara, meta = processar(ds, "L", restrito)

    assert meta["suspeita"]
    assert anatomia.size > 0 and mascara.size > 0


def test_cobertura_normal_nao_e_suspeita():
    _, _, meta = processar(dicom_sintetico(mama_sintetica()), "L", PARAMETROS)

    assert not meta["suspeita"]
    assert 0.25 <= meta["cobertura_mascara"] <= 0.92


def test_saida_e_8_bits_com_mascara_binaria_e_lado_maior_limitado():
    anatomia, mascara, meta = processar(dicom_sintetico(mama_sintetica()), "L", PARAMETROS)

    assert anatomia.dtype == np.uint8 and mascara.dtype == np.uint8
    assert set(np.unique(mascara)) <= {0, 255}
    assert max(anatomia.shape) == 256
    assert anatomia.shape == mascara.shape
    assert (meta["altura_saida"], meta["largura_saida"]) == anatomia.shape


def test_crop_e_retangular_e_nao_apaga_pixels_fora_da_mascara():
    # Marcador claro fora da mama, mas dentro da bounding box: sai da
    # máscara (maior componente), porém continua na anatomia.
    pixels = mama_sintetica()
    pixels[40:47, 150:157] = 4000
    anatomia, mascara, meta = processar(dicom_sintetico(pixels), "L", Parametros(fator_mascara=1))

    y, x = 43 - meta["bbox_y0"], 153 - meta["bbox_x0"]
    assert meta["escala"] == 1.0 and not meta["espelhada"]
    assert mascara[y, x] == 0
    assert anatomia[y, x] == 255
    # Margem de 12 px em volta da silhueta (limitada pela borda da imagem).
    linhas = np.flatnonzero((pixels > 0).any(axis=1) & ~((pixels == 4000).any(axis=1)))
    assert abs(meta["bbox_y0"] - (linhas[0] - 12)) <= 1


def test_mascara_nao_erode_onde_a_mama_encosta_na_borda():
    # A parede torácica encosta na coluna 0: a máscara tem que chegar nela.
    _, mascara, meta = processar(
        dicom_sintetico(mama_sintetica()), "L", Parametros(fator_mascara=1)
    )

    assert meta["bbox_x0"] == 0
    assert (mascara[:, 0] > 0).sum() > 0.8 * (mascara[:, 1:20] > 0).any(axis=1).sum()


def test_redimensionar_nunca_amplia():
    imagem = np.zeros((100, 60), dtype=np.uint8)
    mascara = np.ones((100, 60), dtype=bool)

    saida, saida_mascara, escala = redimensionar(imagem, mascara, 1536)

    assert escala == 1.0
    assert saida.shape == (100, 60) and saida_mascara.shape == (100, 60)


def test_mascara_reduzida_bate_com_a_cheia():
    ds = dicom_sintetico(mama_sintetica(altura=800, largura=480, raio=340))
    _, _, cheia = processar(ds, "L", Parametros(fator_mascara=1))
    _, _, reduzida = processar(ds, "L", Parametros(fator_mascara=4))

    for borda in ("bbox_y0", "bbox_y1", "bbox_x0", "bbox_x1"):
        assert abs(cheia[borda] - reduzida[borda]) <= 4
    assert abs(cheia["cobertura_mascara"] - reduzida["cobertura_mascara"]) < 0.02
