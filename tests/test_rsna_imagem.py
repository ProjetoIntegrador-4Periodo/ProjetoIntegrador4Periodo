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
    criar_mascara,
    fundo_escuro,
    garantir_fundo_escuro,
    nivel_do_ar_ruidoso,
    parede_a_direita,
    processar,
    redimensionar,
    sem_estruturas_finas,
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


def com_ar_ruidoso(pixels, semente=0):
    """Como o aparelho 210: ar em ~1% da escala com ruído, só os cantos em 0."""
    rng = np.random.default_rng(semente)
    altura, largura = pixels.shape
    ar = np.clip(rng.normal(40, 8, pixels.shape), 1, None)
    imagem = np.where(pixels > 0, pixels, ar)
    yy, xx = np.mgrid[:altura, :largura]
    canto = 40
    for cy, cx in ((0, 0), (0, largura), (altura, 0), (altura, largura)):
        imagem[(yy - cy) ** 2 + (xx - cx) ** 2 < canto**2] = 0
    return imagem.astype(np.uint16)


@pytest.mark.parametrize("parede", ["esquerda", "direita"])
def test_ar_ruidoso_nao_entra_na_mascara_nem_decide_o_espelhamento(parede):
    ds = dicom_sintetico(com_ar_ruidoso(mama_sintetica(parede=parede)))

    _, mascara, meta = processar(ds, "L", PARAMETROS)

    assert meta["espelhada"] is (parede == "direita")
    assert not parede_a_direita(mascara > 0)
    assert not meta["suspeita"]


def chapa_com_letras(pixels, recuo=45):
    """
    Moldura fina e clara da chapa e um "L" de traço fino preso nela.

    A moldura fica fora dos cantos que ``com_ar_ruidoso`` zera, fechada como
    no 210, e cruza a mama na parede: sem a correção, a letra e a borda da
    direita entram no maior componente e o preenchimento leva a máscara até lá.
    """
    com_chapa = pixels.copy()
    altura, largura = pixels.shape
    cima, baixo, direita = recuo, altura - recuo, largura - recuo
    claro = 3000
    com_chapa[cima : cima + 2, :direita] = claro
    com_chapa[baixo - 2 : baixo, :direita] = claro
    com_chapa[cima:baixo, direita - 2 : direita] = claro
    com_chapa[cima + 5 : cima + 30, direita - 20 : direita - 17] = claro  # haste do "L"
    com_chapa[cima + 27 : cima + 30, direita - 20 : direita] = claro  # pé, preso na borda
    return com_chapa


def test_borda_da_chapa_e_letras_saem_da_mascara_sem_cortar_a_mama():
    # Chapa mais larga que a mama, como no 210: o ar domina o histograma.
    mama = mama_sintetica(largura=320, raio=195)
    ds = dicom_sintetico(com_ar_ruidoso(chapa_com_letras(mama)))
    imagem, _ = garantir_fundo_escuro(*aplicar_radiometria(ds))

    mascara = criar_mascara(imagem)

    assert not mascara[50:75, 255:258].any(), "letra dentro da máscara"
    assert not mascara[100:300, 230:].any(), "borda da direita (e o ar até ela) na máscara"
    assert not mascara[353:357, 160:].any(), "borda de baixo longe da mama"
    # Linha da pele: todo o tecido fora dos cantos (raio 40) que com_ar_ruidoso zera.
    tecido = (mama > 0)[45:355]
    assert mascara[45:355][tecido].mean() > 0.999, "a máscara cortou tecido da mama"
    assert mascara[45:355, 0].all(), "a parede torácica encostada na borda foi erodida"


def test_linha_na_borda_da_imagem_sai_e_a_parede_fica():
    # Como no 210: a borda da chapa é a última linha do DICOM e encosta na mama.
    mascara = np.zeros((200, 150), bool)
    yy, xx = np.mgrid[:200, :150]
    mascara[((yy - 100) / 95) ** 2 + (xx / 100) ** 2 <= 1] = True  # mama na parede (x = 0)
    mascara[194:, :] = True  # linha da chapa, 6 px, na borda de baixo

    aberta = sem_estruturas_finas(mascara, 4)

    assert not aberta[196:, 60:].any(), "linha da chapa na borda sobreviveu"
    assert aberta[10:190, 0].all(), "a parede encostada na borda foi comida"
    assert aberta[mascara & (yy < 190)].mean() > 0.99


def test_ar_saturado_nao_muda_o_limiar():
    ds = dicom_sintetico(mama_sintetica())
    imagem, _ = aplicar_radiometria(ds)

    assert nivel_do_ar_ruidoso(imagem / imagem.max()) == 0.0


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
