"""Testa o pré-processamento de mamografia com DICOMs sintéticos.

Nenhuma imagem de paciente: a "mama" é uma meia-elipse com textura,
encostada na borda esquerda como a parede torácica.
"""

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")
pytest.importorskip("skimage")
pydicom = pytest.importorskip("pydicom")

from pydicom.dataset import Dataset, FileMetaDataset  # noqa: E402
from pydicom.uid import ExplicitVRLittleEndian  # noqa: E402
from scipy import ndimage  # noqa: E402

from src.data.preprocessamento import (  # noqa: E402
    apply_radiometric_transform,
    background_is_dark,
    create_representation,
    crop_to_mask,
    disk_closing,
    mm_to_pixels,
    preprocess_dataset,
    robust_normalize,
    standardize_geometry,
)


def mama_sintetica(altura=400, largura=200, raio=150, semente=0):
    """Meia-elipse clara sobre fundo escuro, em valores de 12 bits."""
    rng = np.random.default_rng(semente)
    yy, xx = np.mgrid[:altura, :largura]
    dentro = ((yy - altura / 2) / raio) ** 2 + (xx / raio) ** 2 <= 1

    imagem = np.where(dentro, 2500 + rng.normal(0, 150, (altura, largura)), 40.0)
    return np.clip(imagem, 0, 4095).astype(np.uint16)


def dicom_sintetico(pixels, photometric="MONOCHROME2", **tags):
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
    ds.ImagerPixelSpacing = [0.085, 0.085]
    ds.ImageLaterality = "L"

    for nome, valor in tags.items():
        setattr(ds, nome, valor)

    ds.PixelData = pixels.astype(np.uint16).tobytes()
    return ds


def test_mm_to_pixels_e_impar_e_tem_minimo_de_3():
    assert mm_to_pixels(0.6, 0.085) == 7
    assert mm_to_pixels(0.6, 0.1) == 7  # 6 vira 7
    assert mm_to_pixels(0.01, 0.1) == 3


def test_mm_to_pixels_recusa_espacamento_invalido():
    with pytest.raises(ValueError):
        mm_to_pixels(0.6, 0.0)


def test_robust_normalize_fica_em_0_1_e_zera_imagem_constante():
    imagem = np.arange(100, dtype=np.float32).reshape(10, 10)
    normalizada = robust_normalize(imagem)

    assert normalizada.min() == 0.0
    assert normalizada.max() == 1.0
    assert not robust_normalize(np.full((5, 5), 7.0)).any()


def test_disk_closing_bate_com_binary_closing_do_scipy():
    rng = np.random.default_rng(1)
    mascara = ndimage.binary_dilation(rng.random((80, 60)) > 0.97, iterations=2)
    raio = 4

    yy, xx = np.mgrid[-raio : raio + 1, -raio : raio + 1]
    disco = xx**2 + yy**2 <= raio**2

    esperado = ndimage.binary_closing(mascara, structure=disco)

    np.testing.assert_array_equal(disk_closing(mascara, raio), esperado)


def test_crop_to_mask_respeita_margem_e_borda():
    imagem = np.zeros((50, 50))
    mascara = np.zeros((50, 50), dtype=bool)
    mascara[20:30, 0:10] = True

    recorte, recorte_mascara = crop_to_mask(imagem, mascara, margin=5)

    assert recorte.shape == (20, 15)
    assert recorte_mascara.sum() == mascara.sum()


def test_background_is_dark_detecta_inversao():
    imagem = mama_sintetica().astype(np.float32)

    assert background_is_dark(imagem)
    assert not background_is_dark(imagem.max() - imagem)


def test_monochrome1_e_invertido():
    pixels = mama_sintetica()
    imagem, caminho = apply_radiometric_transform(
        dicom_sintetico(pixels.max() - pixels, photometric="MONOCHROME1")
    )

    assert caminho == "modality_only+inverte_mono1"
    assert background_is_dark(imagem)


def test_janelamento_quando_nao_ha_voi_lut():
    ds = dicom_sintetico(mama_sintetica(), WindowCenter=2000, WindowWidth=4000)
    _, caminho = apply_radiometric_transform(ds)

    assert caminho == "windowing_after_modality"


def test_preprocess_dataset_entrega_imagem_cropada_em_0_1():
    imagem, mascara, meta = preprocess_dataset(dicom_sintetico(mama_sintetica()))

    assert imagem.shape == mascara.shape
    assert imagem.dtype == np.float32
    assert imagem.min() >= 0.0 and imagem.max() <= 1.0
    assert not imagem[~mascara].any()
    assert 0.25 <= mascara.mean() <= 0.92
    assert meta["laterality"] == "L"
    assert meta["pixel_spacing_mm"] == pytest.approx(0.085)


def test_preprocess_dataset_corrige_orientacao_quando_as_tags_mentem():
    # MONOCHROME2 com a escala invertida: só a checagem medida pega.
    pixels = mama_sintetica()
    _, _, meta = preprocess_dataset(dicom_sintetico(pixels.max() - pixels))

    assert meta["radiometric_path"].endswith("+corrige_orientacao")


def test_representacao_d_tem_tres_canais_em_0_1():
    imagem, mascara, meta = preprocess_dataset(dicom_sintetico(mama_sintetica()))
    rep = create_representation(imagem, mascara, "D", 0.6, 0.01, meta["pixel_spacing_mm"])

    assert rep.shape == (*imagem.shape, 3)
    assert rep.min() >= 0.0 and rep.max() <= 1.0
    np.testing.assert_array_equal(rep[:, :, 0], imagem)


def test_representacao_desconhecida_falha():
    with pytest.raises(ValueError):
        create_representation(np.zeros((4, 4)), np.ones((4, 4), bool), "X", 0.6, 0.01, 0.1)


@pytest.mark.parametrize("lado", ["L", "R"])
def test_standardize_geometry_ancora_a_mama_na_esquerda(lado):
    imagem, mascara, _ = preprocess_dataset(dicom_sintetico(mama_sintetica()))
    rep = np.stack([imagem] * 3, axis=-1)

    if lado == "R":
        rep, mascara = rep[:, ::-1], mascara[:, ::-1]

    canvas, canvas_mascara, info = standardize_geometry(rep, mascara, lado, 192, 96)

    assert canvas.shape == (192, 96, 3)
    assert canvas_mascara.shape == (192, 96)
    assert info["flipped"] is (lado == "R")
    # A parede torácica fica à esquerda depois de padronizar.
    assert canvas_mascara[:, :48].sum() > canvas_mascara[:, 48:].sum()
