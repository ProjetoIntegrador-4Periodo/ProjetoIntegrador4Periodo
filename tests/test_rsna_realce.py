"""Testa a representação "realce_3canais" e a geração da pasta de realce, sem dados de paciente."""

from dataclasses import replace

import pytest

np = pytest.importorskip("numpy")
cv2 = pytest.importorskip("cv2")
pytest.importorskip("scipy")
pytest.importorskip("yaml")

from src.data.rsna_entrada import (  # noqa: E402
    ConfigAugmentation,
    ConfigEntrada,
    aumentar,
    conferir_pasta_realce,
    ler_realce,
    montar_canvas,
    preparar_entrada,
    representar,
)

REALCE = ConfigEntrada(representacao="realce_3canais")
SPACING = 0.1  # mm: o top-hat de 0,6 mm vira um disco de 7 px


def tecido(forma=(200, 200), semente=0):
    """Tecido uniforme com ruído leve, como o detector."""
    rng = np.random.default_rng(semente)
    return np.clip(rng.normal(100, 3, forma), 0, 255).astype(np.uint8)


def mama(forma=(240, 180)):
    """Meia-elipse encostada à esquerda (máscara) sobre fundo com marcador claro."""
    yy, xx = np.mgrid[: forma[0], : forma[1]]
    dentro = ((yy - forma[0] / 2) / (forma[0] / 2 - 4)) ** 2 + (xx / (forma[1] - 10)) ** 2 <= 1
    anatomia = tecido(forma)
    anatomia[~dentro] = 30
    anatomia[5:15, forma[1] - 20 : forma[1] - 5] = 255  # "L MLO" fora da máscara
    return anatomia, np.where(dentro, 255, 0).astype(np.uint8)


def test_tophat_realca_ponto_pequeno_e_ignora_estrutura_grande():
    anatomia = tecido()
    anatomia[48:51, 48:51] = 200  # calcificação de 3x3 px (0,3 mm), menor que o disco
    yy, xx = np.mgrid[:200, :200]
    grande = (yy - 130) ** 2 + (xx - 130) ** 2 <= 30**2  # massa de 6 mm, maior que o disco
    anatomia[grande] = 200
    mascara = np.full(anatomia.shape, 255, np.uint8)

    g = representar(anatomia, mascara, SPACING, REALCE)[:, :, 1]

    assert g[48:51, 48:51].mean() > 200, "a calcificação tem de saturar o canal G"
    # Tecido uniforme: o percentil do G é o próprio ruído, então compara-se com ele.
    comum = np.ones_like(grande)
    comum[30:70, 30:70] = False
    comum[(yy - 130) ** 2 + (xx - 130) ** 2 <= 40**2] = False
    assert g[grande].mean() <= g[comum].mean() + 5, "a massa grande não pode aparecer no top-hat"


def test_tophat_nao_cria_linha_na_borda_da_imagem():
    # Tecido que clareia em direção à parede (borda esquerda), como o peitoral na CC,
    # com calcificações no miolo para o percentil do G cair no sinal, como no real.
    rampa = np.linspace(160, 80, 120)[None, :].repeat(200, axis=0)
    for y in range(10, 190, 9):
        for x in (40, 70, 100):
            rampa[y : y + 3, x : x + 3] += 60
    anatomia = np.clip(rampa, 0, 255).astype(np.uint8)
    mascara = np.full(anatomia.shape, 255, np.uint8)

    g = representar(anatomia, mascara, SPACING, REALCE)[:, :, 1]

    assert g[10:190, 45:65].max() == 0 and g[11, 41] > 200, "sinal só nas calcificações"
    assert g[:, :5].max() == 0, "linha artificial do top-hat na parede"


def test_fundo_zerado_nos_tres_canais():
    anatomia, mascara = mama()

    realce = representar(anatomia, mascara, SPACING, REALCE)

    assert realce.shape == (*anatomia.shape, 3) and realce.dtype == np.uint8
    assert (realce[mascara == 0] == 0).all()
    assert (realce[:, :, 0][mascara > 0] == anatomia[mascara > 0]).all(), "R é a anatomia"


def test_clahe_e_top_hat_nao_dependem_do_que_esta_fora_da_mascara():
    anatomia, mascara = mama()
    outra = anatomia.copy()
    outra[mascara == 0] = np.random.default_rng(1).integers(0, 256, (mascara == 0).sum())

    a = representar(anatomia, mascara, SPACING, REALCE)
    b = representar(outra, mascara, SPACING, REALCE)

    np.testing.assert_array_equal(a, b)
    assert (a[:, :, 2][mascara == 0] == 0).all(), "CLAHE fora da máscara"


def test_clahe_nao_cria_borda_artificial_na_pele():
    # Tecido uniforme: sem borda real, o CLAHE não pode inventar um degrau na pele.
    anatomia, mascara = mama()
    anatomia[mascara > 0] = 100
    b = representar(anatomia, mascara, SPACING, REALCE)[:, :, 2]
    dentro = b[mascara > 0].astype(int)
    assert dentro.max() - dentro.min() <= 2


@pytest.mark.parametrize("deslocamento", [0, 1, 2, 3])
def test_filtro_de_maximo_preserva_ponto_de_1_pixel_no_canvas(deslocamento):
    representada = np.zeros((1536, 700, 3), np.uint8)
    representada[:, :, 0] = 80
    y, x = 701 + deslocamento, 301 + deslocamento
    representada[y, x, 0] = 255  # mesmo ponto no R (INTER_AREA)
    representada[y, x, 1] = 255  # e no G (máximo antes de reduzir)
    mascara = np.full((1536, 700), 255, np.uint8)

    canvas, _, escala = montar_canvas(representada, mascara, 768, 512, canais_de_pico=(1,))

    assert escala == pytest.approx(0.5)
    assert canvas[:, :, 1].max() == 255, "o ponto do G sumiu na redução"
    assert canvas[:, :, 0].max() < 255, "sem o máximo, o INTER_AREA dilui o ponto"


def test_augmentation_fotometrica_nao_afeta_g_e_b_por_padrao():
    rng = np.random.default_rng(0)
    canvas = rng.integers(1, 256, (96, 64, 3)).astype(np.uint8)
    mascara = np.ones((96, 64), bool)
    # Geometria neutra, brilho e contraste fortes: só o R pode mudar.
    so_foto = ConfigAugmentation(
        ativa=True, prob_flip_vertical=0, rotacao_graus=0, escala=(1, 1), translacao=0,
        brilho=0.3, contraste=(0.5, 0.5),
    )  # fmt: skip

    saida, _ = aumentar(canvas, mascara, so_foto, np.random.default_rng(3))

    np.testing.assert_array_equal(saida[:, :, 1:], canvas[:, :, 1:])
    assert not np.array_equal(saida[:, :, 0], canvas[:, :, 0])
    todos = replace(so_foto, canais_fotometricos=(0, 1, 2))
    saida, _ = aumentar(canvas, mascara, todos, np.random.default_rng(3))
    assert not np.array_equal(saida[:, :, 2], canvas[:, :, 2]), "configurável para os três"


def test_geometria_da_augmentation_e_igual_nos_tres_canais():
    anatomia, mascara = mama()
    realce = representar(anatomia, mascara, SPACING, REALCE)
    igual = np.dstack([realce[:, :, 0]] * 3)
    canvas, dentro, _ = montar_canvas(igual, mascara, 96, 64)
    geo = ConfigAugmentation(ativa=True, brilho=0, contraste=(1, 1))

    saida, _ = aumentar(canvas, dentro, geo, np.random.default_rng(5))

    np.testing.assert_array_equal(saida[:, :, 0], saida[:, :, 1])
    np.testing.assert_array_equal(saida[:, :, 0], saida[:, :, 2])


def test_png_gravado_e_identico_ao_calculado_na_hora_e_reproduzivel(tmp_path):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pydicom")
    from scripts.rsna_realce_gerar import gerar_uma

    anatomia, mascara = mama()
    (tmp_path / "anatomia").mkdir()
    (tmp_path / "mascara").mkdir()
    cv2.imwrite(str(tmp_path / "anatomia" / "7.png"), anatomia)
    cv2.imwrite(str(tmp_path / "mascara" / "7.png"), mascara)
    tarefa = {
        "image_id": 7,
        "png_anatomia": "anatomia/7.png",
        "png_mascara": "mascara/7.png",
        "spacing_mm": SPACING,
        "etapa1_dir": str(tmp_path),
        "entrada": REALCE,
    }

    bytes_ = []
    for pasta in ("a", "b"):
        assert gerar_uma({**tarefa, "saida_dir": str(tmp_path / pasta)})["status"] == "ok"
        bytes_.append((tmp_path / pasta / "realce" / "7.png").read_bytes())

    assert bytes_[0] == bytes_[1], "a mesma entrada tem de gravar os mesmos bytes"
    salvo = ler_realce(tmp_path / "a", 7)
    np.testing.assert_array_equal(salvo, representar(anatomia, mascara, SPACING, REALCE))
    np.testing.assert_array_equal(
        preparar_entrada(anatomia, mascara, SPACING, REALCE),
        preparar_entrada(anatomia, mascara, SPACING, REALCE, representada=salvo),
    )
    del pd


def test_pasta_gerada_com_outro_realce_e_recusada(tmp_path):
    yaml = pytest.importorskip("yaml")
    (tmp_path / "config_usada.yaml").write_text(
        yaml.safe_dump({"realce": {"tophat_mm": 0.6, "mediana": 3, "percentil_tophat": 99.9,
                                    "clahe_clip": 2.0, "clahe_tiles": [8, 8]}})  # fmt: skip
    )
    conferir_pasta_realce(tmp_path, REALCE)

    outro = replace(REALCE, realce=replace(REALCE.realce, clahe_clip=3.0))
    with pytest.raises(ValueError, match="outro realce"):
        conferir_pasta_realce(tmp_path, outro)
