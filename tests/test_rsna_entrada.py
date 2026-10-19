"""Testa a Etapa 2 (PNG da Etapa 1 -> entrada do modelo) com imagens sintéticas."""

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("cv2")
pytest.importorskip("yaml")

from src.data.rsna_entrada import (  # noqa: E402
    ConfigAugmentation,
    ConfigEntrada,
    ConfigEtapa2,
    aumentar,
    carregar_config,
    montar_canvas,
    preparar_entrada,
)
from src.utils.paths import CONFIGS_DIR  # noqa: E402

ENTRADA = ConfigEntrada(altura=96, largura=64)
AUG = ConfigAugmentation(ativa=True)


def mama_png(altura=300, largura=120, semente=0):
    """Meia-elipse encostada à esquerda, como sai da Etapa 1, e um marcador fora dela."""
    rng = np.random.default_rng(semente)
    yy, xx = np.mgrid[:altura, :largura]
    dentro = ((yy - altura / 2) / (altura / 2 - 4)) ** 2 + (xx / (largura - 8)) ** 2 <= 1
    anatomia = np.where(dentro, rng.integers(80, 220, (altura, largura)), 30).astype(np.uint8)
    anatomia[5:15, largura - 20 : largura - 5] = 255  # "L MLO"
    mascara = np.where(dentro, 255, 0).astype(np.uint8)
    return anatomia, mascara


def caixa(mascara):
    linhas, colunas = np.flatnonzero(mascara.any(1)), np.flatnonzero(mascara.any(0))
    return linhas[0], linhas[-1] + 1, colunas[0], colunas[-1] + 1


@pytest.mark.parametrize("forma", [(300, 120), (100, 150), (96, 64), (40, 400)])
def test_canvas_sem_distorcao_encostado_a_esquerda_e_centrado(forma):
    anatomia = np.full(forma, 200, np.uint8)
    mascara = np.full(forma, 255, np.uint8)

    canvas, dentro, escala = montar_canvas(anatomia, mascara, 96, 64)

    assert canvas.shape == dentro.shape == (96, 64)
    y0, y1, x0, x1 = caixa(dentro)
    assert x0 == 0, "a mama encosta na parede (esquerda)"
    assert escala == pytest.approx(min(96 / forma[0], 64 / forma[1]))
    # Mesma escala nos dois eixos: sem esticar, só o arredondamento de 1 px.
    assert abs((y1 - y0) - forma[0] * escala) <= 1
    assert abs((x1 - x0) - forma[1] * escala) <= 1
    assert y1 - y0 == 96 or x1 - x0 == 64, "ocupa o canvas em pelo menos um eixo"
    assert abs(y0 - (96 - y1)) <= 1, "padding igual em cima e embaixo"
    assert (canvas[~dentro] == 0).all() and (canvas[dentro] > 0).all()


def test_fundo_zerado_fora_da_mascara_some_com_o_marcador():
    anatomia, mascara = mama_png()

    canvas, dentro, _ = montar_canvas(anatomia, mascara, 96, 64)

    assert (canvas[~dentro] == 0).all()
    assert canvas[:6, 50:].max() == 0, "o marcador no canto não pode aparecer"


def test_entrada_cinza_repete_o_canal_e_normaliza_como_imagenet():
    anatomia, mascara = mama_png()

    x = preparar_entrada(anatomia, mascara, 0.1, ENTRADA)

    assert x.shape == (3, 96, 64) and x.dtype == np.float32
    reconstruido = (
        x * np.array(ENTRADA.desvio)[:, None, None] + np.array(ENTRADA.media)[:, None, None]
    )
    np.testing.assert_allclose(reconstruido[0], reconstruido[1], atol=1e-5)
    np.testing.assert_allclose(reconstruido[0], reconstruido[2], atol=1e-5)
    # Fundo preto vira -média/desvio em cada canal.
    fundo = -np.array(ENTRADA.media) / np.array(ENTRADA.desvio)
    np.testing.assert_allclose(x[:, 0, -1], fundo, rtol=1e-5)


def test_sem_augmentation_e_deterministica():
    anatomia, mascara = mama_png()
    a = preparar_entrada(anatomia, mascara, 0.1, ENTRADA, ConfigAugmentation(ativa=False))
    b = preparar_entrada(anatomia, mascara, 0.1, ENTRADA)
    np.testing.assert_array_equal(a, b)


def test_augmentation_nunca_espelha_na_horizontal():
    anatomia, mascara = mama_png()
    canvas, dentro, _ = montar_canvas(anatomia, mascara, 96, 64)

    # Só o flip ligado: o resultado é o flip vertical exato, nunca o horizontal.
    so_flip = ConfigAugmentation(
        ativa=True, prob_flip_vertical=1.0, rotacao_graus=0, escala=(1, 1), translacao=0,
        brilho=0, contraste=(1, 1),
    )  # fmt: skip
    saida, _ = aumentar(canvas, dentro, so_flip, np.random.default_rng(0))
    np.testing.assert_array_equal(saida, canvas[::-1])

    # Com tudo ligado, a parede continua à esquerda em todos os sorteios.
    for semente in range(100):
        _, m = aumentar(canvas, dentro, AUG, np.random.default_rng(semente))
        meio = m.shape[1] // 2
        assert m[:, :meio].sum() > m[:, meio:].sum(), f"semente {semente}"


def test_augmentation_deixa_o_tecido_encostado_na_borda_esquerda():
    anatomia, mascara = mama_png()
    canvas, dentro, _ = montar_canvas(anatomia, mascara, 96, 64)
    # Rotação máxima e translação para a direita: sem o realinhamento, abre vão.
    cfg = ConfigAugmentation(ativa=True, rotacao_graus=10, translacao=0.2)

    for semente in range(100):
        saida, m = aumentar(canvas, dentro, cfg, np.random.default_rng(semente))
        assert m[:, 0].any(), f"semente {semente}: vão entre a borda e o tecido"
        assert (saida[~m] == 0).all()


@pytest.mark.parametrize("brilho", [0.0, 0.3])
def test_fundo_continua_preto_apos_augmentation(brilho):
    anatomia, mascara = mama_png()
    canvas, dentro, _ = montar_canvas(anatomia, mascara, 96, 64)
    cfg = ConfigAugmentation(ativa=True, brilho=brilho)

    for semente in range(50):
        saida, m = aumentar(canvas, dentro, cfg, np.random.default_rng(semente))
        assert saida.dtype == np.uint8 and saida.shape == canvas.shape
        assert (saida[~m] == 0).all(), f"semente {semente}"


def test_augmentation_ativa_sem_rng_falha():
    anatomia, mascara = mama_png()
    with pytest.raises(ValueError, match="rng"):
        preparar_entrada(anatomia, mascara, 0.1, ENTRADA, AUG)


def test_representacao_com_nome_invalido_falha():
    with pytest.raises(ValueError, match="desconhecida"):
        ConfigEntrada(representacao="rgb")


def test_config_do_repositorio_carrega_e_volta_do_checkpoint():
    cfg = carregar_config(CONFIGS_DIR / "etapa2.yaml")

    assert (cfg.entrada.altura, cfg.entrada.largura) == (768, 512)
    assert cfg.entrada.representacao == "cinza"
    assert cfg.augmentation.ativa is False

    salvo = cfg.para_checkpoint()
    assert set(salvo) == {"entrada", "augmentation"}
    volta = ConfigEtapa2.de_dict(salvo)
    assert volta.entrada == cfg.entrada and volta.augmentation == cfg.augmentation
