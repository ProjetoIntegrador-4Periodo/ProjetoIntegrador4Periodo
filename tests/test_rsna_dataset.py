"""Testa o Dataset da Etapa 2 sobre uma pasta sintética no formato da Etapa 1."""

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
cv2 = pytest.importorskip("cv2")
torch = pytest.importorskip("torch")

from src.data.rsna_dataset import MamografiaRSNA  # noqa: E402
from src.data.rsna_entrada import ConfigAugmentation, ConfigEntrada  # noqa: E402

ENTRADA = ConfigEntrada(altura=96, largura=64)


@pytest.fixture
def etapa1(tmp_path):
    (tmp_path / "anatomia").mkdir()
    (tmp_path / "mascara").mkdir()
    linhas = []
    for image_id, cancer in ((1, 0), (2, 1)):
        anatomia = np.full((300, 120), 150, np.uint8)
        mascara = np.full((300, 120), 255, np.uint8)
        cv2.imwrite(str(tmp_path / "anatomia" / f"{image_id}.png"), anatomia)
        cv2.imwrite(str(tmp_path / "mascara" / f"{image_id}.png"), mascara)
        linhas.append(
            {
                "image_id": image_id,
                "png_anatomia": f"anatomia/{image_id}.png",
                "png_mascara": f"mascara/{image_id}.png",
                "spacing_resultante_mm": 0.1,
                "cancer": cancer,
            }
        )
    return tmp_path, pd.DataFrame(linhas)


def test_item_tem_forma_tipo_e_rotulo_e_nao_grava_nada(etapa1):
    pasta, manifest = etapa1
    antes = sorted(pasta.rglob("*"))

    dataset = MamografiaRSNA(manifest, pasta, ENTRADA)
    imagem, rotulo = dataset[1]

    assert len(dataset) == 2
    assert imagem.shape == (3, 96, 64) and imagem.dtype == torch.float32
    assert rotulo.item() == 1.0
    assert sorted(pasta.rglob("*")) == antes


def test_augmentation_desligada_ignora_e_ligada_varia(etapa1):
    pasta, manifest = etapa1
    fixo = MamografiaRSNA(manifest, pasta, ENTRADA, ConfigAugmentation(ativa=False))
    torch.manual_seed(0)
    assert torch.equal(fixo[0][0], fixo[0][0])

    variado = MamografiaRSNA(manifest, pasta, ENTRADA, ConfigAugmentation(ativa=True))
    assert not torch.equal(variado[0][0], variado[0][0])


def test_manifest_sem_coluna_falha(etapa1):
    pasta, manifest = etapa1
    with pytest.raises(ValueError, match="spacing_resultante_mm"):
        MamografiaRSNA(manifest.drop(columns="spacing_resultante_mm"), pasta, ENTRADA)


def test_realce_da_pasta_e_igual_ao_calculado_na_hora(etapa1, tmp_path):
    from dataclasses import asdict

    import yaml

    from src.data.rsna_entrada import caminho_realce, ler_pngs, para_png, representar

    pasta, manifest = etapa1
    realce = ConfigEntrada(altura=96, largura=64, representacao="realce_3canais")
    realce_dir = tmp_path / "realce_dir"
    (realce_dir / "realce").mkdir(parents=True)
    for r in manifest.itertuples():
        anatomia, mascara = ler_pngs(pasta, r.png_anatomia, r.png_mascara)
        rep = representar(anatomia, mascara, r.spacing_resultante_mm, realce)
        cv2.imwrite(str(realce_dir / caminho_realce(r.image_id)), para_png(rep))
    parametros = {
        k: list(v) if isinstance(v, tuple) else v for k, v in asdict(realce.realce).items()
    }
    (realce_dir / "config_usada.yaml").write_text(yaml.safe_dump({"realce": parametros}))

    da_pasta = MamografiaRSNA(manifest, pasta, realce, realce_dir=realce_dir)
    na_hora = MamografiaRSNA(manifest, pasta, realce)

    assert da_pasta.realce_dir is not None and na_hora.realce_dir is None
    for i in range(len(manifest)):
        assert torch.equal(da_pasta[i][0], na_hora[i][0])
