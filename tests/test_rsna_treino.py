"""Testa o laço de treino com um modelo minúsculo na CPU e a montagem dos modelos pela config."""

from dataclasses import replace
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
torch = pytest.importorskip("torch")
pytest.importorskip("timm")

from torch import nn  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

from src.models.rsna_ema import ModeloEMA  # noqa: E402
from src.models.rsna_modelo import criar_modelo, criar_otimizador  # noqa: E402
from src.training.rsna_config import (  # noqa: E402
    ConfigModelo,
    ConfigTreino,
    carregar_config_treino,
    carregar_manifest,
    recusar_teste,
)
from src.training.rsna_treino import (  # noqa: E402
    Execucao,
    epoca_de_treino,
    planejar_orcamento,
    tamanhos_dos_grupos,
)
from src.training.rsna_treino import treinar as treinar_em  # noqa: E402
from src.utils.paths import CONFIGS_DIR, PROJECT_ROOT  # noqa: E402

CPU = torch.device("cpu")


def treinar(*args, **kwargs):
    """Sempre na CPU: lá o treino é determinístico byte a byte."""
    return treinar_em(*args, dispositivo=CPU, **kwargs)


def manifest_sintetico(n=24, split="train"):
    cancer = (np.arange(n) % 4 == 0).astype(int)
    return pd.DataFrame(
        {
            "image_id": np.arange(n) + (1000 if split == "val" else 0),
            "patient_id": np.arange(n) // 2 + (500 if split == "val" else 0),
            "laterality": np.where(np.arange(n) % 2, "R", "L"),
            "view": "CC",
            "cancer": cancer,
            "in_negative_pool": True,
            "difficult_negative_case": 0,
            "biopsy": 0,
            "split": split,
            "status": "ok",
        }
    )


class Sintetico(Dataset):
    """Imagem 3x16x16 fixa por image_id; positivos um pouco mais claros."""

    def __init__(self, manifest):
        self.manifest = manifest.reset_index(drop=True)

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, i):
        linha = self.manifest.iloc[i]
        g = torch.Generator().manual_seed(int(linha.image_id))
        imagem = torch.randn(3, 16, 16, generator=g) + float(linha.cancer)
        return imagem, torch.tensor(float(linha.cancer))


def mini_modelo(semente=0):
    torch.manual_seed(semente)
    return nn.Sequential(
        nn.Conv2d(3, 4, 3),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(4, 1),
    )


def cfg_mini(**hp):
    padrao = {
        "timm_nome": "mini",
        "batch_real": 2,
        "acumulacao": 2,
        "lr": 1e-2,
        "warmup_epocas": 0.5,
    }
    modelo = ConfigModelo(**{**padrao, **hp})
    return ConfigTreino(
        modelo="mini", modelos={"mini": modelo}, precisao="fp32", bootstrap_amostras=20
    )


def execucao_padrao(tmp_path, epocas=3, modo="final"):
    return Execucao(
        nome="mini", modo=modo, semente=3, run_dir=tmp_path / "run", checkpoint_dir=tmp_path / "ckpt",
        epocas=epocas, paciencia=None, workers=0,
    )  # fmt: skip


def fabrica(manifest, _treino):
    return Sintetico(manifest)


# ---------------------------------------------------------------------------
# Teste hold-out
# ---------------------------------------------------------------------------


def test_treino_e_ajuste_recusam_o_teste(tmp_path):
    todos = pd.concat(
        [manifest_sintetico(8), manifest_sintetico(8, "val"), manifest_sintetico(8, "test")]
    )
    todos.to_csv(tmp_path / "manifest_processado.csv", index=False)

    assert set(carregar_manifest(tmp_path).split) == {"train", "val"}
    with pytest.raises(ValueError, match="teste hold-out"):
        carregar_manifest(tmp_path, ("train", "val", "test"))
    with pytest.raises(ValueError, match="teste hold-out"):
        recusar_teste(todos)
    with pytest.raises(ValueError, match="teste hold-out"):
        treinar(
            mini_modelo(),
            cfg_mini(),
            execucao_padrao(tmp_path),
            todos,
            manifest_sintetico(8, "val"),
            fabrica,
        )
    # Só a avaliação final pede o teste.
    for script in (
        "scripts/rsna_treinar.py",
        "src/training/rsna_execucao.py",
        "src/training/rsna_treino.py",
    ):
        assert "permitir_teste" not in (PROJECT_ROOT / script).read_text(encoding="utf-8"), script


# ---------------------------------------------------------------------------
# Modelos e otimizadores pela config
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("nome", ["convnext_tiny", "resnet50"])
def test_cada_modelo_e_montado_pela_config(nome):
    from timm.layers import DropPath

    cfg = carregar_config_treino(CONFIGS_DIR / "treino.yaml").com_modelo(nome)
    modelo = criar_modelo(cfg.do_modelo, pretrained=False)

    assert modelo.pretrained_cfg["architecture"] == cfg.do_modelo.timm_nome.split(".")[0]
    assert modelo(torch.zeros(2, 3, 64, 64)).shape == (2, 1), "1 logit"
    caminhos = [m.drop_prob for m in modelo.modules() if isinstance(m, DropPath)]
    assert (max(caminhos, default=0) > 0) == (cfg.do_modelo.drop_path > 0)

    otimizador = criar_otimizador(modelo, cfg.do_modelo)
    esperado = {"adamw": torch.optim.AdamW, "sgd": torch.optim.SGD}[cfg.do_modelo.otimizador]
    assert isinstance(otimizador, esperado)
    sem_decay = otimizador.param_groups[1]
    assert sem_decay["weight_decay"] == 0 and all(p.ndim == 1 for p in sem_decay["params"])


def test_set_troca_so_o_modelo_atual_e_recusa_chave_errada():
    cfg = carregar_config_treino(CONFIGS_DIR / "treino.yaml").com_modelo("resnet50")
    novo = cfg.com_hiperparametros({"lr": 0.05, "acumulacao": 6})
    assert novo.do_modelo.lr == 0.05 and novo.do_modelo.batch_efetivo == 72
    assert novo.modelos["convnext_tiny"] == cfg.modelos["convnext_tiny"]
    with pytest.raises(ValueError, match="desconhecidos"):
        cfg.com_hiperparametros({"learning_rate": 1})


# ---------------------------------------------------------------------------
# Acumulação de gradiente
# ---------------------------------------------------------------------------


def test_grupos_de_acumulacao_cobrem_a_epoca():
    assert tamanhos_dos_grupos(10, 2, 4) == [8, 2]
    assert tamanhos_dos_grupos(36, 12, 3) == [36]
    assert sum(tamanhos_dos_grupos(8580, 12, 3)) == 8580


@pytest.mark.parametrize(("batch_real", "acumulacao"), [(2, 4), (4, 2), (1, 8)])
def test_batch_real_com_acumulacao_equivale_ao_batch_efetivo(batch_real, acumulacao):
    dataset = Sintetico(manifest_sintetico(10))
    criterio = nn.BCEWithLogitsLoss(reduction="sum", pos_weight=torch.tensor(3.0))

    def pesos_depois(batch, acc):
        modelo = mini_modelo()
        modelo[2].p = 0.0  # sem dropout, para comparar gradientes
        otimizador = torch.optim.SGD(modelo.parameters(), lr=0.1)
        carregador = DataLoader(dataset, batch_size=batch, sampler=list(range(10)))
        epoca_de_treino(modelo, carregador, otimizador, None, None, criterio, acc, CPU)
        return torch.cat([p.detach().flatten() for p in modelo.parameters()])

    torch.testing.assert_close(pesos_depois(batch_real, acumulacao), pesos_depois(8, 1))


# ---------------------------------------------------------------------------
# Retomada
# ---------------------------------------------------------------------------


class Queda(Exception):
    pass


@pytest.mark.parametrize(
    ("hp", "segmento"),
    [({}, None), ({"ema": True, "ema_decaimento": 0.9}, 10)],
    ids=["por_epocas", "por_passos_com_ema"],
)
def test_retomar_da_o_mesmo_treino_que_nao_parar(tmp_path, hp, segmento):
    treino, validacao = manifest_sintetico(24), manifest_sintetico(12, "val")

    def execucao(pasta):
        return replace(execucao_padrao(pasta), imagens_por_segmento=segmento)

    def cfg_mini_():
        return cfg_mini(**hp)

    direto = treinar(
        mini_modelo(), cfg_mini_(), execucao(tmp_path / "a"), treino, validacao, fabrica
    )

    def cai_na_primeira(linha):
        if linha["epoca"] == 0:
            raise Queda

    with pytest.raises(Queda):
        treinar(mini_modelo(), cfg_mini_(), execucao(tmp_path / "b"), treino, validacao, fabrica,
                ao_fim_da_epoca=cai_na_primeira)  # fmt: skip
    assert not (tmp_path / "b" / "run" / "metricas_finais.json").exists()
    with pytest.raises(FileExistsError):
        treinar(mini_modelo(), cfg_mini_(), execucao(tmp_path / "b"), treino, validacao, fabrica)
    retomado = treinar(
        mini_modelo(1),
        cfg_mini_(),
        execucao(tmp_path / "b"),
        treino,
        validacao,
        fabrica,
        retomar=True,
    )

    def pesos(pasta: Path, chave="modelo"):
        salvo = torch.load(pasta / "ckpt" / "ultimo.pt", weights_only=False)
        return salvo[chave] if chave == "modelo" else salvo[chave]["modelo"]

    for tipo in ("modelo", "ema") if hp else ("modelo",):
        for chave, valor in pesos(tmp_path / "a", tipo).items():
            torch.testing.assert_close(valor, pesos(tmp_path / "b", tipo)[chave], rtol=0, atol=0)
    a = pd.read_csv(tmp_path / "a" / "run" / "historico.csv")
    b = pd.read_csv(tmp_path / "b" / "run" / "historico.csv")
    pd.testing.assert_series_equal(a.loss_treino, b.loss_treino)
    assert direto["melhor_epoca"] == retomado["melhor_epoca"]
    assert direto["validacao"] == retomado["validacao"]


def test_saidas_padronizadas_do_run(tmp_path):
    treino, validacao = manifest_sintetico(24), manifest_sintetico(12, "val")
    resumo = treinar(
        mini_modelo(), cfg_mini(), execucao_padrao(tmp_path, epocas=2), treino, validacao, fabrica
    )
    run = tmp_path / "run"

    imagens = pd.read_csv(run / "predicoes_val_imagem.csv")
    mamas = pd.read_csv(run / "predicoes_val_mama.csv")
    assert list(imagens.columns) == [
        "image_id",
        "patient_id",
        "laterality",
        "view",
        "cancer",
        "prob",
    ]
    assert list(mamas.columns) == [
        "prediction_id",
        "patient_id",
        "laterality",
        "cancer",
        "prob",
        "n_imagens",
    ]
    assert len(imagens) == 12 and len(mamas) == 12
    assert len(pd.read_csv(run / "historico.csv")) == 2
    assert resumo["batch_efetivo"] == 4 and set(resumo["ic95_validacao"]) >= {
        "f1_melhor",
        "f2_melhor",
        "esp_sens90",
        "roc_auc",
    }
    assert resumo["metrica_selecao"] == "f2_melhor"
    assert resumo["limiar_selecao"] == resumo["validacao"]["limiar_f2"]
    assert (tmp_path / "ckpt" / "melhor.pt").is_file()


def test_early_stopping_para_e_retomar_nao_passa_do_ponto(tmp_path):
    treino, validacao = manifest_sintetico(24), manifest_sintetico(12, "val")
    # lr 0: o modelo não muda, então nenhuma época melhora depois da primeira.
    parado = replace(execucao_padrao(tmp_path, epocas=10), paciencia=2)
    resumo = treinar(
        mini_modelo(), cfg_mini(lr=0.0, lr_minimo=0.0), parado, treino, validacao, fabrica
    )
    assert resumo["epocas_treinadas"] == 3 and resumo["parou_por_paciencia"]

    de_novo = treinar(
        mini_modelo(),
        cfg_mini(lr=0.0, lr_minimo=0.0),
        parado,
        treino,
        validacao,
        fabrica,
        retomar=True,
    )
    assert de_novo["epocas_treinadas"] == 3, "retomar depois do early stopping não treina mais"


def test_numeros_em_notacao_cientifica_sem_ponto_viram_float():
    from src.training.rsna_execucao import ler_alteracoes

    # yaml.safe_load("2e-5") devolve texto; o --set e a config precisam virar número.
    alteracoes = ler_alteracoes(["lr=2e-5", "batch_real=8", "betas=[0.9, 0.99]"])
    assert alteracoes["lr"] == pytest.approx(2e-5), "gravado como número no config_treino.yaml"
    cfg = carregar_config_treino(CONFIGS_DIR / "treino.yaml").com_hiperparametros(alteracoes)
    assert cfg.do_modelo.lr == pytest.approx(2e-5) and isinstance(cfg.do_modelo.lr, float)
    assert cfg.do_modelo.batch_real == 8 and cfg.do_modelo.betas == (0.9, 0.99)
    assert ConfigModelo(timm_nome="x", weight_decay="5e-2").weight_decay == pytest.approx(0.05)
    with pytest.raises(ValueError, match="não é um número"):
        ConfigModelo(timm_nome="x", lr="rapido")


def test_metrica_de_selecao_invalida_e_recusada():
    with pytest.raises(ValueError, match="metrica_selecao"):
        replace(cfg_mini(), metrica_selecao="acuracia")
    assert replace(cfg_mini(), metrica_selecao="roc_auc").metrica_selecao == "roc_auc"
    assert carregar_config_treino(CONFIGS_DIR / "treino.yaml").metrica_selecao == "f2_melhor"


# ---------------------------------------------------------------------------
# Orçamento por passos
# ---------------------------------------------------------------------------


def test_orcamento_por_passos_iguala_os_passos_entre_proporcoes():
    # 60 positivos e 180 negativos no pool: a época de referência (1:10, limitada
    # pelo pool) tem 240 imagens = 60 passos de batch efetivo 4.
    treino = manifest_sintetico(240)
    base = replace(cfg_mini(), epocas_ajuste=12, max_epocas=30)

    for proporcao in (1, 3, 10):
        cfg = base.com_hiperparametros({"negativos_por_positivo": proporcao})
        ajuste = planejar_orcamento(cfg, treino, "ajuste")
        assert (ajuste.validacoes, ajuste.passos_por_validacao) == (12, 60), proporcao
        assert ajuste.passos_total == 720 and ajuste.imagens_por_segmento == 240
        assert planejar_orcamento(cfg, treino, "final").passos_total == 30 * 60

    por_epocas = replace(base, orcamento="epocas").com_hiperparametros(
        {"negativos_por_positivo": 1}
    )
    epocas = planejar_orcamento(por_epocas, treino, "ajuste")
    assert (epocas.passos_por_validacao, epocas.imagens_por_segmento) == (30, None), "1:1 = 120"
    assert (
        planejar_orcamento(replace(base, max_passos_ajuste=100), treino, "ajuste").validacoes == 2
    )
    fixo = planejar_orcamento(replace(base, passos_por_validacao=7), treino, "ajuste")
    assert (fixo.passos_por_validacao, fixo.imagens_por_segmento, fixo.passos_total) == (7, 28, 84)


def test_treino_por_passos_valida_no_mesmo_passo_em_qualquer_proporcao(tmp_path):
    treino, validacao = manifest_sintetico(240), manifest_sintetico(12, "val")
    historicos = {}
    for proporcao in (1, 10):
        cfg = cfg_mini(negativos_por_positivo=proporcao)
        orcamento = planejar_orcamento(replace(cfg, max_epocas=2), treino, "final")
        execucao = replace(
            execucao_padrao(tmp_path / str(proporcao), epocas=orcamento.validacoes),
            imagens_por_segmento=orcamento.imagens_por_segmento,
        )
        treinar(mini_modelo(), cfg, execucao, treino, validacao, fabrica)
        historicos[proporcao] = pd.read_csv(execucao.run_dir / "historico.csv")

    assert list(historicos[1].passo) == list(historicos[10].passo) == [60, 120]
    assert list(historicos[1].epocas_amostragem) == [2.0, 4.0], "1:1: 120 imagens por época"
    assert list(historicos[10].epocas_amostragem) == [1.0, 2.0]
    pd.testing.assert_series_equal(historicos[1].lr, historicos[10].lr)  # mesmo scheduler


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------


def test_ema_segue_a_media_movel_depois_do_aquecimento():
    modelo = nn.Sequential(nn.Conv2d(3, 4, 3), nn.BatchNorm2d(4))
    ema = ModeloEMA(modelo, decaimento=0.9, aquecimento=1)
    inicial = {k: v.clone() for k, v in ema.modelo.state_dict().items()}
    assert not any(p.requires_grad for p in ema.modelo.parameters())

    modelo.train()
    modelo(torch.randn(8, 3, 8, 8))  # muda as estatísticas do BatchNorm
    with torch.no_grad():
        for parametro in modelo.parameters():
            parametro.add_(1.0)
    ema.atualizar(modelo)  # passo 0, no aquecimento: d = 0, a EMA copia o modelo
    atual = modelo.state_dict()
    for nome, valor in ema.modelo.state_dict().items():
        assert torch.equal(valor, atual[nome]), nome
        assert not torch.equal(valor, inicial[nome]) or not valor.is_floating_point()

    with torch.no_grad():
        for parametro in modelo.parameters():
            parametro.add_(1.0)
    modelo(torch.randn(8, 3, 8, 8))
    antes = {k: v.clone() for k, v in ema.modelo.state_dict().items()}
    ema.atualizar(modelo)  # passo 1, aquecido: d = 0,9
    atual = modelo.state_dict()
    for nome, valor in ema.modelo.state_dict().items():
        if valor.is_floating_point():
            torch.testing.assert_close(valor, 0.9 * antes[nome] + 0.1 * atual[nome])
        else:
            assert torch.equal(valor, atual[nome]), "num_batches_tracked é copiado"


def test_com_ema_a_validacao_e_o_melhor_pt_usam_os_pesos_da_ema(tmp_path):
    treino, validacao = manifest_sintetico(24), manifest_sintetico(12, "val")
    cfg = cfg_mini(ema=True, ema_decaimento=0.9)
    treinar(mini_modelo(), cfg, execucao_padrao(tmp_path, epocas=1), treino, validacao, fabrica)

    melhor = torch.load(tmp_path / "ckpt" / "melhor.pt", weights_only=False)
    ultimo = torch.load(tmp_path / "ckpt" / "ultimo.pt", weights_only=False)
    assert melhor["pesos"] == "ema"
    for chave, valor in ultimo["ema"]["modelo"].items():
        torch.testing.assert_close(melhor["modelo"][chave], valor, rtol=0, atol=0)
    assert any(
        not torch.equal(ultimo["modelo"][k], v) for k, v in ultimo["ema"]["modelo"].items()
    ), "a EMA difere dos pesos crus"
    # 24 imagens, batch efetivo 4: 6 passos, e a EMA andou em todos.
    assert ultimo["ema"]["passos"] == 6
    inicial = mini_modelo().state_dict()
    assert all(not torch.equal(inicial[k], v) for k, v in melhor["modelo"].items())

    rede = mini_modelo()
    rede.load_state_dict(melhor["modelo"])
    rede.eval()
    dados = Sintetico(validacao)
    with torch.no_grad():
        prob = torch.sigmoid(rede(torch.stack([dados[i][0] for i in range(len(dados))]))).squeeze(1)
    salvas = pd.read_csv(tmp_path / "run" / "predicoes_val_imagem.csv").prob
    np.testing.assert_allclose(salvas, prob.numpy(), rtol=1e-5)
