"""Testa limpeza, divisão e seleção do fold 0 com um train.csv sintético."""

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("sklearn")

from src.data.rsna_divisao import (  # noqa: E402
    COLUNAS_MANIFEST,
    IDADE_AUSENTE,
    dividir,
    excluir_manual,
    faixa_etaria,
    fundir_estratos_raros,
    limpar,
    selecionar_fold0,
    tabela_pacientes,
    verificar_vazamento,
)

LIMITES = [40, 50, 60, 70, 80]


def train_sintetico(n_pacientes=600, semente=0):
    """4 imagens por paciente (L/R x CC/MLO), ~8% das pacientes com câncer."""
    rng = np.random.default_rng(semente)
    linhas = []
    image_id = 1000
    for patient_id in range(1, n_pacientes + 1):
        site = int(rng.integers(1, 3))
        idade = float(rng.integers(35, 90)) if rng.random() > 0.01 else np.nan
        lado_cancer = rng.choice(["L", "R"]) if rng.random() < 0.08 else None
        for lado in ("L", "R"):
            for view in ("CC", "MLO"):
                cancer = int(lado == lado_cancer)
                linhas.append(
                    {
                        "site_id": site,
                        "patient_id": patient_id,
                        "image_id": image_id,
                        "laterality": lado,
                        "view": view,
                        "age": idade,
                        "cancer": cancer,
                        "biopsy": int(cancer or rng.random() < 0.03),
                        "implant": 0,
                        "machine_id": 49,
                        "difficult_negative_case": bool(not cancer and rng.random() < 0.1),
                    }
                )
                image_id += 1
    return pd.DataFrame(linhas)


def pipeline(df, pool=1000, semente=42):
    pacientes = tabela_pacientes(df, LIMITES)
    estrato, _ = fundir_estratos_raros(pacientes, LIMITES, 10)
    df = df.copy()
    df["faixa_etaria"] = df["patient_id"].map(pacientes["faixa_etaria"])
    divisao = dividir(df, estrato, 0.15, 5, 0, semente)
    return divisao, selecionar_fold0(divisao, pool, semente)


def test_limpeza_conta_o_que_cada_regra_removeu():
    df = train_sintetico(50)
    df.loc[0, "view"] = "AT"
    df.loc[df.patient_id == 2, "implant"] = 1

    limpo, relatorio = limpar(df, ["CC", "MLO"], remover_implante=True)

    assert len(limpo) == len(df) - 1 - 4
    assert relatorio[0]["imagens"] == 1 and relatorio[0]["pacientes_removidas"] == 0
    assert relatorio[1]["imagens"] == 4 and relatorio[1]["pacientes_removidas"] == 1
    assert relatorio[1]["mamas_removidas"] == 2


def test_faixa_etaria_e_idade_ausente():
    faixas = faixa_etaria(pd.Series([39, 40, 79, 80, np.nan]), LIMITES)
    assert list(faixas) == ["<40", "40-49", "70-79", "80+", IDADE_AUSENTE]


def test_estratos_raros_sao_fundidos_sem_misturar_cancer_e_site():
    df = train_sintetico(600)
    pacientes = tabela_pacientes(df, LIMITES)
    estrato, fusoes = fundir_estratos_raros(pacientes, LIMITES, 10)

    assert fusoes, "com ~8% de câncer em 600 pacientes deve haver faixa rara"
    contagem = estrato.value_counts()
    # Grupos câncer+site com 10+ pacientes não podem ter estrato menor que 10.
    grupos = pacientes.groupby(["cancer", "site_id"]).size()
    for (cancer, site), total in grupos.items():
        if total >= 10:
            prefixo = f"{cancer}_{site}_"
            assert contagem[contagem.index.str.startswith(prefixo)].min() >= 10
    # Câncer e site preservados exatamente no prefixo da chave.
    esperado = pacientes["cancer"].astype(str) + "_" + pacientes["site_id"].astype(str)
    assert (estrato.str.split("_").str[:2].str.join("_") == esperado).all()


def test_sem_vazamento_e_proporcoes():
    divisao, _ = pipeline(train_sintetico(600))

    verificar_vazamento(divisao)
    pacientes = divisao.groupby("patient_id")["split"].first()
    assert pacientes.eq("test").mean() == pytest.approx(0.15, abs=0.01)
    assert set(divisao.loc[divisao.split == "test", "fold"]) == {-1}
    assert set(divisao.loc[divisao.split != "test", "fold"]) == {0, 1, 2, 3, 4}
    assert set(divisao.loc[divisao.fold == 0, "split"]) == {"val"}


def test_verificar_vazamento_falha_com_paciente_em_dois_splits():
    divisao, _ = pipeline(train_sintetico(100), pool=100)
    divisao.loc[divisao.index[0], "split"] = "test"
    divisao.loc[divisao.index[1], "split"] = "train"

    with pytest.raises(AssertionError):
        verificar_vazamento(divisao)


def test_pool_inclui_todos_os_negativos_obrigatorios():
    divisao, manifest = pipeline(train_sintetico(600), pool=1000)
    treino = manifest[manifest.split == "train"]

    obrigatorios = divisao[
        (divisao.split == "train")
        & (divisao.cancer == 0)
        & (divisao.difficult_negative_case | (divisao.biopsy == 1))
    ]
    assert set(obrigatorios.image_id) <= set(treino[treino.in_negative_pool].image_id)
    assert treino.in_negative_pool.sum() == 1000
    assert (treino[treino.cancer == 1].in_negative_pool == False).all()  # noqa: E712
    # Todos os positivos do treino entram; val e teste entram inteiros.
    assert treino.cancer.sum() == divisao[divisao.split == "train"].cancer.sum()
    for split in ("val", "test"):
        assert (manifest.split == split).sum() == (divisao.split == split).sum()
    assert list(manifest.columns) == COLUNAS_MANIFEST


def test_pool_menor_que_os_obrigatorios_falha():
    with pytest.raises(ValueError, match="obrigatórios"):
        pipeline(train_sintetico(600), pool=10)


def test_divisao_e_deterministica_e_nao_depende_da_ordem_do_csv():
    df = train_sintetico(400)
    a_div, a_man = pipeline(df)
    b_div, b_man = pipeline(df.sample(frac=1, random_state=7))

    pd.testing.assert_frame_equal(a_div, b_div)
    pd.testing.assert_frame_equal(a_man, b_man)


def test_semente_diferente_muda_a_divisao():
    df = train_sintetico(400)
    a, _ = pipeline(df, semente=1)
    b, _ = pipeline(df, semente=2)

    assert not a["split"].equals(b["split"])


def test_exclusao_manual_remove_so_as_imagens_listadas():
    divisao, _ = pipeline(train_sintetico(300), pool=300)
    alvo = int(divisao.image_id.iloc[5])

    restante, relatorio = excluir_manual(divisao, [alvo])

    assert alvo not in set(restante.image_id)
    assert len(restante) == len(divisao) - 1
    assert relatorio["imagens"] == 1 and relatorio["pacientes_removidas"] == 0


def test_exclusao_manual_com_id_inexistente_falha():
    divisao, _ = pipeline(train_sintetico(300), pool=300)

    with pytest.raises(ValueError, match="não existe"):
        excluir_manual(divisao, [123])


def test_exclusao_manual_nao_reembaralha_divisao_nem_pool():
    divisao, manifest = pipeline(train_sintetico(600), pool=1000)
    treino = manifest[manifest.split == "train"]
    paciente = int(divisao.patient_id.iloc[0])
    no_pool = int(treino[treino.in_negative_pool].image_id.iloc[-1])
    alvo = [*divisao[divisao.patient_id == paciente].image_id, no_pool]

    restante, _ = excluir_manual(divisao, alvo)
    novo = selecionar_fold0(divisao, 1000, 42, alvo)

    # Split e fold das demais imagens não mudam.
    comum = divisao[~divisao.image_id.isin(alvo)].reset_index(drop=True)
    pd.testing.assert_frame_equal(restante, comum)
    # O pool continua com 1000: só entram substitutos para o que saiu.
    novo_treino = novo[novo.split == "train"]
    assert novo_treino.in_negative_pool.sum() == 1000
    antes = set(treino[treino.in_negative_pool].image_id) - set(alvo)
    depois = set(novo_treino[novo_treino.in_negative_pool].image_id)
    assert antes <= depois
    assert len(depois - antes) <= len(alvo)
    assert not set(alvo) & set(novo.image_id)
