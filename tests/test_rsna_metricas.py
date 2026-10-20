"""Testa a agregação por mama, as métricas e o bootstrap com valores calculados à mão."""

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("sklearn")

from src.evaluation import rsna_metricas as m  # noqa: E402


def test_agregacao_por_mama_pelo_maximo():
    imagens = pd.DataFrame(
        {
            "patient_id": [1, 1, 1, 1, 2],
            "laterality": ["L", "L", "R", "R", "L"],
            "cancer": [0, 1, 0, 0, 0],
            "prob": [0.2, 0.7, 0.1, 0.4, 0.9],
        }
    )
    mamas = m.por_mama(imagens).set_index("prediction_id")

    assert list(mamas.index) == ["1_L", "1_R", "2_L"]
    assert mamas.loc["1_L", "prob"] == 0.7 and mamas.loc["1_L", "cancer"] == 1
    assert mamas.loc["1_R", "prob"] == 0.4 and mamas.loc["1_R", "cancer"] == 0
    assert mamas.loc["1_L", "n_imagens"] == 2


def test_pf1_da_competicao():
    rotulo = np.array([1, 1, 0, 0])
    prob = np.array([0.8, 0.4, 0.2, 0.6])
    # VP = 1,2; FP = 0,8; precisão 0,6; revocação 0,6 -> F1 0,6
    assert m.pf1(rotulo, prob) == pytest.approx(0.6)


def test_f1_melhor_e_limiar():
    rotulo = np.array([1, 1, 0, 0, 0])
    prob = np.array([0.9, 0.5, 0.6, 0.2, 0.1])
    # limiar 0,9: F1 2/3; limiar 0,5: VP 2, FP 1 -> F1 0,8 (o melhor)
    f1, limiar = m.f1_melhor(rotulo, prob)
    assert f1 == pytest.approx(0.8) and limiar == pytest.approx(0.5)
    assert m.no_limiar(rotulo, prob, limiar)["f1"] == pytest.approx(0.8)


def test_fbeta_pesa_mais_a_sensibilidade_e_tem_limiar_proprio():
    rotulo = np.array([1, 1, 0, 0, 0, 0])
    prob = np.array([0.9, 0.3, 0.8, 0.5, 0.4, 0.1])
    # limiar 0,9: P 1, R 0,5 -> F1 0,667, F2 0,556
    # limiar 0,3: P 0,4, R 1 -> F1 0,571, F1.5 1,3/1,9 = 0,684, F2 2/2,6 = 0,769
    metricas = m.calcular(pd.DataFrame({"cancer": rotulo, "prob": prob}))

    assert metricas["f1_melhor"] == pytest.approx(2 / 3) and metricas["limiar_f1"] == 0.9
    assert metricas["f1_5_melhor"] == pytest.approx(1.3 / 1.9) and metricas["limiar_f1_5"] == 0.3
    assert metricas["f2_melhor"] == pytest.approx(2 / 2.6) and metricas["limiar_f2"] == 0.3
    # sensibilidade >= 90% exige as 2 positivas: limiar 0,3, 3 FP de 4 -> esp 0,25
    assert metricas["esp_sens90"] == pytest.approx(0.25) and metricas["limiar_sens90"] == 0.3
    fixo = m.no_limiar(rotulo, prob, 0.3)
    assert fixo["f2"] == pytest.approx(2 / 2.6) and fixo["sensibilidade"] == 1.0
    # beta extra pela config
    assert "f3_melhor" in m.calcular(pd.DataFrame({"cancer": rotulo, "prob": prob}), [3])


def test_limiar_do_ponto_de_operacao_da_metrica_de_selecao():
    validacao = {"limiar_f1": 0.9, "limiar_f2": 0.3, "limiar_f1_5": 0.3, "limiar_sens90": 0.2}
    assert m.limiar_da_selecao("f2_melhor", validacao) == 0.3
    assert m.limiar_da_selecao("f1_melhor", validacao) == 0.9
    assert m.limiar_da_selecao("esp_sens90", validacao) == 0.2
    assert m.limiar_da_selecao("roc_auc", validacao) == 0.3, "sem limiar: o do maior beta"
    assert m.metricas_de_selecao() == (
        "f1_melhor",
        "f1_5_melhor",
        "f2_melhor",
        "esp_sens90",
        "roc_auc",
    )


def test_auc_e_sensibilidade_com_especificidade_90():
    rotulo = np.r_[np.ones(10), np.zeros(10)].astype(int)
    prob = np.r_[np.linspace(0.55, 1.0, 10), np.linspace(0.0, 0.6, 10)]
    metricas = m.calcular(pd.DataFrame({"cancer": rotulo, "prob": prob}))

    assert 0.9 < metricas["roc_auc"] < 1.0
    assert 0.9 < metricas["pr_auc"] <= 1.0
    # Com no máximo 1 FP (especificidade 90%): o limiar fica acima de 0,533 (o
    # 2º maior negativo), e os positivos >= 0,6 contam: 9 de 10.
    assert metricas["sens_esp90"] == pytest.approx(0.9)
    separadas = m.calcular(
        pd.DataFrame({"cancer": rotulo, "prob": np.r_[np.ones(10), np.zeros(10)]})
    )
    assert (
        separadas["roc_auc"] == 1.0
        and separadas["sens_esp90"] == 1.0
        and separadas["f1_melhor"] == 1.0
    )


def test_bootstrap_reproduzivel_e_contem_o_valor():
    rng = np.random.default_rng(0)
    rotulo = (rng.random(400) < 0.1).astype(int)
    prob = np.clip(rotulo * 0.3 + rng.random(400) * 0.7, 0, 1)
    mamas = pd.DataFrame({"cancer": rotulo, "prob": prob})

    a = m.bootstrap(mamas, amostras=300, semente=7)
    b = m.bootstrap(mamas, amostras=300, semente=7)
    assert a == b
    pontual = m.calcular(mamas)
    for nome in m.METRICAS:
        assert a[nome]["ic_inferior"] <= pontual[nome] <= a[nome]["ic_superior"], nome
    com_limiar = m.bootstrap(
        mamas, amostras=50, semente=7, limiares={"selecao": 0.5, "sens90": 0.3}
    )
    assert {"selecao_f2", "selecao_sensibilidade", "sens90_especificidade"} <= set(com_limiar)


def test_recortes_por_mama_com_atributos():
    imagens = pd.DataFrame(
        {
            "image_id": range(8),
            "patient_id": [1, 1, 2, 2, 3, 3, 4, 4],
            "laterality": ["L", "L", "L", "L", "R", "R", "R", "R"],
            "view": ["CC", "MLO"] * 4,
            "cancer": [1, 1, 0, 0, 0, 0, 1, 0],
            "prob": [0.9, 0.2, 0.7, 0.1, 0.3, 0.2, 0.4, 0.8],
            "site_id": [1, 1, 1, 1, 2, 2, 2, 2],
            "machine_id": [49, 49, 49, 21, 29, 29, 48, 48],
            "age": [45, 45, 72, 72, None, None, 55, 55],
            "difficult_negative_case": [0, 0, 1, 0, 0, 0, 0, 0],
            "spacing_estimado": [False, False, False, True, True, True, False, False],
        }
    )
    mamas = m.atributos_por_mama(imagens).set_index("patient_id")

    assert mamas.loc[2, "negativo_dificil"] and mamas.loc[2, "spacing_estimado"]
    assert mamas.loc[3, "faixa_etaria"] == "ausente" and mamas.loc[2, "faixa_etaria"] == "70-79"
    assert mamas.loc[4, "prob"] == 0.8 and mamas.loc[4, "cancer"] == 1

    tabela = m.recortes(mamas.reset_index(), limiar=0.75).set_index(["recorte", "valor"])
    assert tabela.loc[("todas", "todas"), "n_mamas"] == 4
    assert tabela.loc[("site_id", 2), "n_positivas"] == 1
    # limiar 0,75: positivas 0,9 e 0,8 passam (sens 1); negativos 0,7 e 0,3 não (esp 1)
    assert tabela.loc[("todas", "todas"), "sensibilidade_limiar"] == 1.0
    assert tabela.loc[("negativos", "difíceis"), "n_mamas"] == 3
