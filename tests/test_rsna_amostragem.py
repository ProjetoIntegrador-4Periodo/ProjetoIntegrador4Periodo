"""Testa a amostragem por época do treino, com uma tabela sintética."""

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

from src.training.rsna_amostragem import indices_da_epoca, indices_do_segmento  # noqa: E402


@pytest.fixture
def treino():
    rng = np.random.default_rng(0)
    n_pos, n_pool, n_fora = 50, 2000, 300
    cancer = np.r_[np.ones(n_pos), np.zeros(n_pool + n_fora)].astype(int)
    pool = np.r_[np.zeros(n_pos, bool), np.ones(n_pool, bool), np.zeros(n_fora, bool)]
    dificil = np.zeros(len(cancer), int)
    dificil[n_pos : n_pos + 400] = 1  # 400 negativos difíceis no pool
    return pd.DataFrame(
        {
            "image_id": rng.permutation(len(cancer)),
            "cancer": cancer,
            "in_negative_pool": pool,
            "difficult_negative_case": dificil,
            "biopsy": 0,
        }
    )


def test_todos_os_positivos_e_proporcao_de_negativos_do_pool(treino):
    indices = indices_da_epoca(
        treino, semente=1, epoca=0, negativos_por_positivo=10, peso_dificil=2
    )
    escolhidas = treino.iloc[indices]

    assert len(set(indices)) == len(indices), "sem repetição dentro da época"
    assert (escolhidas.cancer == 1).sum() == 50, "todos os positivos, toda época"
    assert (escolhidas.cancer == 0).sum() == 500, "1:10"
    assert escolhidas[escolhidas.cancer == 0].in_negative_pool.all(), "negativos só do pool"


def test_sorteio_deterministico_por_semente_e_epoca(treino):
    def sorteio(semente, epoca):
        return indices_da_epoca(treino, semente, epoca, 10, 2)

    np.testing.assert_array_equal(sorteio(1, 3), sorteio(1, 3))
    assert not np.array_equal(sorteio(1, 3), sorteio(1, 4)), "épocas diferentes variam"
    assert not np.array_equal(sorteio(1, 3), sorteio(2, 3)), "sementes diferentes variam"
    # semente + época não colide entre sementes (1, 2) e (2, 1)
    assert not np.array_equal(np.sort(sorteio(1, 2)), np.sort(sorteio(2, 1)))


@pytest.mark.parametrize(("peso", "faixa"), [(1.0, (0.95, 1.05)), (2.0, (1.6, 2.0))])
def test_negativo_dificil_sai_com_frequencia_maior(treino, peso, faixa):
    # Peso de sorteio sem reposição: a frequência fica abaixo do peso, porque
    # quem já saiu não volta (aqui, 25% do pool por época dá ~1,76x com peso 2).
    vezes = np.zeros(len(treino))
    for epoca in range(200):
        vezes[indices_da_epoca(treino, 1, epoca, 10, peso)] += 1
    dificil = (treino.difficult_negative_case == 1).to_numpy()
    comum = (treino.cancer == 0).to_numpy() & treino.in_negative_pool.to_numpy() & ~dificil
    razao = vezes[dificil].mean() / vezes[comum].mean()
    assert faixa[0] < razao < faixa[1], f"peso {peso}: razão de frequência {razao:.2f}"


def test_segmentos_de_tamanho_fixo_cortam_o_fluxo_de_epocas(treino):
    def epoca(e):
        return indices_da_epoca(treino, 1, e, 1, 2)  # 1:1, 100 imagens por época

    fluxo = np.concatenate([epoca(e) for e in range(6)])
    segmentos = [indices_do_segmento(epoca, k, 250) for k in range(2)]
    np.testing.assert_array_equal(np.concatenate(segmentos), fluxo[:500])
    np.testing.assert_array_equal(indices_do_segmento(epoca, 1, 250), fluxo[250:500])
    # Segmento do tamanho da época: o segmento k é a época k.
    np.testing.assert_array_equal(indices_do_segmento(epoca, 3, 100), epoca(3))
