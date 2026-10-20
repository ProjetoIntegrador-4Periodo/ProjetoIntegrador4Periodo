"""Testa a tabela de comparação sobre uma pasta runs/ sintética."""

import json

import pytest

pd = pytest.importorskip("pandas")
yaml = pytest.importorskip("yaml")

from scripts.rsna_comparar import (  # noqa: E402
    comparar,
    contar_tentativas,
    ler_runs,
    ler_tentativas,
    markdown,
    markdown_ajuste,
    resumir_ajuste,
)


def gravar_run(pasta, modelo, rep, semente, f1, modo="final", batch=36):
    pasta.mkdir(parents=True)
    metricas = {"pf1": f1 / 2, "f1_melhor": f1, "roc_auc": 0.8, "pr_auc": 0.3, "sens_esp90": 0.5}
    resumo = {
        "modo": modo,
        "modelo": modelo,
        "semente": semente,
        "batch_efetivo": batch,
        "melhor_epoca": 4,
        "epocas_treinadas": 9,
        "segundos_total": 7200,
        "validacao": metricas,
        "ic95_validacao": {
            k: {"ic_inferior": v - 0.1, "ic_superior": v + 0.1} for k, v in metricas.items()
        },
    }
    (pasta / "metricas_finais.json").write_text(json.dumps(resumo))
    (pasta / "config_etapa2.yaml").write_text(yaml.safe_dump({"entrada": {"representacao": rep}}))
    (pasta / "maquina.json").write_text(json.dumps({"gpu": "GPU X"}))


def test_media_desvio_por_modelo_e_representacao_e_tentativas(tmp_path):
    for semente, f1 in ((1, 0.30), (2, 0.40), (3, 0.50)):
        gravar_run(
            tmp_path / f"resnet50_cinza_seed{semente}", "resnet50", "cinza", semente, f1, batch=32
        )
    gravar_run(
        tmp_path / "resnet50_realce_3canais_seed1", "resnet50", "realce_3canais", 1, 0.45, batch=32
    )
    for n in (1, 2):
        gravar_run(
            tmp_path / "ajuste" / f"resnet50_cinza_tentativa{n}",
            "resnet50",
            "cinza",
            0,
            0.2,
            "ajuste",
        )
    gravar_run(
        tmp_path / "ajuste" / "convnext_tiny_cinza_tentativa1",
        "convnext_tiny",
        "cinza",
        0,
        0.2,
        "ajuste",
    )

    runs = ler_runs(tmp_path)
    assert len(runs) == 4, "as tentativas de ajuste não entram como treino final"
    tentativas = contar_tentativas(tmp_path)
    assert tentativas == {"resnet50": 2, "convnext_tiny": 1}

    tabela = comparar(runs, tentativas).set_index(["modelo", "representacao"])
    cinza = tabela.loc[("resnet50", "cinza")]
    assert cinza.n_seeds == 3 and cinza.seeds == "1,2,3" and cinza.batch_efetivo == "32"
    assert cinza.f1_melhor_media == pytest.approx(0.40)
    assert cinza.f1_melhor_desvio == pytest.approx(0.10)
    assert cinza.f1_melhor_ic_inf == pytest.approx(0.30) and cinza.tentativas_ajuste == 2
    assert pd.isna(
        tabela.loc[("resnet50", "realce_3canais")].f1_melhor_desvio
    ), "1 seed: sem desvio"
    assert "0.400 ± 0.100 [0.300; 0.500]" in markdown(tabela.reset_index())


def gravar_tentativa(runs, n, semente, f2, lr, melhor_epoca, parou=False, epocas=12):
    pasta = runs / "ajuste" / f"convnext_tiny_cinza_tentativa{n}"
    pasta.mkdir(parents=True)
    resumo = {
        "modo": "ajuste",
        "modelo": "convnext_tiny",
        "semente": semente,
        "metrica_selecao": "f2_melhor",
        "melhor_epoca": melhor_epoca,
        "epocas_treinadas": epocas,
        "parou_por_paciencia": parou,
        "segundos_total": 3600,
        "validacao": {"f2_melhor": f2, "roc_auc": 0.8},
    }
    (pasta / "metricas_finais.json").write_text(json.dumps(resumo))
    (pasta / "config_treino.yaml").write_text(
        yaml.safe_dump({"alteracoes": {"lr": lr}, "ajuste": {"epocas": 12}})
    )
    (pasta / "config_etapa2.yaml").write_text(
        yaml.safe_dump({"entrada": {"representacao": "cinza"}})
    )
    pd.DataFrame({"val_f2_melhor": [f2 / 2] * (epocas - 1) + [f2]}).to_csv(
        pasta / "historico.csv", index=False
    )


def test_tabela_do_ajuste_com_media_diferenca_e_ainda_melhorando(tmp_path):
    gravar_tentativa(tmp_path, 1, 0, 0.30, 1e-4, melhor_epoca=11)
    gravar_tentativa(tmp_path, 2, 1, 0.34, 1e-4, melhor_epoca=6, parou=True, epocas=11)
    gravar_tentativa(tmp_path, 3, 0, 0.40, 5e-5, melhor_epoca=4, parou=True, epocas=9)

    tentativas = ler_tentativas(tmp_path)
    assert len(tentativas) == 3
    assert list(tentativas.ainda_melhorando) == [True, False, False]

    resumo = resumir_ajuste(tentativas).set_index("configuracao")
    assert list(resumo.index) == ["lr=5e-05", "lr=0.0001"], "ordenado pelo F2 médio"
    assert resumo.loc["lr=0.0001", "f2_melhor_media"] == pytest.approx(0.32)
    assert resumo.loc["lr=0.0001", "f2_melhor_diferenca"] == pytest.approx(0.04)
    assert resumo.loc["lr=0.0001", "seeds"] == "0,1"
    assert resumo.loc["lr=0.0001", "ainda_melhorando"] == "sim,não"
    assert pd.isna(resumo.loc["lr=5e-05", "f2_melhor_diferenca"]), "1 seed: sem diferença"
    texto = markdown_ajuste(tentativas, resumo.reset_index())
    assert "0.320 (Δ 0.040)" in texto and "convnext_tiny_cinza_tentativa3" in texto
