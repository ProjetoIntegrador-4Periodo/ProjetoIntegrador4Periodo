"""Métricas por mama do RSNA: agregação pelo máximo, pF1, F-beta binarizado, AUCs e bootstrap.

Unidade de avaliação: a mama (``prediction_id = <patient_id>_<laterality>``,
como na competição). A probabilidade da mama é o máximo das suas imagens, e o
rótulo é 1 se qualquer imagem dela tiver câncer.

Métricas (``calcular``):

- ``pf1``: F1 probabilístico da competição. VP = soma das probabilidades nas
  mamas positivas, FP = soma nas negativas; precisão = VP/(VP+FP), revocação =
  VP/positivas.
- ``f1_melhor``, ``f1_5_melhor``, ``f2_melhor`` (um por beta): o maior F-beta
  binarizado entre todos os limiares, com o limiar que o atinge em
  ``limiar_f1``, ``limiar_f1_5``, ``limiar_f2`` (prob >= limiar é positivo).
  Beta > 1 pesa mais a sensibilidade: no F2, a revocação vale 4 vezes a
  precisão. F1, F1.5 e F2 são sempre calculados; ``betas`` acrescenta outros.
- ``roc_auc`` e ``pr_auc`` (precisão média).
- ``sens_esp90``: maior sensibilidade com especificidade >= 90%.
- ``esp_sens90``: maior especificidade com sensibilidade >= 90%, com o limiar
  em ``limiar_sens90``.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

BETAS = (1.0, 1.5, 2.0)
COLUNAS_IMAGEM = ["image_id", "patient_id", "laterality", "view", "cancer", "prob"]
COLUNAS_MAMA = ["prediction_id", "patient_id", "laterality", "cancer", "prob", "n_imagens"]


def nome_fbeta(beta: float) -> str:
    """1 -> f1, 1.5 -> f1_5, 2 -> f2."""
    return f"f{beta:g}".replace(".", "_")


def todos_os_betas(betas: Iterable[float] = ()) -> tuple[float, ...]:
    return tuple(sorted({*BETAS, *(float(b) for b in betas)}))


def metricas_reportadas(betas: Iterable[float] = ()) -> tuple[str, ...]:
    """As métricas sem limiar do ``calcular``, na ordem das tabelas."""
    fbetas = tuple(f"{nome_fbeta(b)}_melhor" for b in todos_os_betas(betas))
    return ("pf1", *fbetas, "roc_auc", "pr_auc", "sens_esp90", "esp_sens90")


def metricas_de_selecao(betas: Iterable[float] = ()) -> tuple[str, ...]:
    """Opções da métrica de seleção (early stopping, checkpoint e escolha no ajuste)."""
    fbetas = tuple(f"{nome_fbeta(b)}_melhor" for b in todos_os_betas(betas))
    return (*fbetas, "esp_sens90", "roc_auc")


METRICAS = metricas_reportadas()


def limiar_da_selecao(metrica: str, validacao: dict, betas: Iterable[float] = ()) -> float:
    """
    Limiar do ponto de operação da métrica de seleção, tirado das métricas da validação.

    ROC AUC não tem limiar: usa o do maior beta (F2 com os betas padrão).
    """
    if metrica == "esp_sens90":
        return float(validacao["limiar_sens90"])
    if metrica == "roc_auc":
        metrica = f"{nome_fbeta(todos_os_betas(betas)[-1])}_melhor"
    return float(validacao[f"limiar_{metrica.removesuffix('_melhor')}"])


def por_mama(imagens: pd.DataFrame) -> pd.DataFrame:
    """Predição por imagem -> por mama, pelo máximo da probabilidade e do rótulo."""
    mamas = (
        imagens.groupby(["patient_id", "laterality"], sort=True)
        .agg(cancer=("cancer", "max"), prob=("prob", "max"), n_imagens=("prob", "size"))
        .reset_index()
    )
    mamas.insert(0, "prediction_id", mamas.patient_id.astype(str) + "_" + mamas.laterality)
    mamas["cancer"] = mamas.cancer.astype(int)
    return mamas[COLUNAS_MAMA]


def pf1(rotulo: np.ndarray, prob: np.ndarray) -> float:
    positivas = rotulo.sum()
    vp = prob[rotulo == 1].sum()
    fp = prob[rotulo == 0].sum()
    if positivas == 0 or vp + fp == 0:
        return 0.0
    precisao, revocacao = vp / (vp + fp), vp / positivas
    return (
        0.0
        if precisao + revocacao == 0
        else float(2 * precisao * revocacao / (precisao + revocacao))
    )


def fbeta(precisao, revocacao, beta: float) -> np.ndarray:
    """F-beta = (1 + b²) P R / (b² P + R), 0 onde P = R = 0."""
    precisao = np.asarray(precisao, dtype=float)
    revocacao = np.asarray(revocacao, dtype=float)
    b2 = beta**2
    soma = b2 * precisao + revocacao
    return np.divide((1 + b2) * precisao * revocacao, soma, out=np.zeros_like(soma), where=soma > 0)


def fbeta_melhor(rotulo: np.ndarray, prob: np.ndarray, beta: float = 1.0) -> tuple[float, float]:
    """(maior F-beta binarizado, limiar). Sem positivas, (0, 1)."""
    if rotulo.sum() == 0:
        return 0.0, 1.0
    precisao, revocacao, limiares = precision_recall_curve(rotulo, prob)
    # O último ponto da curva (precisão 1, revocação 0) não tem limiar.
    valores = fbeta(precisao[:-1], revocacao[:-1], beta)
    melhor = int(np.argmax(valores))
    return float(valores[melhor]), float(limiares[melhor])


def f1_melhor(rotulo: np.ndarray, prob: np.ndarray) -> tuple[float, float]:
    return fbeta_melhor(rotulo, prob, 1.0)


def no_limiar(
    rotulo: np.ndarray, prob: np.ndarray, limiar: float, betas: Iterable[float] = ()
) -> dict[str, float]:
    """Matriz de confusão e F-betas num limiar fixo (o escolhido na validação, no teste)."""
    previsto = prob >= limiar
    vp = int((previsto & (rotulo == 1)).sum())
    fp = int((previsto & (rotulo == 0)).sum())
    fn = int((~previsto & (rotulo == 1)).sum())
    vn = int((~previsto & (rotulo == 0)).sum())
    precisao = vp / (vp + fp) if vp + fp else 0.0
    sensibilidade = vp / (vp + fn) if vp + fn else float("nan")
    especificidade = vn / (vn + fp) if vn + fp else float("nan")
    revocacao = sensibilidade if vp + fn else 0.0
    return {
        **{nome_fbeta(b): float(fbeta(precisao, revocacao, b)) for b in todos_os_betas(betas)},
        "precisao": precisao,
        "sensibilidade": sensibilidade,
        "especificidade": especificidade,
        "vp": vp,
        "fp": fp,
        "fn": fn,
        "vn": vn,
    }


def sens_com_especificidade(
    rotulo: np.ndarray, prob: np.ndarray, especificidade: float = 0.9
) -> float:
    if rotulo.sum() == 0 or rotulo.sum() == len(rotulo):
        return float("nan")
    fpr, tpr, _ = roc_curve(rotulo, prob)
    return float(tpr[fpr <= 1 - especificidade].max())


def esp_com_sensibilidade(
    rotulo: np.ndarray, prob: np.ndarray, sensibilidade: float = 0.9
) -> tuple[float, float]:
    """(maior especificidade com sensibilidade >= ``sensibilidade``, limiar que a atinge)."""
    if rotulo.sum() == 0 or rotulo.sum() == len(rotulo):
        return float("nan"), float("nan")
    fpr, tpr, limiares = roc_curve(rotulo, prob, drop_intermediate=False)
    # O 1º ponto que alcança a sensibilidade é o de maior limiar, logo o de menos FP.
    i = int(np.argmax(tpr >= sensibilidade - 1e-12))
    return float(1 - fpr[i]), float(limiares[i])


def calcular(mamas: pd.DataFrame, betas: Iterable[float] = ()) -> dict[str, float]:
    rotulo = mamas.cancer.to_numpy().astype(int)
    prob = mamas.prob.to_numpy().astype(float)
    ambas = 0 < rotulo.sum() < len(rotulo)
    resultado: dict[str, float] = {"pf1": pf1(rotulo, prob)}
    for beta in todos_os_betas(betas):
        valor, limiar = fbeta_melhor(rotulo, prob, beta)
        resultado[f"{nome_fbeta(beta)}_melhor"] = valor
        resultado[f"limiar_{nome_fbeta(beta)}"] = limiar
    especificidade, limiar_sens90 = esp_com_sensibilidade(rotulo, prob)
    resultado.update(
        {
            "roc_auc": float(roc_auc_score(rotulo, prob)) if ambas else float("nan"),
            "pr_auc": float(average_precision_score(rotulo, prob)) if ambas else float("nan"),
            "sens_esp90": sens_com_especificidade(rotulo, prob),
            "esp_sens90": especificidade,
            "limiar_sens90": limiar_sens90,
            "n_mamas": len(rotulo),
            "n_positivas": int(rotulo.sum()),
        }
    )
    return resultado


def bootstrap(
    mamas: pd.DataFrame,
    amostras: int = 2000,
    semente: int = 0,
    metricas: tuple[str, ...] = METRICAS,
    nivel: float = 0.95,
    limiares: dict[str, float] | None = None,
    betas: Iterable[float] = (),
) -> dict[str, dict[str, float]]:
    """
    IC por percentil, sorteando mamas com reposição.

    Com ``limiares`` ({ponto de operação: limiar}), inclui os F-betas, a
    sensibilidade e a especificidade em cada limiar fixo, como
    ``<ponto>_<medida>`` (a avaliação do teste usa os limiares da validação).
    Reamostras sem nenhuma positiva são descartadas.
    """
    rotulo = mamas.cancer.to_numpy().astype(int)
    prob = mamas.prob.to_numpy().astype(float)
    limiares = limiares or {}
    medidas = (*(nome_fbeta(b) for b in todos_os_betas(betas)), "sensibilidade", "especificidade")
    nomes = [*metricas, *(f"{ponto}_{k}" for ponto in limiares for k in medidas)]
    rng = np.random.default_rng(semente)
    valores: dict[str, list[float]] = {nome: [] for nome in nomes}
    for _ in range(amostras):
        sorteio = rng.integers(0, len(rotulo), len(rotulo))
        r, p = rotulo[sorteio], prob[sorteio]
        if r.sum() == 0 or r.sum() == len(r):
            continue
        amostra = calcular(pd.DataFrame({"cancer": r, "prob": p}), betas)
        for ponto, limiar in limiares.items():
            fixo = no_limiar(r, p, limiar, betas)
            amostra.update({f"{ponto}_{k}": fixo[k] for k in medidas})
        for nome in nomes:
            valores[nome].append(amostra[nome])
    alfa = (1 - nivel) / 2 * 100
    return {
        nome: {
            "ic_inferior": float(np.nanpercentile(v, alfa)),
            "ic_superior": float(np.nanpercentile(v, 100 - alfa)),
            "reamostras": len(v),
        }
        for nome, v in valores.items()
        if v
    }


# ---------------------------------------------------------------------------
# Recortes (avaliação do teste)
# ---------------------------------------------------------------------------

FAIXAS = [0, 40, 50, 60, 70, 80, 200]
NOMES_FAIXAS = ["<40", "40-49", "50-59", "60-69", "70-79", "80+"]


def atributos_por_mama(imagens: pd.DataFrame) -> pd.DataFrame:
    """
    Predição e atributos por mama, para os recortes.

    A mama é difícil se alguma imagem dela for ``difficult_negative_case``; tem
    spacing estimado se alguma imagem tiver; o aparelho é o mais frequente.
    """
    mamas = por_mama(imagens)
    chave = ["patient_id", "laterality"]
    grupos = imagens.groupby(chave, sort=True)
    extras = pd.DataFrame(
        {
            "site_id": grupos.site_id.first(),
            "machine_id": grupos.machine_id.agg(lambda s: s.mode().iloc[0]),
            "age": grupos.age.first(),
            "negativo_dificil": grupos.difficult_negative_case.max().astype(bool),
            "spacing_estimado": grupos.spacing_estimado.max().astype(bool),
        }
    ).reset_index()
    mamas = mamas.merge(extras, on=chave, how="left")
    faixa = pd.cut(mamas.age, FAIXAS, right=False, labels=NOMES_FAIXAS).astype(object)
    mamas["faixa_etaria"] = faixa.where(mamas.age.notna(), "ausente")
    return mamas


def _linha_do_recorte(recorte: str, valor, mamas: pd.DataFrame, limiar: float) -> dict:
    rotulo = mamas.cancer.to_numpy().astype(int)
    prob = mamas.prob.to_numpy().astype(float)
    ambas = 0 < rotulo.sum() < len(rotulo)
    fixo = no_limiar(rotulo, prob, limiar)
    return {
        "recorte": recorte,
        "valor": valor,
        "n_mamas": len(rotulo),
        "n_positivas": int(rotulo.sum()),
        "roc_auc": float(roc_auc_score(rotulo, prob)) if ambas else float("nan"),
        "pr_auc": float(average_precision_score(rotulo, prob)) if ambas else float("nan"),
        "f1_limiar": fixo["f1"],
        "f2_limiar": fixo["f2"],
        "sensibilidade_limiar": fixo["sensibilidade"],
        "especificidade_limiar": fixo["especificidade"],
    }


def recortes(
    mamas: pd.DataFrame,
    limiar: float,
    colunas: tuple[str, ...] = ("site_id", "faixa_etaria", "machine_id", "spacing_estimado"),
) -> pd.DataFrame:
    """
    Métricas por subgrupo de mamas, num limiar fixo da validação.

    Negativos difíceis: positivas contra só os negativos difíceis (e contra os
    negativos comuns, para comparar); a especificidade ali é a taxa de acerto
    nos negativos que enganaram o rastreamento.
    """
    linhas = [_linha_do_recorte("todas", "todas", mamas, limiar)]
    for coluna in colunas:
        for valor, grupo in mamas.groupby(coluna, sort=True):
            linhas.append(_linha_do_recorte(coluna, valor, grupo, limiar))
    positivas = mamas[mamas.cancer == 1]
    negativas = mamas[mamas.cancer == 0]
    for nome, filtro in (
        ("difíceis", negativas.negativo_dificil),
        ("comuns", ~negativas.negativo_dificil),
    ):
        grupo = pd.concat([positivas, negativas[filtro]])
        linhas.append(_linha_do_recorte("negativos", nome, grupo, limiar))
    return pd.DataFrame(linhas)
