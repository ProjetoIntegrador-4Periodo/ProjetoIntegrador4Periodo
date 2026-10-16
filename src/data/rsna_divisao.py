"""Limpeza, divisão por paciente e seleção do fold 0 do RSNA.

Tudo aqui é determinístico: as tabelas são ordenadas por ``patient_id`` e
``image_id`` antes de qualquer sorteio, e todo sorteio recebe a semente da
configuração. A orquestração (ler CSV, imprimir, gravar) fica em
``scripts/rsna_dividir.py``.

Os rótulos auxiliares (BIRADS, biopsy, invasive, difficult_negative_case)
servem só para limpeza, seleção do pool e análise; nunca entram no modelo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

IDADE_AUSENTE = "ausente"

# Colunas do manifest do fold 0, na ordem em que são gravadas.
COLUNAS_MANIFEST = [
    "patient_id",
    "image_id",
    "laterality",
    "view",
    "age",
    "site_id",
    "machine_id",
    "cancer",
    "difficult_negative_case",
    "biopsy",
    "split",
    "fold",
    "in_negative_pool",
]


def ordenar(df: pd.DataFrame) -> pd.DataFrame:
    """Ordem canônica: é ela que torna os sorteios reproduzíveis."""
    return df.sort_values(["patient_id", "image_id"]).reset_index(drop=True)


def _contagem(df: pd.DataFrame) -> dict[str, int]:
    return {
        "imagens": len(df),
        "mamas": df[["patient_id", "laterality"]].drop_duplicates().shape[0],
        "pacientes": df["patient_id"].nunique(),
    }


def _aplicar_regra(
    atual: pd.DataFrame, remover: pd.Series, nome: str
) -> tuple[pd.DataFrame, dict]:
    """Remove as linhas marcadas e conta imagens, mamas e pacientes afetadas/removidas."""
    remover = remover.loc[atual.index]
    removidas = atual[remover]
    restantes = atual[~remover]

    mamas_antes = set(zip(atual.patient_id, atual.laterality, strict=True))
    mamas_depois = set(zip(restantes.patient_id, restantes.laterality, strict=True))

    linha = {
        "regra": nome,
        **_contagem(removidas),
        "positivas_removidas": int(removidas["cancer"].sum()),
        "mamas_removidas": len(mamas_antes - mamas_depois),
        "pacientes_removidas": len(set(atual.patient_id) - set(restantes.patient_id)),
    }
    return restantes, linha


def limpar(
    df: pd.DataFrame, views: list[str], remover_implante: bool
) -> tuple[pd.DataFrame, list[dict]]:
    """
    Aplica as regras fixas de limpeza em sequência e conta o que cada uma removeu.

    Mamas e pacientes são contadas de dois jeitos: "afetadas" (perderam ao
    menos uma imagem) e "removidas" (saíram por completo da base).
    """
    regras = [("view fora de " + "/".join(views), ~df["view"].isin(views))]

    if remover_implante:
        regras.append(("implant == 1", df["implant"] == 1))

    relatorio = []
    atual = df
    for nome, remover in regras:
        atual, linha = _aplicar_regra(atual, remover, nome)
        relatorio.append(linha)

    return ordenar(atual), relatorio


def excluir_manual(df: pd.DataFrame, image_ids: list[int]) -> tuple[pd.DataFrame, dict]:
    """
    Remove a lista manual da config (image_id com o motivo em comentário).

    Roda **depois** da divisão, de propósito: tirar uma paciente antes do
    sorteio muda a sequência para todas as outras (medido: 1 paciente a menos
    moveu 28% das pacientes de split e 68% de fold). Aplicada depois, uma
    exclusão nova só remove as próprias linhas, e os PNGs já gerados e os
    treinos já feitos continuam valendo. Id inexistente é erro de digitação.
    """
    inexistentes = sorted(set(image_ids) - set(df["image_id"]))
    if inexistentes:
        raise ValueError(f"image_id da exclusão manual não existe após a limpeza: {inexistentes}")

    restantes, linha = _aplicar_regra(
        df, df["image_id"].isin(image_ids), "exclusão manual (config)"
    )
    return ordenar(restantes), linha


def faixa_etaria(idade: pd.Series, limites: list[int]) -> pd.Series:
    """``[40, 50, ...]`` vira ``<40, 40-49, ..., 80+``; idade ausente vira categoria própria."""
    rotulos = [f"<{limites[0]}"]
    rotulos += [f"{a}-{b - 1}" for a, b in zip(limites[:-1], limites[1:], strict=True)]
    rotulos += [f"{limites[-1]}+"]

    faixa = pd.cut(idade, [-np.inf, *limites, np.inf], right=False, labels=rotulos)
    return faixa.astype(object).where(idade.notna(), IDADE_AUSENTE).astype(str)


def tabela_pacientes(df: pd.DataFrame, limites: list[int]) -> pd.DataFrame:
    """
    Uma linha por paciente. Câncer da paciente = qualquer imagem com câncer.

    Site e idade são constantes por paciente no RSNA; a função confere isso
    em vez de assumir, porque um site duplicado quebraria a estratificação.
    """
    grupo = df.groupby("patient_id", sort=True)

    if (grupo["site_id"].nunique() > 1).any():
        raise ValueError("Há paciente com imagens em mais de um site.")
    if (grupo["age"].nunique() > 1).any():
        raise ValueError("Há paciente com idades diferentes entre imagens.")

    pacientes = grupo.agg(
        cancer=("cancer", "max"), site_id=("site_id", "first"), age=("age", "first")
    )
    pacientes["faixa_etaria"] = faixa_etaria(pacientes["age"], limites)
    return pacientes


def fundir_estratos_raros(
    pacientes: pd.DataFrame, limites: list[int], minimo: int
) -> tuple[pd.Series, list[str]]:
    """
    Chave ``câncer_site_faixa``, fundindo faixas raras com a vizinha.

    Câncer e site ficam sempre exatos; só a faixa etária é agrupada. Dentro
    de cada câncer + site, enquanto houver faixa com menos de ``minimo``
    pacientes, ela é fundida com a vizinha adjacente menor. A faixa
    "ausente" não tem vizinha natural e é fundida com a faixa mais numerosa.
    """
    ordem = list(faixa_etaria(pd.Series([limites[0] - 1, *limites], dtype=float), limites))
    faixa_final = pacientes["faixa_etaria"].copy()
    fusoes = []

    for (cancer, site), grupo in pacientes.groupby(["cancer", "site_id"], sort=True):
        contagem = grupo["faixa_etaria"].value_counts()
        # Cada bloco é uma lista de faixas originais que viram um estrato só.
        blocos = [[f] for f in ordem if f in contagem.index]
        tamanho = {f: int(contagem[f]) for f in contagem.index}

        def total(bloco, tamanho=tamanho):
            return sum(tamanho[f] for f in bloco)

        if IDADE_AUSENTE in tamanho and tamanho[IDADE_AUSENTE] < minimo and blocos:
            maior = max(blocos, key=total)
            maior.append(IDADE_AUSENTE)
        elif IDADE_AUSENTE in tamanho:
            blocos.append([IDADE_AUSENTE])

        etarios = [b for b in blocos if b != [IDADE_AUSENTE]]
        while len(etarios) > 1:
            menor = min(range(len(etarios)), key=lambda i: total(etarios[i]))
            if total(etarios[menor]) >= minimo:
                break
            vizinhos = [i for i in (menor - 1, menor + 1) if 0 <= i < len(etarios)]
            alvo = min(vizinhos, key=lambda i: total(etarios[i]))
            a, b = sorted((menor, alvo))
            etarios[a] = etarios[a] + etarios[b]
            del etarios[b]

        blocos = etarios + [b for b in blocos if b == [IDADE_AUSENTE]]

        for bloco in blocos:
            if len(bloco) == 1:
                continue
            nome = "|".join(bloco)
            linhas = grupo.index[grupo["faixa_etaria"].isin(bloco)]
            faixa_final.loc[linhas] = nome
            fusoes.append(f"cancer={cancer} site={site}: {nome} ({total(bloco)} pacientes)")

    chave = (
        pacientes["cancer"].astype(str) + "_" + pacientes["site_id"].astype(str) + "_" + faixa_final
    )
    return chave, fusoes


def dividir(
    df: pd.DataFrame,
    estrato_paciente: pd.Series,
    frac_teste: float,
    n_folds: int,
    fold_validacao: int,
    semente: int,
) -> pd.DataFrame:
    """
    Hold-out de pacientes e folds com StratifiedGroupKFold.

    O hold-out é sorteado na tabela de pacientes (uma linha por paciente),
    então 15% vale em pacientes, não em imagens. Os folds rodam sobre as
    imagens com ``groups=patient_id``, como pedido; o estrato de cada
    imagem é o da sua paciente.

    Devolve ``df`` com ``estrato``, ``split`` (train/val/test) e ``fold``
    (-1 no teste).
    """
    df = ordenar(df)
    estrato_paciente = estrato_paciente.sort_index()

    _, teste = train_test_split(
        estrato_paciente.index.to_numpy(),
        test_size=frac_teste,
        stratify=estrato_paciente.to_numpy(),
        random_state=semente,
    )

    df["estrato"] = df["patient_id"].map(estrato_paciente)
    df["fold"] = -1

    resto = df[~df["patient_id"].isin(teste)]
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=semente)

    for fold, (_, indices) in enumerate(
        sgkf.split(resto, resto["estrato"], groups=resto["patient_id"])
    ):
        df.loc[resto.index[indices], "fold"] = fold

    df["split"] = np.where(
        df["fold"] == -1, "test", np.where(df["fold"] == fold_validacao, "val", "train")
    )
    return df


def verificar_vazamento(df: pd.DataFrame) -> None:
    """Falha se alguma paciente aparecer em mais de um split ou fold."""
    for coluna in ("split", "fold"):
        por_paciente = df.groupby("patient_id")[coluna].nunique()
        vazadas = por_paciente[por_paciente > 1]
        if len(vazadas):
            raise AssertionError(
                f"{len(vazadas)} pacientes em mais de um {coluna}: " f"{list(vazadas.index[:5])}"
            )


def distribuicao(df: pd.DataFrame, por: str) -> pd.DataFrame:
    """Pacientes, imagens, câncer, site e faixa etária por grupo (split ou fold)."""
    pac = df.groupby([por, "patient_id"]).agg(
        cancer=("cancer", "max"), site_id=("site_id", "first"), faixa=("faixa_etaria", "first")
    )
    tabela = pd.DataFrame(
        {
            "pacientes": pac.groupby(level=0).size(),
            "imagens": df.groupby(por).size(),
            "pac_cancer_%": pac.groupby(level=0)["cancer"].mean() * 100,
            "img_cancer_%": df.groupby(por)["cancer"].mean() * 100,
        }
    )

    for site in sorted(df["site_id"].unique()):
        tabela[f"site{site}_%"] = pac.groupby(level=0)["site_id"].apply(
            lambda s, site=site: (s == site).mean() * 100
        )

    faixas = pd.crosstab(pac.index.get_level_values(0), pac["faixa"], normalize="index") * 100
    faixas.columns = [f"{c}_%" for c in faixas.columns]
    return pd.concat([tabela, faixas], axis=1).round(2)


def selecionar_fold0(
    df: pd.DataFrame,
    pool_negativos: int,
    semente: int,
    excluir_imagens: list[int] | None = None,
) -> pd.DataFrame:
    """
    Imagens do fold 0: teste e validação inteiros; treino com todos os
    positivos e um pool fixo de negativos.

    O pool inclui obrigatoriamente os negativos do treino com
    ``difficult_negative_case`` ou ``biopsy``, e completa com negativos
    comuns na ordem de uma fila sorteada. A subamostragem por época fica no
    DataLoader.

    ``df`` é a divisão **antes** da exclusão manual: a fila é sorteada sobre
    todos os candidatos e só depois os excluídos saem dela. Assim, excluir
    uma imagem do pool puxa a próxima da fila e não mexe nas outras.
    """
    df = ordenar(df)
    treino = df["split"] == "train"
    negativo = df["cancer"] == 0
    obrigatorio = treino & negativo & (df["difficult_negative_case"] | (df["biopsy"] == 1))
    comum = treino & negativo & ~obrigatorio

    fila = pd.Series(
        np.random.default_rng(semente).permutation(int(comum.sum())), index=df.index[comum]
    )

    excluida = df["image_id"].isin(excluir_imagens or [])
    obrigatorio &= ~excluida
    fila = fila[~excluida.loc[fila.index]]

    n_obrigatorios = int(obrigatorio.sum())
    if n_obrigatorios > pool_negativos:
        raise ValueError(
            f"{n_obrigatorios} negativos obrigatórios não cabem no pool de {pool_negativos}."
        )

    faltam = pool_negativos - n_obrigatorios
    if faltam > len(fila):
        raise ValueError(
            f"Pool de {pool_negativos} pede {faltam} negativos comuns; há só {len(fila)}."
        )

    df["in_negative_pool"] = obrigatorio
    df.loc[fila.nsmallest(faltam).index, "in_negative_pool"] = True

    manter = ~excluida & (
        (df["split"] != "train") | (df["cancer"] == 1) | df["in_negative_pool"]
    )
    return df.loc[manter, COLUNAS_MANIFEST].reset_index(drop=True)
