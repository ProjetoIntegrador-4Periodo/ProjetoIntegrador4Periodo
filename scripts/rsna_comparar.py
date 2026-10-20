"""Comparação dos treinos finais e tabela das tentativas de ajuste.

Treinos finais: lê runs/<modelo>_<rep>_seed<N>/metricas_finais.json das duas
máquinas (depois de juntadas no git), com média e desvio entre seeds por modelo
e representação, e conta as tentativas de ajuste em runs/ajuste/.

Ajuste (``--ajuste``): uma linha por tentativa e, por configuração (as
alterações do ``--set``), a média e a diferença entre seeds, e se a melhor
validação estava no fim do orçamento (ainda melhorando). Rode a partir da raiz:

    python -m scripts.rsna_comparar             # runs/comparacao.csv e .md
    python -m scripts.rsna_comparar --ajuste    # runs/ajuste/ajuste.csv e .md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.evaluation.rsna_metricas import METRICAS
from src.training.rsna_config import caminho_do_repo, carregar_config_treino
from src.utils.paths import CONFIGS_DIR


def ler_runs(runs_dir: Path) -> pd.DataFrame:
    linhas = []
    for arquivo in sorted(runs_dir.glob("*/metricas_finais.json")):
        resumo = json.loads(arquivo.read_text(encoding="utf-8"))
        if resumo.get("modo") != "final":
            continue
        config = arquivo.parent / "config_etapa2.yaml"
        representacao = yaml.safe_load(config.read_text(encoding="utf-8"))["entrada"][
            "representacao"
        ]
        maquina = json.loads((arquivo.parent / "maquina.json").read_text(encoding="utf-8"))
        linha = {
            "run": arquivo.parent.name,
            "modelo": resumo["modelo"],
            "representacao": representacao,
            "semente": resumo["semente"],
            "batch_efetivo": resumo["batch_efetivo"],
            "melhor_epoca": resumo["melhor_epoca"],
            "epocas_treinadas": resumo["epocas_treinadas"],
            "horas": resumo["segundos_total"] / 3600,
            "gpu": maquina.get("gpu"),
        }
        for nome in METRICAS:
            linha[nome] = resumo["validacao"].get(nome)
            ic = resumo.get("ic95_validacao", {}).get(nome, {})
            linha[f"{nome}_ic_inf"] = ic.get("ic_inferior")
            linha[f"{nome}_ic_sup"] = ic.get("ic_superior")
        linhas.append(linha)
    return pd.DataFrame(linhas)


def contar_tentativas(runs_dir: Path) -> dict[str, int]:
    """Tentativas de ajuste concluídas, por modelo (as duas representações juntas)."""
    contagem: dict[str, int] = {}
    for arquivo in (runs_dir / "ajuste").glob("*_tentativa*/metricas_finais.json"):
        modelo = json.loads(arquivo.read_text(encoding="utf-8"))["modelo"]
        contagem[modelo] = contagem.get(modelo, 0) + 1
    return contagem


def comparar(runs: pd.DataFrame, tentativas: dict[str, int]) -> pd.DataFrame:
    linhas = []
    for (modelo, representacao), grupo in runs.groupby(["modelo", "representacao"], sort=True):
        linha = {
            "modelo": modelo,
            "representacao": representacao,
            "seeds": ",".join(str(s) for s in sorted(grupo.semente)),
            "n_seeds": len(grupo),
            "batch_efetivo": ",".join(str(b) for b in sorted(set(grupo.batch_efetivo))),
            "tentativas_ajuste": tentativas.get(modelo, 0),
            "epocas_media": grupo.epocas_treinadas.mean(),
            "horas_media": grupo.horas.mean(),
            "gpu": ",".join(sorted({str(g) for g in grupo.gpu})),
        }
        for nome in METRICAS:
            valores = grupo[nome].astype(float)
            linha[f"{nome}_media"] = valores.mean()
            linha[f"{nome}_desvio"] = valores.std(ddof=1) if len(valores) > 1 else np.nan
            # IC95 médio entre as seeds (o bootstrap é por seed, sobre as mamas da validação)
            linha[f"{nome}_ic_inf"] = grupo[f"{nome}_ic_inf"].astype(float).mean()
            linha[f"{nome}_ic_sup"] = grupo[f"{nome}_ic_sup"].astype(float).mean()
        linhas.append(linha)
    return pd.DataFrame(linhas)


def markdown(tabela: pd.DataFrame) -> str:
    cabecalho = [
        "modelo",
        "representação",
        "seeds",
        "batch efetivo",
        "tentativas de ajuste",
        *METRICAS,
    ]
    linhas = ["| " + " | ".join(cabecalho) + " |", "|" + "---|" * len(cabecalho)]
    for r in tabela.itertuples():
        celulas = [
            r.modelo,
            r.representacao,
            f"{r.n_seeds} ({r.seeds})",
            r.batch_efetivo,
            str(r.tentativas_ajuste),
        ]
        for nome in METRICAS:
            media, desvio = getattr(r, f"{nome}_media"), getattr(r, f"{nome}_desvio")
            inf, sup = getattr(r, f"{nome}_ic_inf"), getattr(r, f"{nome}_ic_sup")
            texto = f"{media:.3f}" + ("" if np.isnan(desvio) else f" ± {desvio:.3f}")
            if not np.isnan(inf):
                texto += f" [{inf:.3f}; {sup:.3f}]"
            celulas.append(texto)
        linhas.append("| " + " | ".join(celulas) + " |")
    nota = (
        "\n\nMétricas por mama na validação (fold 0), na melhor época de cada seed: média ± desvio "
        "entre seeds e, entre colchetes, o IC95 por bootstrap médio entre as seeds.\n"
    )
    return "\n".join(linhas) + nota


# ---------------------------------------------------------------------------
# Ajuste
# ---------------------------------------------------------------------------


def ler_tentativas(runs_dir: Path) -> pd.DataFrame:
    """Uma linha por tentativa de ajuste concluída, com as métricas da melhor validação."""
    linhas = []
    for arquivo in sorted((runs_dir / "ajuste").glob("*_tentativa*/metricas_finais.json")):
        pasta = arquivo.parent
        resumo = json.loads(arquivo.read_text(encoding="utf-8"))
        treino = yaml.safe_load((pasta / "config_treino.yaml").read_text(encoding="utf-8"))
        etapa2 = yaml.safe_load((pasta / "config_etapa2.yaml").read_text(encoding="utf-8"))
        alteracoes = treino.get("alteracoes") or {}
        metrica = resumo["metrica_selecao"]
        historico = pd.read_csv(pasta / "historico.csv")
        serie = historico[f"val_{metrica}"].astype(float)
        linha = {
            "tentativa": pasta.name,
            "modelo": resumo["modelo"],
            "representacao": etapa2["entrada"]["representacao"],
            "configuracao": ", ".join(f"{k}={v}" for k, v in sorted(alteracoes.items()))
            or "padrão",
            "semente": resumo["semente"],
            "metrica_selecao": metrica,
            "melhor_epoca": resumo["melhor_epoca"],
            "epocas_treinadas": resumo["epocas_treinadas"],
            "validacoes_previstas": treino["ajuste"]["epocas"],
            "parou_por_paciencia": resumo["parou_por_paciencia"],
            # Melhor validação numa das 2 últimas do orçamento e sem early stopping.
            "ainda_melhorando": bool(
                not resumo["parou_por_paciencia"]
                and resumo["melhor_epoca"] >= resumo["epocas_treinadas"] - 2
            ),
            "selecao_ultima": serie.iloc[-1],
            "horas": resumo["segundos_total"] / 3600,
        }
        for nome in METRICAS:
            linha[nome] = resumo["validacao"].get(nome)
        linhas.append(linha)
    return pd.DataFrame(linhas)


def resumir_ajuste(tentativas: pd.DataFrame) -> pd.DataFrame:
    """Por configuração: média e diferença (máx - mín) entre seeds, ordenado pela seleção."""
    linhas = []
    chave = ["modelo", "representacao", "configuracao"]
    for (modelo, representacao, configuracao), grupo in tentativas.groupby(chave, sort=True):
        metrica = grupo.metrica_selecao.iloc[0]
        linha = {
            "modelo": modelo,
            "representacao": representacao,
            "configuracao": configuracao,
            "seeds": ",".join(str(s) for s in sorted(grupo.semente)),
            "n_seeds": len(grupo),
            "tentativas": ",".join(grupo.tentativa.str.rsplit("_", n=1).str[-1]),
            "metrica_selecao": metrica,
            "ainda_melhorando": ",".join("sim" if v else "não" for v in grupo.ainda_melhorando),
            "melhor_epoca": ",".join(
                f"{a}/{b}" for a, b in zip(grupo.melhor_epoca, grupo.epocas_treinadas, strict=True)
            ),
        }
        for nome in METRICAS:
            valores = grupo[nome].astype(float)
            linha[f"{nome}_media"] = valores.mean()
            linha[f"{nome}_diferenca"] = (
                valores.max() - valores.min() if len(valores) > 1 else np.nan
            )
        linhas.append(linha)
    tabela = pd.DataFrame(linhas)
    if tabela.empty:
        return tabela
    ordem = f"{tabela.metrica_selecao.iloc[0]}_media"
    return tabela.sort_values(ordem, ascending=False).reset_index(drop=True)


def markdown_ajuste(tentativas: pd.DataFrame, resumo: pd.DataFrame) -> str:
    def numero(valor) -> str:
        return "-" if valor is None or valor != valor else f"{valor:.3f}"

    cabecalho = ["configuração", "seeds", *METRICAS, "melhor val./treinadas", "ainda melhorando"]
    linhas = ["| " + " | ".join(cabecalho) + " |", "|" + "---|" * len(cabecalho)]
    for r in resumo.itertuples():
        celulas = [f"{r.configuracao} ({r.representacao})", f"{r.n_seeds} ({r.seeds})"]
        for nome in METRICAS:
            media, dif = getattr(r, f"{nome}_media"), getattr(r, f"{nome}_diferenca")
            celulas.append(numero(media) + ("" if dif != dif else f" (Δ {dif:.3f})"))
        celulas += [r.melhor_epoca, r.ainda_melhorando]
        linhas.append("| " + " | ".join(celulas) + " |")
    por_tentativa = [
        "| tentativa | configuração | seed | " + " | ".join(METRICAS) + " | melhor val. |"
    ]
    por_tentativa.append("|" + "---|" * (len(METRICAS) + 4))
    for r in tentativas.itertuples():
        valores = " | ".join(numero(getattr(r, nome)) for nome in METRICAS)
        por_tentativa.append(
            f"| {r.tentativa} | {r.configuracao} | {r.semente} | {valores} | "
            f"{r.melhor_epoca}/{r.epocas_treinadas} |"
        )
    nota = (
        "\n\nMétricas por mama na validação (fold 0), na melhor validação de cada tentativa: "
        "média entre seeds e, entre parênteses, a diferença entre a maior e a menor. "
        'Ordenado pela métrica de seleção. "ainda melhorando": a melhor validação foi uma das '
        "2 últimas do orçamento, sem early stopping.\n\n"
    )
    return "\n".join(linhas) + nota + "\n".join(por_tentativa) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=CONFIGS_DIR / "treino.yaml")
    parser.add_argument("--ajuste", action="store_true", help="tabela das tentativas de ajuste")
    args = parser.parse_args()
    cfg = carregar_config_treino(args.config)
    runs_dir = caminho_do_repo(cfg.caminhos.get("runs_dir", "runs"))

    if args.ajuste:
        tentativas = ler_tentativas(runs_dir)
        if tentativas.empty:
            raise SystemExit(f"nenhuma tentativa concluída em {runs_dir / 'ajuste'}")
        resumo = resumir_ajuste(tentativas)
        texto = markdown_ajuste(tentativas, resumo)
        resumo.to_csv(runs_dir / "ajuste" / "ajuste.csv", index=False)
        (runs_dir / "ajuste" / "ajuste.md").write_text(texto, encoding="utf-8")
        print(texto)
        print(f"{runs_dir / 'ajuste' / 'ajuste.csv'}\n{runs_dir / 'ajuste' / 'ajuste.md'}")
        return

    runs = ler_runs(runs_dir)
    if runs.empty:
        raise SystemExit(f"nenhum treino final em {runs_dir}")
    tabela = comparar(runs, contar_tentativas(runs_dir))
    tabela.to_csv(runs_dir / "comparacao.csv", index=False)
    (runs_dir / "comparacao.md").write_text(markdown(tabela), encoding="utf-8")
    print(markdown(tabela))
    print(f"{runs_dir / 'comparacao.csv'}\n{runs_dir / 'comparacao.md'}")


if __name__ == "__main__":
    main()
