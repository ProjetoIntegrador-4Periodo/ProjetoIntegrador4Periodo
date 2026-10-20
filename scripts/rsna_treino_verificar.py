"""Verificações antes de qualquer treino completo: smoke test, sanidade e maior batch.

Cada medição roda num processo próprio, para a memória de uma não contaminar a
outra. Não grava nada em runs/ nem no MLflow. Rode a partir da raiz:

    python -m scripts.rsna_treino_verificar maior-batch   # maior batch_real que cabe na GPU
    python -m scripts.rsna_treino_verificar smoke         # 1 época curta, cada modelo x representação
    python -m scripts.rsna_treino_verificar sanidade      # decorar 64 imagens até loss ~0

Opções: --modelos convnext_tiny resnet50 (padrão: todos da config),
--representacoes cinza realce_3canais. Resultados em caminhos.verificacao_dir.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.models.rsna_modelo import criar_modelo, criar_otimizador
from src.training.rsna_amostragem import indices_da_epoca
from src.training.rsna_execucao import (
    carregar_configs,
    dados_do_treino,
    fabrica_de_datasets,
    ler_alteracoes,
)
from src.training.rsna_treino import Execucao, epoca_de_treino, escolher_precisao, treinar
from src.utils.paths import CONFIGS_DIR, PROJECT_ROOT
from src.utils.seed import set_seed

BATCHES = (4, 8, 12, 16, 20, 24, 32, 48, 64)
N_SMOKE_TREINO, N_SMOKE_VAL = 500, 200
N_SANIDADE, EPOCAS_SANIDADE, LOSS_DECORADA = 64, 100, 0.02
PREFIXO = "RESULTADO "


# ---------------------------------------------------------------------------
# Medições (processo filho)
# ---------------------------------------------------------------------------


def medir_batch(config: Path, modelo: str, batch: int) -> dict:
    """3 passos de treino com imagens aleatórias no tamanho do canvas."""
    cfg, cfg2 = carregar_configs(config, modelo, alteracoes={"batch_real": batch, "acumulacao": 1})
    dispositivo = torch.device("cuda")
    autocast, usa_scaler = escolher_precisao(cfg.precisao, dispositivo)
    rede = criar_modelo(cfg.do_modelo, pretrained=False).to(
        dispositivo, memory_format=torch.channels_last
    )
    imagens = torch.randn(batch * 3, 3, cfg2.entrada.altura, cfg2.entrada.largura)
    dataset = torch.utils.data.TensorDataset(imagens, torch.zeros(batch * 3))
    carregador = torch.utils.data.DataLoader(
        dataset, batch_size=batch, sampler=list(range(batch * 3))
    )
    torch.cuda.reset_peak_memory_stats()
    try:
        epoca_de_treino(
            rede, carregador, criar_otimizador(rede, cfg.do_modelo), None,
            torch.amp.GradScaler("cuda") if usa_scaler else None,
            torch.nn.BCEWithLogitsLoss(reduction="sum"), 1, dispositivo, autocast, True,
        )  # fmt: skip
        torch.cuda.synchronize()
    except torch.OutOfMemoryError:
        return {"modelo": modelo, "batch_real": batch, "cabe": False}
    return {
        "modelo": modelo,
        "batch_real": batch,
        "cabe": True,
        "vram_pico_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "vram_total_gb": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2),
    }


def medir_smoke(config: Path, modelo: str, representacao: str, saida: Path) -> dict:
    """1 época com ~500 imagens do treino (amostragem real) e ~200 da validação."""
    cfg, cfg2 = carregar_configs(config, modelo, representacao)
    hp = cfg.do_modelo
    treino, validacao = dados_do_treino(cfg2)
    epoca_real = indices_da_epoca(treino, 0, 0, hp.negativos_por_positivo, hp.peso_negativo_dificil)
    # Múltiplos do batch, como na época real (8.580 = 715 x 12): um último batch
    # menor tem formato novo e dispara outro cudnn.benchmark no meio da medição.
    subconjunto = epoca_real[: N_SMOKE_TREINO // hp.batch_efetivo * hp.batch_efetivo]
    val = validacao.sample(N_SMOKE_VAL // (2 * hp.batch_real) * 2 * hp.batch_real, random_state=0)
    set_seed(0, deterministic=False)
    torch.backends.cudnn.benchmark = True
    execucao = Execucao(
        nome=f"smoke/{modelo}_{representacao}", modo="smoke", semente=0,
        run_dir=saida / "smoke" / f"{modelo}_{representacao}",
        checkpoint_dir=saida / "smoke" / f"{modelo}_{representacao}" / "ckpt",
        epocas=1, paciencia=None, workers=int(cfg2.carregamento.get("workers", 8)), salvar_checkpoints=False,
    )  # fmt: skip
    inicio = time.perf_counter()
    treinar(
        criar_modelo(hp), cfg, execucao, treino, val, fabrica_de_datasets(cfg2),
        amostrador=lambda _epoca: subconjunto,
    )  # fmt: skip
    historico = pd.read_csv(execucao.run_dir / "historico.csv").iloc[0]
    # Estimativa pela vazão em regime: a partida (workers, cudnn.benchmark) pesa
    # muito em 500 imagens e pouco numa época de 8.580.
    treino_s, val_s = historico.imagens_s_regime, historico.imagens_s_validacao_regime
    epoca_completa = len(epoca_real) / treino_s + len(validacao) / val_s
    return {
        "modelo": modelo,
        "representacao": representacao,
        "batch": f"{hp.batch_real}x{hp.acumulacao}={hp.batch_efetivo}",
        "imagens_s_treino": round(treino_s, 1),
        "imagens_s_validacao": round(val_s, 1),
        "imagens_s_treino_com_partida": round(historico.imagens_s, 1),
        "vram_pico_gb": round(historico.vram_pico_gb, 2),
        "loss": round(historico.loss_treino, 4),
        "segundos_smoke": round(time.perf_counter() - inicio, 1),
        "imagens_smoke": f"{len(subconjunto)} treino / {len(val)} validação",
        "imagens_por_epoca": len(epoca_real),
        "imagens_validacao": len(validacao),
        "min_por_epoca_estimado": round(epoca_completa / 60, 1),
        f"horas_{cfg.max_epocas}_epocas": round(epoca_completa * cfg.max_epocas / 3600, 1),
    }


class Decorou(Exception):
    """Interrompe o teste de sanidade assim que a loss chega ao alvo."""


def medir_sanidade(config: Path, modelo: str, saida: Path, extras: dict | None = None) -> dict:
    """
    64 imagens fixas (metade positiva): a loss tem de ir a ~0.

    Só confirma que o laço aprende: sem augmentation, drop path, weight decay nem
    EMA, sem acumulação e com warmup curto (sem warmup, o ConvNeXt com AdamW 1e-4
    colapsa para uma resposta constante e não decora).
    """
    cfg, cfg2 = carregar_configs(config, modelo, "cinza")
    cfg = cfg.com_hiperparametros(
        {
            "drop_path": 0.0,
            "weight_decay": 0.0,
            # Curto, mas não 1 época: com 6 passos o ConvNeXt (AdamW 1e-4) colapsa.
            "warmup_epocas": 5,  # 30 passos (64 imagens, batch 12)
            "scheduler": "constante",
            "pos_weight": 1.0,
            "ema": False,
            # Sem acumulação: com 64 imagens e batch 36 seriam só 2 passos por época.
            "acumulacao": 1,
        }
    )
    extras_originais, extras = extras, dict(extras or {})
    if "precisao" in extras:
        cfg = replace(cfg, precisao=extras.pop("precisao"))
    if extras:
        cfg = cfg.com_hiperparametros(extras)
    treino, _ = dados_do_treino(cfg2)
    metade = N_SANIDADE // 2
    pequeno = pd.concat(
        [
            treino[treino.cancer == 1].sample(metade, random_state=0),
            treino[treino.cancer == 0].sample(metade, random_state=0),
        ]
    ).reset_index(drop=True)
    sem_aug = fabrica_de_datasets(cfg2)
    variante = modelo + "".join(f"_{k}{v}" for k, v in sorted((extras_originais or {}).items()))
    set_seed(0, deterministic=False)
    torch.backends.cudnn.benchmark = True
    perdas: list[float] = []

    def parar_quando_decorar(linha):
        perdas.append(linha["loss_treino"])
        if linha["loss_treino"] < LOSS_DECORADA:
            raise Decorou

    execucao = Execucao(
        nome=f"sanidade/{variante}", modo="sanidade", semente=0,
        run_dir=saida / "sanidade" / variante, checkpoint_dir=saida / "sanidade" / variante / "ckpt",
        epocas=EPOCAS_SANIDADE, paciencia=None, workers=int(cfg2.carregamento.get("workers", 8)),
        salvar_checkpoints=False,
    )  # fmt: skip
    inicio = time.perf_counter()
    with contextlib.suppress(Decorou):
        treinar(
            criar_modelo(cfg.do_modelo), cfg, execucao, pequeno, pequeno,
            lambda manifest, _treino: sem_aug(manifest, False),
            amostrador=lambda epoca: np.random.default_rng(epoca).permutation(len(pequeno)),
            ao_fim_da_epoca=parar_quando_decorar,
        )  # fmt: skip
    historico = pd.read_csv(execucao.run_dir / "historico.csv")
    return {
        "modelo": modelo,
        "otimizador": cfg.do_modelo.otimizador,
        "lr": cfg.do_modelo.lr,
        "precisao": cfg.precisao,
        "warmup_epocas": cfg.do_modelo.warmup_epocas,
        "epocas": len(perdas),
        "loss_inicial": round(perdas[0], 4),
        "loss_final": round(perdas[-1], 4),
        "decorou": perdas[-1] < LOSS_DECORADA,
        "roc_auc_no_subconjunto": round(historico.val_roc_auc.iloc[-1], 4),
        "segundos": round(time.perf_counter() - inicio, 1),
        "perdas": [round(p, 4) for p in perdas],
    }


# ---------------------------------------------------------------------------
# Orquestração (processo principal)
# ---------------------------------------------------------------------------


def em_processo_novo(config: Path, *argumentos: str) -> dict:
    comando = [
        sys.executable,
        "-m",
        "scripts.rsna_treino_verificar",
        "--config",
        str(config),
        "_interno",
        *argumentos,
    ]
    processo = subprocess.run(comando, cwd=PROJECT_ROOT, capture_output=True, text=True)
    for linha in processo.stdout.splitlines():
        if linha.startswith(PREFIXO):
            return json.loads(linha.removeprefix(PREFIXO))
    sys.stderr.write(processo.stdout[-3000:] + processo.stderr[-3000:])
    raise RuntimeError(f"falhou: {' '.join(argumentos)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=CONFIGS_DIR / "treino.yaml")
    parser.add_argument("acao", choices=["maior-batch", "smoke", "sanidade", "_interno"])
    parser.add_argument("argumentos", nargs="*")
    parser.add_argument("--modelos", nargs="*")
    parser.add_argument("--representacoes", nargs="*", default=["cinza", "realce_3canais"])
    parser.add_argument(
        "--set", nargs="*", default=[], metavar="CHAVE=VALOR",
        help="sanidade: troca hiperparâmetros (e precisao) para diagnosticar",
    )  # fmt: skip
    args = parser.parse_args()

    cfg, _ = carregar_configs(args.config)
    saida = Path(cfg.caminhos["verificacao_dir"])
    saida.mkdir(parents=True, exist_ok=True)

    if args.acao == "_interno":
        tipo, *resto = args.argumentos
        if tipo == "batch":
            resultado = medir_batch(args.config, resto[0], int(resto[1]))
        elif tipo == "smoke":
            resultado = medir_smoke(args.config, resto[0], resto[1], saida)
        else:
            resultado = medir_sanidade(args.config, resto[0], saida, json.loads(resto[1]))
        print(PREFIXO + json.dumps(resultado), flush=True)
        return

    modelos = args.modelos or sorted(cfg.modelos)
    linhas = []
    if args.acao == "maior-batch":
        for modelo in modelos:
            for batch in BATCHES:
                linha = em_processo_novo(args.config, "batch", modelo, str(batch))
                print(linha, flush=True)
                linhas.append(linha)
                if not linha["cabe"]:
                    break
        tabela = pd.DataFrame(linhas)
        for modelo, grupo in tabela.groupby("modelo"):
            cabem = grupo[grupo.cabe]
            maior = cabem.batch_real.max() if len(cabem) else None
            print(f"{modelo}: maior batch_real que cabe = {maior}")
    elif args.acao == "smoke":
        for modelo in modelos:
            for representacao in args.representacoes:
                linha = em_processo_novo(args.config, "smoke", modelo, representacao)
                print(linha, flush=True)
                linhas.append(linha)
        tabela = pd.DataFrame(linhas)
    else:
        for modelo in modelos:
            linha = em_processo_novo(
                args.config, "sanidade", modelo, json.dumps(ler_alteracoes(args.set))
            )
            print({k: v for k, v in linha.items() if k != "perdas"}, flush=True)
            print(f"  loss por época: {linha['perdas']}", flush=True)
            linhas.append(linha)
        tabela = pd.DataFrame(linhas)

    destino = saida / f"{args.acao}.csv"
    tabela.to_csv(destino, index=False)
    print(f"\n{tabela.drop(columns=['perdas'], errors='ignore').to_string(index=False)}\n{destino}")


if __name__ == "__main__":
    main()
