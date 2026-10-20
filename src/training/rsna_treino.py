"""Laço de treino do RSNA: amostragem, acumulação, EMA, validação por mama e retomada.

``treinar`` recebe o modelo e uma fábrica de Datasets, então os testes rodam o
mesmo laço com um modelo minúsculo na CPU. Os scripts montam tudo a partir das
configs (treino + Etapa 2) com ``src.training.rsna_execucao``.

Orçamento (``planejar_orcamento``): o treino é dividido em segmentos, e cada
segmento termina numa validação.

- ``orcamento: epocas``: o segmento é uma época de amostragem da proporção do
  modelo (1:1 tem ~5x menos imagens que 1:10).
- ``orcamento: passos``: o segmento tem um número fixo de passos do otimizador,
  por padrão o de uma época de referência (proporção 1:10), qualquer que seja a
  proporção do treino. Tentativas com proporções diferentes comparam o mesmo
  número de passos, e a validação, o early stopping, o warmup e o scheduler
  ficam na mesma escala.

Na tabela ``historico.csv``, ``epoca`` é o número do segmento (da validação).

Reprodutibilidade: a lista de imagens de cada segmento, a augmentation e o
dropout dependem só de (semente, segmento). Retomar do ``ultimo.pt`` dá o mesmo
treino que não ter parado (byte a byte na CPU; na GPU, a menos do não
determinismo do cuDNN).
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.evaluation import rsna_metricas as metricas
from src.models.rsna_ema import ModeloEMA
from src.models.rsna_modelo import criar_otimizador, criar_scheduler, passos_de_aquecimento
from src.training.rsna_amostragem import indices_da_epoca, indices_do_segmento
from src.training.rsna_config import ConfigTreino, recusar_teste
from src.training.rsna_registro import Registro, curvas_roc_pr, gravar_json

Fabrica = Callable[[pd.DataFrame, bool], Dataset]
Amostrador = Callable[[int], np.ndarray]


@dataclass
class Execucao:
    """Onde e como um treino roda. ``nome`` é também o nome do run no MLflow."""

    nome: str
    modo: str  # final | ajuste | smoke | sanidade
    semente: int
    run_dir: Path
    checkpoint_dir: Path
    epocas: int  # número de validações (segmentos)
    paciencia: int | None  # validações sem melhora até parar
    workers: int = 8
    salvar_checkpoints: bool = True
    # Imagens por segmento no orçamento por passos; None: uma época de amostragem.
    imagens_por_segmento: int | None = None


@dataclass(frozen=True)
class Orcamento:
    unidade: str  # passos | epocas
    validacoes: int
    passos_por_validacao: int
    passos_total: int
    imagens_por_segmento: int | None
    paciencia: int


def planejar_orcamento(
    cfg: ConfigTreino, treino: pd.DataFrame, modo: str, semente: int = 0
) -> Orcamento:
    """
    Validações, passos e paciência de um treino ``final`` ou de ``ajuste``.

    Por passos: ``passos_por_validacao`` vazio vale os passos de uma época de
    referência (todos os positivos + ``proporcao_referencia`` negativos por
    positivo, no batch efetivo do modelo); ``max_passos`` vazio vale
    ``max_epocas`` (ou ``ajuste.epocas``) épocas de referência. O total é
    arredondado para cima até uma validação inteira.
    """
    hp = cfg.do_modelo
    ajuste = modo == "ajuste"
    epocas = cfg.epocas_ajuste if ajuste else cfg.max_epocas
    max_passos = cfg.max_passos_ajuste if ajuste else cfg.max_passos
    paciencia = cfg.paciencia_ajuste if ajuste else cfg.paciencia

    def passos_de(n_imagens: int) -> int:
        return len(tamanhos_dos_grupos(n_imagens, hp.batch_real, hp.acumulacao))

    def imagens_da_epoca(proporcao: float) -> int:
        return len(indices_da_epoca(treino, semente, 0, proporcao, hp.peso_negativo_dificil))

    if cfg.orcamento == "epocas":
        passos = passos_de(imagens_da_epoca(hp.negativos_por_positivo))
        return Orcamento("epocas", epocas, passos, passos * epocas, None, paciencia)
    if cfg.passos_por_validacao:
        por_validacao = int(cfg.passos_por_validacao)
        imagens = por_validacao * hp.batch_efetivo
    else:
        imagens = imagens_da_epoca(cfg.proporcao_referencia)
        por_validacao = passos_de(imagens)
    validacoes = math.ceil(int(max_passos or epocas * por_validacao) / por_validacao)
    return Orcamento(
        "passos", validacoes, por_validacao, validacoes * por_validacao, imagens, paciencia
    )


# ---------------------------------------------------------------------------
# Precisão e passos
# ---------------------------------------------------------------------------


def escolher_precisao(precisao: str, dispositivo: torch.device) -> tuple[torch.dtype | None, bool]:
    """(dtype do autocast ou None, usa GradScaler). Na CPU, sempre fp32."""
    if dispositivo.type != "cuda" or precisao == "fp32":
        return None, False
    if precisao == "bf16" or (precisao == "auto" and torch.cuda.is_bf16_supported()):
        return torch.bfloat16, False
    return torch.float16, True


def tamanhos_dos_grupos(n_total: int, batch_real: int, acumulacao: int) -> list[int]:
    """Imagens em cada passo do otimizador; o último grupo da época pode ser menor."""
    micro = [min(batch_real, n_total - i) for i in range(0, n_total, batch_real)]
    return [sum(micro[i : i + acumulacao]) for i in range(0, len(micro), acumulacao)]


def epoca_de_treino(
    modelo: nn.Module,
    carregador: DataLoader,
    otimizador: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    scaler: torch.amp.GradScaler | None,
    criterio: nn.Module,
    acumulacao: int,
    dispositivo: torch.device,
    autocast: torch.dtype | None = None,
    channels_last: bool = False,
    ema: ModeloEMA | None = None,
) -> dict[str, float]:
    """
    Uma passada (um segmento) com acumulação de gradiente; a EMA anda a cada passo.

    A loss de cada micro-batch é a soma dividida pelo tamanho do grupo de
    acumulação, então ``batch_real x acumulacao`` dá o mesmo gradiente que um
    batch único do tamanho efetivo (exato sem BatchNorm; com BatchNorm, as
    estatísticas são as do micro-batch).
    """
    modelo.train()
    n_total = len(carregador.sampler)
    grupos = tamanhos_dos_grupos(n_total, carregador.batch_size, acumulacao)
    soma_loss, vistas, inicio = 0.0, 0, time.perf_counter()
    aquecido, vistas_aquecido = None, 0
    otimizador.zero_grad(set_to_none=True)
    for i, (imagens, rotulos) in enumerate(carregador):
        imagens = imagens.to(dispositivo, non_blocking=True)
        if channels_last:
            imagens = imagens.contiguous(memory_format=torch.channels_last)
        rotulos = rotulos.to(dispositivo, non_blocking=True)
        with torch.autocast(dispositivo.type, dtype=autocast, enabled=autocast is not None):
            logits = modelo(imagens).squeeze(1)
        loss_soma = criterio(logits.float(), rotulos)
        grupo = grupos[i // acumulacao]
        loss = loss_soma / grupo
        (scaler.scale(loss) if scaler else loss).backward()
        soma_loss += float(loss_soma.detach())
        vistas += len(rotulos)
        fim_do_grupo = (i + 1) % acumulacao == 0 or vistas == n_total
        if fim_do_grupo:
            if scaler:
                scaler.step(otimizador)
                scaler.update()
            else:
                otimizador.step()
            otimizador.zero_grad(set_to_none=True)
            if ema is not None:
                ema.atualizar(modelo)
            if scheduler is not None:
                scheduler.step()
            if aquecido is None:
                # Depois do 1º passo: sem a partida dos workers e o cudnn.benchmark.
                if dispositivo.type == "cuda":
                    torch.cuda.synchronize()
                aquecido, vistas_aquecido = time.perf_counter(), vistas
    if dispositivo.type == "cuda":
        torch.cuda.synchronize()
    fim = time.perf_counter()
    segundos = fim - inicio
    em_regime = vistas - vistas_aquecido
    return {
        "loss_treino": soma_loss / max(vistas, 1),
        "segundos_treino": segundos,
        "imagens_s": vistas / segundos,
        "imagens_s_regime": (
            em_regime / (fim - aquecido) if em_regime and fim > aquecido else float("nan")
        ),
    }


@torch.no_grad()
def prever(
    modelo: nn.Module,
    carregador: DataLoader,
    dispositivo: torch.device,
    autocast: torch.dtype | None = None,
    channels_last: bool = False,
    tempos: dict[str, float] | None = None,
) -> np.ndarray:
    """Probabilidades na ordem do carregador; ``tempos`` recebe a vazão sem o 1º batch."""
    modelo.eval()
    saidas = []
    inicio = None
    for imagens, _ in carregador:
        imagens = imagens.to(dispositivo, non_blocking=True)
        if channels_last:
            imagens = imagens.contiguous(memory_format=torch.channels_last)
        with torch.autocast(dispositivo.type, dtype=autocast, enabled=autocast is not None):
            logits = modelo(imagens).squeeze(1)
        saidas.append(torch.sigmoid(logits.float()).cpu().numpy())
        if inicio is None:
            inicio, vistas_antes = time.perf_counter(), len(saidas[0])
    if tempos is not None and inicio is not None:
        vistas = sum(len(s) for s in saidas) - vistas_antes
        segundos = time.perf_counter() - inicio
        tempos["imagens_s_regime"] = vistas / segundos if vistas and segundos > 0 else float("nan")
    return np.concatenate(saidas) if saidas else np.zeros(0)


def predicoes_por_imagem(manifest: pd.DataFrame, prob: np.ndarray) -> pd.DataFrame:
    imagens = manifest[metricas.COLUNAS_IMAGEM[:-1]].copy()
    imagens["cancer"] = imagens.cancer.astype(int)
    imagens["prob"] = prob
    return imagens


# ---------------------------------------------------------------------------
# Treino
# ---------------------------------------------------------------------------


def _carregador(
    dataset: Dataset,
    indices: np.ndarray | range,
    batch: int,
    workers: int,
    semente: int,
    dispositivo: torch.device,
) -> DataLoader:
    from src.data.rsna_dataset import iniciar_worker

    return DataLoader(
        dataset,
        batch_size=batch,
        sampler=list(indices),
        num_workers=workers,
        pin_memory=dispositivo.type == "cuda",
        worker_init_fn=iniciar_worker if workers else None,
        persistent_workers=False,
        generator=torch.Generator().manual_seed(semente),
    )


def semente_da_epoca(semente: int, epoca: int) -> int:
    return int(np.random.SeedSequence([semente, epoca, 1]).generate_state(1)[0])


def treinar(
    modelo: nn.Module,
    cfg: ConfigTreino,
    execucao: Execucao,
    treino: pd.DataFrame,
    validacao: pd.DataFrame,
    fabrica: Fabrica,
    registro: Registro | None = None,
    params: dict[str, Any] | None = None,
    tags: dict[str, Any] | None = None,
    amostrador: Amostrador | None = None,
    dispositivo: torch.device | None = None,
    retomar: bool = False,
    ao_fim_da_epoca: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """
    Treina até ``execucao.epocas`` validações ou o early stopping e grava runs/ e checkpoints.

    ``amostrador(e)`` dá as imagens da época de amostragem ``e``; com
    ``execucao.imagens_por_segmento``, os segmentos são fatias de tamanho fixo
    do fluxo dessas épocas. Com ``ema`` na config, a validação e o
    ``melhor.pt`` usam os pesos da EMA.

    Devolve o resumo (melhor validação, métricas e, nos modos final e ajuste,
    os intervalos de confiança).
    """
    recusar_teste(treino)
    recusar_teste(validacao)
    hp = cfg.do_modelo
    dispositivo = dispositivo or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    registro = registro or Registro({})
    autocast, usa_scaler = escolher_precisao(cfg.precisao, dispositivo)
    channels_last = cfg.channels_last and dispositivo.type == "cuda"
    modelo = modelo.to(dispositivo)
    if channels_last:
        modelo = modelo.to(memory_format=torch.channels_last)

    if amostrador is None:

        def amostrador(epoca: int) -> np.ndarray:
            return indices_da_epoca(
                treino, execucao.semente, epoca, hp.negativos_por_positivo, hp.peso_negativo_dificil
            )

    n_epoca = len(amostrador(0))
    tamanho = execucao.imagens_por_segmento
    if tamanho is None:
        segmento_de = amostrador
    else:

        def segmento_de(k: int) -> np.ndarray:
            return indices_do_segmento(amostrador, k, tamanho)

    n_segmento = len(segmento_de(0))
    passos = len(tamanhos_dos_grupos(n_segmento, hp.batch_real, hp.acumulacao))
    otimizador = criar_otimizador(modelo, hp)
    scheduler = criar_scheduler(otimizador, hp, passos, execucao.epocas)
    ema = (
        ModeloEMA(modelo, hp.ema_decaimento, passos_de_aquecimento(hp, passos)) if hp.ema else None
    )
    avaliado = ema.modelo if ema else modelo
    scaler = torch.amp.GradScaler("cuda") if usa_scaler else None
    criterio = nn.BCEWithLogitsLoss(
        reduction="sum", pos_weight=torch.tensor(hp.pos_weight, device=dispositivo)
    )

    estado = {
        "epoca": -1,
        "melhor": -math.inf,
        "melhor_epoca": -1,
        "sem_melhora": 0,
        "historico": [],
        "run_id": None,
    }
    execucao.run_dir.mkdir(parents=True, exist_ok=True)
    execucao.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ultimo = execucao.checkpoint_dir / "ultimo.pt"
    if retomar:
        if not ultimo.is_file():
            raise FileNotFoundError(f"Nada para retomar: {ultimo} não existe")
        # Na CPU: carregado na GPU, o checkpoint inteiro ficaria na VRAM durante o treino.
        salvo = torch.load(ultimo, map_location="cpu", weights_only=False)
        modelo.load_state_dict(salvo["modelo"])
        otimizador.load_state_dict(salvo["otimizador"])
        scheduler.load_state_dict(salvo["scheduler"])
        if scaler and salvo.get("scaler"):
            scaler.load_state_dict(salvo["scaler"])
        if ema is not None:
            ema.load_state_dict(salvo["ema"])
        estado = salvo["estado"]
        del salvo
        print(f"retomando de {ultimo}: validação {estado['epoca'] + 1} de {execucao.epocas}")
    elif ultimo.is_file():
        raise FileExistsError(f"{ultimo} já existe: use --retomar para continuar ou apague a pasta")

    tags = {**(tags or {}), "checkpoint_dir": execucao.checkpoint_dir}
    registro.iniciar(execucao.nome, params or {}, tags, estado["run_id"])
    estado["run_id"] = registro.run_id
    lote_val = hp.batch_real * 2
    validacao = validacao.reset_index(drop=True)
    ds_treino, ds_val = fabrica(treino, True), fabrica(validacao, False)
    carregador_val = _carregador(
        ds_val, range(len(ds_val)), lote_val, execucao.workers, 0, dispositivo
    )

    try:
        for epoca in range(estado["epoca"] + 1, execucao.epocas):
            if execucao.paciencia is not None and estado["sem_melhora"] >= execucao.paciencia:
                break  # retomado depois de já ter parado pelo early stopping
            semente = semente_da_epoca(execucao.semente, epoca)
            torch.manual_seed(semente)
            if dispositivo.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            indices = segmento_de(epoca)
            carregador = _carregador(
                ds_treino, indices, hp.batch_real, execucao.workers, semente, dispositivo
            )
            linha: dict[str, Any] = {"epoca": epoca, "lr": otimizador.param_groups[0]["lr"]}
            linha.update(
                epoca_de_treino(
                    modelo, carregador, otimizador, scheduler, scaler, criterio,
                    hp.acumulacao, dispositivo, autocast, channels_last, ema,
                )  # fmt: skip
            )
            linha["passo"] = (epoca + 1) * passos
            linha["epocas_amostragem"] = (epoca + 1) * n_segmento / n_epoca
            inicio = time.perf_counter()
            tempos: dict[str, float] = {}
            prob = prever(avaliado, carregador_val, dispositivo, autocast, channels_last, tempos)
            linha["segundos_validacao"] = time.perf_counter() - inicio
            linha["imagens_s_validacao_regime"] = tempos.get("imagens_s_regime", float("nan"))
            imagens = predicoes_por_imagem(validacao, prob)
            mamas = metricas.por_mama(imagens)
            linha.update({f"val_{k}": v for k, v in metricas.calcular(mamas, cfg.betas_f).items()})
            if dispositivo.type == "cuda":
                linha["vram_pico_gb"] = torch.cuda.max_memory_allocated() / 2**30
            valor = linha[f"val_{cfg.metrica_selecao}"]
            valor = -math.inf if valor is None or valor != valor else valor
            melhorou = valor > estado["melhor"]
            if melhorou:
                estado.update(melhor=valor, melhor_epoca=epoca, sem_melhora=0)
                imagens.to_csv(execucao.run_dir / "predicoes_val_imagem.csv", index=False)
                mamas.to_csv(execucao.run_dir / "predicoes_val_mama.csv", index=False)
                if execucao.salvar_checkpoints:
                    torch.save(
                        {
                            "modelo": avaliado.state_dict(),
                            "pesos": "ema" if ema else "modelo",
                            "epoca": epoca,
                            "metricas": linha,
                            "params": params,
                            "tags": {k: str(v) for k, v in tags.items()},
                        },
                        execucao.checkpoint_dir / "melhor.pt",
                    )
            else:
                estado["sem_melhora"] += 1
            linha["melhorou"] = melhorou
            estado["historico"].append(linha)
            estado["epoca"] = epoca
            pd.DataFrame(estado["historico"]).to_csv(
                execucao.run_dir / "historico.csv", index=False
            )
            registro.epoca(
                {k: v for k, v in linha.items() if isinstance(v, int | float)}, linha["passo"]
            )
            if execucao.salvar_checkpoints:
                torch.save(
                    {
                        "modelo": modelo.state_dict(),
                        "otimizador": otimizador.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "scaler": scaler.state_dict() if scaler else None,
                        "ema": ema.state_dict() if ema else None,
                        "estado": estado,
                        "params": params,
                    },
                    ultimo,
                )
            print(
                f"validação {epoca} (passo {linha['passo']}): loss {linha['loss_treino']:.4f} | "
                f"val {cfg.metrica_selecao} {linha[f'val_{cfg.metrica_selecao}']:.4f} "
                f"roc_auc {linha['val_roc_auc']:.4f} esp@sens90 {linha['val_esp_sens90']:.4f} | "
                f"{linha['imagens_s']:.1f} img/s | {linha['segundos_treino'] + linha['segundos_validacao']:.0f} s"
                + (f" | {linha['vram_pico_gb']:.2f} GB" if "vram_pico_gb" in linha else "")
                + (" *" if melhorou else ""),
                flush=True,
            )
            if ao_fim_da_epoca:
                ao_fim_da_epoca(linha)
            if execucao.paciencia is not None and estado["sem_melhora"] >= execucao.paciencia:
                print(f"early stopping: {execucao.paciencia} validações sem melhora")
                break
    except BaseException:
        registro.encerrar("KILLED")
        raise

    resumo = resumir(cfg, execucao, estado)
    resumo["passos_por_validacao"] = passos
    resumo["imagens_por_validacao"] = n_segmento
    resumo["ema"] = hp.ema
    resumo["mlflow_run_id"] = registro.run_id
    gravar_json(execucao.run_dir / "metricas_finais.json", resumo)
    if (execucao.run_dir / "predicoes_val_mama.csv").is_file():
        mamas = pd.read_csv(execucao.run_dir / "predicoes_val_mama.csv")
        if 0 < mamas.cancer.sum() < len(mamas):
            curvas_roc_pr(mamas, execucao.run_dir / "curvas_roc_pr.png", execucao.nome)
    registro.resumo(
        {f"melhor_{k}": v for k, v in resumo["validacao"].items() if isinstance(v, int | float)}
    )
    registro.artefatos(execucao.run_dir)
    registro.encerrar()
    return resumo


def resumir(cfg: ConfigTreino, execucao: Execucao, estado: dict[str, Any]) -> dict[str, Any]:
    """Métricas da melhor época, com IC por bootstrap (por mama) nos modos final e ajuste."""
    historico = pd.DataFrame(estado["historico"])
    melhor = (
        historico[historico.epoca == estado["melhor_epoca"]].iloc[0] if len(historico) else None
    )
    validacao = {}
    if melhor is not None:
        validacao = {
            k.removeprefix("val_"): melhor[k] for k in historico.columns if k.startswith("val_")
        }
    resumo: dict[str, Any] = {
        "nome": execucao.nome,
        "modo": execucao.modo,
        "semente": execucao.semente,
        "modelo": cfg.modelo,
        "timm_nome": cfg.do_modelo.timm_nome,
        "batch_real": cfg.do_modelo.batch_real,
        "acumulacao": cfg.do_modelo.acumulacao,
        "batch_efetivo": cfg.do_modelo.batch_efetivo,
        "metrica_selecao": cfg.metrica_selecao,
        "limiar_selecao": (
            metricas.limiar_da_selecao(cfg.metrica_selecao, validacao, cfg.betas_f)
            if validacao
            else None
        ),
        "melhor_epoca": int(estado["melhor_epoca"]),
        "melhor_passo": int(melhor["passo"]) if melhor is not None else 0,
        "epocas_treinadas": len(historico),
        "parou_por_paciencia": bool(
            execucao.paciencia is not None and estado["sem_melhora"] >= execucao.paciencia
        ),
        "segundos_total": (
            float(historico.segundos_treino.sum() + historico.segundos_validacao.sum())
            if len(historico)
            else 0.0
        ),
        "validacao": validacao,
        "ic95_validacao": {},
    }
    caminho = execucao.run_dir / "predicoes_val_mama.csv"
    if execucao.modo in ("final", "ajuste") and caminho.is_file():
        mamas = pd.read_csv(caminho)
        if 0 < mamas.cancer.sum() < len(mamas):
            resumo["ic95_validacao"] = metricas.bootstrap(
                mamas,
                cfg.bootstrap_amostras,
                cfg.semente_bootstrap,
                metricas.metricas_reportadas(cfg.betas_f),
                betas=cfg.betas_f,
            )
    return resumo
