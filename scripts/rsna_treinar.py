"""Treino do RSNA (fold 0): ajuste de hiperparâmetros ou treino final.

Usa só o treino e a validação; o teste hold-out nunca é carregado. Rode a partir da raiz:

    # tentativa de ajuste (orçamento de ajuste.epocas épocas de 1:10 em passos),
    # trocando hiperparâmetros na hora
    python -m scripts.rsna_treinar --modo ajuste --modelo convnext_tiny --representacao cinza \\
        --seed 0 --set lr=5e-5 negativos_por_positivo=4

    # treino final com a config do configs/treino.yaml (sem --set), uma seed por vez
    python -m scripts.rsna_treinar --modo final --modelo convnext_tiny --representacao cinza --seed 1

    # caiu? o mesmo comando com --retomar (no ajuste, informe também --tentativa N)
    python -m scripts.rsna_treinar --modo final --modelo convnext_tiny --representacao cinza --seed 1 --retomar

Saídas: runs/<modelo>_<rep>_seed<N>/ (final) ou runs/ajuste/<modelo>_<rep>_tentativa<N>/
(ajuste), checkpoints em caminhos.checkpoints_dir/<mesmo caminho>/ e o run no MLflow.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.evaluation.rsna_metricas import metricas_reportadas
from src.models.rsna_modelo import criar_modelo
from src.training.rsna_config import caminho_do_repo
from src.training.rsna_execucao import (
    caminho_relativo,
    carregar_configs,
    dados_do_treino,
    fabrica_de_datasets,
    ler_alteracoes,
    proxima_tentativa,
    registrar_inicio,
)
from src.training.rsna_registro import Registro
from src.training.rsna_treino import Execucao, planejar_orcamento, treinar
from src.utils.paths import CONFIGS_DIR
from src.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--modo", choices=["ajuste", "final"], required=True)
    parser.add_argument("--config", type=Path, default=CONFIGS_DIR / "treino.yaml")
    parser.add_argument("--modelo", help="chave de `modelos` (padrão: `modelo` da config)")
    parser.add_argument(
        "--representacao", choices=["cinza", "realce_3canais"], help="padrão: etapa2.yaml"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--set", nargs="*", default=[], metavar="CHAVE=VALOR", help="só no ajuste")
    parser.add_argument("--retomar", action="store_true", help="continua do ultimo.pt")
    parser.add_argument("--tentativa", type=int, help="número da tentativa a retomar (ajuste)")
    args = parser.parse_args()

    if args.modo == "final" and args.set:
        parser.error("o treino final usa só o configs/treino.yaml: grave a config escolhida lá")
    if args.modo == "ajuste" and args.retomar and args.tentativa is None:
        parser.error("no ajuste, --retomar precisa de --tentativa N")

    alteracoes = ler_alteracoes(args.set)
    cfg, cfg2 = carregar_configs(args.config, args.modelo, args.representacao, alteracoes)
    runs_dir = caminho_do_repo(cfg.caminhos.get("runs_dir", "runs"))
    tentativa = args.tentativa
    if args.modo == "ajuste" and tentativa is None:
        tentativa = proxima_tentativa(
            runs_dir / "ajuste", f"{cfg.modelo}_{cfg2.entrada.representacao}"
        )
    relativo = caminho_relativo(args.modo, cfg, cfg2, args.seed, tentativa)
    run_dir = runs_dir / relativo
    checkpoint_dir = Path(cfg.caminhos["checkpoints_dir"]) / relativo
    if (run_dir / "metricas_finais.json").is_file() and not args.retomar:
        raise SystemExit(f"{run_dir} já foi concluído; apague a pasta para treinar de novo")

    print(
        f"{relativo} | {cfg.do_modelo.timm_nome} | batch {cfg.do_modelo.batch_real} x "
        f"{cfg.do_modelo.acumulacao} = {cfg.do_modelo.batch_efetivo}"
        + (f" | alterações: {alteracoes}" if alteracoes else "")
    )
    params, tags = registrar_inicio(
        run_dir, checkpoint_dir, cfg, cfg2, args.modo, args.seed, alteracoes, args.retomar
    )
    set_seed(args.seed, deterministic=False)
    torch.backends.cudnn.benchmark = True
    treino, validacao = dados_do_treino(cfg2)
    orcamento = planejar_orcamento(cfg, treino, args.modo, args.seed)
    print(
        f"orçamento por {orcamento.unidade}: {orcamento.validacoes} validações x "
        f"{orcamento.passos_por_validacao} passos = {orcamento.passos_total} passos | "
        f"paciência {orcamento.paciencia} | seleção por {cfg.metrica_selecao}"
        + (f" | EMA {cfg.do_modelo.ema_decaimento}" if cfg.do_modelo.ema else "")
    )
    execucao = Execucao(
        nome=str(relativo),
        modo=args.modo,
        semente=args.seed,
        run_dir=run_dir,
        checkpoint_dir=checkpoint_dir,
        epocas=orcamento.validacoes,
        paciencia=orcamento.paciencia,
        workers=int(cfg2.carregamento.get("workers", 8)),
        imagens_por_segmento=orcamento.imagens_por_segmento,
    )
    resumo = treinar(
        criar_modelo(cfg.do_modelo),
        cfg,
        execucao,
        treino,
        validacao,
        fabrica_de_datasets(cfg2),
        Registro(cfg.mlflow),
        params,
        tags,
        retomar=args.retomar,
    )
    v = resumo["validacao"]
    print(
        f"melhor validação {resumo['melhor_epoca']} (passo {resumo['melhor_passo']}): "
        + " | ".join(f"{k} {v.get(k, float('nan')):.4f}" for k in metricas_reportadas(cfg.betas_f))
        + f" | limiar de {cfg.metrica_selecao} {resumo['limiar_selecao']}\n{run_dir}"
    )


if __name__ == "__main__":
    main()
