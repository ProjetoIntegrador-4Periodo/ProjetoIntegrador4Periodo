"""Avaliação final no teste hold-out: uma vez, no fim, nas configurações escolhidas pela validação.

Carrega o melhor checkpoint de um treino final (os pesos da EMA, se ela estava
ligada) e mede no teste, por mama, em dois pontos de operação escolhidos na
validação:

- ``selecao``: o limiar da métrica de seleção (o do F2 com a config padrão);
- ``sens90``: o maior limiar com sensibilidade >= 90% na validação.

Para cada um: F-betas, sensibilidade e especificidade, com IC por bootstrap, e
recortes por site, faixa etária, aparelho, negativos difíceis e spacing
estimado. A entrada do
modelo vem das configs gravadas no próprio run (só os caminhos vêm da config
atual). Recusa rodar duas vezes no mesmo run. Rode a partir da raiz:

    python -m scripts.rsna_avaliar_teste --run runs/convnext_tiny_cinza_seed1 --uso-unico-confirmado

Saídas em runs/<run>/teste/: metricas_teste.json, recortes_teste.csv,
predicoes_teste_imagem.csv e predicoes_teste_mama.csv (e as métricas no run do MLflow).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import yaml

from src.data.rsna_dataset import MamografiaRSNA
from src.data.rsna_entrada import ConfigEtapa2
from src.evaluation import rsna_metricas as metricas
from src.models.rsna_modelo import criar_modelo
from src.training.rsna_config import ConfigModelo, carregar_manifest
from src.training.rsna_execucao import carregar_configs
from src.training.rsna_registro import Registro, gravar_json
from src.training.rsna_treino import _carregador, escolher_precisao, predicoes_por_imagem, prever
from src.utils.paths import CONFIGS_DIR

COLUNAS_RECORTE = ["site_id", "machine_id", "age", "difficult_negative_case", "spacing_estimado"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", type=Path, required=True, help="pasta runs/<modelo>_<rep>_seed<N>")
    parser.add_argument("--config", type=Path, default=CONFIGS_DIR / "treino.yaml")
    parser.add_argument("--uso-unico-confirmado", action="store_true", required=True)
    args = parser.parse_args()

    run = args.run
    saida = run / "teste"
    if saida.exists():
        raise SystemExit(f"{saida} já existe: o teste deste run já foi usado")
    resumo = json.loads((run / "metricas_finais.json").read_text(encoding="utf-8"))
    if resumo["modo"] != "final":
        raise SystemExit("só treinos finais vão para o teste")
    treino_usado = yaml.safe_load((run / "config_treino.yaml").read_text(encoding="utf-8"))
    etapa2_usada = yaml.safe_load((run / "config_etapa2.yaml").read_text(encoding="utf-8"))
    execucao = json.loads((run / "execucao.json").read_text(encoding="utf-8"))
    limiares = {
        "selecao": float(resumo["limiar_selecao"]),
        "sens90": float(resumo["validacao"]["limiar_sens90"]),
    }
    betas = treino_usado["treino"]["betas_f"]

    cfg_atual, cfg2_atual = carregar_configs(args.config)
    cfg2 = ConfigEtapa2.de_dict({**etapa2_usada, "caminhos": cfg2_atual.caminhos})
    hp = {k: v for k, v in treino_usado["hiperparametros"].items() if k != "batch_efetivo"}
    cfg_modelo = ConfigModelo(**{k: tuple(v) if isinstance(v, list) else v for k, v in hp.items()})

    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelo = criar_modelo(cfg_modelo, pretrained=False)
    checkpoint = torch.load(
        Path(execucao["checkpoint_dir"]) / "melhor.pt", map_location="cpu", weights_only=False
    )
    modelo.load_state_dict(checkpoint["modelo"])
    autocast, _ = escolher_precisao(treino_usado["treino"]["precisao"], dispositivo)
    channels_last = treino_usado["treino"]["channels_last"] and dispositivo.type == "cuda"
    modelo = modelo.to(dispositivo)
    if channels_last:
        modelo = modelo.to(memory_format=torch.channels_last)

    teste = carregar_manifest(cfg2.caminhos["etapa1_dir"], ("test",), permitir_teste=True)
    dataset = MamografiaRSNA.da_config(cfg2, teste, treino=False)
    carregador = _carregador(
        dataset,
        range(len(dataset)),
        cfg_modelo.batch_real * 2,
        int(cfg2.carregamento.get("workers", 8)),
        0,
        dispositivo,
    )
    print(
        f"{run.name}: {len(teste)} imagens do teste | limiares da validação: "
        f"{resumo['metrica_selecao']} {limiares['selecao']:.4f}, sens90 {limiares['sens90']:.4f}"
    )
    prob = prever(modelo, carregador, dispositivo, autocast, channels_last)

    imagens = predicoes_por_imagem(teste, prob)
    com_atributos = imagens.join(teste[COLUNAS_RECORTE])
    mamas = metricas.atributos_por_mama(com_atributos)
    rotulo, p = mamas.cancer.to_numpy(), mamas.prob.to_numpy()
    resultado = {
        "run": run.name,
        "metrica_selecao": resumo["metrica_selecao"],
        "limiares_validacao": limiares,
        "sem_limiar": metricas.calcular(mamas, betas),
        "pontos_de_operacao": {
            ponto: metricas.no_limiar(rotulo, p, limiar, betas)
            for ponto, limiar in limiares.items()
        },
        "ic95": metricas.bootstrap(
            mamas,
            cfg_atual.bootstrap_amostras,
            cfg_atual.semente_bootstrap,
            metricas.metricas_reportadas(betas),
            limiares=limiares,
            betas=betas,
        ),
        "checkpoint_epoca": checkpoint["epoca"],
        "checkpoint_pesos": checkpoint.get("pesos", "modelo"),
    }
    saida.mkdir()
    imagens.to_csv(saida / "predicoes_teste_imagem.csv", index=False)
    mamas[metricas.COLUNAS_MAMA].to_csv(saida / "predicoes_teste_mama.csv", index=False)
    tabela = pd.concat(
        [
            metricas.recortes(mamas, limiar).assign(ponto_de_operacao=ponto, limiar=limiar)
            for ponto, limiar in limiares.items()
        ],
        ignore_index=True,
    )
    tabela.to_csv(saida / "recortes_teste.csv", index=False)
    gravar_json(saida / "metricas_teste.json", resultado)

    registro = Registro(cfg_atual.mlflow)
    if registro.ativo and resumo.get("mlflow_run_id"):
        registro.iniciar(run.name, {}, {"avaliado_no_teste": "sim"}, resumo["mlflow_run_id"])
        registro.resumo(
            {
                f"teste_{k}": v
                for k, v in resultado["sem_limiar"].items()
                if isinstance(v, int | float)
            }
        )
        for ponto, valores in resultado["pontos_de_operacao"].items():
            registro.resumo({f"teste_{ponto}_{k}": v for k, v in valores.items()})
        registro.artefatos(saida, "teste")
        registro.encerrar()

    print(json.dumps(resultado["sem_limiar"], indent=2))
    print(json.dumps(resultado["pontos_de_operacao"], indent=2))
    print(pd.DataFrame(tabela).to_string(index=False))
    print(saida)


if __name__ == "__main__":
    main()
