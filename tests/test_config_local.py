"""Caminhos da máquina vêm do configs/local.yaml, fora do git."""

import pytest
import yaml

from src.utils.config_local import da_maquina
from src.utils.paths import CONFIGS_DIR, PROJECT_ROOT


def test_local_completa_a_config_e_a_config_vence(tmp_path):
    local = tmp_path / "local.yaml"
    local.write_text(
        yaml.safe_dump({"etapa2": {"etapa1_dir": "/a", "realce_dir": "/b", "x": None}})
    )
    caminhos = da_maquina("etapa2", {"realce_dir": "/c", "vazio": ""}, local)
    assert caminhos == {"etapa1_dir": "/a", "realce_dir": "/c"}


def test_sem_local_o_erro_diz_o_que_fazer(tmp_path):
    caminhos = da_maquina("treino", {"runs_dir": "runs"}, tmp_path / "nao_existe.yaml")
    assert caminhos["runs_dir"] == "runs"
    with pytest.raises(KeyError, match="local.exemplo.yaml"):
        caminhos["checkpoints_dir"]


def test_configs_versionadas_sem_caminhos_de_maquina():
    """Nenhum caminho absoluto nas configs do git; o exemplo tem todas as chaves usadas."""
    for nome in ("etapa2.yaml", "treino.yaml", "pipeline_rsna.yaml"):
        texto = (CONFIGS_DIR / nome).read_text(encoding="utf-8")
        assert "/mnt/" not in texto and "/media/" not in texto and "sqlite:" not in texto, nome
    exemplo = yaml.safe_load((CONFIGS_DIR / "local.exemplo.yaml").read_text(encoding="utf-8"))
    assert set(exemplo["etapa2"]) == {"etapa1_dir", "realce_dir", "verificacao_dir"}
    assert set(exemplo["treino"]) == {"checkpoints_dir", "verificacao_dir"}
    assert set(exemplo["mlflow"]) == {"tracking_uri", "artefatos_dir"}
    assert set(exemplo["pipeline_rsna"]) == {"rsna_dir", "saida_dir"}
    assert "maquina" in exemplo and "maquina" not in (CONFIGS_DIR / "treino.yaml").read_text()


def test_local_yaml_fora_do_git():
    regras = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "configs/local.yaml" in regras
