"""Garante que os caminhos do projeto resolvem para a estrutura esperada."""

from src.utils.paths import ALL_DATA_DIRS, DATA_DIR, PROJECT_ROOT


def test_project_root_contem_marcadores_do_repo():
    assert (PROJECT_ROOT / "pyproject.toml").is_file()
    assert (PROJECT_ROOT / "src").is_dir()


def test_diretorios_de_dados_ficam_sob_data_dir():
    for directory in ALL_DATA_DIRS:
        assert directory.parent == DATA_DIR
