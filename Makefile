# Atalhos do dia a dia. Rode `make ajuda` para ver a lista.
# Interpretador alvo — sobrescreva se o seu tiver outro nome: make setup PYTHON=python3
PYTHON ?= python3.11

.PHONY: ajuda setup lint format test check limpar hooks

ajuda:  ## Mostra os comandos disponíveis
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup:  ## Cria a venv e instala as dependências de desenvolvimento
	$(PYTHON) -m venv .venv
	./.venv/bin/pip install --upgrade pip
	./.venv/bin/pip install -r requirements-dev.txt
	@echo "Pronto. Ative com: source .venv/bin/activate"

hooks:  ## Instala os ganchos de pre-commit
	pre-commit install

lint:  ## Verifica estilo e erros estáticos
	ruff check .
	black --check .

format:  ## Formata o código e corrige o que o ruff conseguir
	ruff check . --fix
	black .

test:  ## Roda a suíte de testes com cobertura
	pytest --cov=src --cov-report=term-missing

check: lint test  ## Roda tudo que o CI roda (use antes de abrir o PR)

limpar:  ## Remove caches e arquivos temporários
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov
