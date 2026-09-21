## O que este PR faz

<!-- Uma ou duas frases. Se fecha uma issue, escreva: Closes #<número> -->

## Tipo

- [ ] `feat` — funcionalidade nova
- [ ] `fix` — correção
- [ ] `exp` — experimento / treino / EDA
- [ ] `docs` — documentação
- [ ] `refactor` — reorganização sem mudar comportamento
- [ ] `chore` / `ci` — infraestrutura, dependências, pipeline

## Como foi validado

<!-- Como o revisor confirma que funciona: comando rodado, notebook, print. -->

## Resultados (preencher só em PR de experimento)

| Item | Valor |
|------|-------|
| Config usada | `configs/____.yaml` |
| Dataset / split | |
| pF1 | |
| AUROC | |
| Sensibilidade / Especificidade | |
| Run no MLflow | |

<!-- Comparando com outra abordagem do time? Diga com qual e o que mudou. -->

## Checklist

- [ ] Rodei `make check` (lint + testes) e passou
- [ ] Não subi imagem médica, dado de paciente, peso de modelo nem `.env`
- [ ] Notebooks estão sem saídas salvas
- [ ] Atualizei a documentação afetada (README, `docs/`, docstrings)
- [ ] A branch está atualizada com a `main`
