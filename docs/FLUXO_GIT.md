# Fluxo de trabalho no Git

Referência rápida. O detalhe está em [CONTRIBUTING.md](../CONTRIBUTING.md).

---

## O ciclo

```
main (protegida)
 │
 ├──> git checkout -b feat/minha-tarefa
 │        │
 │        ├── commit
 │        ├── commit
 │        │
 │        └──> git push -u origin feat/minha-tarefa
 │                  │
 │                  └──> Pull Request
 │                          │
 │                          ├── CI: lint + testes + checagem de dados
 │                          ├── Revisão do Davi (CODEOWNERS)
 │                          └── Squash and merge
 │                                  │
 └<────────────────────────────────┘
```

---

## Sequência de comandos

```bash
# 1. partir de uma main atualizada
git checkout main && git pull origin main

# 2. criar a branch
git checkout -b exp/dsp-eda-densidade

# 3. trabalhar e commitar
git add .
git commit -m "exp(eda): analisa distribuição de densidade mamária"

# 4. validar como o CI valida
make check

# 5. sincronizar com a main
git fetch origin && git rebase origin/main

# 6. enviar e abrir o PR
git push -u origin exp/dsp-eda-densidade
gh pr create --fill
```

---

## O que a proteção da `main` exige

| Regra | Efeito prático |
|-------|----------------|
| Pull Request obrigatório | `git push origin main` é recusado |
| 1 aprovação | O PR só mergeia depois de aprovado |
| Revisão de CODEOWNERS | O Davi é adicionado como revisor automaticamente |
| Aprovações dispensadas a cada push | Commit novo depois da aprovação exige revisar de novo |
| Conversas resolvidas | Nenhum comentário pendente no merge |
| CI verde | `lint`, `test` e `seguranca-dados` precisam passar |
| Sem force-push | O histórico da `main` não pode ser reescrito |
| Sem exclusão | A `main` não pode ser apagada |

Administrador consegue destravar uma situação emergencial, mas isso é exceção —
não o caminho normal.

---

## Comandos de emergência

```bash
# commitei na main local sem querer
git branch feat/salva-trabalho
git reset --hard origin/main
git checkout feat/salva-trabalho

# quero desfazer o último commit mantendo as alterações
git reset --soft HEAD~1

# quero jogar fora tudo que não foi commitado
git restore .

# atualizar a branch depois de um rebase
git push --force-with-lease     # nunca --force puro

# ver em que pé está
git status
git log --oneline --graph --all -15
```
