# Como contribuir

Este documento é o combinado do time. Vale para todo mundo, inclusive quem
administra o repositório.

---

## Regra número 1

**Ninguém commita direto na `main`.** A branch está protegida no GitHub: todo
código entra por Pull Request, com revisão e CI verde. Se você tentar
`git push origin main`, o GitHub recusa.

---

## 1. Preparar o ambiente (uma vez só)

O projeto exige **Python 3.12**. Confira o que você tem com `python3.12 --version`.

Se não tiver:

| Sistema | Comando |
|---------|---------|
| macOS | `brew install python@3.12` |
| Ubuntu / WSL | `sudo apt install python3.12 python3.12-venv` |
| Windows | Baixe em [python.org/downloads](https://www.python.org/downloads/) e marque *Add to PATH* |

Não é capricho: numpy, scipy e shap deixaram de publicar versões para o 3.11, e
ficar para trás nesses três significa perder correções durante o semestre.

```bash
git clone https://github.com/ProjetoIntegrador-4Periodo/ProjetoIntegrador4Periodo.git
cd ProjetoIntegrador4Periodo

make setup          # cria a .venv e instala as dependências
source .venv/bin/activate
make hooks          # instala os ganchos de pre-commit

cp .env.example .env   # preencha suas credenciais do Kaggle
```

No Windows, ative a venv com `.venv\Scripts\activate`.

Sem `make` disponível? Os comandos equivalentes estão dentro do `Makefile`.

Se o seu Python 3.12 atende por outro nome, passe na chamada:
`make setup PYTHON=python3`.

---

## 2. Criar a branch

Sempre a partir de uma `main` atualizada:

```bash
git checkout main
git pull origin main
git checkout -b <tipo>/<descricao-curta>
```

### Nomes de branch

| Prefixo | Quando usar | Exemplo |
|---------|-------------|---------|
| `feat/` | Funcionalidade nova | `feat/dataloader-dicom` |
| `fix/` | Correção de bug | `fix/normalizacao-pixel-dicom` |
| `exp/` | Experimento, treino, EDA | `exp/dsp-eda-metadados-rsna` |
| `docs/` | Documentação | `docs/atualiza-readme` |
| `refactor/` | Reorganização sem mudar comportamento | `refactor/extrai-transforms` |
| `chore/` | Infra, dependências, configuração | `chore/atualiza-torch` |
| `ci/` | Pipeline e automação | `ci/adiciona-cache-pip` |

Regras: minúsculas, hífen no lugar de espaço, sem acento.

**O GitHub recusa branch fora desses prefixos.** Não é só convenção — é regra do
repositório. `git push -u origin minha-branch-teste` volta com
`Cannot create ref due to creations being restricted`. Renomeie com
`git branch -m <novo-nome>` e empurre de novo.

**Em branch de experimento (`exp/`), inclua suas iniciais.** Como combinamos que
mais de uma pessoa pode atacar a mesma tarefa para comparar resultados, é isso
que impede duas branches com o mesmo nome — `exp/dsp-gradcam` e
`exp/bnm-gradcam` convivem sem conflito.

---

## 3. Commitar

Usamos [Conventional Commits](https://www.conventionalcommits.org/pt-br/):

```
<tipo>(<escopo opcional>): <descrição no imperativo, minúscula, sem ponto final>
```

Exemplos reais:

```bash
git commit -m "feat(data): adiciona leitor de DICOM com janelamento"
git commit -m "fix(train): corrige cálculo do pF1 quando não há positivos"
git commit -m "exp(xai): compara Grad-CAM e Grad-CAM++ no split de validação"
git commit -m "docs: descreve como baixar o dataset RSNA"
```

Tipos: `feat`, `fix`, `exp`, `docs`, `refactor`, `test`, `chore`, `ci`, `perf`.

Commits pequenos e com significado próprio. `wip`, `ajustes` e `teste` não
dizem nada para quem revisa — e o histórico é a memória do projeto.

---

## 4. Antes de abrir o PR

```bash
make check          # roda lint + testes, igual ao CI
```

Se o lint reclamar de formatação, `make format` conserta a maior parte.

Depois, atualize sua branch com a `main`:

```bash
git fetch origin
git rebase origin/main
```

Deu conflito? Resolva, `git add` nos arquivos e `git rebase --continue`. Se
preferir, `git merge origin/main` também serve — o time não exige histórico
linear.

---

## 5. Abrir o Pull Request

```bash
git push -u origin <sua-branch>
gh pr create --fill        # ou abra pelo site
```

O template de PR abre sozinho. Preencha de verdade — principalmente a seção de
resultados, quando for experimento. É o que permite comparar as abordagens do
time lado a lado.

Marque como **Draft** enquanto ainda estiver trabalhando. Isso sinaliza que
ainda não é hora de revisar.

### O que acontece depois

1. O **CI roda** (lint, testes e a checagem de dados sensíveis). Precisa ficar verde.
2. O **Davi é adicionado como revisor automaticamente** (via `CODEOWNERS`).
3. Ele revisa, comenta e pede ajustes se necessário.
4. Toda conversa precisa estar resolvida antes do merge.
5. **O merge é feito pelo revisor**, com *Squash and merge*.
6. A branch é apagada automaticamente após o merge.

Não faça merge do próprio PR.

---

## 6. Revisando o PR de outra pessoa

Todo mundo pode (e deve) comentar, mesmo sem ser o revisor obrigatório.

O que olhar:

- O código faz o que o PR diz que faz?
- Dá para entender sem perguntar para quem escreveu?
- Tem caminho de arquivo fixo na mão? Tem dado de paciente? Tem segredo?
- O resultado do experimento está registrado de forma que dê para reproduzir?
- Os números batem com o que foi reportado?

Comentário de revisão é sobre o código, nunca sobre a pessoa. Sugira, não exija:
"o que acha de extrair isso para uma função?" funciona melhor que "isso está errado".

---

## 7. Dados, modelos e segredos

Três coisas **nunca** entram no repositório:

1. **Imagens médicas** (`.dcm`, `.png` de mamografia, qualquer dado de paciente)
2. **Pesos de modelo** (`.pt`, `.pth`, `.ckpt`, `.onnx`)
3. **Credenciais** (`.env`, `kaggle.json`, tokens)

O `.gitignore` já bloqueia isso, o `pre-commit` avisa antes do commit e o CI
recusa o PR se algo passar. Se mesmo assim algo vazar para o histórico, **avise
o Davi imediatamente** — remover exige reescrever o histórico e é melhor fazer cedo.

Onde cada coisa vai: dados em `data/` (local), pesos em MLflow ou GitHub Release,
credenciais no seu `.env`.

---

## 8. Dúvidas frequentes

**Esqueci e commitei na `main` local.** Nada perdido:

```bash
git branch feat/minha-branch     # salva o trabalho numa branch
git reset --hard origin/main     # devolve a main ao estado do remoto
git checkout feat/minha-branch
```

**Meu PR está com conflito.** Faça `git fetch origin && git rebase origin/main`,
resolva, e `git push --force-with-lease` (nunca `--force` puro).

**O CI falhou e não entendi.** Abra a aba *Checks* do PR e leia o passo vermelho.
`make check` local reproduz quase tudo.

**Preciso do trabalho de alguém que ainda não foi mergeado.** Crie sua branch a
partir da dele e diga isso na descrição do PR.
