# 🩺 Detecção de Câncer de Mama por Visão Computacional com Inteligência Artificial

[![CI](https://github.com/ProjetoIntegrador-4Periodo/ProjetoIntegrador4Periodo/actions/workflows/ci.yml/badge.svg)](https://github.com/ProjetoIntegrador-4Periodo/ProjetoIntegrador4Periodo/actions/workflows/ci.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![Código: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Licença: MIT](https://img.shields.io/badge/licen%C3%A7a-MIT-green.svg)](LICENSE)

Projeto desenvolvido na disciplina de **Projeto Integrador**, com foco na aplicação de técnicas de Inteligência Artificial e Visão Computacional para auxiliar na detecção de câncer de mama em exames de mamografia digital.

O projeto tem como objetivo investigar métodos capazes de identificar padrões associados a lesões mamárias, oferecendo suporte à análise médica por meio de modelos de aprendizado profundo aliados a técnicas de explicabilidade e avaliação de confiança.

> ⚠️ Trabalho acadêmico. Não é, e não deve ser usado como, ferramenta de diagnóstico médico.

---

## Objetivo

Desenvolver um sistema baseado em Inteligência Artificial para análise de imagens mamográficas, utilizando técnicas de Visão Computacional e Deep Learning para identificar possíveis lesões e fornecer explicações sobre as decisões realizadas pelo modelo.

Além da obtenção de um bom desempenho preditivo, o projeto busca garantir que as decisões produzidas pela inteligência artificial sejam transparentes, interpretáveis e confiáveis, contribuindo para pesquisas na área de diagnóstico assistido por computador (CAD – Computer-Aided Diagnosis).

---

## Objetivos Específicos

- Desenvolver modelos de classificação para imagens de mamografia;
- Aplicar técnicas de aprendizado profundo na identificação de lesões mamárias;
- Gerar explicações visuais das decisões tomadas pelos modelos;
- Avaliar a qualidade das explicações utilizando métricas quantitativas;
- Medir o grau de confiança e calibração das previsões realizadas;
- Comparar diferentes métodos de explicabilidade e avaliar sua eficiência;
- Produzir uma solução reprodutível, organizada e documentada para fins acadêmicos.

---

## Escopo do Projeto

O projeto contempla todas as etapas necessárias para o desenvolvimento de uma solução baseada em Inteligência Artificial aplicada à análise de mamografias, incluindo:

- preparação e organização dos dados;
- treinamento de modelos de Deep Learning;
- avaliação de desempenho utilizando métricas apropriadas;
- geração de explicações das predições;
- validação da confiabilidade das previsões;
- documentação dos experimentos e dos resultados obtidos.

---

## Tecnologias Utilizadas

| Camada | Ferramentas |
|--------|-------------|
| Linguagem | Python 3.12 |
| Deep Learning | PyTorch, torchvision, timm |
| Imagens médicas | pydicom, pylibjpeg, OpenCV, Pillow |
| Análise de dados | NumPy, pandas, scikit-learn, SciPy |
| Explicabilidade (XAI) | Grad-CAM, Captum, SHAP |
| Métricas e calibração | torchmetrics (pF1, AUROC, ECE, Brier) |
| Visualização | Matplotlib, Seaborn |
| Experimentos | MLflow |
| Qualidade de código | Ruff, Black, pytest, pre-commit |
| CI/CD | GitHub Actions |

---

## Base de Dados

**[RSNA Breast Cancer Detection](https://www.kaggle.com/competitions/rsna-breast-cancer-detection)** — base principal do projeto.

| | |
|---|---|
| Instituição | RSNA / Kaggle (2022) |
| Tipo de imagem | Mamografia digital (FFDM), formato DICOM |
| Volume | ~20 mil pacientes, mais de 54 mil imagens |
| Rótulos | Presença ou ausência de câncer, densidade mamária, idade |

A escolha considerou o prazo da disciplina: o RSNA é a base mais direta para treinar e validar a classificação, liberando tempo para aprofundar a **explicabilidade**. **CBIS-DDSM** e **VinDr-Mammo** seguem como candidatas para validação externa e para confrontar as explicações com máscaras reais de lesão, já que o RSNA não disponibiliza ROI.

O comparativo completo das três bases está em [`docs/Levantamento Datasets.md`](docs/Levantamento%20Datasets.md).

Instruções de download em [`data/README.md`](data/README.md). **Os dados não são versionados neste repositório.**

---

## Estrutura do Projeto

```
ProjetoIntegrador4Periodo/
├── .github/
│   ├── ISSUE_TEMPLATE/        # Modelos de tarefa, experimento e bug
│   ├── workflows/ci.yml       # Lint, testes e checagem de dados sensíveis
│   ├── CODEOWNERS             # Revisores automáticos por área
│   └── pull_request_template.md
├── configs/                   # YAMLs de experimento (hiperparâmetros)
├── data/                      # Dados locais — conteúdo NÃO versionado
│   ├── raw/                   #   download original do Kaggle
│   ├── interim/               #   etapas intermediárias
│   ├── processed/             #   dataset pronto para treino
│   └── external/              #   CBIS-DDSM, VinDr-Mammo
├── docs/                      # Documentação, escopo, requisitos, referencial
├── models/                    # Pesos treinados — NÃO versionados
├── notebooks/                 # Exploração (padrão: <etapa>-<iniciais>-<tema>)
├── reports/figures/           # Gráficos e mapas de calor gerados
├── scripts/                   # Scripts executáveis de ponta a ponta
├── src/                       # Código-fonte reutilizável
│   ├── data/                  #   leitura de DICOM e preparação
│   ├── features/              #   transformações e augmentations
│   ├── models/                #   arquitetura, treino e inferência
│   ├── explainability/        #   Grad-CAM, SHAP, LIME
│   ├── evaluation/            #   métricas, calibração, avaliação de XAI
│   └── utils/                 #   caminhos, sementes, helpers
├── tests/                     # Testes automatizados (pytest)
├── CONTRIBUTING.md            # Como contribuir — leia antes do primeiro PR
├── SECURITY.md                # Dados sensíveis e credenciais
├── Makefile                   # make setup / lint / test / check
├── pyproject.toml             # Configuração de ruff, black e pytest
└── requirements.txt           # Dependências de execução
```

---

## Como começar

```bash
git clone https://github.com/ProjetoIntegrador-4Periodo/ProjetoIntegrador4Periodo.git
cd ProjetoIntegrador4Periodo

make setup                 # cria a .venv e instala tudo
source .venv/bin/activate  # Windows: .venv\Scripts\activate
make hooks                 # ganchos de pre-commit

cp .env.example .env       # preencha suas credenciais do Kaggle
make check                 # confirma que o ambiente está ok
```

Comandos disponíveis: `make ajuda`.

---

## Fluxo de contribuição

**A `main` é protegida — ninguém commita direto nela.** Todo código entra por Pull Request.

```bash
git checkout main && git pull origin main
git checkout -b feat/minha-tarefa
# ... trabalhe, commite ...
make check
git push -u origin feat/minha-tarefa
gh pr create --fill
```

O CI roda, o revisor (definido no `CODEOWNERS`) aprova e faz o merge.

Leia [**CONTRIBUTING.md**](CONTRIBUTING.md) antes do primeiro PR — convenções de branch, de commit e o que nunca pode ser commitado.
Resumo visual do fluxo em [`docs/FLUXO_GIT.md`](docs/FLUXO_GIT.md).

---

## Documentação

| Documento | Conteúdo |
|-----------|----------|
| [CONTRIBUTING.md](CONTRIBUTING.md) | Padrões de branch, commit, PR e revisão |
| [SECURITY.md](SECURITY.md) | Dados sensíveis, credenciais e o que fazer se vazar |
| [docs/EQUIPE.md](docs/EQUIPE.md) | Papéis, divisão por área, rituais e próximos passos |
| [CITATION.cff](CITATION.cff) | Como citar este trabalho |
| [docs/FLUXO_GIT.md](docs/FLUXO_GIT.md) | Fluxo Git e comandos de emergência |
| [docs/Levantamento Datasets.md](docs/Levantamento%20Datasets.md) | Comparativo RSNA × CBIS-DDSM × VinDr-Mammo |
| [docs/Escopo do Projeto.docx](docs/Escopo%20do%20Projeto.docx) | Escopo formal da disciplina |
| [docs/RF_RNF_RD.xlsx](docs/RF_RNF_RD.xlsx) | Requisitos funcionais, não funcionais e de domínio |
| [docs/Artigo Referencial Teorico.pdf](docs/Artigo%20Referencial%20Teorico.pdf) | Referencial teórico |

---

## Integrantes

| Integrante | GitHub | Papel |
|------------|--------|-------|
| Kaique Vinicius Geska | [@KaiqueGeska](https://github.com/KaiqueGeska) | Scrum Master |
| Jhonatan da Silva Margraf | [@Jhonatan-Margraf](https://github.com/Jhonatan-Margraf) | Líder |
| Davi Specia | [@davidogral](https://github.com/davidogral) | CI/CD e integração |
| Bruno Nava Mainardi | [@mainardikk](https://github.com/mainardikk) | MLOps |
| Victor de Oliveira Ibarrola | — | Desenvolvimento |
| Thales Serschon | [@thalescr77](https://github.com/thalescr77) | Desenvolvimento |

Detalhes das responsabilidades em [docs/EQUIPE.md](docs/EQUIPE.md).

---

## Licença

O código e a documentação deste repositório estão sob a [Licença MIT](LICENSE).

A licença **não cobre os dados**: RSNA, CBIS-DDSM e VinDr-Mammo têm termos de uso próprios, aceitos individualmente por cada integrante no Kaggle, no TCIA e no PhysioNet. Nenhuma imagem médica é redistribuída aqui.

Projeto desenvolvido para fins acadêmicos na disciplina de **Projeto Integrador**. Os modelos aqui desenvolvidos não constituem ferramenta de diagnóstico médico.
