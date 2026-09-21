# Equipe, papéis e forma de trabalho

Decisões tomadas em reunião de equipe e registradas aqui para que qualquer
pessoa (inclusive o professor) entenda como o projeto se organiza.

---

## Integrantes e papéis

| Integrante | GitHub | Papel | Responsabilidade principal |
|------------|--------|-------|----------------------------|
| Kaique Vinicius Geska | [@KaiqueGeska](https://github.com/KaiqueGeska) | Scrum Master | Organiza as etapas e a divisão das tarefas |
| Jhonatan da Silva Margraf | [@Jhonatan-Margraf](https://github.com/Jhonatan-Margraf) | Líder | Apoio na organização geral do projeto |
| Davi Specia | [@davidogral](https://github.com/davidogral) | CI/CD e integração | Pipeline, revisão dos PRs, merge, proteção da `main`, consolidação da EDA |
| Bruno Nava Mainardi | [@mainardikk](https://github.com/mainardikk) | MLOps | Rastreamento de experimentos, reprodutibilidade, versionamento de modelos |
| Victor de Oliveira Ibarrola | *(a adicionar na organização)* | Desenvolvimento | — |
| Thales Serschon | [@thalescr77](https://github.com/thalescr77) | Desenvolvimento | — |

Kaique conduz o Scrum com **apoio do Davi na parte técnica**.

---

## Divisão do trabalho

As atividades são divididas **por área**, não por pessoa fixa:

- Dados / EDA
- Pré-processamento
- Modelagem
- Avaliação
- Explicabilidade (XAI)
- MLOps
- CI/CD
- Documentação

**Mais de uma pessoa pode trabalhar na mesma tarefa.** Isso é intencional: gera
abordagens diferentes para o mesmo problema e permite comparar resultados. Por
isso branches e notebooks de experimento levam as iniciais de quem fez — veja
[CONTRIBUTING.md](../CONTRIBUTING.md).

Quando duas pessoas atacam a mesma tarefa, cada uma abre o seu PR com os números
no template. A comparação vira discussão no PR e a escolha fica registrada.

---

## Dataset

Ficou definido o uso do **RSNA Breast Cancer Detection** como base principal.

Motivo: dentro do prazo da disciplina, o RSNA é o caminho mais direto — volume
grande, dados de rastreamento clínico real e rótulo de câncer já disponível.
Isso libera tempo para o que é o diferencial do trabalho: a **explicabilidade**,
que recebe foco maior na etapa seguinte.

O levantamento completo das três bases avaliadas (RSNA, CBIS-DDSM e VinDr-Mammo)
está em [Levantamento Datasets.md](Levantamento%20Datasets.md). CBIS-DDSM e
VinDr-Mammo seguem como candidatas para validação externa e para avaliar
explicações contra máscaras reais de lesão, já que o RSNA não traz ROI.

---

## Rituais

- **Reunião semanal** — dia definido por enquete entre quinta, sábado e domingo.
- **Organização das etapas** — conduzida pelo Kaique.
- **Discussão técnica** — acontece no Pull Request, que fica como registro.

---

## Próximos passos

- [ ] Realizar a EDA sobre o RSNA
- [ ] Definir o dia fixo da reunião semanal (enquete)
- [ ] Kaique organizar as próximas etapas e a divisão das tarefas
- [ ] Consolidar os resultados das EDAs individuais (Davi)
- [ ] Adicionar o Victor à organização no GitHub
