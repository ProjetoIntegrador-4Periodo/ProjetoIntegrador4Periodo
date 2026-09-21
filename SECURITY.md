# Segurança e dados sensíveis

Este projeto usa imagens médicas. Mesmo sendo um dataset público e anonimizado,
tratamos os dados com o cuidado que dado de saúde exige.

## O que nunca entra no repositório

- Imagens de mamografia (`.dcm`, `.png`, `.jpg` derivados de exame)
- Qualquer metadado que identifique paciente
- Pesos de modelo treinado
- Credenciais: `.env`, `kaggle.json`, tokens, chaves

Três camadas impedem isso: o `.gitignore`, o gancho de `pre-commit` e o job
`seguranca-dados` do CI, que recusa o Pull Request.

## Se um segredo vazar no histórico

1. **Considere o segredo comprometido.** O repositório é público.
2. **Revogue imediatamente** — gere um token novo no Kaggle e descarte o antigo.
3. **Avise o Davi.** Remover do histórico exige reescrita (`git filter-repo`) e
   force-push coordenado com todo o time.

Apagar o arquivo em um commit novo **não resolve**: ele continua no histórico.

## Uso dos dados

O RSNA Breast Cancer Detection tem termos de uso próprios, aceitos no Kaggle
por cada integrante individualmente. Os dados não são redistribuídos por este
repositório — cada pessoa baixa com a própria credencial.

## Escopo clínico

Este é um trabalho **acadêmico**. Os modelos aqui não são, e não devem ser
apresentados como, ferramenta de diagnóstico médico.
