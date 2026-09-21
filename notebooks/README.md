# Notebooks

Espaço para exploração. Código que for reaproveitado deve migrar para `src/`.

## Convenção de nome

```
<etapa>-<iniciais>-<descrição-curta>.ipynb
```

Exemplos:

- `01-dsp-eda-metadados-rsna.ipynb`
- `01-bnm-eda-distribuicao-densidade.ipynb`
- `02-kvg-preprocessamento-dicom.ipynb`

Prefixos de etapa:

| Prefixo | Etapa |
|---------|-------|
| `00` | Setup e verificação de ambiente |
| `01` | EDA (análise exploratória) |
| `02` | Pré-processamento |
| `03` | Modelagem |
| `04` | Avaliação |
| `05` | Explicabilidade |

Como mais de uma pessoa pode fazer a mesma tarefa para comparar resultados, **as
iniciais no nome são obrigatórias** — é o que evita conflito e deixa claro de
quem é cada análise.

## Regras

1. **Nunca** commite um notebook com as saídas preenchidas. O `pre-commit`
   limpa isso automaticamente (`nbstripout`); instale com `make hooks`.
2. Não salve imagem de paciente no notebook.
3. Caminho de dados vem de `src.utils.paths`, nunca `../../data` na mão.
4. Notebook que virou conclusão → resuma no PR e mova a lógica para `src/`.
