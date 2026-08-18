# Levantamento de Datasets para Detecção de Câncer de Mama por Visão Computacional

**Disciplina:** Projeto Integrador

---

# Introdução

O desenvolvimento de modelos de Inteligência Artificial para detecção de câncer de mama depende diretamente da disponibilidade de bases de dados de qualidade. Esses conjuntos de dados fornecem as imagens necessárias para o treinamento, validação e teste dos modelos, além de informações clínicas que permitem avaliar sua capacidade de identificar padrões associados à presença de lesões mamárias.

Este documento apresenta um levantamento dos três datasets propostos para utilização no projeto:

- RSNA Breast Cancer Detection
- CBIS-DDSM
- VinDr-Mammo

O objetivo é analisar suas principais características, vantagens, limitações e possíveis aplicações no desenvolvimento do projeto.

---

# Comparativo Geral

| Característica | RSNA Breast Cancer Detection | CBIS-DDSM | VinDr-Mammo |
|----------------|------------------------------|------------|-------------|
| Instituição | RSNA / Kaggle | The Cancer Imaging Archive (TCIA) | PhysioNet / VinDr |
| Ano | 2022 | 2017 | 2023 |
| Tipo de imagem | Mamografia Digital (FFDM) | Mamografia Digitalizada | Mamografia Digital (FFDM) |
| Formato | DICOM | DICOM | DICOM |
| Diagnóstico | ✅ Sim | ✅ Sim | ✅ Sim |
| Máscaras (ROI) | ❌ Não | ✅ Sim | ✅ Sim (Bounding Boxes) |
| BI-RADS | ❌ Não | ❌ Não | ✅ Sim |
| Densidade Mamária | ✅ Sim | ❌ Não | ✅ Sim |
| Download Gratuito | ✅ Sim | ✅ Sim | ✅ Sim |
| Cadastro Necessário | Kaggle | Não | PhysioNet |

---

# 1. RSNA Breast Cancer Detection

**Link Oficial**

https://www.kaggle.com/competitions/rsna-breast-cancer-detection

## Descrição

O RSNA Breast Cancer Detection é um dataset disponibilizado durante uma competição promovida pela Radiological Society of North America (RSNA) em parceria com o Kaggle. Seu objetivo é incentivar o desenvolvimento de modelos capazes de detectar câncer de mama em exames de mamografia utilizando técnicas de Inteligência Artificial.

A base reúne exames provenientes de programas reais de rastreamento do câncer de mama, oferecendo um ambiente bastante próximo da prática clínica.

## Principais Características

- Mamografias digitais (FFDM)
- Arquivos em formato DICOM
- Aproximadamente 20 mil pacientes
- Mais de 54 mil imagens
- Informações sobre idade das pacientes
- Informações sobre densidade mamária
- Classificação indicando presença ou ausência de câncer

## Vantagens

- Grande quantidade de imagens
- Dados provenientes de ambiente clínico real
- Excelente qualidade das imagens
- Amplamente utilizada em pesquisas recentes

## Limitações

- Não possui segmentação das lesões
- Não disponibiliza máscaras (ROI)
- Não permite comparação direta entre mapas de ativação e localização real do tumor

## Aplicações

- Treinamento de modelos de classificação
- Validação de desempenho
- Avaliação da capacidade de generalização

---

# 2. CBIS-DDSM

**Link Oficial**

https://www.cancerimagingarchive.net/collection/cbis-ddsm/

## Descrição

O CBIS-DDSM (Curated Breast Imaging Subset of DDSM) é uma versão revisada da tradicional base DDSM, organizada e disponibilizada pelo The Cancer Imaging Archive (TCIA).

É uma das bases mais utilizadas em pesquisas envolvendo Inteligência Artificial aplicada à mamografia por disponibilizar, além das imagens, anotações detalhadas das regiões lesionadas.

## Principais Características

- Arquivos em formato DICOM
- Mamografias digitalizadas
- Aproximadamente 3.500 imagens
- Diagnóstico benigno ou maligno
- Máscaras das regiões lesionadas (ROI)
- Informações sobre massas e microcalcificações

## Vantagens

- Possui ground truth das lesões
- Excelente para segmentação
- Muito utilizada na literatura científica
- Permite validação quantitativa de métodos de explicabilidade

## Limitações

- Base relativamente antiga
- Imagens provenientes de mamografias digitalizadas
- Menor diversidade em comparação com bases mais recentes

## Aplicações

- Treinamento de modelos
- Avaliação de desempenho
- Estudos de Explainable AI (XAI)
- Comparação entre mapas de calor e regiões reais das lesões

---

# 3. VinDr-Mammo

**Link Oficial**

https://physionet.org/content/vindr-mammo/1.0.0/

## Descrição

O VinDr-Mammo é um dataset público desenvolvido pelo grupo VinDr e disponibilizado por meio da plataforma PhysioNet.

A base reúne exames completos de mamografia digital anotados por radiologistas especialistas, seguindo o padrão internacional BI-RADS.

Atualmente é considerada uma das bases públicas mais modernas para pesquisas em detecção de câncer de mama.

## Principais Características

- Mamografia Digital (FFDM)
- Arquivos em formato DICOM
- 5.000 exames
- Aproximadamente 20.000 imagens
- Classificação BI-RADS
- Densidade mamária
- Bounding Boxes das lesões

## Vantagens

- Base recente
- Imagens digitais de alta qualidade
- Informações clínicas completas
- Diversidade de pacientes
- Localização das lesões

## Limitações

- Requer cadastro no PhysioNet
- Menor quantidade de exames quando comparada ao RSNA

## Aplicações

- Validação externa
- Estudos de generalização
- Avaliação de desempenho em imagens modernas
- Desenvolvimento de modelos de detecção

---

# Comparação dos Datasets

| Critério | Dataset Mais Adequado |
|----------|-----------------------|
| Maior quantidade de imagens | RSNA Breast Cancer Detection |
| Melhor qualidade das imagens | VinDr-Mammo |
| Possui máscaras das lesões | CBIS-DDSM |
| Possui BI-RADS | VinDr-Mammo |
| Melhor para estudos de explicabilidade | CBIS-DDSM |
| Melhor para treinamento em larga escala | RSNA Breast Cancer Detection |
| Melhor para validação externa | VinDr-Mammo |

---

# Conclusão

Os três datasets apresentam características complementares e podem contribuir de maneiras distintas para o desenvolvimento de modelos de Inteligência Artificial voltados à detecção de câncer de mama.

O **RSNA Breast Cancer Detection** destaca-se pelo grande volume de exames e pela representatividade dos dados clínicos, sendo adequado para treinamento e avaliação de modelos de classificação.

O **CBIS-DDSM** fornece anotações detalhadas das regiões lesionadas, tornando-se especialmente útil para estudos que envolvem localização de lesões e avaliação de métodos de explicabilidade.

O **VinDr-Mammo** disponibiliza mamografias digitais modernas acompanhadas de informações clínicas relevantes, permitindo validar modelos em um cenário mais atual e diversificado.

A análise dessas bases permite compreender suas características e selecionar os conjuntos de dados mais adequados para cada etapa do desenvolvimento do projeto.

---

# Referências

- **RSNA Breast Cancer Detection.** Kaggle. Disponível em: <https://www.kaggle.com/competitions/rsna-breast-cancer-detection>
- **CBIS-DDSM.** The Cancer Imaging Archive (TCIA). Disponível em: <https://www.cancerimagingarchive.net/collection/cbis-ddsm/>
- **VinDr-Mammo.** PhysioNet. Disponível em: <https://physionet.org/content/vindr-mammo/1.0.0/>