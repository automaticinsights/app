Este notebook implementa uma análise estatística e visual para dados de ansiedade pré e pós-intervenção. Ele foi desenvolvido para processamento em Google Colab, utilizando uma abordagem que combina visualizações em estilo neon com análises estatísticas robustas para avaliar mudanças nos níveis de ansiedade entre diferentes grupos de participantes.

## Funcionalidades Principais

O notebook oferece as seguintes funcionalidades:

- Processamento e validação de dados de ansiedade
- Geração de resumos estatísticos
- Visualizações especializadas com tema escuro e cores neon:
  - Gráficos de densidade (KDE) para níveis de ansiedade
  - Gráficos de violino para comparação entre grupos
  - Gráficos de dispersão para análise de correlação
  - Árvores de decisão para previsão de ansiedade pós-intervenção

## Estrutura do Código

O código está organizado em seções funcionais:

1. **Importações e Configurações**: Importa bibliotecas essenciais (pandas, matplotlib, seaborn, scikit-learn) e define constantes para análise.
2. **Funções de Visualização**: Implementa funções especializadas para geração de gráficos com estética neon sobre fundo preto.
3. **Script Principal**: Orquestra o carregamento de dados, processamento e geração de resultados.

## Constantes e Parâmetros

- `OUTPUT_PATH`: Diretório para armazenamento dos resultados
- `PARTICIPANT_ID_COLUMN`: Coluna para identificação dos participantes
- `GROUP_COLUMN`: Coluna que identifica os grupos de intervenção
- `ANXIETY_PRE_COLUMN` e `ANXIETY_POST_COLUMN`: Colunas com medidas de ansiedade
- `BOOTSTRAP_RESAMPLES`: Número de reamostragens para intervalos de confiança
- `N_CLUSTERS`: Número de clusters para análises de agrupamento
- `RANDOM_STATE`: Semente para reprodutibilidade

## Métodos de Análise Implementados

1. **Análise Descritiva**: Resumos estatísticos dos dados de ansiedade
2. **Visualização de Distribuições**: Gráficos KDE para visualizar distribuições pré e pós-intervenção
3. **Comparação Entre Grupos**: Visualizações específicas para comparar resultados entre grupos
4. **Modelagem Preditiva**: Árvore de decisão para identificar fatores que influenciam mudanças na ansiedade

## Visualizações Produzidas

O notebook gera e salva as seguintes visualizações:

- `kde_plot.png`: Distribuição geral dos níveis de ansiedade pré e pós
- `grouped_kde_plot.png`: Distribuições separadas por grupo de intervenção
- `violin_plot.png`: Comparação da distribuição de ansiedade entre grupos
- `scatter_plot.png`: Correlação entre medidas de ansiedade pré e pós por grupo
- `decision_tree.png`: Modelo de árvore de decisão para previsão de ansiedade pós-intervenção

## Requisitos

- Python 3.x
- pandas
- matplotlib
- seaborn
- scikit-learn
- statsmodels
- numpy
- scipy
- Google Colab (para execução original)

## Como Utilizar

1. Monte o Google Drive utilizando `drive.mount('/content/drive')`
2. Configure o diretório de saída em `OUTPUT_PATH`
3. Execute o notebook para processar os dados e gerar visualizações
4. Consulte os resultados no diretório especificado em `OUTPUT_PATH`

## Dados de Exemplo

O notebook inclui dados sintéticos para demonstração, contendo:
- IDs de participantes
- Grupos de intervenção (Grupo A, Grupo B, Controle)
- Medidas de ansiedade pré e pós-intervenção

## Personalização

Para utilizar seus próprios dados, substitua a string `synthetic_data` ou modifique o código para carregar dados externos. Certifique-se de que seu conjunto de dados contenha as colunas necessárias conforme definido nas constantes.

## Autor

Hélio Craveiro Pessoa Júnior
