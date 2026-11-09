# ADR-0013 — Os experimentos rastreados no MLflow

**Status:** aceita

## Contexto

As decisões das Partes 3 e 4 saíram de medidas, e não de opinião:
- treinar com o rótulo do dia, e não esperar o rótulo amadurecer (ADR-0007);
- os parâmetros dos modelos (ADR-0008);
- tirar a idade do modelo (ADR-0011).

Os números estão nas ADRs, mas refazer uma medida pedia um script avulso, e
nada guardava junto com que dados, com que parâmetros e com que código cada
número saiu. Na Parte 7, o modelo novo vai ser comparado com o atual, e isso
pede um histórico em que cada treino tenha os dados, os parâmetros e as
métricas dele, lado a lado.

## Decisão

- **O MLflow rastreia os experimentos**, na versão 3.17.2, fixada como as outras. A 3.18 saiu na semana em que esta parte foi feita, e ficou de fora até ter algum uso.
- **O rastreio fica ao lado dos dados**, em `dados/mlflow/`, com um banco SQLite (`mlflow.db`) e a pasta dos artefatos, e não vai para o Git. Ele sai da mesma semente, como os dados. É um banco, e não uma pasta de arquivos, porque o registro de modelos da ADR-0014 pede um banco.
- **Cada variante vira uma corrida por modelo**, refeita no tempo, mês a mês, como na avaliação (`rastreio/experimentos.py`):

  | Variante | O que ela muda | Modelos |
  |---|---|---|
  | `padrao` | nada: os modelos como ficaram | as regras, a logística e o LightGBM |
  | `com_a_verdade` | aprende com a verdade, que nenhum banco tem | o LightGBM |
  | `esperando_14_dias` | espera 14 dias o rótulo amadurecer | o LightGBM |
  | `esperando_28_dias` | espera 28 dias | o LightGBM |
  | `com_a_idade` | recebe a idade, como na Parte 3 | o LightGBM |

- **Cada corrida guarda o que é preciso para refazê-la e para compará-la**:
  - **os dados**: a configuração da simulação, que dá os mesmos dados byte a byte;
  - **o treino**: o rótulo, a espera, as variáveis da entrada e as que ficaram de fora, o aquecimento, a calibração;
  - **o modelo**: os parâmetros do LightGBM ou da logística;
  - **a decisão**: o alerta e os custos;
  - **o teste**: os meses;
  - **o código**: o commit, que o MLflow anota sozinho quando roda num repositório Git.
- **As métricas** são as da avaliação, com os meses de teste juntos:
  - a fila: a precisão média, a área ROC, os golpes pegos e o dinheiro salvo no alerta, e os golpes previstos;
  - cada golpe: a precisão média e os golpes pegos no alerta;
  - a decisão pelo custo: o custo, o perdido, os golpes parados, e os honestos segurados e bloqueados;
  - cada faixa de idade: os golpes pegos, e os honestos segurados e bloqueados.

  A precisão média de cada mês vai também, como passo, para o MLflow desenhar a curva mês a mês. As tabelas da avaliação vão em JSON, como artefatos.
- **O comando é um só**: `antifraude-pix experimentos` refaz todas as variantes e grava as corridas. O MLflow as mostra com `mlflow ui`.
- **Os avisos do MLflow sobre o próprio código** são liberados nos testes um a um, pelo texto, e não todos de uma vez: um ao importar o MLflow, e um do jeito como ele usa o SQLAlchemy 2.1. Qualquer outro aviso continua sendo erro.

## Consequências

- **As medidas que decidiram o desenho viram corridas**, com a precisão média que as ADRs registram. De outubro a dezembro, nos dados padrão:

  | Variante | Precisão média |
  |---|--:|
  | `padrao`, as regras de hoje | 0,424 |
  | `padrao`, a regressão logística | 0,870 |
  | `padrao`, o LightGBM | 0,932 |
  | `com_a_verdade` | 0,948 |
  | `esperando_14_dias` | 0,928 |
  | `esperando_28_dias` | 0,928 |
  | `com_a_idade` | 0,934 |

- **A Parte 7 compara com histórico.** O modelo candidato vira uma corrida, com os mesmos campos da corrida do modelo atual.
- **O MLflow traz junto o pandas, o SQLAlchemy e o Flask**, que o projeto não usava. O pandas só aparece na fronteira com o MLflow; o código converte para o Polars logo na entrada.
- **Os testes ficaram mais lentos**: cada teste do rastreio cria um banco SQLite numa pasta temporária, e a suíte passou de um minuto para perto de dois.
