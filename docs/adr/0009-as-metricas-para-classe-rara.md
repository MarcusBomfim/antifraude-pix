# ADR-0009 — As métricas para classe rara: o topo da fila, o alerta do mês, cada golpe e a calibração

**Status:** aceita

## Contexto

Na simulação padrão, há um golpe em cada 1.200 Pix. Com essa raridade, as métricas de costume enganam:
- **a taxa de acerto**: o modelo que nunca acusa ninguém acerta 99,92%;
- **a área sob a curva ROC**: mede a fila inteira, e o banco só olha o começo dela. As regras de hoje têm 0,92 de área ROC e pegam metade dos golpes;
- **o total**: o modelo que pega toda invasão de conta e nenhum falso parente parece bom.

O banco também não segura quantos Pix quiser. Cada Pix segurado é um cliente esperando a confirmação, e alguém do outro lado para confirmar.

## Decisão

- **A precisão média** (a área sob a curva de precisão e recall) é a métrica principal, porque pesa o começo da fila.
- **O alerta**: em cada mês, o banco segura os 0,5% de Pix de maior pontuação. No alerta, a avaliação mede:
  - os golpes pegos, entre todos os golpes do mês;
  - o dinheiro salvo, entre todo o dinheiro dos golpes, porque pegar o golpe de R$ 3.000 vale mais que pegar o de R$ 150;
  - os clientes honestos segurados para cada golpe pego.
- **No empate, quem não cabe inteiro não entra.** Se os Pix empatados na fronteira do alerta não cabem todos, nenhum deles é segurado. O banco não segura metade dos Pix de três pontos. As regras, que dão a mesma soma a muitos Pix, seguram menos que os 0,5%.
- **Cada golpe é medido separado**, contra o Pix legítimo: a precisão média e os golpes pegos no alerta.
- **A calibração é medida contra a verdade**: os golpes previstos (a soma das probabilidades) contra os que houve, e a parte que foi golpe em cada faixa de probabilidade prevista.
- **Os três meses de teste contam juntos**, cada um com o seu modelo; o alerta é sempre do mês.

## Consequências

- **O resultado, de outubro a dezembro, com 948 golpes:**

  | | Precisão média | Golpes pegos | Dinheiro salvo | Honestos por golpe |
  |---|--:|--:|--:|--:|
  | as regras de hoje | 0,42 | 53% | 76% | 1,7 |
  | a regressão logística | 0,88 | 93% | 97% | 5,6 |
  | o LightGBM | 0,93 | 97% | 99% | 5,3 |

- **Os honestos por golpe enganam quando os modelos seguram números diferentes.** As regras seguram menos de 0,5%, porque os empatados não cabem, e por isso seguram menos honestos e pegam menos golpes. A comparação justa é a curva: quantos golpes cada modelo pega para cada parte de Pix segurada.
- **O falso parente e a venda falsa são os difíceis**: 0,81 e 0,79 de precisão média no LightGBM, contra 0,93 na falsa central e 0,99 na invasão.
- **Os números são de um simulador**, e um banco de verdade fica abaixo deles. O que vale levar daqui é a comparação entre os modelos, a diferença entre os golpes e o método de medir.
- **O alerta de 0,5% é um orçamento, e não uma decisão.** Na Parte 4, a decisão de segurar sai do custo de cada lado, Pix a Pix.
