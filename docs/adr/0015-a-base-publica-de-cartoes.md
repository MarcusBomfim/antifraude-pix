# ADR-0015 — O mesmo processo numa base real: a de cartões da ULB

**Status:** aceita

## Contexto

Tudo até aqui foi medido num simulador. O simulador permite o que nenhum banco tem, a verdade ao lado do rótulo atrasado, mas também levanta uma suspeita legítima: o processo funciona fora dele, ou só no mundo que ele mesmo inventou?

Não existe base pública de Pix. A base pública de fraude mais usada é a de cartões da ULB (Universidade Livre de Bruxelas), coletada com a Worldline (Dal Pozzolo e outros, 2015):
- são compras com cartão de clientes europeus, em dois dias de setembro de 2013;
- são 284.807 compras e 492 fraudes, 0,17%;
- as variáveis originais viraram 28 componentes principais, de V1 a V28, sem nome, para proteger os clientes. Só o tempo e o valor vêm como são.

## Decisão

- **A base vem do OpenML** (conjunto 1597), e é conferida pelo MD5 publicado lá, antes de ganhar o nome definitivo. Ela fica em `dados/cartoes/`, e não no repositório. O comando `antifraude-pix cartoes` a baixa se faltar, e roda o processo (`cartoes/`).
- **O mesmo processo, com o mínimo de adaptação**:
  - **a validação no tempo**: o treino no primeiro dia, e o teste no segundo. Sortear o teste entre os dois dias deixaria o modelo aprender com a fraude de amanhã;
  - **os modelos**: a regressão logística e o LightGBM, com os parâmetros escolhidos para o Pix, sem nenhum ajuste nos cartões. O ponto de partida, no lugar das regras de hoje, é o valor sozinho;
  - **o LightGBM** aprende com todas as fraudes e uma compra legítima em cada dez, com peso dez;
  - **a calibração** é pelo método de Platt, em um quinto das compras do treino, que o modelo não vê;
  - **a medida** usa as mesmas funções do Pix: a precisão média, o alerta de 0,5% das compras do dia, a calibração e o custo de cada jeito de decidir. A avaliação dos cartões sai no mesmo formato da do Pix, e as métricas não mudam uma linha.
- **O que não vai**:
  - o rótulo atrasado: a base só tem o rótulo final;
  - a explicação em português: V1 a V28 não têm nome;
  - os erros por idade: a base não tem idade.
- **As corridas vão para o MLflow**, num experimento só dos cartões (`cartoes-ulb`), com a origem e o MD5 da base entre os parâmetros.

## Consequências

- **O processo funciona na base real.** No segundo dia, com 140.021 compras e 211 fraudes:

  | | Precisão média | Área ROC | Fraudes pegas no alerta | Dinheiro salvo no alerta | Fraudes previstas |
  |---|--:|--:|--:|--:|--:|
  | o valor sozinho | 0,002 | 0,44 | 1% | 14% | 236 |
  | a regressão logística | 0,66 | 0,960 | 80% | 62% | 241 |
  | o LightGBM | 0,79 | 0,985 | 84% | 67% | 252 |

  O LightGBM ganha da logística, como no Pix, e com os parâmetros do Pix.
- **A base real é mais difícil que o simulador**: 0,79 de precisão média, contra 0,93 no Pix. É o que o README avisava desde a Parte 3, e a ordem dos modelos não muda.
- **O valor sozinho aponta para o lado errado.** A fraude com cartão nesta base é pequena: a mediana é € 9,21, contra € 20,33 da compra honesta. No Pix, o golpe pede mais; no cartão, uma explicação comum é que quem tem um cartão roubado o testa com compras pequenas. A área ROC do valor fica abaixo de 0,5.
- **A taxa de fraude muda de um dia para o outro**: 0,19% no primeiro dia e 0,15% no segundo. Por isso a probabilidade, calibrada no primeiro dia, previu 252 fraudes onde houve 211. Acima de 10% de chance, ela continua certa (previu 23% e foi fraude 23%; previu 86% e foi 85%). O erro está embaixo, onde a taxa de base pesa. É a mudança que a Parte 7 vai vigiar.
- **Pelo custo, a decisão salva menos que no Pix**:

  | | Custo |
  |---|--:|
  | aprovar tudo | € 26.889 |
  | segurar o alerta de 0,5% | € 21.247 |
  | bloquear de 50% para cima | € 14.789 |
  | **decidir pelo custo** | **€ 11.613** |

  Decidir pelo custo corta 57% do custo de aprovar tudo, contra 94% no Pix. Com fraudes de poucos euros, confirmar custaria mais que a fraude, e a decisão deixa passar a fraude pequena, de propósito. Os custos são os mesmos do Pix, em euros: uma simplificação.
- **O que vale levar daqui é o método.** Os números de uma base de 2013, com variáveis anonimizadas, não dizem como um modelo se sairia num banco hoje. Dizem que o processo se sustenta fora do simulador: o tempo, a calibração, a medida e a decisão.
