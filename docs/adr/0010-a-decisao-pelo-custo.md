# ADR-0010 — A decisão pelo custo: aprovar, segurar para confirmar ou bloquear

**Status:** aceita

## Contexto

A Parte 3 deixou cada Pix com uma probabilidade calibrada de golpe. Falta decidir o que fazer com ela, e os dois jeitos de costume erram:
- **o limiar de 0,5** é o da classificação: bloqueia o Pix com mais de 50% de chance e aprova o resto. Ele trata igual o Pix de R$ 50 e o de R$ 5.000, e o golpe que custa R$ 5.000 passa com 40% de chance;
- **o alerta de 0,5%** da ADR-0009 segura sempre a mesma parte dos Pix. É um orçamento, e não uma decisão: segura o Pix de R$ 30 com cara de golpe e deixa passar o de R$ 3.000 um pouco abaixo da linha.

O banco tem três saídas para cada Pix, e não duas. Além de aprovar e bloquear, ele pode segurar o Pix alguns minutos e confirmar com o cliente: desde 2021, o regulamento do Pix deixa o banco reter o Pix para analisar o risco, por até 30 minutos de dia e 60 de noite. Cada saída erra de um jeito, e cada erro custa um preço diferente.

## Decisão

- **Cada erro tem um custo, em reais.** São suposições de ordem de grandeza, escritas em `decisao/custo.py`:

  | | Custo | Por quê |
  |---|--:|---|
  | o golpe que passa | o valor dele | é o dinheiro do cliente; e, pela Súmula 479 do STJ, o banco responde pela fraude que terceiros fazem nas operações dele |
  | segurar um Pix para confirmar | R$ 10 | o atendimento, a espera e o incômodo do cliente honesto |
  | bloquear o Pix de um cliente honesto | R$ 150 | a reclamação, a ligação e o cliente que às vezes vai embora |
  | o golpe segurado | 30% do valor | a confirmação para 70% dos golpes: a vítima da invasão diz que não foi ela, mas a vítima convencida confirma |

- **A decisão é a saída de menor custo esperado.** Com a probabilidade `p` e o valor `v`:
  - aprovar custa `p × v`;
  - segurar custa `R$ 10 + p × v × 30%`;
  - bloquear custa `(1 − p) × R$ 150`.

  No empate, fica a saída mais branda: endurecer tem de valer a pena.
- **Por isso o limiar depende do valor:**

  | Valor do Pix | Segura a partir de | Bloqueia a partir de |
  |--:|--:|--:|
  | R$ 50 | 29% de chance | 85% |
  | R$ 300 | 4,8% | 58% |
  | R$ 1.000 | 1,4% | 31% |
  | R$ 3.000 | 0,5% | 13% |

- **A probabilidade é a calibrada** (ADR-0008). É por isso que a calibração importava: uma probabilidade que não é probabilidade multiplica o valor errado.
- **A decisão é medida contra a verdade**, nos mesmos três meses de teste, com o custo de fato de cada Pix. O golpe segurado custa a confirmação e 30% do valor, que é o esperado; a conta não sorteia quais golpes a confirmação para.

## Consequências

- **Decidir pelo custo custa um décimo do que custa o banco sem defesa.** De outubro a dezembro, os golpes somam R$ 1,76 milhão. Com o LightGBM:

  | | Custo | Perdido em golpes | Golpes parados | Honestos segurados | Honestos bloqueados |
  |---|--:|--:|--:|--:|--:|
  | aprovar tudo | R$ 1.757 mil | R$ 1.757 mil | 0 | 0 | 0 |
  | segurar o alerta de 0,5% | R$ 597 mil | R$ 539 mil | 643 | 4.878 | 0 |
  | bloquear de 50% para cima | R$ 268 mil | R$ 268 mil | 766 | 0 | 4 |
  | **decidir pelo custo** | **R$ 101 mil** | **R$ 80 mil** | **842** | **1.645** | **25** |

  Os golpes parados somam os bloqueados e 70% dos segurados.
- **Segurar não basta.** O alerta segura 0,5% dos Pix e ainda perde R$ 539 mil, porque a confirmação deixa passar 30% de cada golpe segurado. O golpe com muita cara de golpe tem de ser bloqueado.
- **A decisão pelo custo mexe em pouco**: segura 0,15% dos Pix e bloqueia 0,07%, menos que o orçamento de 0,5% do alerta.
- **O modelo também importa.** Pelo custo, as regras de hoje custam R$ 432 mil, e a regressão logística, R$ 171 mil.
- **As suposições mudam o custo, e não a escolha.** Testei 27 combinações:
  - a confirmação a R$ 5, R$ 10 e R$ 20;
  - o bloqueio do honesto a R$ 75, R$ 150 e R$ 300;
  - a confirmação parando 50%, 70% e 90% dos golpes.

  Em todas, decidir pelo custo foi o mais barato: de R$ 61 mil a R$ 141 mil, contra R$ 220 mil a R$ 1 milhão do alerta e R$ 268 mil do limiar de 0,5.
- **A decisão herda os defeitos da probabilidade.** Acima de 10%, o modelo é tímido (ADR-0008): prevê 79% onde foi golpe 91%, e bloqueia um pouco menos do que devia.
- **Um banco de verdade mede os próprios custos**: o preço da central, a perda de clientes, o quanto a confirmação para de fato. A conta não muda; mudam os números de `Custos`.
