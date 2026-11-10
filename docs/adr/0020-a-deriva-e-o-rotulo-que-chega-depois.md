# ADR-0020 — A deriva e o rótulo que chega depois

**Status:** aceita

## Contexto

O modelo em produção piora sem avisar. O banco não tem a verdade: tem a contestação, que chega horas depois na invasão de conta e semanas depois na venda falsa. O monitoramento de costume olha a deriva dos dados, isto é, se as variáveis de hoje ainda se parecem com as do treino. É barato, sai na hora e não espera rótulo nenhum. A pergunta é se ele vê o que importa. E o golpe novo é raro: cem Pix num mês de quase 400 mil.

## Decisão

- **O monitor olha cada semana inteira desde o corte do campeão** (`monitoramento/semanas.py`).
  - A semana vai de segunda a domingo. A semana pela metade muda o dia da semana e o tamanho do Pix, e a deriva acusaria o calendário.
  - O monitor roda com o que o banco sabia num dia, o `ate`.
- **A deriva, pelo índice de estabilidade da população** (`monitoramento/deriva.py`).
  - Cada variável da entrada e a probabilidade do campeão são comparadas com as quatro semanas antes do corte, as últimas que ele viu.
  - O número vai em dez faixas, pelos decis da referência. A categoria e o sim ou não vão uma faixa por valor, e o vazio é uma faixa a mais.
  - Cada faixa tem um piso de um centésimo de por cento, para a faixa que a referência não tem não dar um índice infinito.
  - Os limites são os de costume: até 0,1, estável; acima de 0,25, mudou.
- **O rótulo que chega depois** (`monitoramento/rotulo.py`).
  - A curva de chegada, `F(k)`, é a parte das contestações que chega até k dias depois do Pix. Ela se mede na história do próprio banco, sem a verdade, com os Pix que já tiveram os 80 dias do MED.
  - Na simulação padrão, 58% chega no primeiro dia, 86% na primeira semana e 99,6% no primeiro mês.
- **A perda esperada contra a perda conhecida**, nos Pix que o campeão aprovou.
  - **Esperada**: a probabilidade vezes o valor de cada Pix, vezes a parte do rótulo que já chegou na idade dele. É o que o próprio modelo diz que vai perder.
  - **Conhecida**: o valor dos aprovados que o banco já sabe contestados.
  - **A razão entre as duas, acumulada desde o corte, dispara o alarme em 2.** Com o modelo calibrado e o golpe de sempre, as duas andam juntas. O golpe novo passa com a probabilidade baixa, e a perda conhecida sobe sem a esperada subir.
  - Semana a semana, a razão é ruidosa: uma venda falsa de R$ 3 mil muda uma semana. Acumulada, ela se firma em duas semanas.

## Consequências

- **O PSI não vê o golpe novo.** A probabilidade do campeão ficou estável em todas as semanas, com e sem a virada: o maior índice foi 0,005 com ela e 0,006 sem ela. Cem Pix num mês de quase 400 mil não mudam a distribuição de nada.
- **O PSI acusa o que muda, e não o que importa.** A variável que mais mudou, nos dois cenários, foi os dias desde o primeiro Pix ao recebedor, com o índice de 1,3 a 1,8 em novembro. A simulação começa em julho, e a história de cada par de cliente e recebedor só cresce: é um relógio. Num banco, com anos de história, ela andaria devagar. O monitor mostra o índice e diz qual variável mudou, mas não dispara o alarme por ele.
- **O rótulo vê.** Com o rótulo de 1º de dezembro, as quatro semanas de novembro deram:

  | Semana | Com a virada: esperada | conhecida | razão | Sem a virada: esperada | conhecida | razão |
  |---|--:|--:|--:|--:|--:|--:|
  | 03/11 | R$ 4.638 | R$ 18.735 | 4,04 | R$ 5.387 | R$ 6.063 | 1,13 |
  | 10/11 | R$ 3.082 | R$ 7.279 | 2,36 | R$ 3.937 | R$ 921 | 0,23 |
  | 17/11 | R$ 2.283 | R$ 620 | 0,27 | R$ 2.998 | R$ 134 | 0,04 |
  | 24/11 | R$ 1.685 | R$ 3.317 | 1,97 | R$ 2.146 | R$ 0 | 0,00 |
  | **acumulada** | | | **2,56** | | | **0,49** |

  Com a virada, o alarme toca; sem ela, não. A razão acumulada com a virada ficou acima de 2 desde a primeira semana; sem ela, foi de 1,13 a 0,49. Deixado até o fim de dezembro, sem retreino, o campeão da virada só ficou abaixo de 2 uma vez, em 1,95, na semana de 1º de dezembro, e o da simulação padrão não passou de 1,51.
- **Sem a virada, a razão fica abaixo de 1, e a conta é conservadora.** A curva de chegada é a de todas as contestações, e o golpe que passa é, mais que a média, o de rótulo lento, como a venda falsa. A parte que já chegou é superestimada nas semanas recentes, e a razão delas sai baixa. O alarme demora um pouco mais, mas não toca à toa.
- **O monitor precisa do campeão com o dia em que ele foi treinado.** O registro guarda o corte de cada versão, e o `monitorar` o lê de lá.
