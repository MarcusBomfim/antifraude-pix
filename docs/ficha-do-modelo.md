# A ficha do modelo: antifraude-pix

## O que ele faz

Para cada Pix, dá a chance de golpe, calibrada, e decide pelo menor custo esperado: aprovar, segurar para confirmar com o cliente ou bloquear. Cada Pix segurado ou bloqueado sai com as três razões que mais pesaram, em português (ADRs 0010 e 0012).

Entram as variáveis de um Pix; saem a pontuação, a probabilidade, a decisão e a explicação.

## Para que ele não serve

- **Para um banco de verdade, como está.** Ele aprendeu com um simulador; num banco, o processo é o mesmo, mas o treino é com os dados do banco.
- **Para julgar o cliente.** Ele olha o Pix, o hábito e o recebedor. Não é um modelo de crédito, nem de quem a pessoa é.
- **Para decidir sozinho os casos de dúvida.** A confirmação com o cliente e a revisão do analista fazem parte da decisão.

## Os dados

- A simulação padrão: 20.000 clientes, 184 dias a partir de 01/07/2025, semente 2026. A mesma semente dá os mesmos dados, byte a byte.
- O treino vai até a véspera de 01/01/2026: 1.952.800 Pix, dos quais 1.470 o banco já sabia que eram golpe, pela contestação (o rótulo do dia, ADR-0007).
- A probabilidade é calibrada pelo método de Platt em 317.734 Pix maduros, que o modelo não viu (ADR-0008).

## A entrada

32 variáveis, calculadas só com o que o banco sabia antes de cada Pix (ADR-0006):

- **o Pix**: valor, hora, noturno, dia_da_semana, iniciacao, chave, recebedor_pj, recebedor_em_banco_digital, valor_redondo, termina_em_90_centavos, parte_do_saldo, parte_do_limite_noturno;
- **o cliente**: renda_mensal, dias_de_conta, valor_sobre_a_renda;
- **o hábito do cliente**: pix_na_ultima_hora, pix_nas_ultimas_24h, valor_nas_ultimas_24h, minutos_desde_o_ultimo_pix, mediana_de_90_dias, valor_sobre_a_mediana, valor_sobre_o_maior_de_90_dias, pix_noturnos_em_90_dias;
- **o cliente e o recebedor**: vezes_que_pagou_o_recebedor, recebedor_novo, dias_desde_o_primeiro_pix_ao_recebedor;
- **o recebedor**: pix_recebidos_em_24h, pagadores_novos_em_24h, pagadores_novos_em_7_dias, contestacoes_do_recebedor;
- **o aparelho**: minutos_desde_o_aparelho_novo, outros_clientes_no_aparelho.

Fica de fora a idade (ADR-0011): o banco tem o dado, mas a decisão não depende dele.

## O modelo

O LightGBM, com 1.000 árvores de 15 folhas e taxa de aprendizado de 0,02. Aprende com todos os golpes e um Pix legítimo em cada 10, com peso 10.

## A medida, mês a mês

O mesmo processo, refeito em cada mês de teste: o modelo treinado com o que o banco sabia na véspera, e medido contra a verdade do simulador. De 10/2025 a 12/2025, com 1.159.527 Pix e 948 golpes:

| Precisão média | Golpes pegos no alerta | Dinheiro salvo no alerta | Golpes previstos |
|--:|--:|--:|--:|
| 0,932 | 97% | 99% | 929 de 948 |

O alerta são os 0,5% de Pix de cada mês com mais cara de golpe (ADR-0009).

| Golpe | Golpes | Precisão média | Pegos no alerta |
|---|--:|--:|--:|
| falsa central | 342 | 0,930 | 99% |
| falso parente | 180 | 0,799 | 90% |
| invasão de conta | 272 | 0,996 | 100% |
| venda falsa | 154 | 0,778 | 96% |

## A decisão

Confirmar custa R$ 10, bloquear um cliente honesto custa R$ 150, e a confirmação para 70% dos golpes. São suposições, escritas na ADR-0010.

| | Custo | Golpes parados | Honestos segurados | Honestos bloqueados |
|---|--:|--:|--:|--:|
| aprovar tudo | R$ 1.756.886 | 0 | 0 | 0 |
| bloquear de 50% para cima | R$ 268.350 | 766 | 0 | 4 |
| segurar o alerta de 0,5% | R$ 596.843 | 643 | 4.878 | 0 |
| decidir pelo custo | R$ 101.099 | 842 | 1.645 | 25 |

## Os erros por idade

Pela decisão do custo, em cada faixa de idade (ADR-0011):

| Idade | Golpes pegos | Honestos segurados a cada 10 mil | Honestos bloqueados a cada 10 mil |
|---|--:|--:|--:|
| 18 a 34 | 91% | 10,2 | 0,25 |
| 35 a 54 | 93% | 13,3 | 0,11 |
| 55 a 64 | 94% | 17,7 | 0,39 |
| 65 ou mais | 93% | 27,3 | 0,24 |

## Onde ele erra

- **O golpe que o banco demora a saber.** A venda falsa é contestada em semanas, e o modelo aprende menos com ela (ADR-0007).
- **O primeiro golpe para uma laranja nova.** O recebedor já contestado é a razão que mais pesa; antes da primeira contestação, sobra o hábito do cliente (ADR-0012).
- **As pontas da probabilidade.** Acima de 10%, o modelo é tímido: foi golpe mais vezes do que ele previu, e ele bloqueia um pouco menos do que devia (ADR-0008).
- **O cliente honesto de 65 anos ou mais** é segurado para confirmar mais vezes que os outros, porque o golpe contra ele é mais comum (ADR-0011).
- **Os números são de um simulador**, e um banco de verdade fica abaixo deles.

## Como usar

```python
import mlflow

modelo = mlflow.pyfunc.load_model("models:/antifraude-pix@campeao")
decisoes = modelo.predict(variaveis)  # pontuacao, probabilidade, acao e explicacao
```
