# ADR-0019 — A virada: o golpe novo

**Status:** aceita

## Contexto

Um modelo de fraude envelhece de um jeito diferente dos outros: do outro lado há alguém que se adapta. A quadrilha que vê o golpe barrado muda a tática, e o modelo, treinado com o golpe de antes, não a reconhece. Para mostrar o monitoramento funcionando, o simulador precisa de um golpe que muda no meio do caminho. E a mudança tem de ser das que acontecem: não um golpe de outro planeta, mas a mesma quadrilha fazendo diferente.

A falsa central é a candidata natural. Ela já engana a vítima por telefone, com a voz do "banco", e a manda tirar o dinheiro da conta "invadida". O que denuncia o golpe hoje é a pressa: dois a seis Pix em minutos, de boa parte do saldo, para contas novas.

## Decisão

- **Uma data na configuração, a `virada`.** Sem ela, a simulação é a de sempre, byte a byte, e o manifesto também: a chave nem aparece nele. Com ela, toda falsa central que começa naquele dia, ou depois, muda de tática. A virada fora do período é recusada com o motivo.
- **A falsa central parcelada** (`simulacao/golpes.py`):
  - a "central" diz que o dinheiro tem de sair aos poucos, "para o sistema não bloquear";
  - um Pix por dia, de três a seis dias, perto da hora do primeiro (até duas horas antes ou depois);
  - sempre para a mesma "conta segura", pela chave;
  - cada um de um quinto a um terço do saldo.

  Sem a pressa, e com o recebedor que, do segundo Pix em diante, o cliente já pagou, o golpe se parece com um pagamento combinado.
- **Só a tática muda.** O plano dos ataques, as vítimas, os horários, os outros três golpes e a contestação são os mesmos. Os testes conferem que, com a virada, só a falsa central dali em diante tem outros passos.
- **A simulação da virada é a padrão com a virada em 1º de novembro**, gravada em `dados-virada/`. Novembro e dezembro têm o golpe novo, e o modelo treinado em 1º de novembro nunca o viu.

## Consequências

- **O campeão de 1º de novembro deixa passar a falsa central nova.** Pela verdade do simulador, a parte dela que ele pega, por semana, é esta:

  | Semana | 03/11 | 10/11 | 17/11 | 24/11 | 01/12 | 08/12 | 15/12 | 22/12 |
  |---|--:|--:|--:|--:|--:|--:|--:|--:|
  | com a virada | 47% | 63% | 80% | 89% | 87% | 71% | 90% | 75% |
  | sem a virada | 100% | 100% | 93% | 100% | 100% | 96% | 96% | 100% |

  O resto ele não perde de todo: a conta segura que recebe de muitas vítimas acaba chamando a atenção pelas variáveis do recebedor.
- **Antes da virada, os Pix e os golpes são os mesmos da simulação padrão.** Mudam de lugar as contestações falsas, as de Pix honestos por arrependimento ou má-fé, que são sorteadas sobre todos os Pix legítimos. Com mais Pix de golpe depois da virada, o sorteio cai em outros. O campeão de 1º de novembro dos dois cenários, então, não é idêntico.
- **A virada é uma só, e de um golpe só.** Ela serve para provar o monitor, e não para imitar a criatividade de uma quadrilha. Outras viradas, como uma venda falsa que passa a usar contas antigas, cabem no mesmo lugar.
