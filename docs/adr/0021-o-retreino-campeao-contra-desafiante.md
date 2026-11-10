# ADR-0021 — O retreino: o campeão contra o desafiante

**Status:** aceita

## Contexto

O alarme do monitor (ADR-0020) diz que o golpe passa mais do que o modelo espera, mas não diz que um modelo novo resolve. Retreinar por calendário, todo mês, troca o modelo bom por um pior quando o mês foi estranho. Retreinar só quando o alarme toca deixa o modelo velho no lugar até alguém olhar. O que falta é uma regra para a troca: o modelo novo só entra se ganhar do atual, nos mesmos Pix, com o rótulo que o banco tem, e sem piorar nada do que o projeto prometeu.

## Decisão

- **O desafiante é a mesma receita do campeão**: o LightGBM, a calibração e a decisão pelo custo, treinados com o que o banco sabia 14 dias antes do retreino.
- **Os dois decidem os mesmos Pix**, os das duas últimas semanas, que nenhum dos dois viu no treino. São medidos com o rótulo que o banco tem no dia do retreino: o contestado é golpe, e o resto não.
- **O desafiante ganha se passar nas três regras** (`monitoramento/retreino.py`):
  - **a precisão média dele passa a do campeão por pelo menos 0,01.** A diferença menor que isso é ruído de duas semanas;
  - **o custo das decisões dele não passa o do campeão**, com os mesmos preços (ADR-0010);
  - **ele não bloqueia mais clientes honestos de 65 anos ou mais do que o acaso explica** (ADR-0011).

    A comparação é Pix a Pix: `b` são os honestos que só ele bloqueia, e `c`, os que só o campeão bloqueia. A diferença `b - c` acima de duas vezes `√(b + c)` não é sorte.
- **O que é comparado é o que é promovido.** O desafiante treinado 14 dias antes é o que vai para o banco se ganhar. Ele não é treinado de novo com os 14 dias a mais, porque esse seria um modelo que ninguém mediu.
- **O registro guarda os dois** (`rastreio/registro.py`):
  - o desafiante entra no MLflow com a ficha e o apelido `desafiante`;
  - o veredito fica na versão dele, como `promovido` ou `recusado`;
  - só a promoção passa a ele o apelido `campeao`, e a API, que carrega o modelo pelo apelido, passa a decidir com ele;
  - cada versão guarda o dia do corte com que foi treinada, e o monitor e o retreino o leem de lá.
- **O `registrar` aceita o corte**, para pôr no registro o campeão que entrou num dia do passado. A ficha mostra só os meses de teste que terminam até o corte.

## Consequências

- **Com a virada, o desafiante ganha.** O retreino de 1º de dezembro, nos Pix de 17 a 30 de novembro, deu:

  | Com o rótulo de 1º de dezembro | Campeão (1º de novembro) | Desafiante (17 de novembro) |
  |---|--:|--:|
  | precisão média | 0,736 | 0,814 |
  | custo das decisões | R$ 17.626 | R$ 10.907 |
  | contestados pegos | 74 de 81 | 78 de 81 |
  | honestos de 65+ bloqueados | 4 | 6 |

  Dois honestos de 65 anos ou mais a mais estão dentro do acaso. O desafiante virou o campeão.
- **Sem a virada, o desafiante perde, e o campeão fica.** Nos mesmos dias da simulação padrão, a precisão média foi de 0,887 do campeão contra 0,855 do desafiante, e o custo, de R$ 10.166 contra R$ 12.574. Ser mais novo não bastou.
- **A verdade de dezembro dá razão às duas decisões.** Ela é medida com o simulador, que o banco não tem:

  | Dezembro, pela verdade | Com a virada: campeão | desafiante | Sem a virada: campeão | desafiante |
  |---|--:|--:|--:|--:|
  | precisão média | 0,877 | 0,940 | 0,959 | 0,959 |
  | custo das decisões | R$ 72.112 | R$ 33.136 | R$ 33.291 | R$ 34.960 |
  | falsa central pega | 111 de 141 | 130 de 141 | 119 de 121 | 120 de 121 |

  Com a virada, a troca cortou o custo de dezembro pela metade. Sem ela, a troca não teria ganhado nada.
- **As duas semanas da comparação são poucas.** São uns 80 Pix contestados, e a regra da precisão média, com a margem, protege contra o acaso, mas não de todo. Por isso a regra do custo vale junto: para ganhar, o desafiante tem de ordenar melhor e decidir mais barato.
- **O rótulo das duas semanas ainda não chegou todo.** O golpe de rótulo lento, como a venda falsa, conta menos na comparação do que deveria. Vale para os dois modelos, e a comparação continua justa entre eles. Ela só não vê bem esse golpe.
