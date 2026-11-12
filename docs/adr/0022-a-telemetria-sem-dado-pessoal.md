# ADR-0022 — A telemetria sem dado pessoal

**Status:** aceita

## Contexto

O serviço decide cada Pix em décimos de milissegundo (ADR-0018). Para cuidar dele no ar, é preciso ver, de fora:
- quantos Pix ele decide;
- o que decide;
- quanto leva cada etapa;
- o que o cliente responde ao Pix segurado;
- e, quando um Pix sai estranho, o caminho dele dentro do serviço.

É o que o OpenTelemetry dá: traços, métricas e o protocolo OTLP, que qualquer coletor entende.

Só que o Pix é dado pessoal, do pagador e do recebedor, e dado bancário, sob sigilo. Telemetria é a via mais comum de vazamento: ela sai do serviço para outros sistemas, com outros donos, guardada por outro prazo. Traço com id de cliente, caminho de URL com id de transação, mensagem de erro com o valor: todo mundo já viu.

## Decisão

- **Cada pedido à API vira um traço** (`tempo_real/telemetria.py`). O do Pix tem as duas etapas dentro dele: as variáveis, que o estado calcula, e o modelo, que decide. Se o banco manda o cabeçalho `traceparent`, o traço do serviço continua o dele.
- **Oito métricas**:

  | Métrica | O que conta |
  |---|---|
  | `antifraude.pix.decididos` | os Pix, por ação e faixa de valor |
  | `antifraude.pix.valor` | o dinheiro, em reais, por ação |
  | `antifraude.pix.probabilidade` | a chance de golpe que o modelo deu, por ação |
  | `antifraude.decisao.duracao` | o tempo de cada etapa, em milissegundos |
  | `antifraude.segurados` | os Pix esperando o cliente |
  | `antifraude.confirmacoes` | as respostas do cliente ao segurado |
  | `antifraude.contestacoes` | as contestações que chegaram |
  | `antifraude.pedidos.recusados` | os pedidos recusados, pelo código |

  Os limites dos histogramas são os do serviço: a decisão leva décimos de milissegundo, e quase toda probabilidade fica perto de zero.
- **Nada que identifique alguém sai do serviço.** Ficam de fora o cliente, o recebedor, o aparelho, a chave, o id do Pix e o instante do Pix. Os atributos são uma lista fechada, de onze nomes (`telemetria.ATRIBUTOS`), e só ela é usada.
  - O valor exato vai só somado, nas métricas; no traço vai a faixa (até R$ 100, de 100 a 500, de 500 a 2 mil, de 2 a 10 mil, acima).
  - A mensagem de um erro pode trazer um id ("O Pix 123 já foi decidido."). O traço guarda só o código HTTP e o tipo do erro, sem a mensagem e sem a exceção.
- **Quem precisa do traço de um Pix tem o id do traço na resposta.** O banco o guarda junto do Pix, do lado dele, onde o dado pessoal já está. A telemetria não precisa saber de quem é o Pix para mostrar o caminho dele.
- **A telemetria do FastAPI fica desligada.** O FastAPI 0.162 tem a dele, ligada por padrão:
  - o traço guarda o caminho do pedido, e na confirmação o caminho tem o id do Pix;
  - o registro guarda o corpo do pedido recusado na validação.

  O serviço a desliga e passa a ela os mesmos provedores: se alguém a ligar de novo, os spans dela aparecem nos testes. Ela foi achada no contêiner, pelo Jaeger, e não nos testes, que usavam provedores próprios.
- **Sem `OTEL_EXPORTER_OTLP_ENDPOINT`, nada é exportado.** O OpenTelemetry fica no modo que não faz nada, e o serviço roda como antes. Com o endereço, o serviço manda os traços em lote e as métricas a cada cinco segundos, com o nome e a versão do modelo no recurso. O resto vem das variáveis de ambiente de costume do OpenTelemetry.

## Consequências

- **Os testes conferem a telemetria pelos exportadores em memória do próprio SDK**:
  - o traço com as duas etapas, e o id dele na resposta;
  - as métricas de cada ação, do dinheiro e de quem espera o cliente;
  - o traço que continua o do banco;
  - o pedido recusado só pelo código;
  - o Pix fora da amostra, que conta nas métricas e volta sem traço;
  - com os provedores globais, como no contêiner, só a telemetria do serviço sai;
  - e, em trinta Pix com todos os tipos de pedido, nenhum atributo fora da lista e nenhum valor igual a um id, uma chave ou um valor de Pix.
- **O custo foi medido** com o `antifraude-pix latencia`, em 5.000 Pix de dezembro, com a telemetria indo para um coletor de verdade:

  | Em milissegundos | Mediana | p95 | p99 |
  |---|--:|--:|--:|
  | dentro do serviço, sem telemetria | 0,30 | 0,45 | 0,70 |
  | dentro do serviço, com todos os traços | 0,39 | 0,71 | 1,25 |
  | dentro do serviço, com um traço a cada dez | 0,34 | 0,49 | 0,76 |
  | a ida e volta, sem telemetria | 1,21 | 1,56 | 1,94 |
  | a ida e volta, com todos os traços | 1,45 | 2,18 | 4,41 |
  | a ida e volta, com um traço a cada dez | 1,34 | 1,70 | 2,10 |

  Com todos os traços, o p99 da ida e volta mais que dobra: o envio em lote roda num fio que disputa o interpretador com os pedidos. **Com um traço a cada dez, o custo quase some.** É o que o `compose` usa (`OTEL_TRACES_SAMPLER=parentbased_traceidratio`, 0,1), sem mudar o código. O traço que o banco manda é sempre seguido, e as métricas contam todos os Pix.
- **O id do traço só vai na resposta quando o traço foi guardado.** O Pix fora da amostra volta sem ele: o banco não guarda o id de um traço que não existe.
