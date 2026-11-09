# ADR-0018 — A latência medida

**Status:** aceita

## Contexto

O Pix é instantâneo, e a decisão vem antes dele sair. Cada milissegundo do serviço é um milissegundo do cliente esperando, e o banco ainda tem a própria rede e as próprias regras no caminho. Não basta dizer que o serviço é rápido: é preciso medir, com os Pix de verdade, e olhar o mais lento, e não o típico.

## Decisão

- **A medida é feita com os Pix de verdade, na ordem do tempo** (`tempo_real/latencia.py`). O estado sobe com a história até 1º de dezembro. Os Pix de dezembro passam com cada aparelho novo e cada contestação do caminho, e o campeão do registro decide.
- **Duas medidas**:
  - **por dentro**: as variáveis e a decisão, cronometradas cada uma, sem a API;
  - **pela API**: o serviço num servidor uvicorn de verdade, na própria máquina, e cada Pix mandado por HTTP, como o banco mandaria. A ida e volta inclui o JSON, a validação e a rede local. O serviço também diz o tempo que gastou dentro dele.
- **O resumo é a mediana, os percentis 95 e 99, e o máximo.**
- **A primeira medida mudou o desenho.** O caminho da tabela, o mesmo do lote, levava 2,9 ms na mediana para decidir um Pix, e pelo `predict` do MLflow eram 10 ms. Quase tudo era o custo do Polars de preparar uma tabela de uma linha, e não a conta:
  - converter os tipos: 0,3 ms;
  - pontuar: 1 ms, dos quais 0,5 ms do LightGBM pelo scikit-learn;
  - decidir pelo custo: 0,5 ms;
  - explicar: 1 ms, mesmo para o Pix aprovado, que não tem razão nenhuma.
- **O caminho de um Pix só** (`decidir_um`):
  - a linha das árvores sai direto das variáveis, em NumPy;
  - o LightGBM pontua pelo próprio motor, com um fio só, em 0,1 ms;
  - a decisão é a mesma conta do custo (ADR-0010), na mesma ordem, num número só.

  Só o Pix segurado ou bloqueado, que tem razões a explicar, passa pelo caminho da tabela.
- **Os dois caminhos dão o mesmo resultado, e os testes conferem**:
  - a mesma pontuação, a mesma probabilidade, a mesma decisão e a mesma explicação em 3.000 Pix, com custos que seguram e bloqueiam muitos deles;
  - a decisão de um número só igual à da tabela, em valores sorteados pelo hypothesis.

## Consequências

- **O serviço decide um Pix em 0,26 ms na mediana, e pela API, ida e volta, em 1,2 ms.** O comando `antifraude-pix latencia`, com 5.000 Pix em cada medida, deu:

  | Em milissegundos | Mediana | p95 | p99 | Máximo |
  |---|--:|--:|--:|--:|
  | as variáveis | 0,03 | 0,05 | 0,08 | 0,26 |
  | a decisão | 0,15 | 0,23 | 0,41 | 12,52 |
  | a decisão do Pix segurado ou bloqueado (30 Pix) | 8,25 | 9,11 | 11,55 | 12,52 |
  | dentro do serviço | 0,26 | 0,37 | 0,53 | 9,43 |
  | a ida e volta pela API | 1,21 | 1,51 | 1,77 | 55,04 |

- **O mais lento é o Pix que precisa de explicação**, em 8 ms: um em cada 170. Se um dia pesar, a explicação pode sair depois da decisão, porque o cliente espera a decisão, e não o porquê.
- **O máximo da ida e volta é o primeiro pedido**, com a conexão e a primeira chamada de cada biblioteca. O p99 é o que vale para o banco.
- **Os números são de um computador de mesa** (um Ryzen 5 5600), com um processo só e a rede da própria máquina. Num banco, a rede entre os serviços soma milissegundos que esta medida não vê; a parte do serviço é a que está aqui.
