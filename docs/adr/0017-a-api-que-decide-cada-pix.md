# ADR-0017 — A API que decide cada Pix

**Status:** aceita

## Contexto

O banco precisa de uma resposta antes de cada Pix sair: aprovar, segurar para confirmar ou bloquear. Precisa também do porquê, para o analista e para a revisão que a LGPD garante (ADR-0012).

E o banco muda o que sabe o tempo todo: entra um aparelho novo, chega uma contestação, e o cliente confirma o Pix segurado. O serviço tem de ouvir esses eventos, ou as variáveis do próximo Pix saem erradas.

## Decisão

- **Uma API em FastAPI** (`tempo_real/servico.py`), com o estado do banco (ADR-0016) e o campeão do registro do MLflow (ADR-0014):

  | Rota | O que faz |
  |---|---|
  | `POST /pix` | decide o Pix que chegou, e devolve a ação, a chance de golpe, a explicação, o modelo e a versão |
  | `POST /pix/{id}/confirmacao` | o cliente confirmou, ou não, o Pix segurado |
  | `POST /aparelhos` | um aparelho entrou na conta de um cliente |
  | `POST /contestacoes` | a vítima contestou um Pix |
  | `GET /saude` | o modelo que decide, e o tamanho do estado |

- **Só o Pix que saiu entra na história**:
  - o aprovado entra na hora;
  - o segurado entra quando o cliente confirma, no instante da confirmação, que é quando ele sai;
  - o bloqueado nunca entra, porque não aconteceu.

  É a mesma história que o treino lê, com os Pix que saíram.
- **O pedido que não faz sentido é recusado com o motivo**, antes de chegar ao modelo, pelos tipos da própria API:
  - o valor acima do saldo;
  - a chave no Pix pela agência e a conta;
  - o banco que não existe;
  - o instante sem fuso.

  O cliente que não é do banco recebe 404. O Pix de antes do último evento, ou o Pix que já foi decidido, recebe 409.
- **Um cadeado põe os pedidos em fila.** O estado não foi feito para dois Pix ao mesmo tempo, e a conta de cada um leva décimos de milissegundo.
- **O campeão vem do registro pelo apelido**, e não pela versão. O modelo registrado leva o código dele, e o da Parte 5 não tem o caminho de um Pix só (ADR-0018): ele decide pelo `predict` do MLflow, mais lento, e com a mesma resposta, que um teste confere.
- **O comando `antifraude-pix servir`** sobe o estado com toda a história dos dados, carrega o campeão e serve a API. A documentação de cada rota fica em `/docs`, que o FastAPI gera.

## Consequências

- **A API decide com as variáveis do treino.** O teste manda os dois primeiros dias de setembro pela API, com cada aparelho novo e cada contestação, e confere a probabilidade e a decisão de cada um dos 1.734 Pix contra o modelo com as variáveis do cálculo em lote. Para a história da API seguir a do treino, o teste usa um custo de bloqueio tão alto que nenhum Pix é bloqueado, e confirma o Pix segurado no mesmo instante.
- **O modelo e a regra da decisão mudam sem mexer na API**: muda o apelido no registro, e o serviço sobe com o novo campeão.
- **Um processo só, com o estado na memória.** Para mais de um servidor, o estado iria para um banco de dados em memória e o cadeado sairia. As rotas, a regra da história e o teste ficam os mesmos.
- **O teste usa o `httpx2`**, o cliente que o Starlette desta versão pede para o `TestClient`.
