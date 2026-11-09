# ADR-0016 — As variáveis em tempo real, iguais às do treino

**Status:** aceita

## Contexto

No treino, as 33 variáveis saem de uma vez, em lote, com as janelas do Polars sobre a história inteira (ADR-0006). No banco, o Pix chega um de cada vez. A variável tem de sair em microssegundos, antes de o Pix sair, sem reler a história.

São duas contas diferentes para a mesma coisa, e é aí que mora um dos erros mais comuns de modelo em produção: o serviço calcula a variável de um jeito um pouco diferente do treino, e o modelo decide com números que nunca viu. A precisão medida no treino deixa de valer, e nada avisa.

## Decisão

- **O estado do banco guarda o que as variáveis pedem, e só isso** (`tempo_real/estado.py`):
  - de cada cliente, os Pix dos últimos 90 dias: o instante, o valor e se foi noturno;
  - de cada cliente e recebedor, quantas vezes e desde quando;
  - de cada recebedor, os Pix dos últimos 7 dias, e as contestações;
  - de cada aparelho, quando entrou em cada conta;
  - de cada Pix dos últimos 81 dias, o recebedor: a contestação chega até 80 dias depois, pelo MED.

  O que sai da janela mais longa é podado.
- **Cada conta segue a regra do lote**, com busca binária em listas ordenadas pelo tempo:
  - a janela é fechada à esquerda e aberta à direita, `[t − período, t)`: o Pix do mesmo segundo não entra na conta do outro, e o Pix de exatamente uma hora antes entra;
  - o aparelho que entra no instante do Pix já vale para ele;
  - a contestação feita no instante do Pix, não.
- **Os eventos chegam na ordem do tempo**: o Pix, o aparelho novo e a contestação. O evento de antes do último é recusado, porque o estado não volta atrás.
- **A paridade é exata, e não aproximada.** O teste reproduz uma simulação inteira, evento por evento, pelo estado, e compara as 33 variáveis de cada Pix com as do cálculo em lote: até o último bit. Vale para:
  - todos os 87.956 Pix da simulação média;
  - uma simulação longa, de 200 dias, em que o estado poda a história;
  - o estado que começa no meio da história, como o serviço que sobe com tudo até ontem.

  Um exemplo feito à mão confere as bordas: o mesmo segundo, a ponta de cada janela, e o aparelho e a contestação no instante do Pix.
- **O estado faz a divisão como o Polars a faz.** O Polars divide por uma constante multiplicando pelo inverso dela, e o resultado difere da divisão no último bit: 7.521 / 100 dá 75,21, e 7.521 × 0,01 dá 75,21000000000001. A primeira versão do estado dividia, e a paridade falhou nas seis variáveis divididas por constante, por um bit. O estado agora multiplica pelo inverso, com o porquê escrito ao lado, e o teste exige a igualdade. Se uma versão nova do Polars mudar a conta, o teste avisa.

## Consequências

- **A variável do serviço é a do treino, provada Pix a Pix**, e não por amostra.
- **Calcular as variáveis de um Pix leva 0,03 ms** na mediana (ADR-0018).
- **O estado sobe em menos de meio minuto** com os dois milhões de Pix até dezembro, contados um a um na mesma função que o serviço usa. Não há uma segunda carga, que também precisaria de paridade.
- **O estado mora na memória de um processo só.** Num banco, ele ficaria num banco de dados em memória, compartilhado pelos servidores, com a mesma regra de cada conta; a regra e o teste de paridade valem do mesmo jeito.
