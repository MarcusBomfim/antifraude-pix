# ADR-0024 — A reprodução: os eventos do banco no ritmo dele

**Status:** aceita

## Contexto

O painel e a telemetria só mostram alguma coisa com Pix chegando. A medida da latência (ADR-0018) manda os Pix o mais depressa que pode, e confirma todo Pix segurado. Para o painel, isso não serve:
- o ritmo do banco tem a madrugada vazia e o meio-dia cheio;
- a resposta do cliente ao Pix segurado é o que mostra a confirmação funcionando.

## Decisão

- **`antifraude-pix reproduzir` manda à API os eventos de um período** (`tempo_real/reproducao.py`): os Pix, os aparelhos novos e as contestações, na ordem do tempo, como na medida da latência.
- **O ritmo é o do banco vezes a velocidade.** Com 600, uma hora passa em seis segundos, e um dia em dois minutos e meio. Cada evento espera a hora dele; se a API atrasa, a reprodução não dorme até alcançar. Com velocidade 0, vai o mais depressa que a API responde.
- **O serviço sobe com a história até o dia em que a reprodução começa.** É o `servir --ate`. Daí em diante, os eventos chegam pela API.
- **A reprodução faz o papel do cliente no Pix segurado**:
  - o cliente honesto confirma;
  - a vítima de golpe recusa na parte que a confirmação para (`Custos.confirmacao_para`, 70%), sorteada com a semente.
- **Para saber quem é vítima, a reprodução lê a verdade do simulador.** Ela faz o papel do mundo, e não do banco: a API nunca vê a verdade. O sorteio só corre para a vítima, e a sequência dele não depende de quantos honestos foram segurados.
- **A cada dia que vira, a reprodução imprime o placar do dia**: os Pix, os aprovados, os segurados e os confirmados, os bloqueados.

## Consequências

- **O dia 1º de dezembro inteiro, sem pausa, passou em 70 s pela API do contêiner**: 12.491 Pix, 35 segurados (34 confirmados) e 15 bloqueados. É a mesma conta toda vez, porque a semente é a mesma.
- **A contestação de um Pix que não saiu volta 404**, porque ele foi bloqueado ou recusado. A reprodução segue, e o painel conta esses pedidos recusados pelo código.
- **Se a API não subiu com o `--ate` do mesmo dia, a reprodução para no primeiro Pix fora de ordem**, com o motivo e a pergunta do `--ate`.
- **A reprodução é uma ferramenta de demonstração e de fumaça.** Ela não mede nada que a medida da latência já não meça.
