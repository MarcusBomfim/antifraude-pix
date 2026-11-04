# ADR-0006 — As variáveis no ponto do tempo: cada uma com o que o banco sabia antes do Pix

**Status:** aceita

## Contexto

O vazamento é o erro mais caro de um modelo de fraude, porque não aparece na avaliação: aparece em produção. Ele vem de vários lugares:
- a contagem de Pix do dia que inclui o próprio Pix, ou os do resto do dia;
- a mediana calculada com o mês inteiro;
- o recebedor marcado como laranja porque, uma semana depois, foi contestado.

Na avaliação, tudo isso dá ao modelo uma informação que ele não terá na hora de decidir.

A Parte 1 mostrou o que cada golpe deixa de rastro: o valor fora do hábito, a pressa, o recebedor novo, a laranja que recebe de muitos desconhecidos, o aparelho que acabou de entrar. Cada rastro precisa virar uma variável que o banco consegue calcular na hora do Pix, em milissegundos, só com o passado.

## Decisão

- **Trinta e três variáveis, em seis grupos**, num catálogo com o nome, o tipo e uma frase em português para cada uma:
  - **o Pix**: o valor, a hora, a iniciação, a chave, o recebedor empresa ou de banco digital, o valor redondo ou terminado em 90 centavos, a parte do saldo e do limite noturno;
  - **o cliente**: a idade, a renda, os dias de conta, o valor sobre a renda;
  - **o hábito do cliente**: os Pix da última hora e das últimas 24 horas, os minutos desde o Pix anterior, o valor típico de 90 dias e quantas vezes o Pix é ele, o maior Pix de 90 dias, os Pix noturnos;
  - **o cliente e o recebedor**: quantas vezes já pagou, se é novo, há quanto tempo o paga;
  - **o recebedor**, entre todos os clientes do banco: os Pix recebidos em 24 horas, os pagadores novos em 24 horas e em 7 dias, e as contestações que já tinham chegado;
  - **o aparelho**: há quantos minutos entrou na conta, e em quantas contas de outros clientes já tinha entrado.
- **Toda janela no tempo é `[t − período, t)`**, fechada à esquerda e aberta à direita. O Pix não entra nas contas dele mesmo, nem nas de outro Pix do mesmo segundo.
- **O que vem de outra tabela é o que já tinha acontecido:**
  - a contestação conta se chegou antes do Pix;
  - o aparelho de outra conta conta se entrou nela antes;
  - a entrada do próprio aparelho conta até o instante do Pix, porque é a mesma sessão.
- **O que o banco não vê não entra.** O recebedor é visto pelos Pix que os clientes deste banco mandam para ele, e não pelo extrato do banco dele.
- **Prova de que não há vazamento, em três testes:**
  - **o corte:** para qualquer instante, inclusive o exato de um Pix, o de logo depois de uma contestação e o de um aparelho novo, as variáveis calculadas com o banco cortado são idênticas às calculadas com o banco inteiro, Pix por Pix;
  - **os exemplos à mão:** dois clientes, seis Pix e uma quadrilha, com cada valor conferido. E as bordas de cada conta: dois Pix no mesmo segundo, a ponta de cada janela, a contestação e o aparelho que chegam no instante do Pix, a madrugada de domingo;
  - **o código:** o pacote das variáveis não importa o simulador nem a gravação, e não toca em nada chamado verdade.

## Consequências

- **O primeiro Pix de cada cliente não tem passado**, e as variáveis do hábito ficam vazias para ele. O modelo da Parte 3 recebe o vazio como vazio, e não como zero.
- **O começo da simulação tem pouco passado**, porque a história começa ali. No primeiro mês, quase todo recebedor é novo. O treino da Parte 3 descarta esse aquecimento.
- **O comércio popular recebe muitos pagadores novos**, como a laranja. A variável sozinha não basta, e é para isso que existem as outras.
- **O cálculo é em lote**, com o Polars: os 2,4 milhões de Pix da simulação padrão em menos de dez segundos. Na Parte 6, o serviço calcula as mesmas variáveis um Pix por vez, e um teste confere que as duas contas dão o mesmo.
