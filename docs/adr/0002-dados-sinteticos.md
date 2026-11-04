# ADR-0002 — Os dados são de um simulador, do lado de quem paga, com a verdade guardada à parte

**Status:** aceita

## Contexto

Não existe base pública de transações Pix. O que um banco sabe de cada Pix — quem pagou, a quem, quanto, de que aparelho — está protegido pelo sigilo bancário (Lei Complementar 105/2001) e pela LGPD. As bases públicas de fraude que existem são de cartão de crédito, europeias e anonimizadas a ponto de as colunas não terem nome, ou de dinheiro por celular na África, também simuladas.

Fraude em Pix tem uma forma própria, que essas bases não têm:
- o golpe em que a própria vítima faz o Pix, do aparelho dela, convencida por alguém, como o falso parente ou a falsa central;
- a conta laranja, que recebe de muitas vítimas em poucos dias;
- o limite noturno;
- a devolução pelo MED, pedida dias depois.

## Decisão

- **Um simulador próprio**, com as suposições escritas.
  - Cada número que é suposição sobre o mundo tem nome e porquê, e não fica solto no código.
  - O que vale para a simulação inteira mora na `Configuracao`: os clientes, os dias, a taxa de golpes, o limite noturno, o prazo do MED.
  - O que é hábito de cada cliente ou mecânica de cada golpe mora ao lado do código que o usa: a frequência e o valor dos Pix na rotina, a mistura e os passos dos golpes nos golpes.
  - As proporções do cadastro são de ordem de grandeza, e não de precisão: as idades, os estados, que seguem o Censo 2022, e os bancos.
- **O Pix é visto do lado de quem paga.** Os clientes são de um banco só, que decide se cada Pix sai. Do outro lado ficam as contas que recebem:
  - as pessoas que o cliente conhece;
  - os comércios onde ele compra, poucos muito populares e muitos pouco;
  - as contas laranja, cada uma ativa de dois a vinte dias, nascidas três vezes mais nos bancos digitais.
- **O que o banco vê fica separado do que só o simulador sabe.**
  - O banco vê o cadastro, o Pix e, dias depois, a contestação.
  - Só o simulador sabe qual Pix foi golpe, de que tipo, e qual conta é laranja.
  - Essa verdade é gravada à parte, para medir o quanto o modelo acerta e o quanto o rótulo do banco erra. O modelo nunca a lê.
- **A mesma semente dá os mesmos dados, byte a byte.**
  - Cada etapa da simulação tira números de um fluxo próprio, derivado da semente e do nome da etapa: a população, os ataques e a rotina de cada cliente.
  - Uma etapa nova não muda o que as outras sorteiam, e a rotina de um cliente não depende da ordem em que os outros são simulados.
- **Os golpes são mais frequentes que na vida real**: dez ataques por mil clientes ao mês, perto de 0,1% das transações. Com a taxa de verdade, vinte mil clientes dariam poucas centenas de fraudes, pouco para treinar e para medir. A taxa é um parâmetro.

## Consequências

- **O modelo aprende o mundo que o simulador supõe.** Uma suposição errada vira um modelo que acerta no simulador e erra no banco. Duas defesas:
  - as suposições estão escritas e testadas;
  - na Parte 5, o mesmo processo roda numa base pública real, de cartões.
- **Os golpes são do jeito que acontecem, e não do jeito que facilitam.**
  - No falso parente, a vítima usa o próprio aparelho, de dia, para um recebedor novo. Do lado de quem paga, isso parece um Pix comum.
  - Do lado legítimo também há o que confunde: o aparelho trocado, o Pix grande do carro usado, o recebedor novo da feira. O modelo não pode ganhar só com o óbvio.
- **Os dados não vão para o repositório.** Saem da semente com um comando, e o Git guarda só o código que os gera.
