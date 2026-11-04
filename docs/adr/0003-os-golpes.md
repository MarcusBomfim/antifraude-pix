# ADR-0003 — Os golpes: quatro, do jeito que acontecem, e o legítimo que parece golpe

**Status:** aceita

## Contexto

O golpe que mais cresce no Pix não é o de quem invade a conta, e sim o de quem convence a vítima a pagar. Nesse caso, a própria vítima faz o Pix, do celular dela, de dia, com a senha dela. Do lado do banco, o Pix parece legítimo, porque foi o cliente quem pediu.

Um simulador que só tivesse o golpe fácil, de aparelho estranho e de madrugada, daria um modelo que acerta no simulador e não pega o golpe que mais importa. Um simulador em que todo legítimo é bem-comportado daria um modelo que bloqueia todo recebedor novo.

## Decisão

- **Quatro golpes**, cada um com a sua mecânica:

  | Golpe | Quem é a vítima | Quando | Como o dinheiro sai |
  |---|---|---|---|
  | Falso parente | sobretudo os mais velhos | de dia | 1 a 3 Pix de valor redondo para a mesma laranja, cada um menor, minutos depois |
  | Falsa central | sobretudo os mais velhos | horário comercial, de segunda a sexta | 2 a 6 Pix em minutos, cada um levando de 30% a 70% do saldo; às vezes pela agência e conta ditadas |
  | Invasão de conta | qualquer um, os jovens um pouco mais | muitas vezes de madrugada | o golpista entra por um aparelho dele, espera alguns minutos e espalha o saldo por 2 a 4 laranjas, dentro do limite noturno |
  | Venda falsa | sobretudo os mais jovens | das 8h às 23h; mais na Black Friday e no Natal | o preço do anúncio, terminado em 90 centavos; às vezes o "frete" depois |

- **A conta laranja** recebe por alguns dias e some. Ela serve a vários golpes no mesmo período, e é isso que a denuncia do lado de quem recebe.
- **A quadrilha da invasão** usa poucos aparelhos, os mesmos em contas de vítimas diferentes.
- **O saldo manda**:
  - o valor pedido, no falso parente e na venda falsa, sai inteiro ou não sai, porque ninguém manda metade da "emergência";
  - a parte do saldo, na falsa central e na invasão, cabe no que há e no limite noturno;
  - abaixo de R$ 50, o golpista não insiste.
- **O legítimo também parece golpe**, de propósito:
  - um em cada oito clientes troca de celular no semestre;
  - um Pix comum em oito vai para uma pessoa que o cliente nunca pagou, e quase um em dez para um comércio novo;
  - há Pix legítimo de madrugada;
  - de vez em quando sai um Pix grande, de R$ 1.500 a R$ 15.000, quase sempre para alguém novo: o carro usado, a reforma.
- **A rotina executa como o banco.** O cliente e o golpista pedem; o banco processa em ordem:
  - o salário entra até o quinto dia útil;
  - o Pix sem saldo não sai;
  - o noturno não passa do limite.

## Consequências

- **O falso parente é difícil, como é de verdade.** O aparelho é o de sempre, a hora é comum e o recebedor é novo, mas o recebedor novo é comum também. O que sobra para o modelo é o valor fora do hábito, a pressa e a laranja que recebe de muitos.
- **Cada golpe deixa um rastro diferente**, e por isso a medida de cada modelo vem separada por golpe, e não só no total.
- **Os pesos são suposições de ordem de grandeza**, escritas no código com o porquê. Mudar a mistura dos golpes é mudar um dicionário, e os testes dizem o que cada golpe promete.

## Revisão, na Parte 2: o simulador estava fácil demais

Na Parte 2, medi o quanto cada variável, sozinha, separa cada golpe do Pix legítimo, pela área sob a curva ROC. A primeira versão do simulador estava fácil demais:
- só o valor do Pix tinha força 0,98 no falso parente, numa escala em que 0,5 é não separar nada e 1 é separar tudo;
- "o recebedor é uma empresa" separava os quatro golpes, porque toda laranja era pessoa física.

Um modelo que acerta tudo num simulador fácil não prova nada. O simulador mudou em quatro pontos, para ficar como o mundo é:
- **o autônomo**: um Pix comum em vinte paga a diarista, o eletricista, o frete. É uma pessoa nova, de dia, por volta de R$ 250, três em quatro vezes em valor redondo. Do lado do banco, é o falso parente;
- **a ajuda da família**: um Pix comum em quarenta vai para um contato, de R$ 200 a R$ 2.000, quase sempre redondo;
- **a laranja de fachada**: uma laranja em cada cinco é uma empresa, o MEI aberto para receber golpe, com CNPJ ou chave aleatória;
- **o pedido pequeno**: o falso parente começa em R$ 150, "é só pra pagar uma conta que venceu hoje", e não mais em R$ 400.

Com isso, um Pix comum em sete vai para uma pessoa que o cliente nunca pagou. A força do valor sozinho no falso parente caiu para 0,92, e a de "o recebedor é uma empresa" caiu de 0,81 para 0,71. Os testes da rotina e dos golpes conferem cada um dos quatro pontos.
