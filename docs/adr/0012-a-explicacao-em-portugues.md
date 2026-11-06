# ADR-0012 — A explicação em português de cada decisão

**Status:** aceita

## Contexto

Segurar ou bloquear um Pix afeta o cliente, e a decisão é tomada por máquina. A LGPD (art. 20) dá a ele o direito de pedir a revisão de uma decisão tomada só com base em tratamento automatizado. O § 1º manda o banco dar, quando pedidas, informações claras e adequadas sobre os critérios da decisão, "observados os segredos comercial e industrial".

Quem revisa é o analista de fraude, e ele precisa saber o que pesou, em frases, e não em nomes de colunas. "O LightGBM achou" não é explicação: o modelo tem mil árvores.

## Decisão

- **Cada Pix segurado ou bloqueado sai com as suas razões**: as três variáveis que mais empurraram a pontuação para o golpe, cada uma numa frase com o valor que teve no Pix. O Pix aprovado sai só com a chance de golpe.
- **O quanto cada variável empurrou é a conta do próprio modelo**, em `decisao/explicacao.py`:
  - no LightGBM, os valores de Shapley das árvores, que o próprio LightGBM calcula (`pred_contrib`). Não entrou nenhuma biblioteca nova;
  - na regressão logística, o peso de cada coluna vezes o valor padronizado dela, somado nas colunas da variável: as 24 casas da hora viram a hora, e o indicador do vazio volta para a variável;
  - nas regras de hoje, os pontos de cada regra que valeu. A frase é a da regra.
- **A base mais a soma das contribuições é a pontuação**, nos três modelos, e um teste confere isso até a nona casa. A explicação não é uma história contada depois: é a conta que decidiu.
- **A contribuição é contra o Pix típico.** A base é a pontuação de um Pix comum do treino. Empurrar para o golpe quer dizer que, por causa daquela variável, o Pix tem mais cara de golpe que o Pix comum.
- **A frase de cada variável** está em `decisao/frases.py`. Há uma para cada variável que o modelo recebe, e ela escreve o valor como se fala: o dinheiro em reais, o tempo em minutos, horas ou dias, e o singular e o plural.
- **A idade não tem frase.** Ela não entra no modelo (ADR-0011), e por isso nunca vira razão. Pedir a frase da idade dá erro.
- **Só entra quem empurrou para o golpe**, e duas variáveis que viram a mesma frase contam uma vez só.
- **O formato** é uma frase só: a decisão, a chance de golpe e as razões, da que mais pesou para a que menos pesou.
- **O pacote da decisão não lê a verdade.** Ele roda no banco, Pix a Pix: não importa o simulador, a gravação, as análises nem a avaliação, e um teste confere isso.

## Consequências

- **As explicações, nos dados padrão**, como saem de `antifraude-pix decidir`:
  - um golpe bloqueado: "Bloqueado, com 80% de chance de golpe: 2 Pix para este recebedor já tinham sido contestados como golpe; o recebedor não recebeu nenhum Pix de clientes do banco nas 24 horas anteriores; e é o primeiro Pix do cliente para este recebedor.";
  - um golpe segurado: "Segurado para confirmar com o cliente, com 42% de chance de golpe: 2 Pix para este recebedor já tinham sido contestados como golpe; o recebedor recebeu 2 Pix de clientes do banco nas 24 horas anteriores; e 3 clientes do banco pagaram o recebedor pela primeira vez nos 7 dias anteriores.";
  - um Pix honesto segurado: "Segurado para confirmar com o cliente, com 0,8% de chance de golpe: o Pix é 89% da renda mensal do cliente; é o primeiro Pix do cliente para este recebedor; e o Pix é de R$ 7.100,00.".
- **Cada golpe tem as suas razões.** De outubro a dezembro, a primeira razão mais comum de cada grupo de Pix segurados ou bloqueados foi:

  | | Pix | A primeira razão mais comum |
  |---|--:|---|
  | invasão de conta | 271 | o aparelho que acabou de entrar na conta (85%) |
  | falsa central | 326 | o primeiro Pix para o recebedor, feito minutos antes (56%) |
  | falso parente | 152 | o recebedor já contestado como golpe (72%) |
  | venda falsa | 131 | o recebedor já contestado (70%); depois, o preço terminado em 90 centavos (23%) |
  | honestos | 1.670 | o Pix grande para a renda do cliente (87%) |

  Na invasão de conta, o aparelho está entre as três razões de 95% dos Pix (258 de 271). No segundo Pix para a mesma laranja, a razão mais forte passa a ser a laranja que o cliente pagou minutos antes.
- **O recebedor já contestado é a razão que mais aparece.** A laranja recebe de várias vítimas, e a contestação da primeira denuncia as seguintes. O primeiro golpe para uma laranja nova é o mais difícil de explicar, e de pegar.
- **A explicação explica o modelo, e não o mundo.** "O recebedor não recebeu nenhum Pix nas 24 horas anteriores" vira razão porque o modelo aprendeu que a laranja recém-aberta recebe pouco antes do golpe. É o que pesou, e não a prova; o analista lê assim.
- **As variáveis que dizem quase a mesma coisa dividem o peso**, como o recebedor novo, as vezes que o cliente já pagou e os dias desde o primeiro Pix para o recebedor. Só a frase igual é tirada.
- **O texto é para quem revisa.** O que vai ao cliente é escolha do banco: a lei guarda o segredo comercial, e contar ao golpista o que o denunciou ensina o próximo golpe. Um caminho é dizer ao cliente a decisão, o canal da revisão e as razões sobre o próprio hábito dele, e deixar com o analista as razões sobre o recebedor e o aparelho.
- **Explicar custa pouco**: 0,7 milissegundo por Pix, e só os Pix segurados e bloqueados são explicados, um em cada 450. A latência do serviço inteiro é medida na Parte 6, com a API.
