# Antifraude Pix

[![CI](https://github.com/MarcusBomfim/antifraude-pix/actions/workflows/ci.yml/badge.svg)](https://github.com/MarcusBomfim/antifraude-pix/actions/workflows/ci.yml)
[![Licença: MIT](https://img.shields.io/badge/licen%C3%A7a-MIT-blue.svg)](LICENSE)

Detecção de fraude em Pix, do lado do banco de quem paga: o banco tem alguns milissegundos para decidir se cada Pix sai, segura para confirmar ou é bloqueado. O projeto começa por um simulador de transações com os golpes que acontecem no Brasil — o falso parente, a falsa central, a invasão de conta e a venda falsa — e com o rótulo que o banco tem de verdade: a contestação, que chega dias depois e nem sempre chega. Sobre esses dados vêm as variáveis calculadas sem olhar o futuro e, nas próximas partes, os modelos, a decisão pelo custo, a explicação em português e o serviço que pontua em tempo real.

Python 3.13 · uv · NumPy · Polars · Parquet · matplotlib · pytest · hypothesis · ruff · mypy · GitHub Actions

> Projeto independente, de estudo e portfólio. Não tem ligação com nenhum banco nem com o Banco Central. Os dados são de um simulador: não há nenhum dado real de cliente, nem nenhuma transação de verdade.

## O problema

O Pix é instantâneo e irrevogável: depois que sai, o dinheiro só volta se o banco de quem recebeu ainda o encontrar na conta. A decisão de deixar sair é tomada antes, em milissegundos, e erra dos dois lados. Deixar passar o golpe custa o dinheiro da vítima. Segurar o Pix do cliente honesto custa a confiança dele, e o cliente vai para outro banco.

O golpe que mais cresce não é o de quem invade a conta, e sim o de quem convence a vítima a pagar. A vítima faz o Pix do celular dela, de dia, com a senha dela. Do lado do banco, parece um Pix como outro qualquer.

Por isso o problema é difícil, e é o que uma vaga de machine learning em banco cobra:

- **A fraude é rara**: perto de 0,1% dos Pix. Um modelo que nunca acusa nada acerta 99,9%, e não serve para nada.
- **O rótulo chega atrasado.** O banco só sabe que foi golpe quando a vítima reclama: horas depois, na invasão de conta; dias, na venda falsa. Treinar com o que ainda não foi contestado é treinar com golpe marcado como legítimo.
- **O rótulo é sujo.** Parte das vítimas nunca reclama, e há quem conteste um Pix legítimo.
- **O passado não pode ver o futuro.** Cada variável tem de ser calculada só com o que o banco sabia na hora do Pix.
- **O erro custa diferente** para cada lado, e o limiar sai do custo, e não do 0,5.
- **A decisão tem de ser explicada.** A LGPD (art. 20) dá direito à revisão da decisão automatizada.
- **O fraudador muda de tática**, e o modelo envelhece.

## Os dados

Não existe base pública de Pix, por causa do sigilo bancário. Os dados vêm de um simulador próprio, com as suposições escritas e testadas ([ADR-0002](docs/adr/0002-dados-sinteticos.md)).

- **O banco vê o que um banco vê**: o cadastro, os Pix, os aparelhos novos e, dias depois, as contestações. A verdade — qual Pix foi golpe, de que tipo, e qual conta é laranja — fica em outra pasta, para medir. O modelo nunca a lê ([ADR-0004](docs/adr/0004-o-rotulo-atrasado-e-o-contrato-dos-dados.md)).
- **Os golpes são do jeito que acontecem** ([ADR-0003](docs/adr/0003-os-golpes.md)):
  - o falso parente mira os mais velhos, com um a três Pix de valor redondo, de R$ 150 a R$ 3.000, do aparelho da própria vítima;
  - a falsa central liga em horário comercial e esvazia a conta em minutos;
  - a invasão vem de poucos aparelhos da quadrilha, muitas vezes de madrugada, dentro do limite noturno;
  - a venda falsa cresce na Black Friday, com o preço terminado em 90 centavos.
- **A conta laranja** recebe de várias vítimas por alguns dias e some. Quase sempre é uma pessoa; uma em cada cinco é uma empresa de fachada.
- **O legítimo também parece golpe**: troca de celular, paga o autônomo que nunca pagou em valor redondo, ajuda a família, faz Pix de madrugada e, de vez em quando, um Pix grande. Sem isso, o modelo bloquearia todo recebedor novo.
- **A mesma semente dá os mesmos dados, byte a byte.**

A simulação padrão, com a semente 2026:

| | |
|---|---|
| Clientes | 20.000, de julho a dezembro de 2025 |
| Pix | 2.393.510 |
| Pix de golpe | 1.922 (0,08%), em 1.145 ataques |
| Ataques | 358 de venda falsa, 314 de falso parente, 247 de invasão de conta, 226 de falsa central |
| Contas laranja | 204, das quais 41 empresas de fachada |
| Contestados | 1.645 dos Pix de golpe (86%); 277 nunca foram contestados |
| Pix legítimos contestados | 159 |
| Aparelhos novos | 2.656, dos quais 2.409 de clientes que trocaram de celular |

### O retrato

As figuras saem dos dados com `uv run python -m antifraude_pix.analise`.

![A hora do golpe: a falsa central só em horário comercial; a invasão de conta espalhada pela madrugada; o falso parente de dia; a venda falsa parecida com o Pix comum](docs/imagens/horas.png)

A falsa central liga em horário comercial; a invasão de conta acontece a qualquer hora, inclusive de madrugada, quando a vítima dorme. A venda falsa tem a hora do Pix comum: a hora sozinha não a denuncia.

![O tamanho do Pix: a mediana do Pix legítimo é R$ 60; a do falso parente, R$ 560; a da invasão de conta, R$ 935; a da falsa central, R$ 967; a da venda falsa, R$ 345](docs/imagens/valores.png)

O golpe pede mais: de seis a dezesseis vezes a mediana do Pix comum. De madrugada, o golpista da invasão leva o limite noturno inteiro, R$ 1.000, muitas vezes num Pix só.

![Quando o banco fica sabendo: em um dia, três quartos das invasões e das falsas centrais já foram contestadas; a venda falsa demora semanas e para em 55%](docs/imagens/contestacao.png)

A invasão de conta é contestada em horas, e a venda falsa, em semanas — quando é. Um modelo treinado com os dados de ontem ainda não sabe da maior parte da venda falsa da última semana.

![A laranja recebe de desconhecidos: 60% das contas laranja recebem de três ou mais pagadores novos na semana mais movimentada, contra 19% das contas de pessoas comuns; o comércio recebe de muita gente nova](docs/imagens/recebedores.png)

A conta laranja recebe de muita gente que nunca a pagou, em poucos dias. A pessoa comum recebe de quem já a pagava. O comércio também recebe de muita gente nova, e é ele que confunde: o rastro do recebedor ajuda, mas não basta.

## As variáveis

Cada Pix ganha 33 variáveis, calculadas só com o que o banco sabia antes dele ([ADR-0005](docs/adr/0005-a-visao-do-banco-e-o-corte-no-tempo.md), [ADR-0006](docs/adr/0006-as-variaveis-no-ponto-do-tempo.md)). Elas vêm em seis grupos:
- **o Pix**: o valor, a hora, a chave, o valor redondo, a parte do saldo;
- **o cliente**: a idade, a renda, o valor sobre a renda;
- **o hábito do cliente**: os Pix da última hora, o valor típico de 90 dias, quantas vezes o Pix é ele;
- **o cliente e o recebedor**: quantas vezes já pagou, há quanto tempo;
- **o recebedor**, entre todos os clientes do banco: os pagadores novos, as contestações que já tinham chegado;
- **o aparelho**: há quantos minutos entrou, e em quantas contas de outros clientes já tinha entrado.

A figura mede o quanto cada uma, sozinha, separa cada golpe do Pix legítimo. A medida usa a verdade do simulador, de propósito: é uma análise, e o modelo não a lê.

![O que cada variável, sozinha, separa: o valor e o valor fora do hábito chegam a 0,92 em três golpes; o preço terminado em 90 centavos separa toda a venda falsa; o aparelho novo só denuncia a invasão de conta; os pagadores novos do recebedor quase não separam nada sozinhos](docs/imagens/separacao.png)

Cada golpe deixa um rastro diferente:
- **a invasão de conta**: o aparelho que acabou de entrar, e que já esteve em contas de outros clientes. Ele só aparece nela;
- **a venda falsa**: o preço de anúncio;
- **a falsa central**: a pressa, com o Pix da última hora e os poucos minutos desde o anterior;
- **o falso parente**: nenhum rastro próprio. Sobra o valor fora do hábito, e o autônomo também cobra em valor redondo, de alguém que o cliente nunca pagou.

Nenhuma variável basta sozinha, porque o golpe é raro. Para pegar 80% dos golpes só pelo valor, o banco seguraria um Pix legítimo em cada seis: 244 Pix de clientes honestos para cada golpe pego. Juntar as variáveis num modelo e decidir pelo custo são as Partes 3 e 4.

## Roteiro

O projeto é construído em partes, cada uma entregue funcionando, testada e documentada.

| Parte | Conteúdo | |
|--:|---|---|
| 1 | O simulador do Pix: a população, a rotina de cada cliente, os quatro golpes, a contestação atrasada, o contrato dos dados e o retrato | ✅ |
| 2 | As variáveis sem olhar o futuro: o valor fora do hábito, a pressa, o recebedor novo, a conta que recebe de muitos desconhecidos, o aparelho novo — com testes que provam que não há vazamento | ✅ |
| 3 | Os modelos: as regras de hoje como ponto de partida, a regressão logística e o LightGBM; a validação no tempo, com o rótulo que o banco tinha no dia; as métricas para classe rara; a probabilidade calibrada | |
| 4 | A decisão e o porquê: aprovar, segurar para confirmar ou bloquear, pelo custo; a explicação em português de cada decisão; o modelo que não erra mais com os mais velhos | |
| 5 | Os experimentos rastreados no MLflow, o modelo registrado com a sua ficha, e o mesmo processo numa base pública real, de cartões | |
| 6 | A pontuação em tempo real: a API em FastAPI, a latência medida, e o teste de que a API e o treino calculam as mesmas variáveis | |
| 7 | O monitoramento: os dados que mudam, o rótulo que chega depois, o golpe novo, e o retreino comparando o modelo atual com o candidato | |
| 8 | Docker, entrega contínua, OpenTelemetry e o painel | |

## O que já está pronto

**Parte 1 — o simulador do Pix**

- **A população**: os clientes do banco, com idade, estado, renda, conta, limite noturno e o círculo de cada um — os contatos, os comércios preferidos, o aluguel. Do outro lado, as contas que recebem: pessoas, comércios (poucos muito populares, muitos pouco) e as laranjas, ativas de dois a vinte dias, três vezes mais nos bancos digitais.
- **O calendário bancário**: os feriados nacionais, os móveis calculados a partir da Páscoa, os dias úteis e o salário até o quinto dia útil.
- **A rotina de cada cliente**, executada como o banco executa: o salário entra, o aluguel sai, o comércio no almoço e no fim do dia, o Pix sem saldo não sai, e o noturno não passa do limite.
- **Os quatro golpes**, cada um com a sua vítima, a sua hora e a sua mecânica, e a quadrilha da invasão usando os mesmos aparelhos em contas diferentes.
- **A contestação**, com a chance e a demora de cada golpe, dentro do prazo do MED, e os Pix legítimos contestados mesmo assim.
- **O contrato dos dados**: colunas e tipos fixos, categorias fechadas, dinheiro em centavos, instante com o fuso de Brasília; e a validação de fora, que diz todos os problemas de uma vez, em frases.
- **Os dados em Parquet**, o banco numa pasta e a verdade em outra, com o manifesto que guarda a configuração e o SHA-256 de cada arquivo.

**Parte 2 — as variáveis sem olhar o futuro**

- **O simulador mais difícil.** Medir as variáveis mostrou que a primeira versão separava o falso parente só pelo valor. Entraram o autônomo, a ajuda da família, a laranja de fachada e o falso parente que pede pouco (a revisão da [ADR-0003](docs/adr/0003-os-golpes.md)).
- **A visão do banco**, um tipo sem nenhum campo para a verdade:
  - ela tem o cadastro, os Pix, os aparelhos novos e as contestações;
  - a leitura do disco abre só a pasta do banco, e confere o SHA-256 dela;
  - o corte num instante devolve o banco daquele momento, com as contestações que já tinham chegado.
- **Trinta e três variáveis, em seis grupos**, num catálogo com o nome, o tipo e uma frase em português para cada uma.
- **Toda janela no tempo é fechada à esquerda**: o Pix não entra nas contas dele mesmo, nem nas de outro Pix do mesmo segundo.
- **O cálculo em lote**, com o Polars: as variáveis dos 2,4 milhões de Pix da simulação padrão em cerca de seis segundos.
- **A força de cada variável**, sozinha, contra cada golpe: a área sob a curva ROC, com o vazio contado como o menor valor.

## Decisões

As decisões que mudam o desenho do projeto estão registradas em [`docs/adr`](docs/adr):

- [ADR-0001](docs/adr/0001-python-e-as-ferramentas.md) — Python, e as ferramentas que tratam o código de dados como código.
- [ADR-0002](docs/adr/0002-dados-sinteticos.md) — os dados são de um simulador, do lado de quem paga, com a verdade guardada à parte.
- [ADR-0003](docs/adr/0003-os-golpes.md) — os golpes: quatro, do jeito que acontecem, e o legítimo que parece golpe.
- [ADR-0004](docs/adr/0004-o-rotulo-atrasado-e-o-contrato-dos-dados.md) — o rótulo atrasado e sujo, o contrato dos dados, e a verdade em outra pasta.
- [ADR-0005](docs/adr/0005-a-visao-do-banco-e-o-corte-no-tempo.md) — a visão do banco, e o corte no tempo: o que o banco sabia em cada instante.
- [ADR-0006](docs/adr/0006-as-variaveis-no-ponto-do-tempo.md) — as variáveis no ponto do tempo: cada uma com o que o banco sabia antes do Pix.

## Testes

- **O calendário conferido com datas conhecidas e com propriedades sorteadas.** A Páscoa de 2008, 2025 e 2285. A Páscoa sempre num domingo entre 22 de março e 25 de abril, em qualquer ano. O quinto dia útil sempre útil, e com quatro dias úteis antes, em qualquer mês.
- **A população no que ela promete**:
  - os adultos, a renda acima do mínimo, a conta antiga;
  - os contatos e os comércios que existem, sem repetir;
  - o aluguel que cabe na renda;
  - a chave certa para cada tipo de conta;
  - as laranjas mais nos bancos digitais;
  - as idades e os estados nas proporções da população.
- **Os golpes, cada um no seu jeito**:
  - o falso parente e a falsa central mirando os mais velhos;
  - o falso parente pedindo de R$ 150 a R$ 3.000, e o pedido pequeno comum;
  - a falsa central em dia de semana;
  - a venda falsa crescendo na Black Friday;
  - a invasão vindo de poucos aparelhos;
  - o valor redondo, a parte do saldo e o preço de anúncio.
- **A rotina**: a de um cliente sai igual simulada sozinha. O legítimo também troca de celular, paga desconhecidos, faz Pix de madrugada e Pix grande, paga o autônomo como se fosse o falso parente e ajuda a família em valor alto e redondo.
- **A contestação**: o ataque contestado inteiro e de uma vez, cada golpe na proporção dele, a invasão contestada logo e a venda falsa depois de dias.
- **Os dados adulterados**, um para cada regra do contrato — dezenove, do Pix fora do período à contestação depois do prazo do MED — recusados com a frase certa.
- **Qualquer simulação pequena segue o contrato**: de 5 a 60 clientes, de 1 a 40 dias, com qualquer semente, taxa de golpes e data de início, sorteados pelo hypothesis.
- **A gravação**: o que se grava é o que se lê, os mesmos dados dão os mesmos arquivos, e o arquivo mexido depois de gravado é recusado.
- **A visão do banco**:
  - não tem campo para a verdade;
  - o corte guarda só o que veio antes, e deixa de fora a contestação de um Pix antigo feita depois;
  - o instante com fuso é recusado;
  - a leitura funciona com a pasta da verdade apagada;
  - o arquivo do banco mexido é recusado.
- **O vazamento, provado de três jeitos**:
  - **o corte**: as variáveis calculadas com o banco cortado são idênticas às do banco inteiro, Pix por Pix. Vale em qualquer instante, inclusive o exato de um Pix, o de logo depois de uma contestação e o de um aparelho novo, e em minutos sorteados pelo hypothesis;
  - **os exemplos à mão**: dois clientes, seis Pix e uma quadrilha, com cada valor conferido. E as bordas de cada conta: dois Pix no mesmo segundo, a ponta de cada janela, a contestação e o aparelho que chegam no instante do Pix, a madrugada de domingo;
  - **o código**: o pacote das variáveis não importa o simulador nem a gravação, e não toca em nada chamado verdade.
- **Cada rastro, conferido com a verdade**:
  - o aparelho da invasão entrou minutos antes;
  - o aparelho de outras contas só aparece na invasão;
  - o primeiro Pix de cada ataque vai para um recebedor novo;
  - a venda falsa termina em 90 centavos.
- **A força**: a variável que separa tudo vale 1, nos dois sentidos; a que não separa nada, 0,5; o empate vale meio; o vazio é o menor valor.
- **A linha de comando e as figuras**: o que sai, o que é recusado, e as figuras gravadas.

## Como rodar

Pré-requisito: o [uv](https://docs.astral.sh/uv/), que instala o Python certo e as dependências. No Windows:

```bash
winget install --id astral-sh.uv --exact
```

O ambiente, com as versões do `uv.lock`:

```bash
uv sync
```

A simulação padrão, em menos de dois minutos, gravada em `dados/`:

```bash
uv run antifraude-pix simular
```

Outra configuração, por exemplo menor e com outra semente:

```bash
uv run antifraude-pix simular --clientes 2000 --dias 60 --semente 7 --saida dados-pequenos
```

A conferência dos dados gravados:

```bash
uv run antifraude-pix validar
```

As variáveis de cada Pix, calculadas só com o que o banco sabia antes dele, gravadas em `dados/variaveis.parquet`:

```bash
uv run antifraude-pix variaveis
```

As figuras deste README:

```bash
uv run python -m antifraude_pix.analise
```

Os testes, a formatação, o lint e os tipos, como o CI os roda:

```bash
uv run pytest
```

```bash
uv run ruff format --check .
```

```bash
uv run ruff check .
```

```bash
uv run mypy
```

## Estrutura

```text
src/antifraude_pix/
  simulacao/
    configuracao.py     os parâmetros da simulação inteira, cada um com o seu porquê
    calendario.py       os feriados, os dias úteis e o quinto dia útil
    sorteio.py          um fluxo de números para cada etapa e para cada cliente
    populacao.py        os clientes, as contas que recebem e as laranjas
    golpes.py           os quatro golpes: o plano dos ataques e os passos de cada um
    rotina.py           a rotina de cada cliente, executada como o banco executa
    contestacoes.py     o rótulo que chega atrasado
    pix.py              a iniciação e o relógio da simulação
    simulador.py        da semente às tabelas
  dados/
    esquema.py          o contrato: as colunas e os tipos de cada tabela
    validacao.py        o que tem de valer nos dados, conferido de fora
    gravacao.py         o banco numa pasta, a verdade em outra, e o manifesto
    banco.py            a visão do banco, sem a verdade, e o corte num instante
  variaveis/
    catalogo.py         as 33 variáveis: o nome, o grupo, o tipo e a frase de cada uma
    calculo.py          o cálculo em lote, cada variável só com o passado
  analise/
    retrato.py          as contas do retrato de cada golpe
    separacao.py        a força de cada variável, sozinha, contra cada golpe
    figuras.py          as figuras do README
  cli.py                antifraude-pix simular, validar e variaveis
tests/                  o simulador, os dados, as variáveis e as análises
docs/adr/               decisões de arquitetura
docs/imagens/           as figuras do retrato e da força das variáveis
```

## Licença

[MIT](LICENSE).
