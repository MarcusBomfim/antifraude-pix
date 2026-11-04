# Antifraude Pix

[![CI](https://github.com/MarcusBomfim/antifraude-pix/actions/workflows/ci.yml/badge.svg)](https://github.com/MarcusBomfim/antifraude-pix/actions/workflows/ci.yml)
[![Licença: MIT](https://img.shields.io/badge/licen%C3%A7a-MIT-blue.svg)](LICENSE)

Detecção de fraude em Pix, do lado do banco de quem paga: o banco tem alguns milissegundos para decidir se cada Pix sai, segura para confirmar ou é bloqueado. O projeto começa por um simulador de transações com os golpes que acontecem no Brasil — o falso parente, a falsa central, a invasão de conta e a venda falsa — e com o rótulo que o banco tem de verdade: a contestação, que chega dias depois e nem sempre chega. Sobre esses dados vêm, nas próximas partes, as variáveis sem vazamento, os modelos, a decisão pelo custo, a explicação em português e o serviço que pontua em tempo real.

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
  - o falso parente mira os mais velhos, com um a três Pix de valor redondo, do aparelho da própria vítima;
  - a falsa central liga em horário comercial e esvazia a conta em minutos;
  - a invasão vem de poucos aparelhos da quadrilha, muitas vezes de madrugada, dentro do limite noturno;
  - a venda falsa cresce na Black Friday, com o preço terminado em 90 centavos.
- **O legítimo também parece golpe**: troca de celular, paga quem nunca pagou, faz Pix de madrugada e, de vez em quando, um Pix grande. Sem isso, o modelo bloquearia todo recebedor novo.
- **A mesma semente dá os mesmos dados, byte a byte.**

A simulação padrão, com a semente 2026:

| | |
|---|---|
| Clientes | 20.000, de julho a dezembro de 2025 |
| Pix | 2.533.947 |
| Pix de golpe | 2.039 (0,08%), em 1.145 ataques |
| Ataques | 358 de venda falsa, 314 de falso parente, 247 de invasão de conta, 226 de falsa central |
| Contas laranja | 204 |
| Contestados | 1.758 dos Pix de golpe (86%); 281 nunca foram contestados |
| Pix legítimos contestados | 104 |
| Aparelhos novos | 2.687, dos quais 2.440 de clientes que trocaram de celular |

### O retrato

As figuras saem dos dados com `uv run python -m antifraude_pix.analise`.

![A hora do golpe: a falsa central só em horário comercial; a invasão de conta espalhada pela madrugada; o falso parente de dia; a venda falsa parecida com o Pix comum](docs/imagens/horas.png)

A falsa central liga em horário comercial; a invasão de conta acontece a qualquer hora, inclusive de madrugada, quando a vítima dorme. A venda falsa tem a hora do Pix comum: a hora sozinha não a denuncia.

![O tamanho do Pix: a mediana do Pix legítimo é R$ 55; a do falso parente e a da invasão de conta, R$ 1.000; a da falsa central, R$ 1.103; a da venda falsa, R$ 379](docs/imagens/valores.png)

O golpe pede mais: perto de vinte vezes a mediana do Pix comum. A mediana da invasão de conta é exatamente R$ 1.000, que é o limite noturno: de madrugada, o golpista leva o limite inteiro num Pix só.

![Quando o banco fica sabendo: em um dia, três quartos das invasões e das falsas centrais já foram contestadas; a venda falsa demora semanas e para em 56%](docs/imagens/contestacao.png)

A invasão de conta é contestada em horas, e a venda falsa, em semanas — quando é. Um modelo treinado com os dados de ontem ainda não sabe da maior parte da venda falsa da última semana.

![A laranja recebe de desconhecidos: 62% das contas laranja recebem de três ou mais pagadores novos na semana mais movimentada, contra 14% das contas de pessoas comuns; o comércio recebe de muita gente nova](docs/imagens/recebedores.png)

A conta laranja recebe de muita gente que nunca a pagou, em poucos dias. A pessoa comum recebe de quem já a pagava. O comércio também recebe de muita gente nova, e é ele que confunde: o rastro do recebedor ajuda, mas não basta.

## Roteiro

O projeto é construído em partes, cada uma entregue funcionando, testada e documentada.

| Parte | Conteúdo | |
|--:|---|---|
| 1 | O simulador do Pix: a população, a rotina de cada cliente, os quatro golpes, a contestação atrasada, o contrato dos dados e o retrato | ✅ |
| 2 | As variáveis sem olhar o futuro: o valor fora do hábito, a pressa, o recebedor novo, a conta que recebe de muitos desconhecidos, o aparelho novo — com testes que provam que não há vazamento | |
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

## Decisões

As decisões que mudam o desenho do projeto estão registradas em [`docs/adr`](docs/adr):

- [ADR-0001](docs/adr/0001-python-e-as-ferramentas.md) — Python, e as ferramentas que tratam o código de dados como código.
- [ADR-0002](docs/adr/0002-dados-sinteticos.md) — os dados são de um simulador, do lado de quem paga, com a verdade guardada à parte.
- [ADR-0003](docs/adr/0003-os-golpes.md) — os golpes: quatro, do jeito que acontecem, e o legítimo que parece golpe.
- [ADR-0004](docs/adr/0004-o-rotulo-atrasado-e-o-contrato-dos-dados.md) — o rótulo atrasado e sujo, o contrato dos dados, e a verdade em outra pasta.

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
  - a falsa central em dia de semana;
  - a venda falsa crescendo na Black Friday;
  - a invasão vindo de poucos aparelhos;
  - o valor redondo, a parte do saldo e o preço de anúncio.
- **A rotina**: a de um cliente sai igual simulada sozinha, e o legítimo também troca de celular, paga desconhecidos, faz Pix de madrugada e Pix grande.
- **A contestação**: o ataque contestado inteiro e de uma vez, cada golpe na proporção dele, a invasão contestada logo e a venda falsa depois de dias.
- **Os dados adulterados**, um para cada regra do contrato — dezenove, do Pix fora do período à contestação depois do prazo do MED — recusados com a frase certa.
- **Qualquer simulação pequena segue o contrato**: de 5 a 60 clientes, de 1 a 40 dias, com qualquer semente, taxa de golpes e data de início, sorteados pelo hypothesis.
- **A gravação**: o que se grava é o que se lê, os mesmos dados dão os mesmos arquivos, e o arquivo mexido depois de gravado é recusado.
- **A linha de comando e o retrato**: o que sai, o que é recusado, e as figuras gravadas.

## Como rodar

Pré-requisito: o [uv](https://docs.astral.sh/uv/), que instala o Python certo e as dependências. No Windows:

```bash
winget install --id astral-sh.uv --exact
```

O ambiente, com as versões do `uv.lock`:

```bash
uv sync
```

A simulação padrão, em cerca de um minuto, gravada em `dados/`:

```bash
uv run antifraude-pix simular
```

Outra configuração, por exemplo menor e com outra semente:

```bash
uv run antifraude-pix simular --clientes 2000 --dias 60 --semente 7 --saida dados-pequenos
```

A conferência dos dados gravados, e as figuras deste README:

```bash
uv run antifraude-pix validar
```

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
  analise/
    retrato.py          as contas do retrato de cada golpe
    figuras.py          as figuras do README
  cli.py                antifraude-pix simular e validar
tests/                  o simulador, os dados e o retrato
docs/adr/               decisões de arquitetura
docs/imagens/           as figuras do retrato
```

## Licença

[MIT](LICENSE).
