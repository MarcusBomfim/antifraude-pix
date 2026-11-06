# ADR-0011 — A idade fora do modelo, e os erros por idade

**Status:** aceita

## Contexto

O golpe mira os mais velhos. O falso parente e a falsa central escolhem a vítima pela idade (ADR-0003). De outubro a dezembro, os clientes de 65 anos ou mais fazem 10,6% dos Pix e sofrem 27,5% dos golpes: o Pix deles é golpe quatro vezes mais que o de quem tem de 18 a 34 anos.

Na Parte 3, a idade era uma das 33 variáveis dos modelos. Um modelo pode errar com os mais velhos de dois jeitos:
- **deixar passar** mais os golpes contra eles, que são os golpes difíceis;
- **segurar ou bloquear** mais o Pix honesto deles, por serem da idade das vítimas.

As duas coisas importam fora do modelo também:
- a LGPD proíbe o tratamento de dados para discriminação ilícita ou abusiva (art. 6º, IX), e deixa a ANPD auditar os aspectos discriminatórios de uma decisão automatizada (art. 20, § 2º);
- o Estatuto da Pessoa Idosa faz crime dificultar o acesso da pessoa idosa a operações bancárias por causa da idade (art. 96).

## Decisão

- **A medida vem primeiro**: os dois erros em cada faixa de idade, com a decisão pelo custo (ADR-0010), em `avaliacao/justica.py`. As faixas são 18 a 34, 35 a 54, 55 a 64, e 65 ou mais.
- **O modelo que via a idade aprendeu a desconfiar do idoso.** O LightGBM, com a idade e sem ela, de outubro a dezembro:

  | | Golpes pegos (com / sem) | Honestos segurados a cada 10 mil (com / sem) | Honestos bloqueados a cada 10 mil (com / sem) |
  |---|--:|--:|--:|
  | 18 a 34 | 91% / 91% | 7,0 / 10,2 | 0,16 / 0,25 |
  | 35 a 54 | 92% / 93% | 11,6 / 13,3 | 0,07 / 0,11 |
  | 55 a 64 | 95% / 94% | 20,0 / 17,7 | 0,58 / 0,39 |
  | 65 ou mais | 94% / 93% | 42,8 / 27,3 | 3,01 / 0,24 |

  Com a idade, o Pix honesto de quem tem 65 anos ou mais era bloqueado 19 vezes mais que o de quem tem de 18 a 34: foram 37 Pix honestos de clientes idosos bloqueados em três meses. Sem a idade, foram 3, na mesma proporção das outras faixas.
- **A idade sai da entrada dos modelos.** Ela continua no cadastro e no catálogo das variáveis: o banco a tem, e é com ela que se mede se o modelo erra mais com alguém. Os modelos recebem as outras 32 variáveis (`PROTEGIDAS` em `modelos/preparo.py`), e um teste confere que mudar a idade de todos os clientes não muda a pontuação de nenhum Pix.
- **O compromisso, medido em cada parte daqui em diante:** contra os mais velhos, o modelo não pega menos golpes que contra os outros, e não bloqueia mais o Pix honesto.

## Consequências

- **Tirar a idade quase não custa.** O custo da decisão subiu R$ 912 em três meses (0,9%), e os golpes pegos dos mais velhos ficaram iguais: 93% sem a idade, contra 94% com ela, dois golpes em 261. A precisão média do LightGBM foi de 0,934 para 0,932, e a da regressão logística, de 0,876 para 0,870.
- **O honesto de 65 anos ou mais ainda é segurado mais**: 27 a cada 10 mil Pix, contra 10 de quem tem de 18 a 34, quase três vezes. A idade saiu, mas o hábito ficou, e o dos mais velhos se parece com o das vítimas: eles fazem menos Pix por mês, e o Pix grande de quem faz pouco Pix fica mais fora do hábito. Igualar exigiria segurar menos os golpes contra eles, que são quatro vezes mais comuns.
- **E cada Pix segurado de um idoso é golpe mais vezes**: 8,5% dos segurados de 65 anos ou mais eram golpe, contra 7,2% dos de 18 a 34. A espera de alguns minutos é o passo brando, feito para a dúvida, e a dúvida é maior onde o golpe é mais comum.
- **A probabilidade fica calibrada no total, e não em cada faixa.** Sem a idade, o modelo previu 231 golpes contra quem tem 65 anos ou mais, onde houve 261, e 238 contra quem tem de 18 a 34, onde houve 227. Quando o golpe é mais comum numa faixa, não dá para ter as duas coisas ao mesmo tempo: o mesmo erro em cada faixa e a probabilidade certa em cada faixa. A escolha foi o mesmo erro onde ele pesa mais, nos golpes pegos e no bloqueio do honesto.
- **Tirar a variável não basta, e por isso a medida fica.** O hábito, a renda e o tempo de conta carregam um pouco da idade. O que garante o compromisso é medir os erros por faixa a cada modelo novo, e não só a lista de variáveis.
