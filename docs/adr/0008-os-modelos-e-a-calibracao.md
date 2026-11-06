# ADR-0008 — Os modelos: as regras de hoje, a regressão logística e o LightGBM, com a probabilidade calibrada

**Status:** aceita

## Contexto

Um modelo de fraude só vale se for melhor que o que o banco já faz, e o que o banco já faz são regras escritas por analistas. Também não basta ordenar os Pix: a Parte 4 decide pesando a chance de golpe contra o valor e contra o custo de segurar um cliente honesto, e por isso precisa de uma probabilidade que seja uma probabilidade.

## Decisão

- **Três modelos, do mais simples ao mais forte**, todos com a mesma entrada (as 33 variáveis da ADR-0006) e a mesma saída (uma pontuação e a probabilidade calibrada).
- **As regras de hoje** são o ponto de partida. São seis regras escritas à mão, cada uma com os seus pontos, e a pontuação é a soma:

  | Regra | Pontos |
  |---|--:|
  | o Pix de mais de R$ 200 sai de um aparelho que entrou na conta há menos de um dia | 3 |
  | o recebedor já teve um Pix contestado | 3 |
  | o Pix vai para quem o cliente nunca pagou e vale cinco vezes o Pix típico dele | 2 |
  | o Pix de mais de R$ 200 leva metade do saldo | 2 |
  | é o terceiro Pix do cliente na mesma hora | 1 |
  | de noite, o Pix leva quatro quintos do limite noturno | 1 |

- **A regressão logística** recebe tudo em número, na mesma escala:
  - os números em logaritmo;
  - o vazio com um indicador ao lado, para o aparelho de sempre não virar o aparelho que acabou de entrar;
  - a hora, o dia da semana e as categorias em casas;
  - tudo padronizado pela média e pelo desvio do treino, e só do treino.
- **O LightGBM** recebe o número como está, o vazio como vazio e a categoria pelo código. Ele aprende com todos os golpes e um legítimo em cada dez, com peso dez: na validação, deu a mesma precisão em metade do tempo. Os fios são fixos, para o mesmo treino dar as mesmas árvores.
- **Os parâmetros foram escolhidos em setembro, o mês de validação**, e nunca olhando o teste. No LightGBM, a precisão média em setembro foi:

  | Parâmetros | Precisão média |
  |---|--:|
  | 31 folhas, 400 árvores | 0,862 |
  | 15 folhas, 300 árvores | 0,880 |
  | 15 folhas, 300 árvores, golpe com peso 10 | 0,827 |
  | 15 folhas, 300 árvores, golpe com peso 30 | 0,745 |
  | **15 folhas, 1.000 árvores mais lentas** | **0,888** |

  Com poucas centenas de golpes, a árvore grande decora, e dar mais peso ao golpe piora. Na logística, a regularização de 0,1, a de 1 e a de 10 deram o mesmo, 0,847, e a de 0,01 deu 0,826. Ficou a de 0,1, a mais forte entre as melhores.
- **A probabilidade é calibrada pelo método de Platt**: uma regressão logística de uma variável só, a pontuação, ajustada em Pix que o modelo não viu.
  - Os Pix da calibração são um em cada cinco dos maduros, de mais de 28 dias antes do corte, em que o rótulo do banco já quase chegou ao que vai ser.
  - O modelo aprende com todo o resto, inclusive os dias mais recentes.
  - É Platt, e não a isotônica, porque a calibração tem poucos golpes (61 no corte de outubro, 183 no de dezembro), e a isotônica, com poucos pontos, vira escada.
- **A probabilidade é a chance de um golpe de que o banco vai ficar sabendo**, porque é com esse rótulo que ela é calibrada. O golpe que ninguém contesta fica fora dela.

## Consequências

- **A probabilidade fecha a conta.** De outubro a dezembro, o LightGBM previu 942 golpes, e houve 948; a logística previu 883.
- **No meio, a probabilidade bate com o que acontece, e nas pontas o modelo é tímido.** Na faixa de 1% a 3%, o LightGBM previu 1,6% e foi golpe 2,0%; na de 10% a 30%, previu 17% e foi golpe 29%. Na Parte 4, a decisão acontece no meio, onde ela bate.
- **As regras ficam mal calibradas**: com seis regras, a soma tem poucos valores, e a reta de Platt não tem o que ajustar.
- **Os modelos são refeitos a cada corte**, e o treino de um mês leva uns 20 segundos nos dados padrão. Os experimentos rastreados e o modelo registrado vêm na Parte 5.
